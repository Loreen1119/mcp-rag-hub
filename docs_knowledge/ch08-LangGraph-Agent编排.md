# 第 8 章：LangGraph Agent 编排

## 知识点

### 1. 为什么需要 Agent 编排

前几章构建了 RAG 的**管线**（pipeline）—— 固定流程，一步到底。但实际场景中，用户的问题质量参差不齐：

- "RAG 怎么优化" → 直接检索就能找到答案
- "那个啥，就是那个...怎么让 AI 回答更靠谱" → 需要先改写查询再去检索

Agent = 管线 + **决策能力**：根据中间结果决定下一步做什么。

### 2. LangGraph 核心概念

LangGraph 用**有向图**建模 Agent 的决策流程：

| 概念 | 含义 | 类比 |
|------|------|------|
| **State** | 贯穿全图的共享数据容器 | Redux store / 全局变量 |
| **Node** | 图中的一个处理步骤（函数） | pipeline 的一个 stage |
| **Edge** | 节点间的无条件流转 | `A → B` |
| **Conditional Edge** | 根据 State 动态选择下一个节点 | `A → (条件判断) → B or C` |

**State 的更新语义**：

```python
class AgentState(TypedDict):
    query: str                          # 覆盖：每次返回最新值
    retrieved_chunks: Annotated[list, operator.add]  # 累加：多次返回合并
```

- 不带 `Annotated` → **覆盖**（适合单值字段）
- `Annotated[type, operator.add]` → **累加**（适合列表，多次检索结果叠加）

### 3. 本 Agent 的状态图

```
START → analyze_query → retrieve → check_results
                                        ├── [结果充足] → generate_answer → END
                                        └── [结果不足] → rewrite_query → retrieve (循环)
```

五个节点、六个边（含一条条件边），构成了最简单的 RAG Agent 模式：

1. **analyze_query**：初始化状态，记录原始查询
2. **retrieve**：执行完整 RAG 管线（BM25 + Vector → RRF → CE）
3. **check_results**：判断 Top-1 的 CE 分数是否达标
4. **rewrite_query**：LLM 改写查询（口语化 → 书面化、扩展缩写）
5. **generate_answer**：基于检索结果生成最终答案

**关键设计决策**：
- **max_attempts=2**：防止改写循环无限执行（实际生产中改写质量边际递减）
- **CE 阈值=3.0**：ms-marco-MiniLM 的经验值，低于 3 通常不相关
- **operator.add 累加**：改写后的检索结果不会覆盖原始结果，而是合并——第一次检索可能也含有用信息

### 4. 查询改写（Query Rewriting）

这是 RAG Agent 最核心的"智能"体现。用户不会用完美的技术术语提问。

**改写策略**：
| 原始查询 | 改写后 | 策略 |
|----------|--------|------|
| "那个啥，怎么让 AI 不瞎编" | "大语言模型幻觉问题 缓解方法 RAG 检索增强生成" | 口语→术语 |
| "RRF" | "RRF Reciprocal Rank Fusion 排名融合算法" | 扩展缩写 |
| "怎么评价检索好不好" | "信息检索系统评测指标 MRR Hit@K Precision@K" | 补充专业术语 |

**Fallback 设计**：LLM 不可用时（Ollama 未安装），使用规则式改写：将原 query 的关键词 + 同义表达拼接。

### 5. RAG Agent vs 纯 RAG 管线的区别

| | RAG 管线 (Ch4) | RAG Agent (Ch8) |
|---|---|---|
| 流程 | 固定：检索 → 融合 → 重排 | 动态：检索 → 检查 → 可能改写 → 再检索 |
| 输入 | 假设 query 是良好的 | 接受口语化、模糊的 query |
| LLM 角色 | 仅用于重排序 (Cross-Encoder) | 也用于查询改写 + 答案生成 |
| 容错 | 一次检索，不行就返回低分结果 | 多次尝试，自动改写 query |
| 可观测性 | 看分数 | search_log 记录每一步决策 |

**面试表述**：「RAG 管线解决的是'怎么检索'，RAG Agent 解决的是'检索不到怎么办'。条件边是智能的分水岭——从固定流程变成决策流程。」

### 面试速记

- **LangGraph 三要素**：State（数据）、Node（步骤）、Edge（流程控制）
- **条件边是智能关键**：check_results → 结果好走 generate，不好走 rewrite
- **operator.add 累加语义**：多次检索结果合并，不覆盖
- **查询改写**：RAG Agent 最核心的智能——把不专业的查询转成可检索的技术术语
- **Ollama fallback**：LLM 不可用时降级为规则式改写 + 检索拼接，保证系统可用
- **max_attempts**：循环边界，防止无限改写

## 产出文件

- `agent.py` — LangGraph Agent（5 节点 / 6 边 / 1 条件边）

## 关键实现

### ① 条件边

```python
def _decide_next(state: AgentState) -> str:
    if best_score < threshold and attempt < max_attempts:
        return "rewrite_query"
    return "generate_answer"

graph.add_conditional_edges("check_results", _decide_next, {
    "rewrite_query": "rewrite_query",
    "generate_answer": "generate_answer",
})
```

**追问应对**：「条件边的路由函数返回什么？」— 返回目标节点的名字（字符串），LangGraph 根据返回值查路由表，找到对应的 Node。如果路由函数返回了没有映射的字符串，LangGraph 会抛异常。所以 `_decide_next` 和路由表必须一致。

### ② operator.add 累加

```python
retrieved_chunks: Annotated[list[dict], operator.add]
```

**追问应对**：「为什么用 Annotated + operator.add？」— LangGraph 的默认 state 更新是覆盖。如果不加 Annotated，第二次 retrieve 的结果会把第一次的覆盖掉。用 operator.add 实现列表拼接，两次检索结果合并——这很重要，因为改写后的查询可能能找到原始查询漏掉的结果。

### ③ LLM Fallback 降级

```python
def _call_llm(prompt, system=""):
    try:
        import ollama
        response = ollama.chat(model="qwen2.5:7b", messages=messages)
        return response["message"]["content"]
    except Exception:
        return ""  # 调用方做 fallback
```

**追问应对**：「Ollama 不可用时代理还能工作吗？」— 能。查询改写降级为规则式拼接，答案生成降级为检索 Top-3 的原文拼贴。虽然质量不如 LLM 模式，但系统骨架不依赖外部服务。这在生产环境非常重要——LLM 是 best-effort 的增强，不是硬依赖。
