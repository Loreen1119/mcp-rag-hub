# 第 7 章：FastMCP 工具封装

## 知识点

### 1. 什么是 MCP 协议

MCP（Model Context Protocol）是 Anthropic 发布的开放协议，定义了 AI 应用与外部工具/数据源之间的标准通信方式。
在 MCP 出现之前，每个大模型调用外部工具都讲自己的"方言"，后端必须为不同客户端写胶水翻译代码。MCP 相当于一套"标准英语通信系统"——只要 RAG 按 MCP 规范做成标准插座，任何支持 MCP 的客户端插上即可自动发现工具列表和参数 Schema，零胶水对接。

**核心架构**：Client ↔ Server 模式
- **Server**：暴露 Tools（可调用的函数）、Resources（可读取的数据）、Prompts（预设模板）
- **Client**：Claude Desktop、VS Code 插件、自定义 Agent 等

**为什么用 MCP 而不是自己写 API**：
- 标准化：一套协议适配所有 MCP Client，不用为每个前端写胶水代码
- 可发现：Client 自动获取 Server 的工具列表、参数 Schema
- 认证内置：OAuth 2.0 支持开箱即用

### 2. FastMCP 快速搭建

`FastMCP` 对标的正是 FastAPI 的开发体验——写最纯粹的 Python 函数，贴上 `@mcp.tool` 装饰器，底层自动把函数名、docstring、类型注解翻译成大模型能读懂的 JSON Schema。

```python
from fastmcp import FastMCP

mcp = FastMCP("Server Name", version="1.0.0")

@mcp.tool(description="工具描述会作为 LLM 的 tool description 传给模型")
def my_tool(param: str, count: int = 5) -> list[dict]:
    """函数 docstring 也会被用作描述信息。"""
    ...
```

**关键细节**：
- 类型注解自动映射到 JSON Schema（`str` → string, `int` → number, `Annotated[str, "desc"]` → 参数级 description）
- 返回值可以是 dict、list、str、Pydantic model
- 支持 `async def` 异步工具
- 三种传输模式：`stdio`（进程通信）、`sse`（HTTP/SSE）、`streamable-http`

### 3. 工具粒度设计：细比粗好

一个 RAG Server 应该暴露什么工具？

| 工具 | 用途 | 给谁用 |
|------|------|--------|
| `search_knowledge` | 全管线检索 | LLM Agent 做知识问答 |
| `list_documents` | 查看索引范围 | 确认知识库覆盖了哪些文档 |
| `get_chunk` | 按 ID 取原文 | 验证检索结果、Debug 切片质量 |
| `get_chunk_count` | 快速检查索引规模 | 健康检查、监控 |

**设计原则**：每个工具只做一件事，粒度越细越好。大模型本质上是一个擅长排兵布阵的指挥官（Router Agent）——当它需要数数就调 `get_chunk_count`，需要检索就调 `search_knowledge`。工具职责单一、参数干净，从物理层面掐断由于返回体过大导致大模型看晕、参数填错的可能。

### 4. 管线懒加载：长驻进程的保命设计

MCP Server 是长驻进程。RAG 管线加载模型、建索引代价高（SentenceTransformer 冷加载约 3 秒），如果放在模块 `import` 阶段执行，Server 启动时就会卡顿。客户端连接时要等待 Server 就绪，启动慢会触发连接超时导致握手失败。

因此必须**懒加载**——在第一次 Tool 调用时才初始化，而不是 import 时就跑：

```python
_initialized: bool = False

def _ensure_pipeline():
    global _initialized
    if _initialized:
        return
    # 首次调用：加载模型 + 建索引
    ...
    _initialized = True
```

程序启动时秒开完成轻量级握手，直到 Agent 第一次真正调用工具时才装载重型资产。

**追问应对**：「为什么不在 `if __name__ == "__main__"` 里初始化？」— MCP Server 在 Client 连接时是作为子进程启动的，模块 import 阶段不做重操作可以加快启动速度。而且懒加载保证了无论用哪种 transport（stdio/sse），管线只在真正需要时才构建。

### 5. MCP 与 Streamlit App 的本质区别

| 维度 | Streamlit (Ch5) | FastMCP (Ch7) |
|------|-----------------|---------------|
| 消费方 | 人类的肉眼（浏览器） | AI Agent 的脑子（结构化客户端） |
| 输出格式 | HTML/CSS 可视化 | 结构化 JSON |
| 调用方式 | 人工手动输入查询 | Agent 根据需求自主编排调用 |
| 本质 | 前端精装房（Human-to-AI） | 机房裸接口（AI-to-AI） |

**面试表述**：「同一个核心 RAG 算法过滤漏斗，套上 Streamlit 变成给人用的精美 UI，接上 MCP 变成给 Agent 自主编排的超能力工具。这自证了系统的关注点分离——管线本身不关心消费方是人还是模型，两种暴露方式共享同一套核心逻辑。」

### 面试速记

- **MCP 三个核心概念**：Tools(函数调用)、Resources(数据读取)、Prompts(模板)
- **FastMCP ≈ FastAPI for MCP**：装饰器风格，类型注解自动转 Schema
- **懒加载管线**：MCP Server 是长驻进程，首次 Tool 调用时才初始化模型，避免启动超时
- **3 种传输**：stdio（本地 Agent）、sse（HTTP 长轮询）、streamable-http（HTTP 流式）
- **工具粒度**：细比粗好，LLM 擅长组合小工具

---

## 关键实现与代码走读

> 以下代码节选自实际 `src/mcp_server.py`，注释为讲解用。

### ① 核心检索工具封装

```python
@mcp.tool(description="检索知识库。输入自然语言查询，返回 Cross-Encoder 精排后的 Top-K 结果。")
def search_knowledge(
    query: Annotated[str, "自然语言查询，例如：混合检索策略、RAG 的核心优化方向"],
    top_k: Annotated[int, "返回的结果数量，默认 5"] = 5,
) -> list[dict]:
    """执行完整检索管线：BM25 + 向量 → RRF 融合 → Cross-Encoder 精排。"""
    _ensure_pipeline()        # 懒加载拦截

    bm25_results = _bm25.search(query)
    vector_results = _vector.search(query)
    output = _pipeline.run(bm25_results, vector_results, query, ce_top_k=top_k)

    # 直接 inline 构造返回 dict，控制字段和截断长度
    return [
        {
            "rank": i + 1,
            "content": r.chunk.content[:500],      # 截断 500 字，防止返回体过大
            "score": round(r.score, 4),
            "source_doc": r.chunk.metadata.get("source", "unknown"),
            "chunk_index": r.chunk.metadata.get("chunk_index", -1),
            "headings": r.chunk.metadata.get("heading_breadcrumb", ""),
            "rrf_score": r.chunk.metadata.get("rrf_score"),
            "ce_score": r.chunk.metadata.get("ce_score"),
            "chunk_id": r.chunk.chunk_id,
        }
        for i, r in enumerate(output["cross_encoder"])
    ]
```

**追问应对**：「为什么 `search_knowledge` 返回整个管线结果而不是某一路？」— 因为下游 Agent 需要的是"最相关的、经过质量仲裁的 Top-K 答案"，不是"BM25 版本"或"向量版本"。内部的多路融合策略对调用方完全透明——将来换 reranker 或加新路召回，Tool 签名不变，Client 不受影响。

### ② 数据结构化返回的原则

返回 dict 而非 Chunk 对象。MCP 传输层序列化为 JSON，用 dict 控制字段更精确：
- `content[:500]` 截断，避免返回体过大（LLM 上下文窗口有限）
- 附上 `rank`、`score`、`headings` 辅助信息，帮助 Agent 判断相关性
- `rrf_score` 和 `ce_score` 双分并存，供 Debug 时回溯中间阶段

### ③ 辅助工具实现

```python
@mcp.tool(description="列出知识库中已索引的所有文档及其切片数量。")
def list_documents() -> list[dict]:
    _ensure_pipeline()
    from collections import Counter
    counts = Counter(ch.metadata.get("source", "unknown") for ch in _chunks)
    return [
        {"document": doc, "chunk_count": count}
        for doc, count in sorted(counts.items())
    ]


@mcp.tool(description="根据 chunk_id 获取切片的完整内容和元数据。")
def get_chunk(chunk_id: Annotated[str, "切片唯一标识符 (8 位 hex)"]) -> dict | None:
    _ensure_pipeline()
    for ch in _chunks:
        if ch.chunk_id == chunk_id:
            return {"chunk_id": ch.chunk_id, "content": ch.content, "metadata": ch.metadata}
    return None


@mcp.tool(description="返回知识库中已索引的切片总数。")
def get_chunk_count() -> int:
    _ensure_pipeline()
    return len(_chunks)
```

这四个工具就是微服务积木——大模型 Router 想要数数就调 count，想查原文就调 get_chunk，工具职责单一、参数干净。

---

## 面试话术

**面试官**："我看你的简历里写了用 FastMCP 做了工具封装，为什么要引入 MCP 协议？它的工程价值在哪？"

**回答**："引入 MCP 协议，本质上是在解决大模型时代'多端调用导致的接口方言碎片化'问题。MCP 是全行业通用的标准通信协议——我用 FastMCP 把写好的双路精排 RAG 管线做成了标准的 MCP Server，现在任何一个支持 MCP 的客户端连上来，就能自动发现我的工具列表并直接调用，彻底砍掉了为每个前端写胶水代码的工作。

在设计这个 Server 时，我落地了两个核心工程决策：

**第一是工具的细粒度设计**。我没有搞一个包揽所有功能的巨无霸接口，而是拆成了 `search_knowledge`、`get_chunk_count` 等 4 个极简小工具。这就好比给下游的大模型 Router 提供了精准的微服务积木——想要数数就调 count，想要检索就调 search。参数极度干净，从物理层面掐断了由于返回体过大、字段冗余导致大模型产生幻觉的可能。

**第二是管线懒加载机制**。因为我们的 RAG 管线要加载 SentenceTransformer 模型和 ChromaDB 索引，属于重型操作。MCP Server 在 Client 连接时作为子进程瞬间拉起，如果我把加载逻辑放在 `import` 阶段，启动时就会卡顿，可能导致客户端连接超时。我通过全局状态锁实现了懒加载，让服务启动时秒开完成握手，Agent 第一次真正按下检索键时才在后台装载模型。

最终，这个 MCP Server 与我之前写的 Streamlit 前端形成了完美的关注点分离——Streamlit 是面向人类肉眼的'前端精装房'，MCP 是面向 AI Agent 脑子的'结构化裸接口'。同一套核心 RAG 管线，多端解耦复用。"

---

**面试官**："你的工具返回数据结构是怎么设计的？"

**回答**："返回纯 Python dict，而非内部的 Chunk 对象。因为 MCP 传输层是做 JSON 序列化的——Chunk 对象里包含 SentenceTransformer 的 embedding 向量（384 维浮点数组），塞进 JSON 会导致返回体爆炸。我手工控制字段：`content[:500]` 截断正文、保留 `rrf_score` 和 `ce_score` 双分并存供 Debug 回溯。将来换底层数据结构，Tool 的返回 Schema 不变，Client 完全不受影响。"

---

## 产出文件

- `src/mcp_server.py` — FastMCP Server，暴露 4 个 Tool

## 相关章节

- [[ch05-Streamlit前端]] — Streamlit vs MCP，同一管线两种暴露方式
- [[ch08-LangGraph Agent编排]] — MCP Tool 被 LangGraph Agent 作为检索节点调用
- [[ch04-RRF融合与重排序]] — `search_knowledge` 内部调用的 FusionPipeline
