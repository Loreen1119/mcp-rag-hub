# 在 Dify 中复现 RAG 管线：从部署到分块调优的一次实战

> 日期：2026-07-31  
> 关键词：Dify、RAG、知识库、分块策略、混合检索、调优

## 背景

`mcp-rag-hub` 是我从零手写的一个 RAG 评测与实现项目，已经跑通了 BM25 + 向量 + 实体共现图三路召回、RRF 融合、Cross-Encoder 重排的完整管线，自建评测集上 MRR 0.81、Hit@5 100%。

但我一直是从"建造者"视角在看这套系统——我写代码、我跑评测、我看指标。这次我想换一个角度：**用现成的工具（Dify）把同样的能力复现一遍**，从"使用者"视角验证自己写的管线在工程层面是否真的合理。

## 部署 Dify

本地 Docker 部署，遇到第一个坑：C 盘空间爆满（剩几百 MB）把 Docker Desktop 的 WSL2 引擎挤崩了，daemon 反复卡在 "Engine starting"。清出 8 GB 后才恢复。

教训：**Docker Desktop 对磁盘空间极其敏感**，C 盘要留足余量。镜像数据目录一定要配到非系统盘（我配到 D 盘）。

## 配置模型供应商

Dify 的知识库问答需要两类模型：

- **Chat 模型**：DeepSeek `deepseek-chat`（生成答案）
- **Embedding 模型**：通义千问 `text-embedding-v3`（向量化）

DeepSeek 没有自家的 embedding 服务，所以 embedding 必须单独配。我选了阿里云百炼的 `text-embedding-v3`，免费额度足够测试。

## 第一次跑通：README 问答

把项目 `README.md` 导入知识库，参数：

- 分段规则：通用
- 分段最大长度：1024
- 分段重叠：100
- Embedding：text-embedding-v3
- 检索：混合检索（向量 + 全文）+ qwen3-rerank
- Top K：5

问"mcp-rag-hub 的核心特性是什么？"，DeepSeek 准确答出了 7 个特性（多路召回、RRF、Cross-Encoder、LangGraph、FastMCP、评测体系），耗时 8.74 s、404 tokens。

链路通了。但这只是 Markdown 文档——README 有清晰的章节结构，分块天然友好。

## 第二次踩坑：代码文件分块切散

把 `src/retrievers.py` 和 `src/fusion.py` 重命名为 `.txt` 导入（Dify 不支持 `.py`），同样参数处理。

问"retrievers.py 里 BM25 和向量检索是怎么并行执行的？"——AI 直接回答\*\*"没有足够的信息"\*\*。

这不对。代码就在知识库里，为什么召回到不到？

### 召回测试定位

在知识库的"召回测试"里搜 `ThreadPoolExecutor` 和 `asyncio`，结果触目惊心：

| 查询                   | Top 1 召回片段 | 长度   | Score |
| -------------------- | ---------- | ---- | ----- |
| `ThreadPoolExecutor` | `def run(` | 8 字符 | 0.50  |
| `asyncio`            | `}`        | 1 字符 | 0.46  |

召回回来的全是几字符的代码碎片（`def run(`、`continue`、`}`、`bash`），完全失去上下文。

### 根因

按段落（`\n\n`）切分代码时，Python 代码里的空行（函数之间、类之间）被当成段落边界，一段完整的 `def xxx():\n    ThreadPoolExecutor...\n    asyncio.gather...` 被切成了多个无意义的小块：

- 块 1：`def run(`
- 块 2：`ThreadPoolExecutor(max_workers=3) as executor:`
- 块 3：`continue`
- 块 4：`}`

每个小块单独看都没有语义，向量化和 BM25 都无法准确匹配。

## 调优方案

两手并进：

### 1. 加文件级描述

在每个 `.txt` 开头加一段文件级注释，让 RAG 能拿到整文件的语义摘要：

```python
# File: retrievers.py
# 实现多路召回：BM25 + 向量检索 + 实体共现图，三路并行执行。
# 并行机制：使用 ThreadPoolExecutor（线程池）或 asyncio.gather（协程），
# 具体取决于 HybridRetriever 的实现。详见 fusion.py 的 RRF 融合。
```

### 2. 加大分段块

- 分段最大长度：1024 → **2048**
- 分段重叠：100 → **200**

让一段完整的函数实现尽量落在同一个块里。

### 验证

重新导入后，问同样的问题"BM25 和向量检索怎么并行执行？"——这次回答准确：

- ✅ 准确说出"双路召回"概念
- ✅ 区分 BM25 路（精确字面匹配）和 向量语义路
- ✅ 描述完整并行流程：同一查询 → 两路引擎 → 各取 top-K → RRF 融合
- ✅ 主动提到可以扩展到三路召回（知识图谱）

耗时 6.25 s、440 tokens。

## 经验总结

这次实战让我对 RAG 工程化有了几个新认识：

1. **分块策略对代码文件远比对 Markdown 敏感**。Markdown 的章节结构天然适合按段落切，但代码的"段落"（空行分隔）和"语义单元"（一个完整函数）不是一回事。
2. **召回测试是 RAG 调优的第一工具**。不要急着改 Chatbot 的提示词，先看召回回来的是什么——如果召回的就是垃圾，再好的 LLM 也救不回来。
3. **文件级描述是低成本高收益的优化**。一段几十字的注释，让向量化和 BM25 都能拿到文件整体的语义锚点，比改分块算法见效快。
4. **从使用者视角验证自己的系统是有价值的**。我用 Dify 复现了一遍自己写的管线，体感上能直接对上——这说明 mcp-rag-hub 的架构选择（多路召回 + RRF + 重排）在工程层面是站得住的。

## 后续调优：Top K 与 Agent

当天晚上继续测了两项：Top K 调参和 Agent 工具调用。

### Top K 5 → 7：效果提升，成本可控

把 Chatbot 关联知识库的 **Top K 从 5 调到 7**，用同一个问题"混合检索和 RRF 重排序是怎么实现的？"做对照：

| Top K | 回答质量 | 耗时 | Token 花费 |
|---|---|---|---|
| 5 | 答出 RRF 两阶段流程 | 8.74 s | 404 |
| 7 | 额外给出 RRF 公式、\(`k=60\) 默认值、Cross-Encoder 仅对 top K 候选重排 | 8.37 s | 488 |

Top K=7 召回了更多关键片段（包括 RRF 公式本身），回答明显更完整；Token 成本只涨了 21%，效率上没有明显损失。最终把该知识库的默认 Top K 定为 **7**。

### Agent 搜索工具：限流导致成本爆炸

在 Dify 里给同一个应用加了 WebSearch/DuckDuckGo 搜索工具，测试了两个问题：

| 问题 | 结果 | 耗时 | Token 花费 |
|---|---|---|---|
| "mcp-rag-hub 项目最近有什么更新？" | 搜索工具反复重试后均限流，最终 fallback 到知识库 README 回答 | **47.82 s** | **14,392** |
| "2026 年 7 月 31 日广州天气怎么样？" | 同样搜索限流，LLM 用内部知识给出广州 7 月底气候参考 | 16.11 s | 7,848 |

14k tokens 已经接近几十轮纯 RAG 问答的成本。根因是搜索工具遇到限流后，Agent 反复重试（"已深度思考"出现了 5-6 次）。

**临时结论**：免费/默认搜索工具不适合生产测试。下一步要么换成带稳定 API key 的搜索服务，要么先用本地工具（如 Calculator）验证 Agent 的工具调用逻辑，避免无限重试烧 token。

## 当天落地：Python 代码 AST 分块

文章写到一半没有停。基于 Dify 里发现的"代码被通用滑窗切散"问题，我直接给 `mcp-rag-hub` 写了一个 AST 分块方案并让 Cursor 生成实现：

- `src/data_pipeline.py` 新增 `chunk_by_ast()`：按函数/类/模块级边界切分 Python 源码
- `process_document()` 增加文件类型路由：`.md` → 标题面包屑 + 滑窗，`.py` → AST 分块，`.pdf/.txt` → 通用滑窗兜底
- `load_document()` 支持 `.py` 走文本加载
- `process_directory()` 支持 `.py`
- metadata 附加 `source_type`（ast / fallback）、`start_line`、`end_line`
- 类拆分条件没有按原方案用 token 阈值，而是改成 **"方法数 >= 2 就拆"**——因为默认 `CHUNK_SIZE=256` token 偏小，按 token 很难触发拆分，按方法数更实用
- 新增 `tests/test_data_pipeline.py`，**11/11 测试通过**

这次把"使用者视角的痛点"直接转化成了"建造者视角的代码改进"——Dify 里踩的坑，反哺了项目本身。

## 后续

- 尝试**父子分段**（Parent-Child Retrieval），让检索精度和上下文完整性兼得
- 用实现了 AST 分块的 `mcp-rag-hub` 重新构建 Dify 知识库，验证代码问答质量是否进一步提升
- 用 Dify 的 Agent 功能时，先配置稳定的工具 API key，并限制最大迭代次数，防止限流后无限重试
