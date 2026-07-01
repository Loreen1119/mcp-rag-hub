# 第 7 章：FastMCP 工具封装

## 知识点

### 1. MCP 协议是什么

MCP（Model Context Protocol）是 Anthropic 发布的开放协议，定义了 AI 应用与外部工具/数据源之间的标准通信方式。

**核心架构**：Client ↔ Server 模式
- **Server**：暴露 Tools（可调用的函数）、Resources（可读取的数据）、Prompts（预设模板）
- **Client**：Claude Desktop、VS Code 插件、自定义 Agent 等

**为什么用 MCP 而不是自己写 API**：
- 标准化：一套协议适配所有 MCP Client，不用为每个前端写胶水代码
- 可发现：Client 自动获取 Server 的工具列表、参数 Schema
- 认证内置：OAuth 2.0 支持开箱即用

### 2. FastMCP 快速搭建

FastMCP 是 MCP Server 的 Python 高层封装，对标 FastAPI 的开发体验。定义函数 → 加装饰器 → 自动生成 JSON Schema。

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

### 3. 工具粒度设计

一个 RAG Server 应该暴露什么工具？

| 工具 | 用途 | 给谁用 |
|------|------|--------|
| `search_knowledge` | 核心检索 | LLM Agent 做知识问答 |
| `list_documents` | 查看索引范围 | 用户确认知识库覆盖了哪些文档 |
| `get_chunk` | 按 ID 取原文 | 验证检索结果、Debug 切片质量 |
| `get_chunk_count` | 快速检查索引规模 | 健康检查、监控 |

**设计原则**：每个工具只做一件事，粒度细比粗好。
- LLM 更擅长组合细粒度工具，而不是理解大返回体的字段
- 细粒度工具的参数更简单，Schema 更清晰

### 4. 管线懒加载

MCP Server 是长驻进程，模块级初始化在 import 时执行。RAG 管线（加载文档、建索引、加载模型）代价高，应该**懒加载**——在第一次 Tool 调用时才初始化，而不是 import 时就跑。

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

**追问应对**：「为什么不在 `if __name__ == "__main__"` 里初始化？」— MCP Server 在 Client 连接时是作为子进程启动的，模块 import 阶段不做重操作可以加快启动速度。而且懒加载保证了无论用哪种 transport（stdio/sse），管线只在真正需要时才构建。

### 5. MCP 与 Streamlit App 的区别

| | Streamlit (Ch5) | FastMCP (Ch7) |
|---|---|---|
| 交互方 | 人（浏览器） | AI Agent（MCP Client） |
| 输出格式 | HTML/CSS 可视化 | 结构化 JSON |
| 调用方式 | 人工输入查询 | Agent 自动编排调用 |
| 能力边界 | 展示 + 交互 | Tools + Resources + Prompts |

**面试表述**：「第 5 章的 Streamlit 是给人用的 UI，第 7 章的 MCP Server 是给 AI Agent 用的 API。同一个 RAG 管线，两种暴露方式——这体现了关注点分离：管线本身不关心消费方是人还是模型。」

### 面试速记

- **MCP 三个核心概念**：Tools(函数调用)、Resources(数据读取)、Prompts(模板)
- **FastMCP ≈ FastAPI for MCP**：装饰器风格，类型注解自动转 Schema
- **懒加载管线**：MCP Server 是长驻进程，首次 Tool 调用时才初始化模型
- **3 种传输**：stdio（本地 Agent）、sse（HTTP 长轮询）、streamable-http（HTTP 流式）
- **工具粒度**：细比粗好，LLM 擅长组合小工具

## 产出文件

- `src/mcp_server.py` — FastMCP Server，暴露 4 个 Tool

## 关键实现

### ① 核心检索工具

```python
@mcp.tool(description="检索知识库。输入自然语言查询，返回 Cross-Encoder 精排后的 Top-K 结果。")
def search_knowledge(
    query: Annotated[str, "自然语言查询"],
    top_k: Annotated[int, "返回的结果数量，默认 5"] = 5,
) -> list[dict]:
    _ensure_pipeline()
    bm25_results = _bm25.search(query)
    vector_results = _vector.search(query)
    output = _pipeline.run(bm25_results, vector_results, query, ce_top_k=top_k)
    return [_format_result(r, i) for i, r in enumerate(output["cross_encoder"])]
```

**追问应对**：「为什么 search_knowledge 返回整个管线的结果而不是某一路？」— 因为下游 Agent 需要的是"最相关的 Top-K"，而不是"BM25 版本"或"向量版本"。内部的多路融合策略对调用方透明——将来换 reranker 或加新路召回，Tool 签名不变，Client 不受影响。

### ② 数据结构化返回

返回 dict 而非 Chunk 对象。MCP 在传输层序列化为 JSON，用 dict 控制字段更精确：
- `content[:500]` 截断，避免返回体过大（LLM 上下文窗口有限）
- 附上 `rank`、`score`、`headings` 辅助信息，帮助 Agent 判断相关性
