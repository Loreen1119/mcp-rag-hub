"""
FastMCP 工具封装 — 将 RAG 检索管线暴露为 MCP Server。

提供 4 个 Tool:
- search_knowledge: 全管线检索 (BM25 → Vector → Graph → RRF → Cross-Encoder)
- list_documents: 列出已索引的文档清单
- get_chunk: 按 chunk_id 获取切片详情
- get_chunk_count: 返回已索引切片总数

启动方式:
    python src/mcp_server.py          # stdio 模式（供 MCP Client 调用）
    fastmcp run src/mcp_server.py     # FastMCP CLI 启动
"""

from __future__ import annotations

import logging
from typing import List, Annotated

from fastmcp import FastMCP

from config import CE_TOP_K, KG_RRF_WEIGHT, ENABLE_KG
from src.pipeline import get_pipeline

logger = logging.getLogger(__name__)

# ============================================================
# FastMCP Server 实例
# ============================================================

mcp = FastMCP(
    "RAG Knowledge Server",
    version="1.0.0",
    instructions=(
        "RAG 智能知识检索系统 — BM25 + 向量 + 可选图三路融合 + RRF + Cross-Encoder。"
        f"当前 KG 路状态：{'启用' if ENABLE_KG else '关闭'}。"
    ),
)

# ============================================================
# 管线初始化（共享单例，所有 Tool 共享同一批组件）
# ============================================================

_ctx = None


def _ensure_pipeline():
    """懒加载：首次调用时通过 src.pipeline 初始化全部管线组件。"""
    global _ctx
    if _ctx is None:
        _ctx = get_pipeline()
    return _ctx


# ============================================================
# Tool 1: 知识检索
# ============================================================


@mcp.tool(description="BM25+向量+图三路融合检索。输入自然语言查询，返回 Cross-Encoder 精排后的 Top-K 结果。")
def search_knowledge(
    query: Annotated[str, "自然语言查询，例如：混合检索策略、RAG 的核心优化方向"],
    top_k: Annotated[int, "返回的结果数量，默认 5"] = 5,
) -> list[dict]:
    """执行完整检索管线：BM25 + 向量 + 图 → RRF 融合 → Cross-Encoder 精排。"""
    # 参数校验
    if not query or not query.strip():
        raise ValueError("查询不能为空")
    if len(query.strip()) > 500:
        raise ValueError("查询过长（最多 500 字符）")
    if not isinstance(top_k, int) or top_k < 1 or top_k > 50:
        raise ValueError("top_k 必须在 1~50 之间")

    ctx = _ensure_pipeline()
    bm25_results = ctx.bm25.search(query)
    vector_results = ctx.vector.search(query)
    graph_results = ctx.graph.search(query) if ctx.graph else None
    rrf_weights = [1.0, 1.0, KG_RRF_WEIGHT] if ctx.graph else None
    output = ctx.pipeline.run(
        bm25_results, vector_results, query,
        ce_top_k=top_k,
        graph_results=graph_results,
        rrf_weights=rrf_weights,
    )

    return [
        {
            "rank": i + 1,
            "content": r.chunk.content[:500],
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


# ============================================================
# Tool 2: 文档清单
# ============================================================


@mcp.tool(description="列出知识库中已索引的所有文档及其切片数量。")
def list_documents() -> list[dict]:
    """返回已索引文档的清单（来源文件 + 切片数）。"""
    ctx = _ensure_pipeline()

    from collections import Counter
    counts = Counter(ch.metadata.get("source", "unknown") for ch in ctx.chunks)
    return [
        {"document": doc, "chunk_count": count}
        for doc, count in sorted(counts.items())
    ]


# ============================================================
# Tool 3: 获取切片详情
# ============================================================


@mcp.tool(description="根据 chunk_id 获取切片的完整内容和元数据。")
def get_chunk(
    chunk_id: Annotated[str, "切片唯一标识符 (16 位 hex)"],
) -> dict | None:
    """按 chunk_id 查找并返回切片的完整信息。"""
    if not chunk_id or not chunk_id.strip():
        raise ValueError("chunk_id 不能为空")

    ctx = _ensure_pipeline()
    for ch in ctx.chunks:
        if ch.chunk_id == chunk_id:
            return {
                "chunk_id": ch.chunk_id,
                "content": ch.content,
                "metadata": ch.metadata,
            }
    return None


# ============================================================
# Tool 4: 切片总数
# ============================================================


@mcp.tool(description="返回知识库中已索引的切片总数。")
def get_chunk_count() -> int:
    """返回已索引的 Chunk 总数。"""
    ctx = _ensure_pipeline()
    return len(ctx.chunks)


# ============================================================
# 启动入口
# ============================================================


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(message)s")

    # stdio 模式启动（供 MCP Client 如 Claude Desktop 连接）
    mcp.run(transport="stdio")
