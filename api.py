"""FastAPI 服务层 —— 把 RAG 管线暴露成 HTTP 接口 + 自动 OpenAPI 文档。

**定位**：仓库原本只有三个入口 —— Streamlit 界面（`app.py`）、LangGraph Agent
（`agent.py` CLI）、MCP Server（`src/mcp_server.py`，stdio）。缺一个**标准的 HTTP
接口**：既方便第三方/前端接入，也让 `/openapi.json` 成为机器可读的接口契约。

启动：
    uvicorn api:app --host 127.0.0.1 --port 8000
    # 或
    python api.py

文档：
    http://127.0.0.1:8000/docs      （Swagger UI）
    http://127.0.0.1:8000/openapi.json

设计约束（每一条都对应一个踩过的坑，改之前先读）：
1. **`/health` 不许构建管线、不许加载模型。** `get_pipeline()` 冷启动要跑
   `process_directory` + BM25 + embedding 模型，实测量级约 2 分钟。健康检查挂在
   它身上，外部看到的只会是「服务死了」。所以 `/health` 只读 config 常量 + 直接
   查 Chroma 集合的 `count()`，并对结果做 60s 缓存。
2. **`/documents` 默认只回聚合摘要。** `corpora/tidb` 有 9 千级文件，全量清单是
   接近 MB 的 JSON。默认按来源目录聚合，文件清单分页且 `limit` 硬上限 200。
   逐来源的 chunk 数需要扫全量 metadata（10 万行），故只在显式
   `?with_chunk_counts=true` 时计算，并缓存 5 分钟。
3. **`/search` 默认 `mode=rrf`。** `mode=ce` 会懒加载 bge-reranker-base（约 1.1GB，
   首次 30~60 秒），2G 内存机器会直接 OOM —— 它是显式选项，不是默认。
4. **handler 内惰性 import。** 模块顶层只 import 标准库 / fastapi / pydantic /
   config。`agent` / `src.pipeline` 一律在函数体内 import，否则
   `tests/test_api.py` 一被收集就会拉起 torch / chromadb / langgraph，CI 变慢且脆。
5. **可注入。** `/search` 走 `get_ctx()`、`/query` 走 `query_runner()` 两个依赖，
   测试用 `app.dependency_overrides` 换成假实现，不需要建索引、不需要 Ollama。
   测试收尾必须调 `src.pipeline.reset_pipeline()`（模块级单例，不 reset 会串味）。
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Callable, Literal, Optional

# 服务路径 CE 超时：取 CE P95 7973ms ≈ 2 倍。必须在 import config 之前设，
# 否则 config 已读完 env 再改无效。setdefault 允许用户显式覆盖。
os.environ.setdefault("MCP_RAG_CE_TIMEOUT_MS", "15000")

from fastapi import Depends, FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from config import (
    BM25_TOP_K,
    CE_CANDIDATE_K,
    CE_TOP_K,
    CHROMA_PERSIST_DIR,
    CORPUS,
    CROSS_ENCODER_MODEL,
    DOCS_DIR,
    EMBEDDING_MODEL,
    ENABLE_KG,
    ENABLE_RERANKER,
    KG_RRF_WEIGHT,
    LLM_BACKEND,
    LLM_MODEL,
    RRF_K,
    VECTOR_TOP_K,
)

logger = logging.getLogger(__name__)

DOC_SUFFIXES = (".md", ".txt", ".pdf", ".py")
"""`/documents` 认定的文档后缀，与 data_pipeline 的解析范围保持一致。"""

DOC_LIST_MAX_LIMIT = 200
"""`/documents` 文件清单单页硬上限 —— 防止一次吐出近 MB 的 JSON。"""

_CHUNK_COUNT_TTL = 60.0
_chunk_count_cache: dict = {"at": 0.0, "value": None, "status": "missing"}

_SOURCE_CHUNK_TTL = 300.0
_source_chunk_cache: dict = {"at": 0.0, "value": None}

_SCAN_TTL = 60.0
_scan_cache: dict = {"at": 0.0, "value": None}
"""语料目录扫描结果缓存：本机 9 千级文件走一次 `rglob` 约 2 秒，没必要每次请求都扫。"""

app = FastAPI(
    title="MCP-RAG-Hub API",
    version="1.0.0",
    description=(
        "混合检索 RAG 系统的 HTTP 服务层。\n\n"
        "- `/query`：完整 agent 链路（检索 → 质量判断 → 必要时改写 → 生成 → 校验），"
        "返回三态 `answered / insufficient_evidence / generation_unavailable`。\n"
        "- `/search`：只走检索层，可单独选 `bm25 / vector / rrf / ce` 四档之一。\n"
        "- `/health`：**不加载模型**的轻量探针。\n"
        "- `/documents`：语料清单，默认按来源聚合。\n\n"
        "口径说明：本服务不产生新的评测数字，所有指标口径以 `EVAL_REPORT.md` 为准；"
        "生成层三维是裁判自评、未经人工校准，仅作内部参考。"
    ),
)


# ============================================================
# 依赖（测试用 app.dependency_overrides 注入假实现）
# ============================================================


def get_ctx():
    """返回管线上下文（懒加载单例）。**这是最慢的一步**，冷启动约 2 分钟。

    惰性 import：模块顶层 import `src.pipeline` 会连带拉起 chromadb /
    sentence-transformers，测试收集阶段就会变慢。
    """
    from src.pipeline import get_pipeline

    return get_pipeline()


def query_runner() -> Callable[[str], dict]:
    """返回 agent 的单次查询函数（可注入，测试不必起 Ollama）。"""
    from agent import run_query

    return run_query


# ============================================================
# 响应模型
# ============================================================


class RetrievalConfig(BaseModel):
    bm25_top_k: int
    vector_top_k: int
    rrf_k: int
    ce_candidate_k: int
    ce_top_k: int


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"] = Field(
        description="`ok` = 索引集合可读；`degraded` = 索引缺失或不可读（服务本身仍在跑）"
    )
    corpus: str
    index_status: Literal["ready", "missing", "unavailable"]
    chunk_count: Optional[int] = Field(None, description="索引集合的 chunk 总数；不可读时为 null")
    kg_enabled: bool
    reranker_enabled: bool = Field(description="`MCP_RAG_ENABLE_RERANKER`，关掉可省约 1.1GB 内存")
    llm_backend: str = Field(description="`ollama` 或 `deepseek`")
    llm_model: str
    embedding_model: str
    cross_encoder_model: str
    retrieval: RetrievalConfig


class Hit(BaseModel):
    rank: int
    source: str = Field(description="文档相对路径（如 `docs-cn/xxx.md`）")
    headings: str = ""
    score: float
    chunk_id: str = ""
    content: str
    truncated: bool = Field(description="正文是否被截断到 300 字（接口只回片段，不回全文）")


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    mode: Literal["rrf", "bm25", "vector", "ce"] = Field(
        "rrf",
        description=(
            "检索档位。`rrf` = 融合后（默认，不加载 reranker）；`bm25` / `vector` = 单路召回；"
            "`ce` = Cross-Encoder 精排，**会懒加载约 1.1GB reranker，首次 30~60 秒**，"
            "内存 < 4G 的环境不要用"
        ),
    )
    top_k: int = Field(5, ge=1, le=50)


class SearchResponse(BaseModel):
    query: str
    mode: str
    count: int
    elapsed_ms: int
    kg_enabled: bool
    hits: list[Hit]


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)


class QueryResponse(BaseModel):
    query: str
    status: str = Field(
        description="`answered` / `insufficient_evidence` / `generation_unavailable` / `invalid_output`"
    )
    answer: str
    used_source_ids: list[int]
    reason: Optional[str] = None
    evidence: list[Hit]
    attempt: int = Field(description="检索轮次（含查询改写后的重试）")
    rewritten_queries: list[str]
    elapsed_ms: int


class SourceSummary(BaseModel):
    source: str = Field(description="来源目录（`source` 元数据的第一段路径）")
    doc_count: int
    chunk_count: Optional[int] = Field(
        None, description="仅在 `with_chunk_counts=true` 时返回，否则为 null"
    )


class DocumentsResponse(BaseModel):
    corpus: str
    total_docs: int
    total_chunks: Optional[int]
    sources: list[SourceSummary]
    documents: list[str] = Field(description="当前页的文件相对路径")
    limit: int
    offset: int
    has_more: bool
    note: str = ""


# ============================================================
# 轻量索引探测（不加载模型）
# ============================================================


def _index_status() -> tuple[Optional[int], str]:
    """读 Chroma 集合的 chunk 总数。**不构建管线、不加载 embedding 模型。**

    只 import chromadb（几秒）与 `src.index_meta`（纯函数）。失败一律降级成
    `(None, "unavailable")`，绝不让健康检查抛异常。
    """
    now = time.monotonic()
    if now - _chunk_count_cache["at"] < _CHUNK_COUNT_TTL:
        return _chunk_count_cache["value"], _chunk_count_cache["status"]

    if not CHROMA_PERSIST_DIR.exists():
        _chunk_count_cache.update(at=now, value=None, status="missing")
        return None, "missing"

    try:
        import chromadb

        from src.index_meta import corpus_collection_name

        client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
        collection = client.get_collection(corpus_collection_name(CORPUS, EMBEDDING_MODEL))
        count = int(collection.count())
        _chunk_count_cache.update(at=now, value=count, status="ready")
        return count, "ready"
    except Exception as exc:  # noqa: BLE001 - 健康检查不允许抛
        logger.warning("[health] 索引探测失败: %s", exc)
        _chunk_count_cache.update(at=now, value=None, status="unavailable")
        return None, "unavailable"


def _source_chunk_counts() -> dict[str, int]:
    """扫全量 metadata 统计每来源的 chunk 数。**昂贵**，故只在显式开启时调用。

    10 万级行数，秒级耗时 + 数百 MB 峰值内存；结果缓存 5 分钟。
    """
    now = time.monotonic()
    if _source_chunk_cache["value"] is not None and now - _source_chunk_cache["at"] < _SOURCE_CHUNK_TTL:
        return _source_chunk_cache["value"]

    import chromadb

    from src.index_meta import corpus_collection_name

    client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
    collection = client.get_collection(corpus_collection_name(CORPUS, EMBEDDING_MODEL))
    got = collection.get(include=["metadatas"])

    counts: dict[str, int] = {}
    for meta in got.get("metadatas") or []:
        source = str((meta or {}).get("source", "?"))
        key = source.split("/", 1)[0] if "/" in source else "(root)"
        counts[key] = counts.get(key, 0) + 1

    _source_chunk_cache.update(at=now, value=counts)
    return counts


def _scan_corpus() -> list[str]:
    """扫描语料目录，返回相对路径清单（纯文件系统，不加载任何模型）。"""
    now = time.monotonic()
    if _scan_cache["value"] is not None and now - _scan_cache["at"] < _SCAN_TTL:
        return _scan_cache["value"]

    if not DOCS_DIR.exists():
        _scan_cache.update(at=now, value=[])
        return []

    files = [
        p.relative_to(DOCS_DIR).as_posix()
        for p in DOCS_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in DOC_SUFFIXES
    ]
    files.sort()
    _scan_cache.update(at=now, value=files)
    return files


def _hit_from_agent_chunk(chunk: dict) -> Hit:
    """agent 返回的 chunk 已在 `retrieve` 阶段截断到 500 字，这里只做字段映射。"""
    content = chunk.get("content", "")
    return Hit(
        rank=int(chunk.get("rank", 0) or 0),
        source=str(chunk.get("source_doc", "?")),
        headings=str(chunk.get("headings", "") or ""),
        score=float(chunk.get("score", 0.0) or 0.0),
        chunk_id=str(chunk.get("chunk_id", "") or ""),
        content=content,
        truncated=len(content) >= 500,
    )


def _hit_from_result(result, rank: int) -> Hit:
    """把融合层的结果对象转成响应模型（正文截断 300 字）。"""
    content = getattr(result.chunk, "content", "") or ""
    meta = getattr(result.chunk, "metadata", {}) or {}
    return Hit(
        rank=rank,
        source=str(meta.get("source", "?")),
        headings=str(meta.get("heading_breadcrumb", "") or ""),
        score=float(getattr(result, "score", 0.0) or 0.0),
        chunk_id=str(getattr(result.chunk, "chunk_id", "") or ""),
        content=content[:300],
        truncated=len(content) > 300,
    )


# ============================================================
# 路由
# ============================================================


@app.get("/health", response_model=HealthResponse, summary="健康检查（不加载模型）")
def health() -> HealthResponse:
    chunk_count, index_status = _index_status()
    return HealthResponse(
        status="ok" if index_status == "ready" else "degraded",
        corpus=CORPUS,
        index_status=index_status,
        chunk_count=chunk_count,
        kg_enabled=ENABLE_KG,
        reranker_enabled=ENABLE_RERANKER,
        llm_backend=LLM_BACKEND,
        llm_model=LLM_MODEL,
        embedding_model=EMBEDDING_MODEL,
        cross_encoder_model=CROSS_ENCODER_MODEL,
        retrieval=RetrievalConfig(
            bm25_top_k=BM25_TOP_K,
            vector_top_k=VECTOR_TOP_K,
            rrf_k=RRF_K,
            ce_candidate_k=CE_CANDIDATE_K,
            ce_top_k=CE_TOP_K,
        ),
    )


@app.get("/documents", response_model=DocumentsResponse, summary="语料清单（默认聚合摘要）")
def documents(
    limit: int = Query(50, ge=1, le=DOC_LIST_MAX_LIMIT, description="文件清单单页条数"),
    offset: int = Query(0, ge=0),
    with_chunk_counts: bool = Query(
        False, description="是否逐来源统计 chunk 数（需扫全量 metadata，秒级 + 数百 MB 内存）"
    ),
) -> DocumentsResponse:
    files = _scan_corpus()
    grouped: dict[str, int] = {}
    for rel in files:
        key = rel.split("/", 1)[0] if "/" in rel else "(root)"
        grouped[key] = grouped.get(key, 0) + 1

    counts: dict[str, int] = {}
    if with_chunk_counts:
        total, _ = _index_status()
        if total is not None and total > 200_000:
            raise HTTPException(
                status_code=400,
                detail=f"索引 {total} chunk 超过 20 万，逐来源统计会吃数百 MB 内存；请用 /health 的总额或用 --detail 脚本离线统计",
            )
        counts = _source_chunk_counts()

    sources = [
        SourceSummary(source=key, doc_count=doc, chunk_count=counts.get(key))
        for key, doc in sorted(grouped.items(), key=lambda kv: -kv[1])
    ]
    page = files[offset : offset + limit]
    total_chunks, _ = _index_status()

    return DocumentsResponse(
        corpus=CORPUS,
        total_docs=len(files),
        total_chunks=total_chunks,
        sources=sources,
        documents=page,
        limit=limit,
        offset=offset,
        has_more=offset + limit < len(files),
        note="文件名清单分页返回，单页上限 200 条；chunk_count 需 with_chunk_counts=true",
    )


@app.post("/search", response_model=SearchResponse, summary="检索层（可选 bm25/vector/rrf/ce）")
def search(payload: SearchRequest, ctx=Depends(get_ctx)) -> SearchResponse:
    started = time.perf_counter()
    query = payload.query

    if payload.mode == "bm25":
        results = ctx.bm25.search(query)
        hits = [_hit_from_result(r, i + 1) for i, r in enumerate(results[: payload.top_k])]
    elif payload.mode == "vector":
        results = ctx.vector.search(query)
        hits = [_hit_from_result(r, i + 1) for i, r in enumerate(results[: payload.top_k])]
    else:
        bm25_results = ctx.bm25.search(query)
        vector_results = ctx.vector.search(query)
        graph_results = ctx.graph.search(query) if ctx.graph else None
        output = ctx.pipeline.run(
            bm25_results,
            vector_results,
            query,
            ce_top_k=payload.top_k if payload.mode == "ce" else CE_TOP_K,
            graph_results=graph_results,
            rrf_weights=[1.0, 1.0, KG_RRF_WEIGHT] if ctx.graph else None,
        )
        key = "cross_encoder" if payload.mode == "ce" else "rrf"
        results = output.get(key) or []
        hits = [_hit_from_result(r, i + 1) for i, r in enumerate(results[: payload.top_k])]

    return SearchResponse(
        query=query,
        mode=payload.mode,
        count=len(hits),
        elapsed_ms=int((time.perf_counter() - started) * 1000),
        kg_enabled=bool(ctx.graph),
        hits=hits,
    )


@app.post("/query", response_model=QueryResponse, summary="完整 agent 链路（检索→改写→生成→校验）")
def query(
    payload: QueryRequest,
    runner: Callable[[str], dict] = Depends(query_runner),
) -> QueryResponse:
    started = time.perf_counter()
    try:
        result = runner(payload.query)
    except RuntimeError as exc:  # 语料为空 / 管线初始化失败
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    answer_data = result.get("answer", {})
    if not isinstance(answer_data, dict):
        answer_data = {
            "status": "invalid_output",
            "answer": "",
            "used_source_ids": [],
            "reason": "invalid_output",
        }

    chunks = result.get("last_retrieved_chunks", []) or []
    return QueryResponse(
        query=str(result.get("query") or payload.query),
        status=str(answer_data.get("status") or ""),
        answer=str(answer_data.get("answer") or ""),
        used_source_ids=[int(x) for x in (answer_data.get("used_source_ids") or [])],
        reason=answer_data.get("reason"),
        evidence=[_hit_from_agent_chunk(c) for c in chunks],
        attempt=int(result.get("attempt") or 0),
        rewritten_queries=[str(q) for q in (result.get("rewritten_queries") or [])],
        elapsed_ms=int((time.perf_counter() - started) * 1000),
    )


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    # 默认 8010：8000 在本机已被另一个项目（CodeForge）占用，
    # 沿用 8000 会让 `python api.py` 直接 bind 失败。
    uvicorn.run("api:app", host="127.0.0.1", port=8010, reload=False)
