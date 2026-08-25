"""
统一 RAG 管线上下文。

全项目（app / agent / mcp_server / retrieval_eval / kg_builder）通过这里获取
同一批索引组件，消除 5 处各自 init 造成的：
- 重复 process_directory + 重复 VectorRetriever(rebuild=True)
- 内存 _chunks 与 Chroma 持久化结果的数据源分裂
- 并发下重复初始化模型/索引的竞态

生命周期：懒加载单例（threading.Lock 保护），持有
- 文档 Chunk 列表（内存权威源，get_chunk / list_documents 用它）
- 索引内容 hash（复用/重建索引的判据）
- BM25 / Vector / Graph / Fusion 组件

VectorRetriever 在内容 hash / schema 版本未变化时复用已有 Chroma 集合，
仅在摘要不匹配或显式 force_rebuild 时重建。
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from config import EMBEDDING_MODEL, ENABLE_KG
from src.data_pipeline import process_directory, corpus_hash
from src.fusion import FusionPipeline
from src.retrievers import BM25Retriever, VectorRetriever

logger = logging.getLogger(__name__)


@dataclass
class PipelineContext:
    """索引组件集合。get_chunk / list_documents 应消费 chunks（内存权威源）。"""

    chunks: list
    bm25: BM25Retriever
    vector: VectorRetriever
    pipeline: FusionPipeline
    graph: Optional[object] = None


_ctx: Optional[PipelineContext] = None
_ctx_signature: Optional[tuple] = None  # 与 _ctx 绑定的参数签名，参数变化时重建替换
_build_lock = threading.Lock()
_INDEX_META_FILE = "index_meta.json"


def compute_index_meta(docs_dir=None, chunk_size=None, chunk_overlap=None) -> dict:
    """返回索引元数据：docs 目录（resolve 规范化）+ 内容 hash + chunk 参数。

    docs_dir 统一为绝对路径，避免 docs/、./docs、绝对路径等同一目录的
    不同字符串写法导致签名不同、重复构建上下文。
    """
    from config import CHUNK_OVERLAP, CHUNK_SIZE, DOCS_DIR

    docs_dir = Path(docs_dir).resolve() if docs_dir else DOCS_DIR
    chunk_size = chunk_size or CHUNK_SIZE
    chunk_overlap = chunk_overlap or CHUNK_OVERLAP
    return {
        "docs_dir": str(docs_dir),
        "corpus_hash": corpus_hash(docs_dir, chunk_size, chunk_overlap),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
    }


def get_pipeline(
    docs_dir=None,
    chunk_size=None,
    chunk_overlap=None,
    force_rebuild: bool = False,
) -> PipelineContext:
    """返回全局共享的管线上下文（懒加载单例）。

    单例与参数签名绑定：传入 docs_dir / chunk_size / chunk_overlap 与当前
    缓存不同时，会在持锁状态下重建替换，避免测试/多语料拿到旧索引。
    线程安全：并发首次调用时仅一个线程构建，其余等待。
    force_rebuild=True 强制重建向量索引（并更新持久化索引元数据）。
    """
    global _ctx, _ctx_signature

    meta = compute_index_meta(docs_dir, chunk_size, chunk_overlap)
    sig = (meta["docs_dir"], meta["chunk_size"], meta["chunk_overlap"])

    if _ctx is not None and _ctx_signature == sig and not force_rebuild:
        return _ctx

    with _build_lock:
        if _ctx is not None and _ctx_signature == sig and not force_rebuild:
            return _ctx

        # 先丢旧引用（Chroma PersistentClient 无公开 close，靠引用计数销毁，
        # 避免同 persist_dir 双 client 并发打开 SQLite 冲突），再构建新管线。
        _ctx = None
        _ctx_signature = None

        logger.info("初始化 RAG 管线（docs=%s, corpus_hash=%s）...", meta["docs_dir"], meta["corpus_hash"][:12])
        chunks = process_directory(meta["docs_dir"], chunk_size=meta["chunk_size"], overlap=meta["chunk_overlap"])

        if not chunks:
            raise RuntimeError(
                "知识库为空，请先向 docs/ 目录放入文档（支持 .pdf / .md / .txt / .py）后再试。"
            )

        from config import CHROMA_PERSIST_DIR
        from src.kg_retriever import KGRetriever

        bm25 = BM25Retriever(chunks)
        vector = VectorRetriever(
            chunks,
            persist_dir=CHROMA_PERSIST_DIR,
            rebuild=force_rebuild,
            corpus_hash=meta["corpus_hash"],
        )
        graph = KGRetriever(chunks) if ENABLE_KG else None
        pipeline = FusionPipeline()

        _ctx = PipelineContext(
            chunks=chunks,
            bm25=bm25,
            vector=vector,
            pipeline=pipeline,
            graph=graph,
        )
        _ctx_signature = sig
        _save_index_meta(meta)
        logger.info(
            "RAG 管线就绪 — %d 个 Chunk 已索引, collection=%s, KG 路%s",
            len(chunks),
            vector.collection.name,
            "已启用" if graph else "已关闭",
        )
        return _ctx


def get_document_chunks(
    docs_dir=None,
    chunk_size=None,
    chunk_overlap=None,
) -> list:
    """仅加载文档切片（不构建向量索引/不加载模型）。

    kg_builder 等只需切片的调用方用它，避免加载 embedding 模型。
    """
    meta = compute_index_meta(docs_dir, chunk_size, chunk_overlap)
    return process_directory(meta["docs_dir"], chunk_size=meta["chunk_size"], overlap=meta["chunk_overlap"])


# ============================================================
# 索引元数据持久化（纯诊断）
# ============================================================


def _save_index_meta(meta: dict) -> None:
    """把最近一次构建的索引元数据写入 chroma_db/。

    纯诊断用途：活动 collection 由当前 corpus_hash 直接推导，
    本文件不参与任何读取/复用判断。
    """
    try:
        from config import CHROMA_PERSIST_DIR

        CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
        (CHROMA_PERSIST_DIR / _INDEX_META_FILE).write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception as exc:
        logger.warning("索引元数据写入失败: %s", exc)


def reset_pipeline() -> None:
    """清空全局单例（测试/重载场景用）。"""
    global _ctx, _ctx_signature
    _ctx = None
    _ctx_signature = None
