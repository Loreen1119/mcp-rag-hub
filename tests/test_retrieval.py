"""检索层回归测试 — 专门针对「RRF 如何工作」暴露的三处缺陷。

背景（Phase 0 事故复盘）：
- ① 向量检索静默失效：VectorRetriever 复用判据只比 metadata 摘要，构建时把
  chunk_count 写进 metadata 但 add 失败会留下「声明 45 / 实际 0」的空壳集合，
  后续每次启动摘要匹配、静默复用、向量恒返回 0 条，混合检索退化为单路。
- ③ RRF 只融合 BM25 一路时，融合结果里不会出现 vector 来源，单路退化无法察觉。

本文件 3 条测试分别抓住：
1. rebuild 后 metadata.chunk_count == collection.count()（抓空壳集合）
2. 向量检索返回条数 > 0（抓向量失效）
3. RRF 融合结果同时含 bm25 与 vector 来源（防止单路退化，不用魔法数）

测试 1/2 是真实集成测试：会加载 embedding 模型（all-MiniLM-L6-v2）并对
合成语料真实编码，因此能当场抓住「声明条数 ≠ 实际写入条数」的分裂。
"""

from __future__ import annotations

import pytest

from src.models import Chunk, RetrievalResult
from src.retrievers import VectorRetriever
from src.fusion import reciprocal_rank_fusion


# ============================================================
# 合成语料：语义上分属两个主题，保证向量检索有区分度
# ============================================================

_CORPUS = [
    Chunk(
        content="RRF 即倒数排名融合，将 BM25 与向量检索两路排名按 1/(k+rank) 加权求和，消除量纲差异。",
        metadata={"source": "doc_a.md", "chunk_index": 0},
        chunk_id="a0",
    ),
    Chunk(
        content="混合检索策略同时使用稀疏关键词检索与稠密语义检索，通过 RRF 融合两者的排序结果。",
        metadata={"source": "doc_a.md", "chunk_index": 1},
        chunk_id="a1",
    ),
    Chunk(
        content="Cross-Encoder 将 query 与文档拼接后做全注意力计算，比 Bi-Encoder 独立编码更精准。",
        metadata={"source": "doc_a.md", "chunk_index": 2},
        chunk_id="a2",
    ),
    Chunk(
        content="今天天气晴朗，适合出门散步，公园里有很多人在放风筝。",
        metadata={"source": "doc_b.md", "chunk_index": 0},
        chunk_id="b0",
    ),
    Chunk(
        content="红烧肉的做法：五花肉切块焯水，加冰糖炒糖色，小火慢炖一小时。",
        metadata={"source": "doc_b.md", "chunk_index": 1},
        chunk_id="b1",
    ),
]


@pytest.fixture(scope="module")
def vector_retriever(tmp_path_factory):
    """在独立临时目录构建 VectorRetriever，避免污染生产 chroma_db。

    scope=module：构建一次（含模型加载 + embedding），三条测试共享。
    临时目录由 tmp_path_factory 管理，不手动删除（避免触发沙箱批量删除守卫）。
    """
    persist_dir = tmp_path_factory.mktemp("chroma_test")
    vr = VectorRetriever(
        _CORPUS,
        persist_dir=persist_dir,
        corpus_hash="testcorpus",
        collection_name="test_kb",
    )
    return vr


# ============================================================
# 测试 1：rebuild 后 metadata 声明的条数 == 实际 count()
# ============================================================


def test_collection_count_matches_metadata(vector_retriever):
    """抓空壳集合：声明 45 条但实际 0 条会被复用，这里验证 count 与 metadata 一致。"""
    coll = vector_retriever.collection
    declared = int(coll.metadata.get("chunk_count", -1))

    assert declared == len(_CORPUS), (
        f"metadata.chunk_count={declared} 与语料条数 {len(_CORPUS)} 不一致"
    )
    assert coll.count() == declared, (
        f"空壳集合：metadata 声明 {declared} 条，实际 count()={coll.count()}"
    )
    assert coll.count() > 0


# ============================================================
# 测试 2：向量检索返回条数 > 0
# ============================================================


def test_vector_search_returns_results(vector_retriever):
    """抓向量失效：正常情况下向量检索应返回非空结果，且 source 为 vector。"""
    results = vector_retriever.search("RRF 倒数排名融合怎么工作", top_k=5)

    assert len(results) > 0, "向量检索返回 0 条，向量路疑似静默失效"
    assert all(r.source == "vector" for r in results)
    # 语义相关的内容应排在前面：doc_a 的 RRF 内容应命中，而非 doc_b 的无关内容
    top_chunk_ids = [r.chunk.chunk_id for r in results]
    assert any(cid.startswith("a") for cid in top_chunk_ids), (
        f"向量检索未命中语义相关 chunk，实际 top: {top_chunk_ids}"
    )


# ============================================================
# 测试 3：RRF 融合结果同时含 bm25 与 vector 来源
# ============================================================


def test_rrf_fusion_includes_both_sources():
    """防止单路退化：RRF 融合后，结果里应同时出现 bm25 和 vector 两路来源。

    不用魔法数（如「向量贡献 > 0.3」），只断言两路来源都存在于融合结果的
    original_source 中——这是「混合检索没有退化成单路」的最小充分条件。
    """
    bm25_results = [
        RetrievalResult(chunk=_CORPUS[0], score=12.5, source="bm25"),
        RetrievalResult(chunk=_CORPUS[1], score=8.0, source="bm25"),
    ]
    vector_results = [
        RetrievalResult(chunk=_CORPUS[1], score=0.82, source="vector"),
        RetrievalResult(chunk=_CORPUS[2], score=0.71, source="vector"),
    ]

    fused = reciprocal_rank_fusion([bm25_results, vector_results], k=60)

    sources = {r.chunk.metadata.get("original_source") for r in fused}
    assert "bm25" in sources, f"RRF 融合结果缺少 bm25 来源，实际来源: {sources}"
    assert "vector" in sources, f"RRF 融合结果缺少 vector 来源，实际来源: {sources}"
