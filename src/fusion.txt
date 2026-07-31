"""
RRF 融合 + Cross-Encoder 重排序。

RRF: 只看排名不看原始分数，天然消除 BM25（0~几十）和向量余弦相似度（0~1）的量纲差异。
Cross-Encoder: query+doc 拼接送入模型做全注意力计算，比 Bi-Encoder 的独立编码准确得多。
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import List

from sentence_transformers import CrossEncoder

from config import RRF_K, CROSS_ENCODER_MODEL, CE_TOP_K, BM25_TOP_K, VECTOR_TOP_K
from src.models import Chunk, RetrievalResult

logger = logging.getLogger(__name__)

# ============================================================
# RRF 融合
# ============================================================


def reciprocal_rank_fusion(
    rankings: List[List[RetrievalResult]],
    k: int = RRF_K,
    top_n: int | None = None,
    weights: List[float] | None = None,
) -> List[RetrievalResult]:
    """RRF 算法 — 将多路排名转换为倒数权重后相加。

    RRF(chunk) = Σ weight_i * 1 / (k + rank_i(chunk))

    核心优势：不看原始分数的绝对大小，只看排名。
    BM25 得分 0~几十，向量余弦相似度 0~1，直接加权 BM25 会碾压。
    RRF 用排名替代分数，天然消除量纲差异。

    Args:
        rankings: 每路检索的排序结果列表（rank 1 在 list[0]）
        k: 平滑常数，k 越小排名靠前的文档权重越大
        top_n: 返回 Top-N，不传则返回全部
        weights: 每路检索的 RRF 权重，长度需与 rankings 一致；
                 可用于降低图检索等辅助召回路的贡献

    Returns:
        按 RRF 得分降序排列的结果，source 标记为 "rrf"
    """
    scores: defaultdict[str, float] = defaultdict(float)
    chunk_map: dict[str, tuple[Chunk, str]] = {}  # chunk_id → (Chunk, 原始来源)

    if weights is None:
        weights = [1.0] * len(rankings)
    elif len(weights) != len(rankings):
        raise ValueError(f"weights 长度 ({len(weights)}) 必须与 rankings 长度 ({len(rankings)}) 一致")

    for ranking, weight in zip(rankings, weights):
        for rank, result in enumerate(ranking, start=1):
            cid = result.chunk.chunk_id
            scores[cid] += weight * 1.0 / (k + rank)
            if cid not in chunk_map:
                chunk_map[cid] = (result.chunk, result.source)

    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    if top_n:
        sorted_items = sorted_items[:top_n]

    results: List[RetrievalResult] = []
    for cid, rrf_score in sorted_items:
        chunk, original_source = chunk_map[cid]
        meta = dict(chunk.metadata)
        meta["rrf_score"] = round(rrf_score, 6)
        meta["original_source"] = original_source

        results.append(RetrievalResult(
            chunk=Chunk(content=chunk.content, metadata=meta, chunk_id=chunk.chunk_id),
            score=rrf_score,
            source="rrf",
        ))
    return results


# ============================================================
# Cross-Encoder 重排序
# ============================================================


class CrossEncoderReranker:
    """Cross-Encoder 重排序器。

    Bi-Encoder（第 3 章的 VectorRetriever）把 query 和 doc 分别编码再算相似度，
    快但粗糙。Cross-Encoder 把 query+doc 拼接后做一次完整注意力计算，
    准但慢。所以策略是：Bi-Encoder 粗筛 Top-20，Cross-Encoder 精排 Top-5。
    """

    def __init__(self, model_name: str = CROSS_ENCODER_MODEL):
        logger.info("加载 Cross-Encoder 模型: %s", model_name)
        self.model = CrossEncoder(model_name)
        logger.info("Cross-Encoder 加载完成")

    def rerank(
        self,
        query: str,
        candidates: List[RetrievalResult],
        top_k: int = CE_TOP_K,
    ) -> List[RetrievalResult]:
        """对候选列表做 Cross-Encoder 精排。

        Args:
            query: 用户查询
            candidates: RRF 或其他来源的候选结果
            top_k: 返回数量

        Returns:
            按 CE 得分降序排列的 Top-K 结果，source 标记为 "cross_encoder"
        """
        if not candidates:
            return []

        # 构造 query-doc pairs
        pairs = [(query, r.chunk.content) for r in candidates]
        scores = self.model.predict(pairs, show_progress_bar=False)

        # 配对排序
        scored = list(zip(candidates, scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        results: List[RetrievalResult] = []
        for rank, (candidate, ce_score) in enumerate(scored[:top_k]):
            meta = dict(candidate.chunk.metadata)
            meta["ce_score"] = round(float(ce_score), 4)
            meta["ce_rank"] = rank + 1

            results.append(RetrievalResult(
                chunk=Chunk(
                    content=candidate.chunk.content,
                    metadata=meta,
                    chunk_id=candidate.chunk.chunk_id,
                ),
                score=float(ce_score),
                source="cross_encoder",
            ))
        return results


# ============================================================
# 融合管线
# ============================================================


class FusionPipeline:
    """完整的检索融合管线：RRF → Cross-Encoder。

    将第 3 章的两路检索结果串联为一条端到端链路。
    """

    def __init__(self, reranker: CrossEncoderReranker | None = None):
        self.reranker = reranker or CrossEncoderReranker()

    def run(
        self,
        bm25_results: List[RetrievalResult],
        vector_results: List[RetrievalResult],
        query: str,
        rrf_k: int = RRF_K,
        ce_top_k: int = CE_TOP_K,
        graph_results: List[RetrievalResult] | None = None,
        rrf_weights: List[float] | None = None,
    ) -> dict:
        """执行完整融合链路。

        Args:
            bm25_results: BM25 关键词检索结果
            vector_results: ChromaDB 向量检索结果
            query: 用户查询
            rrf_k: RRF 平滑常数
            ce_top_k: Cross-Encoder 返回数量
            graph_results: 可选，GraphRAG 图检索结果（三路融合时传入）
            rrf_weights: 可选，每路检索在 RRF 中的权重；
                         例如 [1.0, 1.0, 0.5] 可降低图检索贡献

        Returns:
            {
                "rrf": List[RetrievalResult],      # RRF 融合后的结果
                "cross_encoder": List[RetrievalResult],  # CE 精排后的 Top-K
            }
        """
        # 阶段 1: RRF 融合（二路或三路）
        rankings = [bm25_results, vector_results]
        if graph_results:
            rankings.append(graph_results)
        fused = reciprocal_rank_fusion(rankings, k=rrf_k, weights=rrf_weights)

        # 阶段 2: Cross-Encoder 重排
        reranked = self.reranker.rerank(query, fused, top_k=ce_top_k)

        return {"rrf": fused, "cross_encoder": reranked}


# ============================================================
# 演示入口
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    from src.data_pipeline import process_directory
    from src.retrievers import BM25Retriever, VectorRetriever
    from src.graph_retriever import GraphRetriever

    # 1. 加载文档 & 建索引
    chunks = process_directory()
    bm25 = BM25Retriever(chunks)
    vector = VectorRetriever(chunks)
    graph_retriever = GraphRetriever(chunks)

    pipeline = FusionPipeline()

    # 2. 测试查询
    queries = [
        "混合检索策略",
        "RAG 的核心优化方向",
    ]

    for q in queries:
        print(f"\n{'='*60}")
        print(f"  Query: {q}")
        print(f"{'='*60}")

        bm25_results = bm25.search(q)
        vector_results = vector.search(q)
        graph_results = graph_retriever.search(q)
        output = pipeline.run(bm25_results, vector_results, q, graph_results=graph_results)

        print(f"\n  [RRF 融合 Top-5]")
        for r in output["rrf"][:5]:
            print(f"    RRF={r.score:.6f}  src={r.chunk.metadata.get('original_source')}"
                  f"  |  {r.chunk}")

        print(f"\n  [Cross-Encoder 精排 Top-3]")
        for r in output["cross_encoder"][:3]:
            print(f"    CE={r.score:.4f}  rank={r.chunk.metadata.get('ce_rank')}"
                  f"  |  {r.chunk}")
