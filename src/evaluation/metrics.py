"""
共享评测指标函数 — 消除 retrieval_eval.py 和 experiments.py 的重复代码。

指标说明:
- MRR: 第一个相关结果出现位置的倒数均值
- Hit@K: Top-K 中至少命中一个相关结果的比例
- Precision@K: Top-K 中相关结果占比
- Recall@K: 所有相关结果中被检索到的比例（分母 = total_golden_sources）
"""

from typing import List

from src.models import RetrievalResult


def load_test_cases(path) -> list[dict]:
    import json

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["test_cases"]


def is_relevant(result: RetrievalResult, golden_sources: list[str]) -> bool:
    """判定检索结果是否相关：source 文件在 golden_sources 中。"""
    source = result.chunk.metadata.get("source", "")
    return source in golden_sources


def mrr(
    results: List[RetrievalResult], golden_sources: list[str], k: int = 10
) -> float:
    """MRR@K — 第一个相关结果的倒数排名。无命中返回 0。"""
    for rank, r in enumerate(results[:k], start=1):
        if is_relevant(r, golden_sources):
            return 1.0 / rank
    return 0.0


def hit_at_k(
    results: List[RetrievalResult], golden_sources: list[str], k: int = 5
) -> float:
    """Hit@K — Top-K 中是否至少有一个相关结果。返回 0 或 1。"""
    for r in results[:k]:
        if is_relevant(r, golden_sources):
            return 1.0
    return 0.0


def precision_at_k(
    results: List[RetrievalResult], golden_sources: list[str], k: int = 5
) -> float:
    """Precision@K — Top-K 中相关结果的比例。"""
    if not results[:k]:
        return 0.0
    hits = sum(1 for r in results[:k] if is_relevant(r, golden_sources))
    return hits / min(k, len(results[:k]))


def recall_at_k(
    results: List[RetrievalResult], golden_sources: list[str], k: int = 10
) -> float:
    """Recall@K — 所有 goldens 中被检索到的比例。

    计数 unique source（文档级），而非 chunk 级。
    因为 golden_sources 是文件名列表，一个文档可能拆成多个 chunk，
    按 chunk 计数会重复，导致 recall > 1.0。
    """
    total_golden = len(golden_sources)
    if total_golden == 0:
        return 0.0
    unique_sources_in_results = set()
    for r in results[:k]:
        if is_relevant(r, golden_sources):
            unique_sources_in_results.add(r.chunk.metadata.get("source", ""))
    return len(unique_sources_in_results) / total_golden
