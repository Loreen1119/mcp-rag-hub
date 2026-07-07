"""
RAG 系统自动化评测。

自实现核心指标（Ragas 在 Windows SSL 层不兼容）：
- MRR (Mean Reciprocal Rank): 第一个相关结果出现位置的倒数均值
- Hit@K: Top-K 中至少命中一个相关结果的比例
- Precision@K: Top-K 中相关结果占比
- Recall@K: 所有相关结果中被检索到的比例
- 阶段贡献分析: BM25 → +Vector → +RRF → +CE 的增量效果

评测 LLM 模式（可选）：如果安装了 Ollama，可计算 Faithfulness。
"""

from __future__ import annotations

import logging
from typing import List, Dict

from config import TEST_QUERIES_FILE, CE_TOP_K, BM25_TOP_K
from src.evaluation.metrics import (
    load_test_cases,
    is_relevant,
    mrr,
    hit_at_k,
    precision_at_k,
    recall_at_k,
)
from src.data_pipeline import process_directory
from src.retrievers import BM25Retriever, VectorRetriever
from src.fusion import FusionPipeline, reciprocal_rank_fusion

logger = logging.getLogger(__name__)

# ============================================================
# 全量评测
# ============================================================


def run_evaluation(
    test_cases: list[dict] | None = None,
    verbose: bool = True,
) -> dict:
    """对全部 test cases 跑四阶段评测，返回汇总指标。"""
    if test_cases is None:
        test_cases = load_test_cases(TEST_QUERIES_FILE)

    # 加载管线（只做一次）
    chunks = process_directory()
    bm25 = BM25Retriever(chunks)
    vector = VectorRetriever(chunks, rebuild=True)
    pipeline = FusionPipeline()

    # 四阶段指标累加
    stages = ["bm25", "vector", "rrf", "cross_encoder"]
    accum: dict[str, dict[str, float]] = {
        s: {"mrr": 0.0, "hit@5": 0.0, "precision@5": 0.0, "recall@5": 0.0}
        for s in stages
    }
    per_query: list[dict] = []

    for tc in test_cases:
        query = tc["query"]
        golden_sources = tc["golden_chunk_sources"]

        # 执行各阶段检索
        bm25_results = bm25.search(query, top_k=BM25_TOP_K)
        vector_results = vector.search(query, top_k=BM25_TOP_K)
        rrf_results = reciprocal_rank_fusion([bm25_results, vector_results])
        ce_results = pipeline.reranker.rerank(query, rrf_results, top_k=CE_TOP_K)

        stage_results = {
            "bm25": bm25_results,
            "vector": vector_results,
            "rrf": rrf_results,
            "cross_encoder": ce_results,
        }

        # 计算各阶段指标
        q_metrics: dict = {"id": tc["id"], "query": query, "category": tc["category"]}
        for stage in stages:
            results = stage_results[stage]
            mrr_val = mrr(results, golden_sources)
            hit = hit_at_k(results, golden_sources, k=5)
            prec = precision_at_k(results, golden_sources, k=5)
            recall = recall_at_k(results, golden_sources, k=5)

            accum[stage]["mrr"] += mrr_val
            accum[stage]["hit@5"] += hit
            accum[stage]["precision@5"] += prec
            accum[stage]["recall@5"] += recall

            q_metrics[f"{stage}_mrr"] = round(mrr_val, 4)
            q_metrics[f"{stage}_hit@5"] = int(hit)
            q_metrics[f"{stage}_precision@5"] = round(prec, 4)
            q_metrics[f"{stage}_recall@5"] = round(recall, 4)

        per_query.append(q_metrics)

        if verbose:
            _print_query_result(q_metrics)

    # 汇总（取平均）
    n = len(test_cases)
    summary = {}
    for stage in stages:
        summary[stage] = {
            "mrr": round(accum[stage]["mrr"] / n, 4),
            "hit@5": round(accum[stage]["hit@5"] / n, 4),
            "precision@5": round(accum[stage]["precision@5"] / n, 4),
            "recall@5": round(accum[stage]["recall@5"] / n, 4),
        }

    if verbose:
        _print_summary(summary)

    return {"summary": summary, "per_query": per_query}


# ============================================================
# 阶段贡献分析
# ============================================================


def ablation_analysis(summary: dict) -> dict:
    """分析每个模块的增量贡献。

    BM25 基线 → +Vector(RRF) → +Cross-Encoder 的逐级提升。
    """
    stages_order = ["bm25", "vector", "rrf", "cross_encoder"]

    print("\n" + "=" * 60)
    print("  Ablation Study")
    print("=" * 60)

    for metric in ["mrr", "hit@5", "precision@5", "recall@5"]:
        print(f"\n  [{metric}]")
        prev = 0.0
        for stage in stages_order:
            val = summary[stage][metric]
            delta = val - prev
            bar = _bar(val, max_val=1.0, width=30)
            print(f"    {stage:<16s}  {val:.4f}  {bar}  (+{delta:+.4f})")
            prev = val

    return {}


def _bar(value: float, max_val: float = 1.0, width: int = 30) -> str:
    filled = int(value / max_val * width)
    return "[" + "#" * filled + "-" * (width - filled) + "]"


# ============================================================
# 输出格式化
# ============================================================


def _print_query_result(qm: dict) -> None:
    print(f"\n{'-'*50}")
    print(f"  [{qm['id']}] {qm['query'][:40]}")
    print(f"  category: {qm['category']}")
    print(f"  {'Stage':<16s} {'MRR':>8s}  {'Hit@5':>6s}  {'Prec@5':>8s}  {'Recall@5':>8s}")
    print(f"  {'-'*52}")
    for stage in ["bm25", "vector", "rrf", "cross_encoder"]:
        print(
            f"  {stage:<16s}"
            f"  {qm[f'{stage}_mrr']:>8.4f}"
            f"  {qm[f'{stage}_hit@5']:>6d}"
            f"  {qm[f'{stage}_precision@5']:>8.4f}"
            f"  {qm[f'{stage}_recall@5']:>8.4f}"
        )


def _print_summary(summary: dict) -> None:
    print(f"\n{'='*60}")
    print("  Summary")
    print(f"{'='*60}")
    print(f"  {'Stage':<16s} {'MRR':>8s}  {'Hit@5':>8s}  {'Prec@5':>8s}  {'Recall@5':>8s}")
    print(f"  {'-'*58}")
    for stage, metrics in summary.items():
        print(
            f"  {stage:<16s}"
            f"  {metrics['mrr']:>8.4f}"
            f"  {metrics['hit@5']:>8.4f}"
            f"  {metrics['precision@5']:>8.4f}"
            f"  {metrics['recall@5']:>8.4f}"
        )


# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)  # 减少日志噪音

    result = run_evaluation(verbose=True)
    ablation_analysis(result["summary"])
