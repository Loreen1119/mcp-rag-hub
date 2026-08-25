"""
RAG 系统自动化评测。

自实现核心指标（Ragas 在 Windows SSL 层不兼容）：
- MRR (Mean Reciprocal Rank): 第一个相关结果出现位置的倒数均值
- Hit@K: Top-K 中至少命中一个相关结果的比例
- Precision@K: Top-K 中相关结果占比
- Recall@K: 所有相关结果中被检索到的比例
- 阶段贡献分析: BM25 → +Vector → +RRF → +CE（+Graph）的增量效果

评测 LLM 模式（可选）：如果安装了 Ollama，可计算 Faithfulness。

命令行参数：
  --split  train | test | all  (默认 all)
  --graph  启用 GraphRAG 图检索（默认关闭）
"""

from __future__ import annotations

import argparse
import logging
from typing import List, Dict

from config import (PROJECT_ROOT, TEST_QUERIES_FILE, TRAIN_QUERIES_FILE,
                    CE_TOP_K, BM25_TOP_K, KG_RRF_WEIGHT)
from src.evaluation.metrics import (
    load_test_cases,
    is_relevant,
    mrr,
    hit_at_k,
    precision_at_k,
    recall_at_k,
)
from src.pipeline import get_pipeline
from src.fusion import reciprocal_rank_fusion

logger = logging.getLogger(__name__)


# ============================================================
# 文件路径映射
# ============================================================


def _resolve_queries_file(split: str) -> str:
    """根据 split 参数返回对应的 JSON 文件路径。"""
    if split == "train":
        return str(TRAIN_QUERIES_FILE)
    elif split == "test":
        return str(TEST_QUERIES_FILE)
    else:  # "all"
        return str(PROJECT_ROOT / "data" / "test_queries_all.json")


# ============================================================
# 全量评测
# ============================================================


def run_evaluation(
    test_cases: list[dict] | None = None,
    verbose: bool = True,
    enable_graph: bool = False,
) -> dict:
    """对全部 test cases 跑四阶段评测（+可选第五阶段 graph），返回汇总指标。

    Args:
        test_cases: 可选，直接传入测试用例列表。
        verbose: 是否打印详细输出。
        enable_graph: 是否启用 GraphRAG 图检索（三路 RRF 融合）。
    """
    if test_cases is None:
        raise ValueError(
            "test_cases 不能为 None，请通过 load_test_cases(_resolve_queries_file(split)) 传入"
        )

    ctx = get_pipeline()
    bm25 = ctx.bm25
    vector = ctx.vector
    pipeline = ctx.pipeline
    graph_retriever = ctx.graph if enable_graph else None

    stages = ["bm25", "vector", "rrf", "cross_encoder"]
    if enable_graph:
        stages.insert(3, "graph")
        stages.insert(4, "hybrid")

    accum: dict[str, dict[str, float]] = {
        s: {"mrr": 0.0, "hit@5": 0.0, "precision@5": 0.0, "recall@5": 0.0}
        for s in stages
    }
    per_query: list[dict] = []

    for tc in test_cases:
        query = tc["query"]
        golden_sources = tc["golden_chunk_sources"]

        bm25_results = bm25.search(query, top_k=BM25_TOP_K)
        vector_results = vector.search(query, top_k=BM25_TOP_K)

        graph_results = None
        if enable_graph:
            graph_results = graph_retriever.search(query, top_k=BM25_TOP_K)

        rrf_results = reciprocal_rank_fusion([bm25_results, vector_results])

        if enable_graph and graph_results:
            hybrid_results = reciprocal_rank_fusion(
                [bm25_results, vector_results, graph_results],
                weights=[1.0, 1.0, KG_RRF_WEIGHT],
            )
        else:
            hybrid_results = reciprocal_rank_fusion([bm25_results, vector_results])

        ce_results = pipeline.reranker.rerank(query, rrf_results, top_k=CE_TOP_K)

        stage_results: dict[str, list] = {
            "bm25": bm25_results,
            "vector": vector_results,
            "rrf": rrf_results,
            "cross_encoder": ce_results,
        }
        if enable_graph:
            stage_results["graph"] = graph_results or []
            stage_results["hybrid"] = hybrid_results or []

        q_metrics: dict = {
            "id": tc["id"], "query": query, "category": tc["category"]
        }
        for stage in stages:
            results = stage_results[stage]
            mrr_val = mrr(results, golden_sources)
            hit = hit_at_k(results, golden_sources, k=5)
            prec = precision_at_k(results, golden_sources, k=5)
            rec = recall_at_k(results, golden_sources, k=5)

            accum[stage]["mrr"] += mrr_val
            accum[stage]["hit@5"] += hit
            accum[stage]["precision@5"] += prec
            accum[stage]["recall@5"] += rec

            q_metrics[f"{stage}_mrr"] = round(mrr_val, 4)
            q_metrics[f"{stage}_hit@5"] = int(hit)
            q_metrics[f"{stage}_precision@5"] = round(prec, 4)
            q_metrics[f"{stage}_recall@5"] = round(rec, 4)

        per_query.append(q_metrics)

        if verbose:
            _print_query_result(q_metrics, enable_graph=enable_graph)

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
        _print_summary(summary, enable_graph=enable_graph)

    return {"summary": summary, "per_query": per_query}


# ============================================================
# 阶段贡献分析
# ============================================================


def ablation_analysis(summary: dict, enable_graph: bool = False) -> dict:
    """分析每个模块的增量贡献。

    默认路径：BM25 → Vector → RRF → Cross-Encoder
    graph 模式：BM25 → Vector → Graph → Hybrid → Cross-Encoder
    """
    stages_order = ["bm25", "vector", "rrf", "cross_encoder"]
    if enable_graph:
        stages_order = ["bm25", "vector", "graph", "hybrid", "cross_encoder"]

    print("\n" + "=" * 60)
    print("  Ablation Study" + (" (with GraphRAG)" if enable_graph else ""))
    print("=" * 60)

    for metric in ["mrr", "hit@5", "precision@5", "recall@5"]:
        print(f"\n  [{metric}]")
        prev = 0.0
        for stage in stages_order:
            if stage not in summary:
                continue
            val = summary[stage][metric]
            delta = val - prev
            bar = _bar(val, max_val=1.0, width=30)
            print(f"    {stage:<16s}  {val:.4f}  {bar}  ({delta:+.4f})")
            prev = val

    return {}


# ============================================================
# 输出格式化
# ============================================================


def _bar(value: float, max_val: float = 1.0, width: int = 30) -> str:
    filled = int(value / max_val * width)
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def _print_query_result(qm: dict, enable_graph: bool = False) -> None:
    stages = ["bm25", "vector", "rrf", "cross_encoder"]
    if enable_graph:
        stages = ["bm25", "vector", "graph", "hybrid", "cross_encoder"]

    print(f"\n{'-'*50}")
    print(f"  [{qm['id']}] {qm['query'][:40]}")
    print(f"  category: {qm['category']}")
    print(f"  {'Stage':<16s} {'MRR':>8s}  {'Hit@5':>6s}  {'Prec@5':>8s}  {'Recall@5':>8s}")
    print(f"  {'-'*52}")
    for stage in stages:
        if f"{stage}_mrr" not in qm:
            continue
        print(
            f"  {stage:<16s}"
            f"  {qm[f'{stage}_mrr']:>8.4f}"
            f"  {qm[f'{stage}_hit@5']:>6d}"
            f"  {qm[f'{stage}_precision@5']:>8.4f}"
            f"  {qm[f'{stage}_recall@5']:>8.4f}"
        )


def _print_summary(summary: dict, enable_graph: bool = False) -> None:
    print(f"\n{'='*60}")
    print("  Summary" + (" (with GraphRAG)" if enable_graph else ""))
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
    parser = argparse.ArgumentParser(description="RAG 检索评测")
    parser.add_argument(
        "--split",
        choices=["train", "test", "all"],
        default="all",
        help="选择评测数据集：train (15条) / test (15条) / all (默认 test_queries.json)",
    )
    parser.add_argument(
        "--graph",
        action="store_true",
        help="启用 GraphRAG 图检索（三路 RRF 融合）",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)

    split = args.split
    queries_file = _resolve_queries_file(split)
    split_label = split

    print(f"\n>>> 加载数据集: {split_label}  ({queries_file})")
    print(f">>> GraphRAG: {'启用' if args.graph else '关闭'}")

    test_cases = load_test_cases(queries_file)
    print(f">>> 共 {len(test_cases)} 条测试用例\n")

    result = run_evaluation(
        test_cases=test_cases,
        verbose=True,
        enable_graph=args.graph,
    )
    ablation_analysis(result["summary"], enable_graph=args.graph)
