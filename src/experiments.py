"""
消融实验与数据分析 — 系统性评估各模块的独立贡献。

实验矩阵:
1. 模块隔离消融: 单独测 BM25/Vector/BM25+Vector(concat)/+RRF/+CE
2. 参数敏感性: chunk_size / overlap / RRF_K / CE_TOP_K 的网格搜索
3. 分 Query 类别: exact_match vs semantic vs mixed 的模块增益差异
4. 延迟分析: 各阶段耗时统计

运行:
    python src/experiments.py              # 运行全部实验
    python src/experiments.py --quick       # 快速模式（跳过参数扫描）
    python src/experiments.py --export      # 仅导出已有数据

输出:
    experiments/ablation_results.json       # 结构化实验数据
    experiments/parameter_sweep.json        # 参数扫描结果
"""

from __future__ import annotations

import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Callable

# 确保项目根目录在 sys.path 中（支持从 src/ 目录直接运行）
_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    RRF_K,
    CE_TOP_K,
    BM25_TOP_K,
    VECTOR_TOP_K,
    DOCS_DIR,
    EXPERIMENTS_DIR,
    TEST_QUERIES_FILE,
)
from src.models import Chunk, RetrievalResult
from src.data_pipeline import process_directory, process_document
from src.retrievers import BM25Retriever, VectorRetriever
from src.fusion import FusionPipeline, reciprocal_rank_fusion

logger = logging.getLogger(__name__)

EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 测试数据加载
# ============================================================


def _load_test_cases() -> list[dict]:
    with open(TEST_QUERIES_FILE, "r", encoding="utf-8") as f:
        return json.load(f)["test_cases"]


def _is_relevant(result: RetrievalResult, golden_sources: list[str]) -> bool:
    source = result.chunk.metadata.get("source", "")
    return source in golden_sources


# ============================================================
# 指标计算
# ============================================================


def _mrr(results: list[RetrievalResult], golden_sources: list[str], k: int = 10) -> float:
    for rank, r in enumerate(results[:k], start=1):
        if _is_relevant(r, golden_sources):
            return 1.0 / rank
    return 0.0


def _hit_at_k(results: list[RetrievalResult], golden_sources: list[str], k: int = 5) -> float:
    for r in results[:k]:
        if _is_relevant(r, golden_sources):
            return 1.0
    return 0.0


def _precision_at_k(results: list[RetrievalResult], golden_sources: list[str], k: int = 5) -> float:
    if not results[:k]:
        return 0.0
    hits = sum(1 for r in results[:k] if _is_relevant(r, golden_sources))
    return hits / min(k, len(results[:k]))


def _evaluate_single(
    results_fn: Callable[[str], list[RetrievalResult]],
    test_cases: list[dict],
) -> dict:
    """对一组 test cases 运行检索函数并汇总指标。"""
    n = len(test_cases)
    total: dict[str, float] = {"mrr": 0.0, "hit@5": 0.0, "precision@5": 0.0}

    for tc in test_cases:
        query = tc["query"]
        golden_sources = tc["golden_chunk_sources"]
        results = results_fn(query)
        total["mrr"] += _mrr(results, golden_sources)
        total["hit@5"] += _hit_at_k(results, golden_sources)
        total["precision@5"] += _precision_at_k(results, golden_sources)

    for k in total:
        total[k] = round(total[k] / n, 4)
    return total


# ============================================================
# 实验 1: 模块隔离消融
# ============================================================


def _dedup_concat(
    results_a: list[RetrievalResult],
    results_b: list[RetrievalResult],
) -> list[RetrievalResult]:
    """去重拼接两路结果（无 RRF，仅按原始分数排序）。

    需要对 BM25 和 Vector 分数做 min-max 归一化到 [0,1] 才能使排序有意义。
    """
    def _normalize(rl: list[RetrievalResult]) -> list[RetrievalResult]:
        if not rl:
            return rl
        scores = [r.score for r in rl]
        smin, smax = min(scores), max(scores)
        if smax == smin:
            return rl
        for r in rl:
            r.score = (r.score - smin) / (smax - smin)
        return rl

    a_norm = _normalize(results_a)
    b_norm = _normalize(results_b)

    seen: set[str] = set()
    merged: list[RetrievalResult] = []
    for r in sorted(a_norm + b_norm, key=lambda x: x.score, reverse=True):
        if r.chunk.chunk_id not in seen:
            seen.add(r.chunk.chunk_id)
            merged.append(r)
    return merged


def run_module_ablation(test_cases: list[dict] | None = None) -> dict:
    """模块隔离消融：分别评估每个模块独立 + 组合的性能。

    返回每模块的 MRR/Hit@5/Prec@5。
    """
    if test_cases is None:
        test_cases = _load_test_cases()

    print("\n" + "=" * 60)
    print("  实验 1: Module Ablation")
    print("=" * 60)

    chunks = process_directory()
    bm25 = BM25Retriever(chunks)
    vector = VectorRetriever(chunks, rebuild=True)
    pipeline = FusionPipeline()

    configs: dict[str, Callable] = {
        "bm25_only": lambda q: bm25.search(q, top_k=BM25_TOP_K),
        "vector_only": lambda q: vector.search(q, top_k=VECTOR_TOP_K),
        "concat_naive": lambda q: _dedup_concat(
            bm25.search(q, top_k=BM25_TOP_K),
            vector.search(q, top_k=VECTOR_TOP_K),
        ),
        "rrf_fusion": lambda q: reciprocal_rank_fusion([
            bm25.search(q, top_k=BM25_TOP_K),
            vector.search(q, top_k=VECTOR_TOP_K),
        ]),
        "full_pipeline": lambda q: pipeline.run(
            bm25.search(q, top_k=BM25_TOP_K),
            vector.search(q, top_k=VECTOR_TOP_K),
            q,
        )["cross_encoder"],
    }

    results: dict = {"description": "模块隔离消融: 各配置独立运行，非累积", "results": {}}
    prev: dict[str, float] = {}

    for name, fn in configs.items():
        metrics = _evaluate_single(fn, test_cases)
        delta = {}
        for m in metrics:
            delta[f"delta_{m}"] = (
                round(metrics[m] - prev[m], 4) if prev else 0.0
            )
        results["results"][name] = {**metrics, **delta}
        prev = metrics

        # 打印
        bar = _bar(metrics["mrr"])
        print(f"  {name:<20s}  MRR={metrics['mrr']:.4f}  "
              f"Hit@5={metrics['hit@5']:.4f}  Prec@5={metrics['precision@5']:.4f}  {bar}")

    _save_json(results, "ablation_results.json")
    return results


# ============================================================
# 实验 2: 分 Query 类别分析
# ============================================================


def run_category_breakdown(test_cases: list[dict] | None = None) -> dict:
    """按 query 类别（exact_match / semantic / mixed）拆分评测。

    揭示各模块对不同难度类型 query 的处理能力差异。
    """
    if test_cases is None:
        test_cases = _load_test_cases()

    print("\n" + "=" * 60)
    print("  实验 2: Per-Category Breakdown")
    print("=" * 60)

    chunks = process_directory()
    bm25 = BM25Retriever(chunks)
    vector = VectorRetriever(chunks, rebuild=True)
    pipeline = FusionPipeline()

    categories: dict[str, list[dict]] = defaultdict(list)
    for tc in test_cases:
        categories[tc["category"]].append(tc)

    results: dict = {"description": "分 query 类别的模块增益矩阵", "categories": {}}

    for cat_name, cat_cases in sorted(categories.items()):
        cat_results: dict = {}
        print(f"\n  [{cat_name}] ({len(cat_cases)} queries)")

        # BM25 only
        bm25_m = _evaluate_single(
            lambda q: bm25.search(q, top_k=BM25_TOP_K), cat_cases
        )
        # Vector only
        vec_m = _evaluate_single(
            lambda q: vector.search(q, top_k=VECTOR_TOP_K), cat_cases
        )
        # Full pipeline
        full_m = _evaluate_single(
            lambda q: pipeline.run(
                bm25.search(q, top_k=BM25_TOP_K),
                vector.search(q, top_k=VECTOR_TOP_K),
                q,
            )["cross_encoder"],
            cat_cases,
        )

        cat_results["bm25_only"] = bm25_m
        cat_results["vector_only"] = vec_m
        cat_results["full_pipeline"] = full_m
        cat_results["bm25_to_full_gain"] = {
            m: round(full_m[m] - bm25_m[m], 4) for m in bm25_m
        }
        cat_results["vector_to_full_gain"] = {
            m: round(full_m[m] - vec_m[m], 4) for m in vec_m
        }

        results["categories"][cat_name] = cat_results

        print(f"    BM25 Only:  MRR={bm25_m['mrr']:.4f}  "
              f"Hit@5={bm25_m['hit@5']:.4f}  Prec@5={bm25_m['precision@5']:.4f}")
        print(f"    Vector Only: MRR={vec_m['mrr']:.4f}  "
              f"Hit@5={vec_m['hit@5']:.4f}  Prec@5={vec_m['precision@5']:.4f}")
        print(f"    Full Pipe:   MRR={full_m['mrr']:.4f}  "
              f"Hit@5={full_m['hit@5']:.4f}  Prec@5={full_m['precision@5']:.4f}")
        print(f"    BM25→Full Δ: MRR={cat_results['bm25_to_full_gain']['mrr']:+.4f}  "
              f"Hit@5={cat_results['bm25_to_full_gain']['hit@5']:+.4f}")

    _save_json(results, "category_breakdown.json")
    return results


# ============================================================
# 实验 3: 参数敏感性扫描
# ============================================================


def _rebuild_and_evaluate(
    test_cases: list[dict],
    chunk_size: int,
    overlap: int,
    rrf_k: int,
    ce_top_k: int,
) -> dict:
    """用指定参数重建管线并评测。"""
    chunks = process_directory(
        dir_path=DOCS_DIR,
        chunk_size=chunk_size,
        overlap=overlap,
    )
    bm25 = BM25Retriever(chunks)
    vector = VectorRetriever(chunks, rebuild=True)

    ce_pipeline = FusionPipeline()

    def full_search(query: str) -> list[RetrievalResult]:
        output = ce_pipeline.run(
            bm25.search(query, top_k=BM25_TOP_K),
            vector.search(query, top_k=VECTOR_TOP_K),
            query,
            rrf_k=rrf_k,
            ce_top_k=ce_top_k,
        )
        return output["cross_encoder"]

    return _evaluate_single(full_search, test_cases)


def run_parameter_sweep(
    test_cases: list[dict] | None = None,
    quick: bool = False,
) -> dict:
    """参数网格扫描。

    默认扫描: chunk_size × overlap × RRF_K
    quick 模式: 仅扫描 RRF_K（最快，不需要重建索引）
    """
    if test_cases is None:
        test_cases = _load_test_cases()

    print("\n" + "=" * 60)
    print("  实验 3: Parameter Sensitivity Sweep")
    print("=" * 60)

    results: dict = {"description": "参数敏感性网格扫描", "sweeps": []}

    if quick:
        # 快速模式：仅扫描 RRF_K 和 CE_TOP_K（不需要重建索引）
        chunks = process_directory()
        bm25 = BM25Retriever(chunks)
        vector = VectorRetriever(chunks, rebuild=True)

        for rrf_k in [30, 60, 120]:
            for ce_k in [3, 5, 10]:
                pipeline = FusionPipeline()

                def make_fn(_rrf_k=rrf_k, _ce_k=ce_k):
                    return lambda q: pipeline.run(
                        bm25.search(q, top_k=BM25_TOP_K),
                        vector.search(q, top_k=VECTOR_TOP_K),
                        q,
                        rrf_k=_rrf_k,
                        ce_top_k=_ce_k,
                    )["cross_encoder"]

                metrics = _evaluate_single(make_fn(), test_cases)
                entry = {
                    "rrf_k": rrf_k,
                    "ce_top_k": ce_k,
                    **metrics,
                }
                results["sweeps"].append(entry)
                print(f"  RRF_K={rrf_k:<5d} CE_K={ce_k:<3d}  "
                      f"MRR={metrics['mrr']:.4f}  "
                      f"Hit@5={metrics['hit@5']:.4f}  "
                      f"Prec@5={metrics['precision@5']:.4f}  {_bar(metrics['mrr'])}")

        # 找最优组合
        best = max(results["sweeps"], key=lambda x: x["mrr"])
        results["best_config"] = {
            "rrf_k": best["rrf_k"],
            "ce_top_k": best["ce_top_k"],
            "mrr": best["mrr"],
        }
        print(f"\n  Best: RRF_K={best['rrf_k']} CE_K={best['ce_top_k']} "
              f"MRR={best['mrr']:.4f}")
    else:
        # 完整模式：扫描 chunk_size × overlap × RRF_K
        chunk_sizes = [256, 512, 1024]
        overlaps = [64, 128, 256]

        for cs in chunk_sizes:
            for ov in overlaps:
                if ov >= cs:
                    continue  # 跳过无效组合
                metrics = _rebuild_and_evaluate(
                    test_cases, chunk_size=cs, overlap=ov,
                    rrf_k=RRF_K, ce_top_k=CE_TOP_K,
                )
                entry = {
                    "chunk_size": cs,
                    "overlap": ov,
                    "rrf_k": RRF_K,
                    "ce_top_k": CE_TOP_K,
                    **metrics,
                }
                results["sweeps"].append(entry)
                print(f"  CS={cs:<5d} OV={ov:<4d}  "
                      f"MRR={metrics['mrr']:.4f}  "
                      f"Hit@5={metrics['hit@5']:.4f}  "
                      f"Prec@5={metrics['precision@5']:.4f}  {_bar(metrics['mrr'])}")

        # 找最优
        best = max(results["sweeps"], key=lambda x: x["mrr"])
        results["best_config"] = {
            "chunk_size": best["chunk_size"],
            "overlap": best["overlap"],
            "mrr": best["mrr"],
        }
        print(f"\n  Best: CS={best['chunk_size']} OV={best['overlap']} "
              f"MRR={best['mrr']:.4f}")

    _save_json(results, "parameter_sweep.json")
    return results


# ============================================================
# 实验 4: 延迟分析
# ============================================================


def run_latency_profile(test_cases: list[dict] | None = None) -> dict:
    """测量各阶段的平均耗时。

    对每个 test case 运行一次完整管线，记录每步耗时。
    """
    if test_cases is None:
        test_cases = _load_test_cases()

    print("\n" + "=" * 60)
    print("  实验 4: Latency Profile")
    print("=" * 60)

    chunks = process_directory()
    bm25 = BM25Retriever(chunks)
    vector = VectorRetriever(chunks, rebuild=True)
    ce_pipeline = FusionPipeline()

    timings: dict[str, list[float]] = defaultdict(list)

    for tc in test_cases:
        query = tc["query"]

        # BM25
        t0 = time.perf_counter()
        bm25_results = bm25.search(query, top_k=BM25_TOP_K)
        timings["bm25_search_ms"].append((time.perf_counter() - t0) * 1000)

        # Vector
        t0 = time.perf_counter()
        vector_results = vector.search(query, top_k=VECTOR_TOP_K)
        timings["vector_search_ms"].append((time.perf_counter() - t0) * 1000)

        # RRF
        t0 = time.perf_counter()
        rrf_results = reciprocal_rank_fusion([bm25_results, vector_results])
        timings["rrf_fusion_ms"].append((time.perf_counter() - t0) * 1000)

        # Cross-Encoder
        t0 = time.perf_counter()
        ce_pipeline.reranker.rerank(query, rrf_results, top_k=CE_TOP_K)
        timings["ce_rerank_ms"].append((time.perf_counter() - t0) * 1000)

    results: dict = {"description": "各阶段平均延迟 (ms)", "timings": {}}
    print(f"\n  {'Stage':<22s} {'Mean':>8s}  {'Min':>8s}  {'Max':>8s}")
    print(f"  {'-'*50}")

    for stage in ["bm25_search_ms", "vector_search_ms", "rrf_fusion_ms", "ce_rerank_ms"]:
        vals = timings[stage]
        mean_val = sum(vals) / len(vals)
        min_val = min(vals)
        max_val = max(vals)
        results["timings"][stage] = {
            "mean_ms": round(mean_val, 2),
            "min_ms": round(min_val, 2),
            "max_ms": round(max_val, 2),
        }
        print(f"  {stage:<22s} {mean_val:>8.2f}  {min_val:>8.2f}  {max_val:>8.2f}")

    # 总耗时
    total_mean = sum(
        results["timings"][s]["mean_ms"]
        for s in results["timings"]
    )
    results["total_mean_ms"] = round(total_mean, 2)
    print(f"  {'-'*50}")
    print(f"  {'Pipeline Total':<22s} {total_mean:>8.2f} ms")

    _save_json(results, "latency_profile.json")
    return results


# ============================================================
# 实验 5: 单 Query 深度分析
# ============================================================


def run_query_deep_dive(test_cases: list[dict] | None = None) -> dict:
    """选取典型 query 做链路追踪：展示每个模块对结果的影响。

    选取 S03（BM25 失败的语义查询）和 E03（BM25 擅长的精确匹配）做对比。
    """
    if test_cases is None:
        test_cases = _load_test_cases()

    print("\n" + "=" * 60)
    print("  实验 5: Query Deep Dive")
    print("=" * 60)

    # 选取代表性 query
    picks = {"E03": None, "S03": None, "M01": None}
    for tc in test_cases:
        if tc["id"] in picks:
            picks[tc["id"]] = tc

    chunks = process_directory()
    bm25 = BM25Retriever(chunks)
    vector = VectorRetriever(chunks, rebuild=True)
    pipeline = FusionPipeline()

    results: dict = {"description": "典型 Query 的全链路追踪", "queries": {}}

    for qid, tc in picks.items():
        if tc is None:
            continue

        query = tc["query"]
        golden = tc["golden_chunk_sources"]

        bm25_r = bm25.search(query, top_k=5)
        vector_r = vector.search(query, top_k=5)
        rrf_r = reciprocal_rank_fusion([bm25_r, vector_r])
        ce_r = pipeline.reranker.rerank(query, rrf_r, top_k=CE_TOP_K)

        def _top_info(rl: list[RetrievalResult], k: int = 5) -> list[dict]:
            return [
                {
                    "rank": i + 1,
                    "score": round(r.score, 4),
                    "source": r.chunk.metadata.get("source", ""),
                    "headings": r.chunk.metadata.get("heading_breadcrumb", ""),
                    "relevant": _is_relevant(r, golden),
                    "preview": r.chunk.content[:80],
                }
                for i, r in enumerate(rl[:k])
            ]

        q_info = {
            "query": query,
            "category": tc["category"],
            "bm25_top5": _top_info(bm25_r),
            "vector_top5": _top_info(vector_r),
            "rrf_top5": _top_info(rrf_r),
            "ce_top5": _top_info(ce_r),
            "bm25_mrr": _mrr(bm25_r, golden),
            "vector_mrr": _mrr(vector_r, golden),
            "rrf_mrr": _mrr(rrf_r, golden),
            "ce_mrr": _mrr(ce_r, golden),
        }
        results["queries"][qid] = q_info

        print(f"\n  [{qid}] {query}")
        print(f"    Category: {tc['category']}")
        print(f"    BM25 MRR={q_info['bm25_mrr']:.4f} → "
              f"Vector MRR={q_info['vector_mrr']:.4f} → "
              f"RRF MRR={q_info['rrf_mrr']:.4f} → "
              f"CE MRR={q_info['ce_mrr']:.4f}")

        # 展示各阶段 Top-3 的来源和相关性
        for stage_name, stage_results in [
            ("BM25", bm25_r),
            ("Vector", vector_r),
            ("CE", ce_r),
        ]:
            top3_sources = []
            for r in stage_results[:3]:
                rel = "[HIT]" if _is_relevant(r, golden) else "[MISS]"
                src = r.chunk.metadata.get("source", "?")
                top3_sources.append(f"{rel} {src}")
            print(f"    {stage_name:<8s} Top-3: {' | '.join(top3_sources)}")

    _save_json(results, "query_deep_dive.json")
    return results


# ============================================================
# 工具函数
# ============================================================


def _bar(value: float, max_val: float = 1.0, width: int = 25) -> str:
    filled = int(value / max_val * width)
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def _load_if_exists(filename: str) -> dict | None:
    path = EXPERIMENTS_DIR / filename
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def _save_json(data: dict, filename: str) -> None:
    path = EXPERIMENTS_DIR / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  → 已保存: {path}")


# ============================================================
# 综合报告
# ============================================================


def generate_report(
    ablation: dict | None = None,
    category: dict | None = None,
    latency: dict | None = None,
) -> str:
    """基于实验结果生成文字分析报告。"""
    lines: list[str] = []

    lines.append("=" * 60)
    lines.append("  RAG 系统消融实验 — 分析报告")
    lines.append("=" * 60)

    # --- 模块消融 ---
    if ablation and "results" in ablation:
        lines.append("\n## 1. 模块贡献分析\n")
        ar = ablation["results"]

        lines.append("| 配置 | MRR | Hit@5 | Prec@5 | Δ MRR |")
        lines.append("|------|-----|-------|--------|-------|")
        for name, metrics in ar.items():
            lines.append(
                f"| {name} | {metrics['mrr']:.4f} | "
                f"{metrics['hit@5']:.4f} | {metrics['precision@5']:.4f} | "
                f"{metrics.get('delta_mrr', 0):+.4f} |"
            )

        # 关键发现
        bm25_mrr = ar.get("bm25_only", {}).get("mrr", 0)
        vector_mrr = ar.get("vector_only", {}).get("mrr", 0)
        full_mrr = ar.get("full_pipeline", {}).get("mrr", 0)
        rrf_mrr = ar.get("rrf_fusion", {}).get("mrr", 0)
        concat_mrr = ar.get("concat_naive", {}).get("mrr", 0)

        lines.append(f"\n**BM25 → Full Pipeline MRR 提升: {full_mrr - bm25_mrr:+.4f}**")
        lines.append(f"**Vector → Full Pipeline MRR 提升: {full_mrr - vector_mrr:+.4f}**")

        if rrf_mrr > concat_mrr:
            lines.append(f"\n[OK] RRF ({rrf_mrr:.4f}) 优于 Naive Concat ({concat_mrr:.4f})，"
                         f"验证了排名融合对消除量纲差异的作用。")
        else:
            lines.append(f"\n[WARN] RRF 未优于 Naive Concat，小语料下排名融合的优势不明显。")

    # --- 分类分析 ---
    if category and "categories" in category:
        lines.append("\n## 2. 分类表现\n")

        lines.append("| 类别 | BM25 MRR | Vector MRR | Full MRR | BM25→Full Δ |")
        lines.append("|------|----------|------------|----------|-------------|")
        for cat_name, cat_data in category["categories"].items():
            b_mrr = cat_data["bm25_only"]["mrr"]
            v_mrr = cat_data["vector_only"]["mrr"]
            f_mrr = cat_data["full_pipeline"]["mrr"]
            gain = cat_data["bm25_to_full_gain"]["mrr"]
            lines.append(f"| {cat_name} | {b_mrr:.4f} | {v_mrr:.4f} | {f_mrr:.4f} | {gain:+.4f} |")

        # 关键发现
        exact = category["categories"].get("exact_match", {})
        semantic = category["categories"].get("semantic", {})

        if exact and semantic:
            e_bm25 = exact.get("bm25_only", {}).get("mrr", 0)
            e_vec = exact.get("vector_only", {}).get("mrr", 0)
            s_bm25 = semantic.get("bm25_only", {}).get("mrr", 0)
            s_vec = semantic.get("vector_only", {}).get("mrr", 0)

            lines.append(f"\n**BM25 在 exact_match 上的优势**: "
                         f"BM25={e_bm25:.4f} vs Vector={e_vec:.4f}")
            lines.append(f"**Vector 在 semantic 上的优势**: "
                         f"Vector={s_vec:.4f} vs BM25={s_bm25:.4f}")

            if s_bm25 < s_vec:
                lines.append("\n[OK] 验证结论：BM25 对专有名词精确匹配更好，"
                             "向量检索对语义相似查询更优。两者互补，混合召回是正确架构。")

    # --- 延迟 ---
    if latency and "timings" in latency:
        lines.append("\n## 3. 延迟分析\n")

        timings = latency["timings"]
        lines.append("| 阶段 | 平均延迟 (ms) |")
        lines.append("|------|-------------|")
        for stage, data in timings.items():
            label = stage.replace("_ms", "").replace("_", " ").title()
            lines.append(f"| {label} | {data['mean_ms']:.2f} |")

        total = latency.get("total_mean_ms", 0)
        lines.append(f"\n**Pipeline 总延迟: {total:.2f} ms**")

        # 瓶颈分析
        if timings:
            slowest = max(timings.items(), key=lambda x: x[1]["mean_ms"])
            lines.append(f"瓶颈阶段: {slowest[0]} ({slowest[1]['mean_ms']:.2f} ms)")

    lines.append(f"\n{'='*60}")
    lines.append("  实验数据文件:")
    lines.append(f"    {EXPERIMENTS_DIR / 'ablation_results.json'}")
    lines.append(f"    {EXPERIMENTS_DIR / 'category_breakdown.json'}")
    lines.append(f"    {EXPERIMENTS_DIR / 'parameter_sweep.json'}")
    lines.append(f"    {EXPERIMENTS_DIR / 'latency_profile.json'}")
    lines.append(f"    {EXPERIMENTS_DIR / 'query_deep_dive.json'}")
    lines.append(f"{'='*60}")

    report = "\n".join(lines)
    with open(EXPERIMENTS_DIR / "report.md", "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n  → 报告已保存: {EXPERIMENTS_DIR / 'report.md'}")

    return report


# ============================================================
# CLI 入口
# ============================================================


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.WARNING)

    quick = "--quick" in sys.argv
    export_only = "--export" in sys.argv

    if export_only:
        # 仅从已有 JSON 生成报告
        ablation = _load_if_exists("ablation_results.json")
        category = _load_if_exists("category_breakdown.json")
        latency = _load_if_exists("latency_profile.json")
        print(generate_report(ablation, category, latency))
        raise SystemExit(0)

    print("=" * 60)
    print("  RAG 消融实验套件")
    mode = "快速" if quick else "完整"
    print(f"  模式: {mode}")
    print(f"  输出目录: {EXPERIMENTS_DIR}")
    print("=" * 60)

    test_cases = _load_test_cases()

    # 实验 1: 模块消融
    ablation = run_module_ablation(test_cases)

    # 实验 2: 分类别
    category = run_category_breakdown(test_cases)

    # 实验 3: 参数扫描
    sweep = run_parameter_sweep(test_cases, quick=quick)

    # 实验 4: 延迟
    latency = run_latency_profile(test_cases)

    # 实验 5: 深度追踪
    deep_dive = run_query_deep_dive(test_cases)

    # 生成报告
    print(generate_report(ablation, category, latency))
