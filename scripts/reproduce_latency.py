"""延迟剖析（复现 experiments.py:run_latency_profile 的测量逻辑）。

## 为什么单独有这个脚本

`src/evaluation/experiments.py` 目前**跑不了** —— `src/retrievers.py:169` 的
`if rebuild:` 分支调用 `self._rebuild_collection(...)` 却没把返回值赋给 `self.collection`，
紧接着的 logger 访问 `self.collection.name` 必然 `AttributeError`。
该 bug 影响 experiments.py 里全部 6 处实验（ablation / category / sweep / quick /
latency / deepdive），记为待修项 **C9**。

本脚本用「复用主库已有索引」绕开那条路径（`rebuild=False` + 显式 collection 名），
**测量点与 `run_latency_profile` 逐行一致**（同 top_k、同计时位置），故数字可比。

> C9 修好之后，本脚本仍可保留 —— 它不重建索引，比 `experiments --quick` 快得多。

## 用法

    set HF_HUB_OFFLINE=1 & set TRANSFORMERS_OFFLINE=1
    .venv\\Scripts\\python.exe -m scripts.reproduce_latency

产物：`experiments/latency_profile.reproduced.json`
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

# 支持从项目根或任意目录运行
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import (  # noqa: E402
    BM25_TOP_K,
    CE_CANDIDATE_K,
    CE_TOP_K,
    CORPUS,
    EXPERIMENTS_DIR,
    GRAPH_TOP_K,
    TEST_QUERIES_FILE,
    VECTOR_TOP_K,
)
from src.data_pipeline import process_directory  # noqa: E402
from src.evaluation.metrics import load_test_cases  # noqa: E402
from src.fusion import FusionPipeline, reciprocal_rank_fusion  # noqa: E402
from src.graph_retriever import GraphRetriever  # noqa: E402
from src.pipeline import open_vector_retriever  # noqa: E402
from src.retrievers import BM25Retriever  # noqa: E402

STAGES = [
    "bm25_search_ms",
    "vector_search_ms",
    "graph_search_ms",
    "rrf_fusion_ms",
    "ce_rerank_ms",
]


def _load_queries(source: str, n: int | None) -> list[str]:
    """取要测延迟的 query 列表。

    source=eval_set（默认）：从 data/eval_set.jsonl 取**可答**样本的 query。
      用评测集的 query 而不是旧的 18 条小集，保证与检索评测同分布、可直接对话。
    source=test_queries：旧的 fastapi-zh 18 条（仅作历史对照，与 TiDB 语料不可混比）。
    """
    if source == "test_queries":
        return [tc["query"] for tc in load_test_cases(TEST_QUERIES_FILE)][: (n or 10 ** 9)]
    p = _PROJECT_ROOT / "data" / "eval_set.jsonl"
    rows = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    qs = [r["query"] for r in rows if not r.get("unanswerable")]  # 可答样本才有意义（会走 CE 精排）
    return qs[: (n or 10 ** 9)]


def _percentile(sorted_vals: list[float], q: float) -> float:
    """nearest-rank 分位数（q=0.5/0.95）。

    用 nearest-rank 而不是线性插值：样本量小的时候它更诚实 ——
    n<=20 时 p95 本来就该贴近最大值（旧实现 ``vals[int(n*0.95)]`` 在 n<=20
    时恒等于 max，且没有原始样本可离线重算，故一并落 ``samples_ms``）。
    """
    import math

    n = len(sorted_vals)
    idx = min(n - 1, max(0, math.ceil(q * n) - 1))
    return sorted_vals[idx]


def main() -> dict:
    import argparse

    ap = argparse.ArgumentParser(description="延迟剖析（分阶段计时）")
    ap.add_argument("--source", choices=["eval_set", "test_queries"], default="eval_set",
                    help="query 来源；默认 eval_set（TiDB 评测集可答样本）")
    ap.add_argument("--n", type=int, default=20, help="测多少条 query（默认 20）")
    ap.add_argument("--warmup", type=int, default=2,
                    help="预热条数（不计入统计；CE 首次前向含惰性初始化，不预热会虚高）")
    ap.add_argument("--with-graph", action="store_true",
                    help="是否把图检索算进链路。生产默认关闭（ENABLE_KG=false / 图路未启用），默认不计。")
    ap.add_argument("--ce-candidate-k", type=int, default=CE_CANDIDATE_K,
                    help="RRF 后送 CE 的候选截断数（生产配置 30）。做对照实验时另跑一档更大的 K "
                         "（如 50 = 未截断），对外表述固定为「候选截断至 30（对照未截断 50）」，"
                         "**不得写成「从 50 优化到 30」**：50 不是历史配置，是实验构造的对照臂。")
    ap.add_argument("--out", default="experiments/latency_profile_tidb.json",
                    help="产物路径（默认 experiments/latency_profile_tidb.json，不覆盖旧的 reproduced 文件）")
    args = ap.parse_args()

    print("=" * 60)
    print("  延迟剖析（分阶段计时）")
    print(f"  语料={CORPUS}  链路={'BM25+向量+图+RRF+CE' if args.with_graph else 'BM25+向量+RRF+CE（生产默认，无图路）'}")
    print("  ⚠️ 必须在**机器空闲**时跑：有负载时全线慢 1.35~2.66 倍（实测 12.8s vs 34s）")
    print("=" * 60)

    chunks = process_directory()
    print(f"  chunks = {len(chunks)}")

    bm25 = BM25Retriever(chunks)
    # ⚠️ 必须走规范入口：corpus_hash 与 collection 名要成对给，否则摘要校验必败、
    # 每次跑都删集合重嵌入 10.4 万 chunk（2026-10-15 那次 78.9 分钟的真身）。
    vector = open_vector_retriever(chunks)
    print(f"  collection = {vector.collection.name}（复用主库索引，不重建）")
    graph_retriever = GraphRetriever(chunks) if args.with_graph else None
    ce_pipeline = FusionPipeline()

    queries = _load_queries(args.source, args.n)
    if not queries:
        raise SystemExit("没有取到 query，检查 --source / data/eval_set.jsonl")
    print(f"  queries = {len(queries)}（source={args.source}）  warmup = {min(args.warmup, len(queries))}")
    print()

    timings: dict[str, list[float]] = defaultdict(list)

    def _one(q: str, record: bool) -> None:
        t0 = time.perf_counter()
        bm25_results = bm25.search(q, top_k=BM25_TOP_K)
        if record:
            timings["bm25_search_ms"].append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        vector_results = vector.search(q, top_k=VECTOR_TOP_K)
        if record:
            timings["vector_search_ms"].append((time.perf_counter() - t0) * 1000)

        rankings = [bm25_results, vector_results]
        if graph_retriever is not None:
            t0 = time.perf_counter()
            rankings.append(graph_retriever.search(q, top_k=GRAPH_TOP_K))
            if record:
                timings["graph_search_ms"].append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        rrf_results = reciprocal_rank_fusion(rankings)
        if record:
            timings["rrf_fusion_ms"].append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        ce_pipeline.reranker.rerank(q, rrf_results, top_k=CE_TOP_K, candidate_cap=args.ce_candidate_k)
        if record:
            timings["ce_rerank_ms"].append((time.perf_counter() - t0) * 1000)

    for q in queries[: args.warmup]:
        _one(q, record=False)
    for q in queries[args.warmup:]:
        _one(q, record=True)

    active_stages = [s for s in STAGES if timings[s]]
    results: dict = {
        "description": "各阶段延迟（mean / P50 / P95 / max，单位 ms）；n_queries 与 percentile_method 一并落盘，引用时必须连样本量一起写",
        "percentile_method": "nearest-rank（P50=ceil(0.5n)、P95=ceil(0.95n)）",
        "note": "由 scripts/reproduce_latency.py 产出；机器空闲时测量",
        "corpus": CORPUS,
        "n_queries": len(timings["bm25_search_ms"]),
        "warmup": args.warmup,
        "ce_candidate_k": args.ce_candidate_k,
        "arm_note": (f"本臂 ce_candidate_k={args.ce_candidate_k}（生产配置 {CE_CANDIDATE_K}）。"
                     "对外统一表述「候选截断至 30（对照臂 K=50）」，不得写成「从 50 优化到 30」："
                     "更大的 K 不是历史配置，是为对照实验构造出来的臂，不是优化前的线上状态。"),
        "with_graph": args.with_graph,
        "source": args.source,
        "timings": {},
    }

    print(f"  {'Stage':<22s} {'Mean':>10s} {'P50':>10s} {'P95':>10s} {'Max':>10s}")
    print(f"  {'-' * 68}")
    for stage in active_stages:
        vals = sorted(timings[stage])
        mean_v, min_v, max_v = sum(vals) / len(vals), vals[0], vals[-1]
        p50 = _percentile(vals, 0.50)
        p95 = _percentile(vals, 0.95)
        results["timings"][stage] = {
            "mean_ms": round(mean_v, 2),
            "min_ms": round(min_v, 2),
            "p50_ms": round(p50, 2),
            "p95_ms": round(p95, 2),
            "max_ms": round(max_v, 2),
            # 原始样本全量留下：口径或分位数算法要改时不必重跑（一次重跑 ~10 min 起）
            "samples_ms": [round(v, 2) for v in vals],
        }
        print(f"  {stage:<22s} {mean_v:>10.2f} {p50:>10.2f} {p95:>10.2f} {max_v:>10.2f}")

    total = sum(v["mean_ms"] for v in results["timings"].values())
    results["total_mean_ms"] = round(total, 2)
    ce_ms = results["timings"].get("ce_rerank_ms", {}).get("mean_ms", 0.0)
    results["ce_share_pct"] = round(ce_ms / total * 100, 1) if total else 0.0
    print(f"  {'-' * 68}")
    print(f"  {'Pipeline Total':<22s} {total:>10.2f}")
    print(f"  CE 占比: {results['ce_share_pct']}%")

    out = Path(args.out)
    if not out.is_absolute():
        out = _PROJECT_ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  → 已保存: {out}")
    return results


if __name__ == "__main__":
    main()
