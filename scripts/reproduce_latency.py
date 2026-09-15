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
    CE_TOP_K,
    CORPUS,
    EMBEDDING_MODEL,
    EXPERIMENTS_DIR,
    GRAPH_TOP_K,
    TEST_QUERIES_FILE,
    VECTOR_TOP_K,
)
from src.data_pipeline import process_directory  # noqa: E402
from src.evaluation.metrics import load_test_cases  # noqa: E402
from src.fusion import FusionPipeline, reciprocal_rank_fusion  # noqa: E402
from src.graph_retriever import GraphRetriever  # noqa: E402
from src.pipeline import corpus_collection_name  # noqa: E402
from src.retrievers import BM25Retriever, VectorRetriever  # noqa: E402

STAGES = [
    "bm25_search_ms",
    "vector_search_ms",
    "graph_search_ms",
    "rrf_fusion_ms",
    "ce_rerank_ms",
]


def main() -> dict:
    print("=" * 60)
    print("  延迟剖析（复现 experiments.py 的测量逻辑）")
    print("=" * 60)

    chunks = process_directory()
    print(f"  chunks = {len(chunks)}")

    bm25 = BM25Retriever(chunks)
    coll = corpus_collection_name(CORPUS, EMBEDDING_MODEL)
    print(f"  collection = {coll}（复用主库索引，不重建）")
    vector = VectorRetriever(chunks, rebuild=False, collection_name=coll)
    graph_retriever = GraphRetriever(chunks)
    ce_pipeline = FusionPipeline()

    test_cases = load_test_cases(TEST_QUERIES_FILE)
    print(f"  test cases = {len(test_cases)}")
    print()

    timings: dict[str, list[float]] = defaultdict(list)

    for tc in test_cases:
        q = tc["query"]

        t0 = time.perf_counter()
        bm25_results = bm25.search(q, top_k=BM25_TOP_K)
        timings["bm25_search_ms"].append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        vector_results = vector.search(q, top_k=VECTOR_TOP_K)
        timings["vector_search_ms"].append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        graph_results = graph_retriever.search(q, top_k=GRAPH_TOP_K)
        timings["graph_search_ms"].append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        rrf_results = reciprocal_rank_fusion([bm25_results, vector_results, graph_results])
        timings["rrf_fusion_ms"].append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        ce_pipeline.reranker.rerank(q, rrf_results, top_k=CE_TOP_K)
        timings["ce_rerank_ms"].append((time.perf_counter() - t0) * 1000)

    results: dict = {
        "description": "各阶段平均延迟 (ms)",
        "note": "由 scripts/reproduce_latency.py 产出（experiments.py 因 retrievers.py 的 rebuild 漏赋值不可用，待修项 C9）",
        "corpus": CORPUS,
        "n_queries": len(test_cases),
        "timings": {},
    }

    print(f"  {'Stage':<22s} {'Mean':>10s} {'Min':>10s} {'Max':>10s}")
    print(f"  {'-' * 56}")
    for stage in STAGES:
        vals = timings[stage]
        mean_v, min_v, max_v = sum(vals) / len(vals), min(vals), max(vals)
        results["timings"][stage] = {
            "mean_ms": round(mean_v, 2),
            "min_ms": round(min_v, 2),
            "max_ms": round(max_v, 2),
        }
        print(f"  {stage:<22s} {mean_v:>10.2f} {min_v:>10.2f} {max_v:>10.2f}")

    total = sum(v["mean_ms"] for v in results["timings"].values())
    results["total_mean_ms"] = round(total, 2)
    ce_ms = results["timings"]["ce_rerank_ms"]["mean_ms"]
    results["ce_share_pct"] = round(ce_ms / total * 100, 1) if total else 0.0
    print(f"  {'-' * 56}")
    print(f"  {'Pipeline Total':<22s} {total:>10.2f}")
    print(f"  CE 占比: {results['ce_share_pct']}%")

    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    out = Path(EXPERIMENTS_DIR) / "latency_profile.reproduced.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  → 已保存: {out}")
    return results


if __name__ == "__main__":
    main()
