"""KG 单路诊断脚本。

对 test_queries.json 中的每个查询跑 KGRetriever.search()，
按 exact_match / semantic / mixed / graph 四类汇总 MRR/Hit@5，
并输出每条的实体抽取、节点匹配、Top-5 召回明细。
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path

from config import DOCS_DIR, TEST_QUERIES_FILE
from src.data_pipeline import process_directory
from src.graph_retriever import extract_entities_from_text
from src.kg_retriever import KGRetriever

logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(message)s")


def compute_mrr_hits(results: list[dict], k: int = 5) -> tuple[float, float]:
    rr_sum = 0.0
    hits = 0
    for r in results:
        rank = None
        for idx, chunk_source in enumerate(r["top_k_sources"][:k], start=1):
            if chunk_source in r["golden_sources"]:
                rank = idx
                break
        if rank is not None:
            rr_sum += 1.0 / rank
            hits += 1
    mrr = rr_sum / len(results) if results else 0.0
    hit = hits / len(results) if results else 0.0
    return mrr, hit


def main() -> None:
    print("=" * 70)
    print("KG 单路诊断")
    print("=" * 70)

    # 1. 加载测试集
    with open(TEST_QUERIES_FILE, encoding="utf-8") as f:
        test_data = json.load(f)
    cases = test_data["test_cases"]

    # 2. 构建 chunks
    print(f"\n[1/4] 处理文档目录: {DOCS_DIR}")
    chunks = process_directory(DOCS_DIR)
    print(f"      共 {len(chunks)} 个 chunk")

    # 3. 初始化 KG 检索器
    print("\n[2/4] 初始化 KGRetriever...")
    kg = KGRetriever(chunks)

    # 4. 逐条诊断
    print("\n[3/4] 对每个查询跑 KG 检索...")
    per_case: list[dict] = []
    for case in cases:
        query = case["query"]
        golden = set(case.get("golden_chunk_sources", []))

        # query 实体（和 KGRetriever 内部用的同一套）
        raw_entities = extract_entities_from_text(query, top_n=20)
        query_entities = [e for e, _ in raw_entities]

        # KG 检索
        results = kg.search(query, top_k=5)
        top_sources = [r.chunk.metadata.get("source", "unknown") for r in results]

        # 找命中位置
        hit_rank = None
        for idx, src in enumerate(top_sources, start=1):
            if src in golden:
                hit_rank = idx
                break

        per_case.append({
            "id": case["id"],
            "category": case["category"],
            "query": query,
            "golden_sources": list(golden),
            "query_entities": query_entities,
            "matched_nodes": [],  # 后面补
            "top_k_sources": top_sources,
            "top_k_scores": [round(r.score, 4) for r in results],
            "hit_rank": hit_rank,
            "num_results": len(results),
        })

    # 补 matched_nodes：用 KGRetriever 内部逻辑
    # 这里简单复刻 _extract_query_entities + _find_matched_nodes
    from src.kg_retriever import (
        _extract_query_entities,
        _find_matched_nodes,
    )
    all_nodes = set(kg.graph.nodes())
    for record in per_case:
        entities = _extract_query_entities(record["query"])
        matched = _find_matched_nodes(entities, all_nodes)
        record["matched_nodes"] = matched

    # 5. 输出每类汇总
    print("\n[4/4] 按查询类型汇总（Top-5）\n")
    by_category = defaultdict(list)
    for record in per_case:
        by_category[record["category"]].append(record)

    print(f"{'类别':<14} {'数量':>4} {'MRR@5':>8} {'Hit@5':>8}")
    print("-" * 40)
    overall = []
    for cat in ["exact_match", "semantic", "mixed", "graph"]:
        group = by_category.get(cat, [])
        if not group:
            continue
        mrr, hit = compute_mrr_hits(group, k=5)
        overall.extend(group)
        print(f"{cat:<14} {len(group):>4} {mrr:>8.4f} {hit:>8.4f}")
    print("-" * 40)
    overall_mrr, overall_hit = compute_mrr_hits(overall, k=5)
    print(f"{'overall':<14} {len(overall):>4} {overall_mrr:>8.4f} {overall_hit:>8.4f}")

    # 6. 输出详细明细
    print("\n" + "=" * 70)
    print("逐条明细")
    print("=" * 70)
    for r in per_case:
        status = f"✅ rank={r['hit_rank']}" if r["hit_rank"] else "❌ miss"
        print(f"\n[{r['id']}] {r['category']} | {status}")
        print(f"  query: {r['query']}")
        print(f"  entities: {r['query_entities'][:8]}")
        print(f"  matched_nodes: {r['matched_nodes'][:8]}")
        print(f"  golden: {r['golden_sources']}")
        print(f"  top5: {list(zip(r['top_k_sources'], r['top_k_scores']))}")

    # 7. 保存报告
    report_path = Path("experiments/kg_diag_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "summary": {
                    cat: {
                        "count": len(by_category.get(cat, [])),
                        "mrr@5": compute_mrr_hits(by_category.get(cat, []), k=5)[0],
                        "hit@5": compute_mrr_hits(by_category.get(cat, []), k=5)[1],
                    }
                    for cat in ["exact_match", "semantic", "mixed", "graph"]
                },
                "overall": {
                    "count": len(overall),
                    "mrr@5": overall_mrr,
                    "hit@5": overall_hit,
                },
                "cases": per_case,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\n报告已保存: {report_path}")


if __name__ == "__main__":
    main()
