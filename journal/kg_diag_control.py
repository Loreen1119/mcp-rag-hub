"""KG 单路对照实验：仅使用科普类文档。

排除个人笔记、实现文档、样例笔记等噪声来源，验证 KG 检索效果差
是否由知识库混杂导致。不修改、不删除任何原始数据。
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path

from config import DOCS_DIR, KG_TRIPLES_FILE, TEST_QUERIES_FILE
from src.data_pipeline import process_directory
from src.kg_retriever import KGRetriever, _extract_query_entities, _find_matched_nodes

logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(message)s")

# 对照实验使用的“干净”文档集合
CLEAN_DOCS = {
    "rag-intro.md",
    "embedding-guide.md",
    "chunking-strategies.md",
    "sample_rag_paper.md",
}

CONTROL_TRIPLES_FILE = KG_TRIPLES_FILE.with_suffix(".control.jsonl")


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


def build_control_triples() -> int:
    """从完整 triples 中过滤出仅属于 CLEAN_DOCS 的记录，写入临时文件。"""
    count = 0
    with open(KG_TRIPLES_FILE, encoding="utf-8") as fin, \
         open(CONTROL_TRIPLES_FILE, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("source_doc") in CLEAN_DOCS:
                fout.write(line + "\n")
                count += 1
    return count


def main() -> None:
    print("=" * 70)
    print("KG 单路对照实验 — 仅科普文档")
    print("=" * 70)
    print(f"\n使用文档: {sorted(CLEAN_DOCS)}")

    # 1. 加载测试集并过滤
    with open(TEST_QUERIES_FILE, encoding="utf-8") as f:
        test_data = json.load(f)
    all_cases = test_data["test_cases"]

    usable_cases = [
        c for c in all_cases
        if all(src in CLEAN_DOCS for src in c.get("golden_chunk_sources", []))
    ]
    skipped = len(all_cases) - len(usable_cases)
    print(f"\n测试集: {len(all_cases)} 条, 可用（golden 全在干净文档中）: {len(usable_cases)} 条, 跳过: {skipped} 条")

    # 2. 构建 chunks 并过滤到干净文档
    print(f"\n[1/4] 处理文档目录: {DOCS_DIR}")
    all_chunks = process_directory(DOCS_DIR)
    chunks = [c for c in all_chunks if c.metadata.get("source") in CLEAN_DOCS]
    print(f"      原始 {len(all_chunks)} 个 chunk, 过滤后 {len(chunks)} 个 chunk")

    # 3. 准备对照 triples 文件
    print("\n[2/4] 构建对照 triples 文件...")
    triple_count = build_control_triples()
    print(f"      从 {KG_TRIPLES_FILE} 过滤出 {triple_count} 条 triples 到 {CONTROL_TRIPLES_FILE}")

    # 4. 初始化 KG 检索器
    print("\n[3/4] 初始化 KGRetriever...")
    kg = KGRetriever(chunks, triples_file=CONTROL_TRIPLES_FILE)

    # 5. 逐条诊断
    print("\n[4/4] 对每个查询跑 KG 检索...")
    per_case: list[dict] = []
    for case in usable_cases:
        query = case["query"]
        golden = set(case.get("golden_chunk_sources", []))

        raw_entities = _extract_query_entities(query)

        results = kg.search(query, top_k=5)
        top_sources = [r.chunk.metadata.get("source", "unknown") for r in results]

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
            "query_entities": raw_entities,
            "matched_nodes": [],
            "top_k_sources": top_sources,
            "top_k_scores": [round(r.score, 4) for r in results],
            "hit_rank": hit_rank,
            "num_results": len(results),
        })

    all_nodes = set(kg.graph.nodes())
    for record in per_case:
        entities = _extract_query_entities(record["query"])
        matched = _find_matched_nodes(entities, all_nodes)
        record["matched_nodes"] = matched

    # 6. 输出汇总
    print("\n按查询类型汇总（Top-5）\n")
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

    # 7. 输出详细明细
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

    # 8. 保存报告
    report_path = Path("experiments/kg_diag_report_control.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "clean_docs": sorted(CLEAN_DOCS),
                "skipped_cases": skipped,
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
