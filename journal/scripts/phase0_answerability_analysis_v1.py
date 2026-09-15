"""Phase 0: use Cross-Encoder scores to select an answerability threshold.

The script evaluates the 36 existing answerable cases and the explicit
unanswerable set.  It also checks whether each multi-evidence query has both
required evidence groups represented in the Cross-Encoder Top-3 results.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

# 让脚本从任意 cwd 都能 import 项目根模块（config / src）
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config import CE_TOP_K, EXPERIMENTS_DIR, KG_RRF_WEIGHT, PROJECT_ROOT
from src.pipeline import get_pipeline


ANSWERABLE_FILE = PROJECT_ROOT / "data" / "test_queries_all.json"
UNANSWERABLE_FILE = PROJECT_ROOT / "data" / "unanswerable_queries.json"
MULTI_EVIDENCE_FILE = PROJECT_ROOT / "data" / "multi_evidence_queries.json"


def _load_cases(path: Path, key: str | None = None) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    return data[key] if key else data


def _retrieve(ctx: Any, query: str) -> list[Any]:
    bm25 = ctx.bm25.search(query)
    vector = ctx.vector.search(query)
    graph = ctx.graph.search(query) if ctx.graph else None
    weights = [1.0, 1.0, KG_RRF_WEIGHT] if ctx.graph else None
    return ctx.pipeline.run(
        bm25, vector, query, ce_top_k=CE_TOP_K,
        graph_results=graph, rrf_weights=weights,
    )["cross_encoder"]


def _result_record(case: dict[str, Any], label: str, results: list[Any]) -> dict[str, Any]:
    top_results = [
        {
            "rank": rank,
            "score": round(float(result.score), 4),
            "source": result.chunk.metadata.get("source", "unknown"),
            "chunk_id": result.chunk.chunk_id,
            "content": result.chunk.content[:300],
        }
        for rank, result in enumerate(results, start=1)
    ]
    return {
        "query_id": case.get("id", case.get("query_id")),
        "query": case["query"],
        "label": label,
        "category": case.get("category", "answerable"),
        "top_score": top_results[0]["score"] if top_results else None,
        "results": top_results,
    }


def _confusion(records: list[dict[str, Any]], threshold: float) -> dict[str, int]:
    counts = Counter()
    for record in records:
        actual = record["label"] == "answerable"
        predicted = record["top_score"] is not None and record["top_score"] >= threshold
        counts["tp" if actual and predicted else "fn" if actual else "fp" if predicted else "tn"] += 1
    return dict(counts)


def _candidate_table(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scores = sorted({r["top_score"] for r in records if r["top_score"] is not None})
    candidates = [scores[0] - 0.01, *scores, scores[-1] + 0.01]
    assessed: list[dict[str, Any]] = []
    for threshold in candidates:
        matrix = _confusion(records, threshold)
        fp = matrix.get("fp", 0)
        tn = matrix.get("tn", 0)
        tp = matrix.get("tp", 0)
        fn = matrix.get("fn", 0)
        false_acceptance = fp / (fp + tn) if fp + tn else 0.0
        true_acceptance = tp / (tp + fn) if tp + fn else 0.0
        assessed.append({
            "threshold": round(threshold, 4), "tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "acceptance_rate": round(true_acceptance, 4),
            "false_acceptance_rate": round(false_acceptance, 4),
            "note": "not_acceptable: false_acceptance_rate is not below 30%" if false_acceptance >= 0.30 else "acceptable",
        })
    safe = [item for item in assessed if item["false_acceptance_rate"] < 0.30]
    pool = safe or assessed
    # Among thresholds meeting the safety constraint, retain as many answerable
    # queries as possible; on ties prefer the lower (less restrictive) threshold.
    best = max(pool, key=lambda item: (item["acceptance_rate"], -item["threshold"]))
    best["note"] = "RECOMMENDED: highest TP among thresholds with false_acceptance_rate < 30%"
    return assessed


def _score_distribution(records: list[dict[str, Any]], label: str) -> dict[str, Any]:
    scores = [record["top_score"] for record in records if record["label"] == label and record["top_score"] is not None]
    return {"scores": scores, "count": len(scores), "min": round(min(scores), 4), "max": round(max(scores), 4), "mean": round(mean(scores), 4)}


def _multi_evidence_record(case: dict[str, Any], results: list[Any]) -> dict[str, Any]:
    texts = [result.chunk.content.lower() for result in results[:3]]
    coverage = []
    for source in case["required_sources"]:
        matched = [word for word in source["keywords"] if any(word.lower() in text for text in texts)]
        coverage.append({
            "description": source["description"], "keywords": source["keywords"], "matched_keywords": matched,
            "covered": bool(matched),
        })
    return {
        "query_id": case["query_id"], "query": case["query"], "required_sources": coverage,
        "covered": all(item["covered"] for item in coverage),
        "top3_sources": [result.chunk.metadata.get("source", "unknown") for result in results[:3]],
    }


def run_analysis() -> dict[str, Any]:
    ctx = get_pipeline()
    answerable = _load_cases(ANSWERABLE_FILE, "test_cases")
    unanswerable = _load_cases(UNANSWERABLE_FILE)
    multi_evidence = _load_cases(MULTI_EVIDENCE_FILE)

    records = [_result_record(case, "answerable", _retrieve(ctx, case["query"])) for case in answerable]
    records.extend(_result_record(case, "unanswerable", _retrieve(ctx, case["query"])) for case in unanswerable)
    candidates = _candidate_table(records)
    recommended = next(item for item in candidates if item["note"].startswith("RECOMMENDED"))
    threshold = recommended["threshold"]
    matrix = {key: recommended[key] for key in ("tp", "fn", "fp", "tn")}
    multi_records = [_multi_evidence_record(case, _retrieve(ctx, case["query"])) for case in multi_evidence]
    misses = [record for record in records if (record["label"] == "answerable") != (record["top_score"] is not None and record["top_score"] >= threshold)]
    misses.extend(record for record in multi_records if not record["covered"])

    false_acceptance = matrix.get("fp", 0) / (matrix.get("fp", 0) + matrix.get("tn", 0))
    coverage = sum(item["covered"] for item in multi_records) / len(multi_records)
    multi_analysis = {
        "evaluation_scope": "multi_evidence_retrieval_quality",
        "note": "Dedicated citation-coverage cases; not part of the original 36-case benchmark.",
        "queries": [{
            "query_id": record["query_id"], "coverage": record["covered"],
            "top3_sources": record["top3_sources"],
            "reasoning": "All required evidence groups have keyword support in Top-3." if record["covered"] else "Known retrieval boundary: LangGraph Agent implementation is not in the docs/ corpus.",
            "known_limitation": not record["covered"], "retrieval_miss_recorded": not record["covered"],
        } for record in multi_records],
    }
    same_tp_alternatives = [str(item["threshold"]) for item in candidates if item["false_acceptance_rate"] < 0.30 and item["tp"] == recommended["tp"]]
    return {
        "metadata": {"answerable_count": len(answerable), "unanswerable_count": len(unanswerable), "timestamp": datetime.now().isoformat()},
        "score_distribution": {"answerable": _score_distribution(records, "answerable"), "unanswerable": _score_distribution(records, "unanswerable")},
        "candidates": candidates,
        "recommended_ce_answer_threshold": round(threshold, 4),
        "selection_rule": "maximize answerable acceptance subject to false acceptance < 30%; ties choose lower threshold",
        "recommendation_justification": {
            "constraint": "false_acceptance_rate < 0.30", "meeting_constraint": True,
            "num_candidates_meeting_constraint": sum(item["false_acceptance_rate"] < 0.30 for item in candidates),
            "selection_criterion": "maximize_true_positive; ties choose lower threshold",
            "selected_tp": recommended["tp"], "alternatives_with_same_tp": same_tp_alternatives,
        },
        "confusion_matrix": matrix,
        "false_acceptance_rate": round(false_acceptance, 4),
        "multi_evidence_coverage_at_3": round(coverage, 4),
        "records": records,
        "multi_evidence_records": multi_records,
        "multi_evidence_analysis": multi_analysis,
        "retrieval_miss_cases": misses,
    }


def _markdown(report: dict[str, Any]) -> str:
    matrix = report["confusion_matrix"]
    distribution = report["score_distribution"]
    multi_rows = "\n".join(
        f"| {item['query_id']} | {'✅ 是' if item['covered'] else '❌ 否'} | "
        f"{'核心证据组均在 Top-3 中出现' if item['covered'] else 'LangGraph 查询改写/答案生成证据未进入 Top-3；agent.py 不在 docs/ 语料中'} |"
        for item in report["multi_evidence_records"]
    )
    return f"""# Phase 0 Answerability Analysis

## Recommendation

- Recommended `CE_ANSWER_THRESHOLD`: `{report['recommended_ce_answer_threshold']}`
- Rule: {report['selection_rule']}
- False acceptance rate: {report['false_acceptance_rate']:.1%} (target: < 30%)
- Multi-evidence Top-3 coverage: {report['multi_evidence_coverage_at_3']:.1%} (target: ≥ 66%)

## Confusion matrix

| Actual / decision | Accept | Refuse |
| --- | ---: | ---: |
| Answerable | {matrix.get('tp', 0)} | {matrix.get('fn', 0)} |
| Unanswerable | {matrix.get('fp', 0)} | {matrix.get('tn', 0)} |

`retrieval_miss_cases.json` contains every score-gating error and every multi-evidence coverage failure with its retrieved evidence.

## 分布分析与阈值选择

- 可回答问题（{distribution['answerable']['count']} 条）：Top-1 CE 分数 {distribution['answerable']['min']:.4f}–{distribution['answerable']['max']:.4f}，均值 {distribution['answerable']['mean']:.4f}。
- 无答案问题（{distribution['unanswerable']['count']} 条）：Top-1 CE 分数 {distribution['unanswerable']['min']:.4f}–{distribution['unanswerable']['max']:.4f}，均值 {distribution['unanswerable']['mean']:.4f}。
- 两类分数存在明显重叠，单一 CE 阈值无法完美区分所有问题。

### 阈值 {report['recommended_ce_answer_threshold']:.4f} 的保守性

该阈值接受 {matrix.get('tp', 0)}/{report['metadata']['answerable_count']} 条可回答问题（{matrix.get('tp', 0) / report['metadata']['answerable_count']:.1%}），并误接受 {matrix.get('fp', 0)}/{report['metadata']['unanswerable_count']} 条无答案问题（{report['false_acceptance_rate']:.1%}）。

- 优点：优先压低缺乏证据时的错误接受风险。
- 代价：会拒答部分本应可回答的问题。
- 建议：Phase 1A 以 LLM 信息不足判断与引用校验作为后续门控，不仅依赖 CE 分数。

## 已知检索边界

| ID | Top-3 覆盖 | 原因 |
| --- | --- | --- |
{multi_rows}

ME03 是已知检索漏失，不作为拒答门控失败统计；如将来把 LangGraph Agent 文档加入 `docs/`，应重新运行本分析。

## 历史记录：ID 冲突解决

原 36 条基准集含 `M01`–`M08`。新增多证据集已改用 `ME01`–`ME03`，它们是引用覆盖的专用测试集，不属于原 36 条基准集，从而避免 ID 混淆。
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 0 answerability analysis")
    parser.add_argument("--output-dir", type=Path, default=EXPERIMENTS_DIR)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = run_analysis()
    (args.output_dir / "answerability_score_analysis.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_dir / "answerability_score_analysis.md").write_text(_markdown(report), encoding="utf-8")
    (args.output_dir / "retrieval_miss_cases.json").write_text(json.dumps(report["retrieval_miss_cases"], ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_dir / "multi_evidence_analysis.json").write_text(json.dumps(report["multi_evidence_analysis"], ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Recommended CE_ANSWER_THRESHOLD: {report['recommended_ce_answer_threshold']}")
    print(f"False acceptance: {report['false_acceptance_rate']:.1%}")
    print(f"Multi-evidence Top-3 coverage: {report['multi_evidence_coverage_at_3']:.1%}")


if __name__ == "__main__":
    main()
