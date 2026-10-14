#!/usr/bin/env python
"""版本混淆独立集评测（[Z]）—— TiDB 中文评测集 · 版本混淆探针 31 条。

为什么单独一个脚本、不并进主集：
  data/eval_set_version.jsonl 是**独立探针集**（data/eval_set_FROZEN.json 明确
  「主集 375 + 版本混淆独立集 31 = 406」）。主集 Hit@5 按 slug 级判定，而同一 slug
  在 docs-cn/ 与 docs-cn-v8.4/ 同时存在、**stem 完全相同** → slug 级无法区分版本，
  必须按 metadata["source"] 的目录前缀（docs-cn/ vs docs-cn-v8.4/）判定。
  故 [Z] 不能复用主集口径，需单独跑、单独报，不与主集 Hit@5 混。

口径：
  - 每条探针的 golden_slug 在语料里都有两个版本页（已核验 31/31 均 new+old 双有）。
  - correct：src 以 docs-cn/ 开头且 stem == golden_slug（正确版本页）。
  - twin   ：src 以 docs-cn-v8.4/ 开头且 stem == golden_slug（旧版本差异页）。
  - **版本混淆干扰率 [Z] = top-5 内出现旧版本页（twin）的探针占比**（与主集
    「版本孪生命中」诊断同口径）。
  - 危害口径：twin_outranks_correct（旧版页压过正确版页，或正确版页未进 top-5
    而旧版进了）—— 这才是用户真会看到错误版本答案的情形。

工程约束（同 scripts/eval_baseline_ladder.py）：
  - 纯本地离线，清空全部 *_PROXY（含 ALL_PROXY），不联网、不重建索引。
  - 逐条落盘 + 按 id 续跑（续跑判据 = 该 id 是否已含目标档）；评测集只读。
  - 复用 eval_baseline_ladder 的 Appender / log / 环境准备，避免第二份实现。

用法：
    python -m scripts.eval_version_confusion --stage all      # 四档全跑（含 CE）
    python -m scripts.eval_version_confusion --stage retrieval # 只跑 BM25/向量/RRF
    python -m scripts.eval_version_confusion --stage ce        # 含 CE
    python -m scripts.eval_version_confusion --stage report    # 只汇总
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

# 复用主脚本的环境准备（离线变量 + 清代理 + sys.path）+ 落盘工具
from scripts.eval_baseline_ladder import (  # noqa: E402
    ROOT,
    Appender,
    log,
    DIAG_TOP,
    RRF_STORE_TOP,
)

from config import (  # noqa: E402
    BM25_TOP_K,
    VECTOR_TOP_K,
    CE_TOP_K,
    CE_CANDIDATE_K,
    CORPUS,
    EMBEDDING_MODEL,
)
from src.fusion import reciprocal_rank_fusion  # noqa: E402
from src.pipeline import corpus_collection_name, get_pipeline  # noqa: E402

EVAL_SET = ROOT / "data" / "eval_set_version.jsonl"
OUT_DIR = ROOT / "experiments" / "version_confusion"
RECORDS = OUT_DIR / "version_confusion_31.jsonl"
SUMMARY_FILE = OUT_DIR / "summary.json"

HEAD_IDS = {f"ver_{i}" for i in range(1, 10)}  # 头部 9 条：差异点精准
STAGES_ALL = ["bm25", "vector", "rrf", "ce"]


def load_eval_set() -> list[dict]:
    return [json.loads(l) for l in open(EVAL_SET, encoding="utf-8") if l.strip()]


def version_of(src: str) -> str:
    """按 source 目录前缀判版本：docs-cn-v8.4/ = old，docs-cn/ = new，其余 = other。"""
    if src.startswith("docs-cn-v8.4/"):
        return "old"
    if src.startswith("docs-cn/"):
        return "new"
    return "other"


def classify(results: list, golden_slug: str) -> tuple[list, dict]:
    """把一路检索结果按版本分类，给出该档的版本混淆指标。"""
    rows: list[dict] = []
    correct_rank = None
    twin_rank = None
    golden_best_rank = None
    for rank, r in enumerate(results, start=1):
        src = r.chunk.metadata.get("source", "")
        slug = Path(src).stem
        ver = version_of(src)
        is_golden_slug = slug == golden_slug
        rows.append(
            {
                "rank": rank,
                "slug": slug,
                "src": src,
                "version": ver,
                "score": round(float(r.score), 6),
                "is_golden_slug": bool(is_golden_slug),
            }
        )
        if is_golden_slug:
            if golden_best_rank is None:
                golden_best_rank = rank
            if ver == "new" and correct_rank is None:
                correct_rank = rank
            if ver == "old" and twin_rank is None:
                twin_rank = rank

    def in_top5(rk):
        return rk is not None and rk <= 5

    correct_in_top5 = in_top5(correct_rank)
    twin_in_top5 = in_top5(twin_rank)
    # 旧版页压过正确版页（正确版未进 top-5 而旧版进了，或旧版排名更靠前）
    twin_outranks_correct = bool(
        twin_in_top5 and (not correct_in_top5 or (twin_rank < correct_rank))
    )
    top5 = rows[:5]
    stats = {
        "n_results": len(rows),
        "correct_rank": correct_rank,
        "twin_rank": twin_rank,
        "golden_best_rank": golden_best_rank,
        "correct_in_top5": bool(correct_in_top5),
        "twin_in_top5": bool(twin_in_top5),
        "any_golden_slug_in_top5": bool(in_top5(golden_best_rank)),
        "twin_outranks_correct": twin_outranks_correct,
        "top1_is_twin": bool(top5 and top5[0]["is_golden_slug"] and top5[0]["version"] == "old"),
        "twin_share_top5": round(sum(1 for x in top5 if x["version"] == "old") / len(top5), 4)
        if top5
        else 0.0,
    }
    return rows, stats


def read_records() -> list[dict]:
    if not RECORDS.exists():
        return []
    return [json.loads(l) for l in open(RECORDS, encoding="utf-8") if l.strip()]


def run(rows: list[dict], do_ce: bool) -> None:
    need_stage = "ce" if do_ce else "rrf"
    done = {r["id"] for r in read_records() if need_stage in r.get("stages", {})}
    todo = [r for r in rows if r["id"] not in done]
    log(f"版本混淆探针：共 {len(rows)} 条，已含 {need_stage} 档 {len(done)}，本次待跑 {len(todo)}")
    if not todo:
        log("已全部完成，跳过（重跑续上逻辑生效）")
        return

    ctx = get_pipeline()
    log(f"管线就绪 — chunks={len(ctx.chunks)} collection={ctx.vector.collection.name} "
        f"count={ctx.vector.collection.count()}")
    if do_ce:
        log("加载 CE 模型（约 1.1GB）...")
        ctx.pipeline.reranker  # 触发懒加载
        log("CE 就绪")

    # 含 CE 的档位重跑时，先把同 id 的旧记录（仅 retrieval）剔除，避免 jsonl 出现重复 id
    if do_ce and RECORDS.exists():
        todo_ids = {x["id"] for x in todo}
        kept = [r for r in read_records() if r["id"] not in todo_ids]
        with open(RECORDS, "w", encoding="utf-8") as fh:
            for r in kept:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    app = Appender(RECORDS)
    t0 = time.time()
    try:
        for i, tc in enumerate(todo, start=1):
            q = tc["query"]
            g = tc["golden_slug"]
            tq = time.time()
            stages: dict[str, dict] = {}

            bm25_res = ctx.bm25.search(q, top_k=DIAG_TOP)
            vec_res = ctx.vector.search(q, top_k=DIAG_TOP)
            rrf_res = reciprocal_rank_fusion(
                [bm25_res[:BM25_TOP_K], vec_res[:VECTOR_TOP_K]], top_n=RRF_STORE_TOP
            )
            for name, res in (("bm25", bm25_res), ("vector", vec_res), ("rrf", rrf_res)):
                rank_rows, stats = classify(res, g)
                stages[name] = {"top": rank_rows, **stats}

            if do_ce:
                ce_res = ctx.pipeline.reranker.rerank(
                    q, rrf_res, top_k=CE_TOP_K, candidate_cap=CE_CANDIDATE_K
                )
                rank_rows, stats = classify(ce_res, g)
                stages["ce"] = {"top": rank_rows, **stats}

            app.write(
                {
                    "id": tc["id"],
                    "query": q,
                    "golden_slug": g,
                    "source": tc.get("source"),
                    "new_version": tc.get("new_version"),
                    "old_version": tc.get("old_version"),
                    "diff_point": tc.get("diff_point"),
                    "stages": stages,
                    "elapsed": round(time.time() - tq, 2),
                    "run_ts": datetime.now().isoformat(timespec="seconds"),
                }
            )
            flag = "干扰" if stages["rrf"]["twin_outranks_correct"] else "ok"
            log(f"  {i}/{len(todo)} {tc['id']:>6} rrf 正确@{stages['rrf']['correct_rank']} "
                f"旧版@{stages['rrf']['twin_rank']} [{flag}]  {round(time.time()-tq,1)}s")
    finally:
        app.close()
    log(f"完成，耗时 {(time.time()-t0)/60:.1f} min")


def summarize() -> dict:
    # 同 id 去重（保留档位更全的那条）
    by_id: dict[str, dict] = {}
    for r in read_records():
        if r["id"] not in by_id or len(r.get("stages", {})) > len(by_id[r["id"]].get("stages", {})):
            by_id[r["id"]] = r
    recs = list(by_id.values())
    stages = [s for s in STAGES_ALL if any(s in r["stages"] for r in recs)]

    def stat(stage, ids=None):
        sub = [r for r in recs if stage in r["stages"] and (ids is None or r["id"] in ids)]
        if not sub:
            return None
        n = len(sub)

        def rt(key):
            return round(sum(1 for r in sub if r["stages"][stage][key]) / n, 4)

        return {
            "n": n,
            "correct_in_top5": rt("correct_in_top5"),
            "twin_in_top5": rt("twin_in_top5"),                    # [Z] 主口径
            "any_golden_slug_in_top5": rt("any_golden_slug_in_top5"),
            "twin_outranks_correct": rt("twin_outranks_correct"),  # 危害口径
            "top1_is_twin": rt("top1_is_twin"),
            "avg_twin_share_top5": round(
                sum(r["stages"][stage]["twin_share_top5"] for r in sub) / n, 4),
            "mrr_correct": round(
                sum((1.0 / r["stages"][stage]["correct_rank"]) if r["stages"][stage]["correct_rank"] else 0.0
                    for r in sub) / n, 4),
            "harmful_ids": [r["id"] for r in sub if r["stages"][stage]["twin_outranks_correct"]],
        }

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "index": {
            "corpus": CORPUS,
            "embedding_model": EMBEDDING_MODEL,
            "collection": corpus_collection_name(CORPUS, EMBEDDING_MODEL),
        },
        "eval_set": {"file": str(EVAL_SET.relative_to(ROOT)), "n": len(recs)},
        "criteria": {
            "correct": "src 以 docs-cn/ 开头且 stem == golden_slug",
            "twin": "src 以 docs-cn-v8.4/ 开头且 stem == golden_slug",
            "Z_primary": "版本混淆干扰率 = top-5 内出现旧版本页(twin)的探针占比",
            "Z_harmful": "twin_outranks_correct = 旧版页压过正确版页（或正确版未进 top-5 而旧版进了）",
            "ce_candidate_k": CE_CANDIDATE_K,
        },
        "stages": {st: stat(st) for st in stages},
        "by_head_tail": {
            st: {
                "head9(精准差异点)": stat(st, HEAD_IDS),
                "tail22(泛query)": stat(st, {r["id"] for r in recs} - HEAD_IDS),
            }
            for st in stages
        },
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_FILE.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="版本混淆独立集评测 [Z]")
    parser.add_argument("--stage", choices=["all", "retrieval", "ce", "report"], default="all")
    args = parser.parse_args()

    rows = load_eval_set()
    log(f"版本混淆探针集：{EVAL_SET.name} 共 {len(rows)} 条")

    if args.stage in ("all", "ce"):
        run(rows, do_ce=True)
    elif args.stage == "retrieval":
        run(rows, do_ce=False)
    if args.stage in ("all", "report", "retrieval", "ce"):
        s = summarize()
        log("===== [Z] 版本混淆汇总 =====")
        for st, blk in s["stages"].items():
            log(f"  {st:>6}: 正确版进top5={blk['correct_in_top5']}  旧版进top5[Z]={blk['twin_in_top5']}  "
                f"旧版压过正确={blk['twin_outranks_correct']}  top1是旧版={blk['top1_is_twin']}  "
                f"MRR(correct)={blk['mrr_correct']}")
        log(f"汇总已写入 {SUMMARY_FILE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
