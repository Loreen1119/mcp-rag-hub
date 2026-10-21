#!/usr/bin/env python
"""五档基线阶梯评测 —— TiDB 中文评测集（主集 375 条）。

档位（逐档跑，各自出 Hit@5）：
  1. BM25 单路（无优化基线，[X] 取这档）
  2. 向量单路（bge-small-zh-v1.5）
  3. RRF 融合（BM25 + 向量）
  4. CE 精排（CE_CANDIDATE_K=30，只跑 200 条子集）
  5. 拒答阈值调优（复用第 4 档中间产物，只改阈值）

口径（与评测集冻结清单 data/eval_set_FROZEN.json 对齐）：
  - 命中判定：检索结果 metadata["source"] 的文件名 stem（即语料 slug）落在
    golden_slugs 中即命中。golden 是文档页级（slug），一份文档拆成多 chunk 时
    任一 chunk 命中即算该 golden 命中。
  - Hit@5：top-5 内至少一个 golden slug 命中。不可答类（unanswerable=true）无
    golden，不计入 Hit@5（单独统计拒答）。
  - 版本孪生页：同一 slug 在 docs-cn/ 与 docs-cn-v8.4/ 同时存在（差异页），
    命中旧版页时额外记 twin=True，供失败模式标注区分（主集 Hit@5 仍算命中，
    版本混淆干扰率由独立探针集 data/eval_set_version.jsonl 单独报）。
  - CE 子集：从「有 golden 的样本」(asktug+doc_fact+miss_kill) 按 source 分层
    随机抽样 200 条，seed=42 固定。拒答判定集 = 全部 40 不可答 + 全部 20 误杀
    探针（误杀率必须在全量 20 条探针上统计，故一并送入 CE 打分，与 200 子集去重）。

工程约束：
  - 纯本地：离线环境变量 + 清空全部 *_PROXY（含 ALL_PROXY）。不联网。
  - 逐条落盘（jsonl 追加），按 id 跳过已完成条目，中断后可原样重跑续上。
  - 不重建索引：直接用 corpus_collection_name(tidb, bge-small-zh-v1.5) 定位的
    既有集合（kb__tidb__bge-small-zh-v1.5__v2）。
  - 评测集只读：本脚本不修改 data/eval_set*.jsonl。

产出目录：experiments/baseline_ladder/
  - retrieval_375.jsonl   档 1/2/3 逐条 rank 明细（含哪条命中/未命中、golden 排第几）
  - ce_scores.jsonl       档 4 逐条 CE top5 + top1 分数（档 5 复用）
  - summary.json          各档汇总 + 阈值遍历结果

用法：
    python -m scripts.eval_baseline_ladder --stage all
    python -m scripts.eval_baseline_ladder --stage retrieval
    python -m scripts.eval_baseline_ladder --stage ce
    python -m scripts.eval_baseline_ladder --stage report
    python -m scripts.eval_baseline_ladder --tag k25test --stage retrieval   # 产物 → experiments/runs/k25test/
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

# ------------------------------------------------------------------
# 离线 / 代理环境（必须在任何 torch / transformers / config 导入之前生效）
# ------------------------------------------------------------------
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("MCP_RAG_CORPUS", "tidb")
for _k in list(os.environ):
    if _k.endswith("_PROXY") or _k.lower() == "all_proxy":
        os.environ.pop(_k, None)
os.environ["NO_PROXY"] = "127.0.0.1,localhost"

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import (  # noqa: E402
    BM25_TOP_K,
    VECTOR_TOP_K,
    CE_TOP_K,
    CE_CANDIDATE_K,
    CE_THRESHOLD,
    CORPUS,
    EMBEDDING_MODEL,
)
from src.fusion import reciprocal_rank_fusion  # noqa: E402
from src.pipeline import corpus_collection_name, get_pipeline  # noqa: E402

EVAL_SET = ROOT / "data" / "eval_set.jsonl"
OUT_DIR = ROOT / "experiments" / "baseline_ladder"
RETRIEVAL_RECORDS = OUT_DIR / "retrieval_375.jsonl"
CE_RECORDS = OUT_DIR / "ce_scores.jsonl"
SUMMARY_FILE = OUT_DIR / "summary.json"

CE_SUBSET_SIZE = 200
CE_SEED = 42
# 任务指定的候选阈值（bge-reranker 量纲）。实测该量纲下 top1 分数严重饱和
# （242/248 落在 0.9~1.0，92% >=0.99），0.3~0.7 全部落在饱和区、无操作空间，
# 故补一段细网格覆盖真正有区分度的 0.95~0.999 窄带，两部分都报告。
THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]
THRESHOLDS_EXTENDED = [0.9, 0.95, 0.97, 0.98, 0.99, 0.995, 0.999]
RRF_STORE_TOP = 30  # 落盘保留的 RRF 排名深度（= CE 候选上限，供失败模式标注）
DIAG_TOP = 50       # 单路检索落盘深度（仅供诊断「golden 排第几」，不参与 Hit@5 / RRF 口径）

# 以下四项可被 CLI 覆盖（实验用；**不改 config.py 的生产默认值**）。
# 动机：做完「加深召回 / 扩候选」这类实验需要改这三个数，但不该动生产配置。
BM25_RECALL = BM25_TOP_K        # 送进 RRF 的 BM25 召回深度
VECTOR_RECALL = VECTOR_TOP_K    # 送进 RRF 的向量召回深度
CE_CANDIDATE = CE_CANDIDATE_K   # RRF 后送 CE 的候选截断数


# ------------------------------------------------------------------
# 数据加载 / 命中判定
# ------------------------------------------------------------------
def load_eval_set() -> list[dict]:
    rows = [json.loads(line) for line in open(EVAL_SET, encoding="utf-8") if line.strip()]
    return rows


def slug_of(result) -> str:
    """检索结果的语料 slug = metadata['source'] 的文件名 stem。"""
    return Path(result.chunk.metadata.get("source", "")).stem


def is_twin(src: str) -> bool:
    """是否命中旧版本差异页（docs-cn-v8.4/）。"""
    return "docs-cn-v8.4/" in (src or "")


def ranking_record(results: list, golden_slugs: list[str]) -> tuple[list, dict]:
    """把一路检索结果转成可落盘的 rank 明细 + 命中统计。"""
    golden = set(golden_slugs or [])
    rows: list[dict] = []
    golden_rank = {g: None for g in golden}
    for rank, r in enumerate(results, start=1):
        src = r.chunk.metadata.get("source", "")
        slug = Path(src).stem
        hit = slug in golden
        rows.append(
            {
                "rank": rank,
                "slug": slug,
                "src": src,
                "score": round(float(r.score), 6),
                "hit": bool(hit),
                "twin": is_twin(src),
            }
        )
        if hit and golden_rank.get(slug) is None:
            golden_rank[slug] = rank
    hit_at5 = any(row["hit"] and row["rank"] <= 5 for row in rows)
    first_hit = next((row["rank"] for row in rows if row["hit"]), None)
    stats = {
        "hit@5": bool(hit_at5),
        "first_hit_rank": first_hit,
        "mrr@10": round(1.0 / first_hit, 4) if first_hit and first_hit <= 10 else 0.0,
        "golden_rank": golden_rank,
        "n_golden_hit_in_pool": sum(1 for v in golden_rank.values() if v is not None),
        "pool_depth": len(rows),
    }
    return rows, stats


# ------------------------------------------------------------------
# 分层抽样
# ------------------------------------------------------------------
def stratified_sample(pool: list[dict], n: int, seed: int) -> tuple[list[dict], dict]:
    """按 source 分层等比随机抽样（最大余数法补足配额），固定 seed。

    pool 内每条按 id 排序后抽样，保证同一 seed 下结果完全可复现。
    """
    by_source: dict[str, list[dict]] = {}
    for r in pool:
        by_source.setdefault(r["source"], []).append(r)
    total = len(pool)
    keys = sorted(by_source)

    quota: dict[str, int] = {}
    remainders: list[tuple[float, str]] = []
    for k in keys:
        exact = n * len(by_source[k]) / total
        quota[k] = int(exact)
        remainders.append((exact - int(exact), k))
    left = n - sum(quota.values())
    remainders.sort(key=lambda x: (-x[0], x[1]))
    for i in range(left):
        quota[remainders[i % len(remainders)][1]] += 1

    rng = random.Random(seed)
    picked: list[dict] = []
    for k in keys:
        items = sorted(by_source[k], key=lambda r: r["id"])
        q = min(quota[k], len(items))
        picked.extend(rng.sample(items, q))
    return picked, quota


# ------------------------------------------------------------------
# 落盘 / 续跑
# ------------------------------------------------------------------
def load_done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done = set()
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            done.add(json.loads(line)["id"])
        except Exception:
            continue
    return done


class Appender:
    """逐条追加写 + flush，保证中断时已完成条目全部落盘。"""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = open(path, "a", encoding="utf-8")

    def write(self, obj: dict) -> None:
        self.fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
        self.fh.flush()

    def close(self) -> None:
        self.fh.close()


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------------------
# 档 1/2/3：BM25 / 向量 / RRF
# ------------------------------------------------------------------
def run_retrieval(rows: list[dict]) -> None:
    done = load_done_ids(RETRIEVAL_RECORDS)
    todo = [r for r in rows if r["id"] not in done]
    log(f"档 1-3 检索：共 {len(rows)} 条，已完成 {len(done)}，本次待跑 {len(todo)}")
    if not todo:
        log("档 1-3 已全部完成，跳过（重跑续上逻辑生效）")
        return

    ctx = get_pipeline()
    log(f"管线就绪 — chunks={len(ctx.chunks)} collection={ctx.vector.collection.name} "
        f"count={ctx.vector.collection.count()}")

    app = Appender(RETRIEVAL_RECORDS)
    t0 = time.time()
    try:
        for i, tc in enumerate(todo, start=1):
            q = tc["query"]
            golden = tc.get("golden_slugs") or []
            tq = time.time()

            # 单路取 DIAG_TOP 深度落盘（诊断 golden 落在第几）；RRF 口径用
            # BM25_RECALL / VECTOR_RECALL（默认=生产配置 20+20，可被 CLI 加深）。
            depth = max(DIAG_TOP, BM25_RECALL, VECTOR_RECALL)
            bm25_res = ctx.bm25.search(q, top_k=depth)
            vec_res = ctx.vector.search(q, top_k=depth)
            rrf_res = reciprocal_rank_fusion(
                [bm25_res[:BM25_RECALL], vec_res[:VECTOR_RECALL]], top_n=RRF_STORE_TOP
            )

            stages = {}
            for name, res in (("bm25", bm25_res), ("vector", vec_res), ("rrf", rrf_res)):
                rank_rows, stats = ranking_record(res, golden)
                stages[name] = {"top": rank_rows, **stats}

            app.write(
                {
                    "id": tc["id"],
                    "source": tc["source"],
                    "unanswerable": bool(tc.get("unanswerable")),
                    "query": q,
                    "golden_slugs": golden,
                    "n_golden": len(golden),
                    "stages": stages,
                    "elapsed": round(time.time() - tq, 2),
                    "run_ts": datetime.now().isoformat(timespec="seconds"),
                }
            )
            if i % 25 == 0 or i == len(todo):
                rate = (time.time() - t0) / i
                eta = rate * (len(todo) - i)
                log(f"  检索 {i}/{len(todo)}  均 {rate:.2f}s/条  ETA {eta/60:.1f}min")
    finally:
        app.close()
    log(f"档 1-3 检索完成，耗时 {(time.time()-t0)/60:.1f} min")


# ------------------------------------------------------------------
# CE 子集与拒答判定集
# ------------------------------------------------------------------
def build_ce_ids(rows: list[dict], scope: str = "subset") -> tuple[list[str], set[str], dict]:
    pool = [r for r in rows if r["source"] in ("asktug", "doc_fact", "miss_kill")]
    if scope == "all":
        # 全量口径：所有有 golden 的样本都跑 CE，使 [X] 与 [Y] 在同一分母（335）上可比。
        # 已有的 200 子集结果会被 run_ce 的续跑逻辑跳过，只补跑差额。
        subset = list(pool)
        quota = {"scope": "all", "n": len(pool)}
    else:
        subset, quota = stratified_sample(pool, CE_SUBSET_SIZE, CE_SEED)
    subset_ids = [r["id"] for r in subset]
    reject_ids = [r["id"] for r in rows if r.get("unanswerable") or r["source"] == "miss_kill"]
    all_ids = sorted(set(subset_ids) | set(reject_ids), key=lambda x: (len(x), x))
    meta = {
        "scope": scope,
        "pool_size": len(pool),
        "subset_size": len(subset_ids),
        "subset_ids": subset_ids,
        "quota_by_source": quota,
        "reject_ids": reject_ids,
        "ce_total": len(all_ids),
        "extra_beyond_subset": sorted(set(all_ids) - set(subset_ids), key=lambda x: (len(x), x)),
        "seed": CE_SEED,
    }
    return all_ids, set(subset_ids), meta


# ------------------------------------------------------------------
# 档 4：CE 精排
# ------------------------------------------------------------------
def run_ce(rows: list[dict], ce_ids: list[str], subset_ids: set[str]) -> None:
    by_id = {r["id"]: r for r in rows}
    done = load_done_ids(CE_RECORDS)
    todo = [i for i in ce_ids if i not in done]
    log(f"档 4 CE 精排：目标 {len(ce_ids)} 条，已完成 {len(done)}，本次待跑 {len(todo)}")
    if not todo:
        log("档 4 已全部完成，跳过")
        return

    ctx = get_pipeline()
    log("加载 CE 模型（首次访问触发懒加载，约 1.1GB）...")
    ctx.pipeline.reranker  # 触发加载
    log("CE 模型就绪")

    app = Appender(CE_RECORDS)
    t0 = time.time()
    try:
        for i, qid in enumerate(todo, start=1):
            tc = by_id[qid]
            q = tc["query"]
            golden = tc.get("golden_slugs") or []
            tq = time.time()

            bm25_res = ctx.bm25.search(q, top_k=max(BM25_RECALL, VECTOR_RECALL))
            vec_res = ctx.vector.search(q, top_k=max(BM25_RECALL, VECTOR_RECALL))
            rrf_res = reciprocal_rank_fusion([bm25_res[:BM25_RECALL], vec_res[:VECTOR_RECALL]],
                                             top_n=max(RRF_STORE_TOP, CE_CANDIDATE))

            # top_k 放到候选上限 → 一次拿到**全部候选**的 CE 分数（而非只 top-5）。
            # 这样 k=30/40/50 的候选深度消融可以全部离线重算，不必重跑 CE。
            ce_all = ctx.pipeline.reranker.rerank(
                q, rrf_res, top_k=CE_CANDIDATE, candidate_cap=CE_CANDIDATE
            )
            rrf_rank_of = {r.chunk.chunk_id: i for i, r in enumerate(rrf_res[:CE_CANDIDATE], start=1)}
            ce_candidates = [
                {
                    "rrf_rank": rrf_rank_of.get(r.chunk.chunk_id),
                    "ce_rank": rank,
                    "slug": Path(r.chunk.metadata.get("source", "")).stem,
                    "src": r.chunk.metadata.get("source", ""),
                    "score": round(float(r.score), 6),
                }
                for rank, r in enumerate(ce_all, start=1)
            ]
            # stats 在**全部候选**上算（旧实现只拿 top-5 算，导致 CE 的 mrr@10 实为 MRR@5，
            # 与其它档不可比）；ce_top 仍只存 top-5，保持既有字段语义不变。
            rank_rows, stats = ranking_record(ce_all, golden) if golden else ([], {})
            if golden:
                ce_rows = [row for row in rank_rows if row["rank"] <= CE_TOP_K]
            else:
                ce_rows = [
                    {"rank": c["ce_rank"], "slug": c["slug"], "src": c["src"], "score": c["score"]}
                    for c in ce_candidates[:CE_TOP_K]
                ]

            app.write(
                {
                    "id": qid,
                    "source": tc["source"],
                    "unanswerable": bool(tc.get("unanswerable")),
                    "query": q,
                    "golden_slugs": golden,
                    "n_golden": len(golden),
                    "in_ce_subset": qid in subset_ids,
                    "n_candidates": len(rrf_res),
                    "ce_top": ce_rows,
                    "ce_candidates": ce_candidates,
                    "mrr_basis": "candidates<=CE_CANDIDATE（全候选，非仅 top-5）",
                    "ce_top1_score": round(float(ce_all[0].score), 4) if ce_all else None,
                    **stats,
                    "elapsed": round(time.time() - tq, 2),
                    "run_ts": datetime.now().isoformat(timespec="seconds"),
                }
            )
            if i % 5 == 0 or i == len(todo):
                rate = (time.time() - t0) / i
                eta = rate * (len(todo) - i)
                log(f"  CE {i}/{len(todo)}  均 {rate:.1f}s/条  ETA {eta/60:.1f}min  "
                    f"(本批累计 {(time.time()-t0)/60:.0f}min)")
    finally:
        app.close()
    log(f"档 4 CE 完成，耗时 {(time.time()-t0)/60:.1f} min")


# ------------------------------------------------------------------
# 汇总
# ------------------------------------------------------------------
def read_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def _hit_rate(recs: list[dict], key: str = "hit@5") -> float:
    if not recs:
        return 0.0
    return round(sum(1 for r in recs if r.get(key)) / len(recs), 4)


def _stage_block(recs: list[dict], stage: str, gold=None) -> dict:
    """某一档的 Hit@K（K=5/20/50，使用该档落盘排名）+ MRR@10。

    gold: id -> set(golden_slugs)。给了就按当前 golden 从落盘 slug 重算命中
    （不信任记录里的 hit 标志，避免 golden 修正后旧标志过期）。
    未给则回落到记录里的 hit 标志，仅为兼容旧调用。
    """

    def _g(rid: str) -> set:
        return gold.get(rid, set()) if gold is not None else None

    def _is_hit(r: dict, x: dict) -> bool:
        g = _g(r["id"])
        return (x["slug"] in g) if g is not None else bool(x["hit"])

    def hit_at(k: int) -> float:
        if not recs:
            return 0.0
        n = 0
        for r in recs:
            rows = r["stages"][stage]["top"]
            if any(_is_hit(r, x) and x["rank"] <= k for x in rows):
                n += 1
        return round(n / len(recs), 4)

    return {
        "evaluated": len(recs),
        "hit@5": hit_at(5),
        "hit@20": hit_at(20),
        "hit@50": hit_at(50),
        "mrr@10": round(
            sum(r["stages"][stage].get("mrr@10", 0.0) for r in recs) / len(recs), 4
        ) if recs else 0.0,
        "by_source": {
            s: {
                "n": len(sub := [r for r in recs if r["source"] == s]),
                "hit@5": round(
                    sum(
                        1
                        for r in sub
                        if any(_is_hit(r, x) and x["rank"] <= 5 for x in r["stages"][stage]["top"])
                    ) / len(sub), 4
                ) if sub else 0.0,
                "hit@20": round(
                    sum(
                        1
                        for r in sub
                        if any(_is_hit(r, x) and x["rank"] <= 20 for x in r["stages"][stage]["top"])
                    ) / len(sub), 4
                ) if sub else 0.0,
            }
            for s in ("asktug", "doc_fact", "miss_kill")
        },
    }


def _score_dist(miss_kill, unans, answerable) -> dict:
    """CE top1 分数分布（判据：阈值调优是否有操作空间的直接证据）。"""
    import statistics

    def stats(recs):
        v = sorted(r["ce_top1_score"] for r in recs if r.get("ce_top1_score") is not None)
        if not v:
            return {}
        return {
            "n": len(v),
            "min": v[0],
            "p10": v[max(0, len(v) // 10)],
            "median": round(statistics.median(v), 4),
            "max": v[-1],
            "share_ge_0.99": round(sum(1 for x in v if x >= 0.99) / len(v), 4),
        }

    return {
        "miss_kill": stats(miss_kill),
        "unanswerable": stats(unans),
        "answerable_asktug_doc_fact": stats(answerable),
    }


def summarize(rows: list[dict] | None = None, ce_scope: str = "subset") -> dict:
    ret = read_records(RETRIEVAL_RECORDS)
    ce = read_records(CE_RECORDS)

    # 命中一律「按当前评测集的 golden 从落盘排名重算」，不信任记录里的 hit 标志：
    # 记录是在各自运行时刻写的，golden 一旦修正（如 v0.1 的三条），旧标志即过期。
    golden_map = {r["id"]: set(r.get("golden_slugs") or []) for r in (rows or [])}

    def _gold(rid: str) -> set:
        return golden_map.get(rid, set())

    def _hit5_from(rec: dict, key: str) -> bool:
        g = _gold(rec["id"])
        return bool(g) and any(x["slug"] in g and x["rank"] <= 5 for x in rec.get(key, []))

    summary: dict = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "ce_scope": ce_scope,
        "index": {
            "corpus": CORPUS,
            "embedding_model": EMBEDDING_MODEL,
            "collection": corpus_collection_name(CORPUS, EMBEDDING_MODEL),
        },
        "eval_set": {
            "file": str(EVAL_SET.relative_to(ROOT)),
            "retrieval_records": len(ret),
            "ce_records": len(ce),
        },
        "criteria": {
            "hit": "检索结果 slug（metadata.source 文件名 stem）∈ golden_slugs",
            "hit@5": "top-5 内至少一个 golden slug 命中；不可答类不计入",
            "ce_candidate_k": CE_CANDIDATE_K,
            "ce_threshold_default": CE_THRESHOLD,
            "ce_scope": "subset=分层抽 200 条；all=全部有 golden 样本。档 4/5 的分母随该值变化。",
            "hit_recompute": "汇总时按当前 golden 从落盘排名重算命中，不使用记录中的 hit 标志",
        },
        "stages": {},
    }

    if ret:
        answerable = [r for r in ret if not r["unanswerable"]]
        summary["stages"]["1_bm25"] = _stage_block(answerable, "bm25", golden_map)
        summary["stages"]["2_vector"] = _stage_block(answerable, "vector", golden_map)
        summary["stages"]["3_rrf"] = _stage_block(answerable, "rrf", golden_map)
        # 版本孪生命中（诊断）
        twin_hits = {
            "bm25": sum(
                1
                for r in answerable
                for row in r["stages"]["bm25"]["top"][:5]
                if row["slug"] in _gold(r["id"]) and row["twin"]
            ),
            "vector": sum(
                1
                for r in answerable
                for row in r["stages"]["vector"]["top"][:5]
                if row["slug"] in _gold(r["id"]) and row["twin"]
            ),
            "rrf": sum(
                1
                for r in answerable
                for row in r["stages"]["rrf"]["top"][:5]
                if row["slug"] in _gold(r["id"]) and row["twin"]
            ),
        }
        summary["twin_hit_top5_counts"] = twin_hits

        # 不可答类 top1 分布（BM25/向量分数，供拒答诊断）
        una = [r for r in ret if r["unanswerable"]]
        if una:
            summary["unanswerable_retrieval"] = {
                "n": len(una),
                "bm25_nonempty": sum(1 for r in una if r["stages"]["bm25"]["top"]),
                "vector_nonempty": sum(1 for r in una if r["stages"]["vector"]["top"]),
            }

    if ce:
        # 分母按当前 scope 重新解析：落盘的 in_ce_subset 是「写入时刻的 subset 定义」，
        # 跨口径复用会把先前单独跑过的条目漏掉 —— 曾导致 scope=all 时算成 327/335。
        if rows is not None:
            _, subset_ids, _ = build_ce_ids(rows, scope=ce_scope)
        else:
            subset_ids = {r["id"] for r in ce if r.get("in_ce_subset")}
        subset = [r for r in ce if r["id"] in subset_ids]
        answerable_subset = [r for r in subset if not r["unanswerable"]]
        miss_kill = [r for r in ce if r["source"] == "miss_kill"]
        unans = [r for r in ce if r["unanswerable"]]

        def _ce_hit(r: dict) -> bool:
            return _hit5_from(r, "ce_top")

        n_ans = max(1, len(answerable_subset))
        summary["stages"]["4_ce"] = {
            "subset_size": len(subset),
            "evaluated": len(answerable_subset),
            "hit@5": round(sum(1 for r in answerable_subset if _ce_hit(r)) / n_ans, 4),
            "mrr@10": round(
                sum(r.get("mrr@10", 0.0) for r in answerable_subset) / n_ans, 4
            ),
            "mrr_note": "新跑的记录：mrr@10 在**全部候选**（<=CE_CANDIDATE）上算，与其它档可比。"
                        "⚠️ 旧记录（本改造之前）只拿 top-5 算，其实等于 MRR@5，数值偏低、与其它档不可比。",
            "by_source": {
                s: {
                    "n": len(sub := [r for r in answerable_subset if r["source"] == s]),
                    "hit@5": round(sum(1 for r in sub if _ce_hit(r)) / max(1, len(sub)), 4),
                }
                for s in ("asktug", "doc_fact", "miss_kill")
            },
            "ce_scored_total": len(ce),
            "ce_scored_miss_kill": len(miss_kill),
            "ce_scored_unanswerable": len(unans),
            "ce_top1_score_distribution": _score_dist(miss_kill, unans, answerable_subset),
        }

        # 档 5：拒答阈值遍历
        sweep = []
        for t in THRESHOLDS + THRESHOLDS_EXTENDED:
            mk_rejected = [r for r in miss_kill if (r.get("ce_top1_score") or 0) < t]
            un_rejected = [r for r in unans if (r.get("ce_top1_score") or 0) < t]
            answered = [r for r in answerable_subset if (r.get("ce_top1_score") or 0) >= t]
            answered_hit = [r for r in answered if _ce_hit(r)]
            sweep.append(
                {
                    "threshold": t,
                    "reject_false_kill_rate": round(len(mk_rejected) / len(miss_kill), 4) if miss_kill else None,
                    "reject_false_kill_n": f"{len(mk_rejected)}/{len(miss_kill)}",
                    "reject_correct_rate": round(len(un_rejected) / len(unans), 4) if unans else None,
                    "reject_correct_n": f"{len(un_rejected)}/{len(unans)}",
                    "net_hit_rate": round(len(answered_hit) / len(answerable_subset), 4) if answerable_subset else None,
                    "net_hit_n": f"{len(answered_hit)}/{len(answerable_subset)}",
                    "answered_n": len(answered),
                }
            )
        summary["stages"]["5_reject_sweep"] = sweep
        default_row = next((s for s in sweep if abs(s["threshold"] - CE_THRESHOLD) < 1e-9), None)
        summary["stages"]["5_reject_sweep_note"] = {
            "A_default": default_row,
            "candidates": sweep,
        }

    SUMMARY_FILE.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_FILE.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # 失败模式标注辅助：对 RRF 档未命中 top-5 的有 golden 样本，按
    # 「golden 是否出现在单路候选池内」自动给出 召回缺失 / 排序压制 的初判。
    if ret:
        hints = []
        for r in ret:
            if r["unanswerable"]:
                continue
            rrf_rows = r["stages"]["rrf"]["top"]
            if any(x["hit"] and x["rank"] <= 5 for x in rrf_rows):
                continue
            in_bm25 = next((x["rank"] for x in r["stages"]["bm25"]["top"] if x["hit"]), None)
            in_vec = next((x["rank"] for x in r["stages"]["vector"]["top"] if x["hit"]), None)
            if in_bm25 is not None or in_vec is not None:
                hint = "排序压制"
            else:
                hint = "召回缺失"
            golden_ranks = {}
            for st, res in (("bm25", r["stages"]["bm25"]), ("vector", r["stages"]["vector"])):
                for g, rk in (res.get("golden_rank") or {}).items():
                    if rk is not None:
                        golden_ranks[f"{st}:{g}"] = rk
            hints.append(
                {
                    "id": r["id"],
                    "source": r["source"],
                    "query": r["query"],
                    "golden_slugs": r["golden_slugs"],
                    "n_golden": r["n_golden"],
                    "auto_hint": hint,
                    "golden_rank_in_single_path": golden_ranks,
                    "rrf_top5": [
                        {"rank": x["rank"], "slug": x["slug"], "hit": x["hit"], "twin": x["twin"]}
                        for x in rrf_rows[:5]
                    ],
                }
            )
        with open(OUT_DIR / "miss_hints.jsonl", "w", encoding="utf-8") as fh:
            for h in hints:
                fh.write(json.dumps(h, ensure_ascii=False) + "\n")
        summary["miss_hints"] = {
            "file": "experiments/baseline_ladder/miss_hints.jsonl",
            "n": len(hints),
            "召回缺失": sum(1 for h in hints if h["auto_hint"] == "召回缺失"),
            "排序压制": sum(1 for h in hints if h["auto_hint"] == "排序压制"),
        }
        SUMMARY_FILE.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    return summary


# ------------------------------------------------------------------
# main
# ------------------------------------------------------------------
def main() -> None:
    global SUMMARY_FILE, OUT_DIR, RETRIEVAL_RECORDS, CE_RECORDS
    global BM25_RECALL, VECTOR_RECALL, CE_CANDIDATE, RRF_STORE_TOP
    parser = argparse.ArgumentParser(description="五档基线阶梯评测")
    parser.add_argument(
        "--stage",
        choices=["all", "retrieval", "ce", "report", "plan"],
        default="all",
        help="plan=只打印 CE 抽样计划；retrieval=档1-3；ce=档4；report=汇总；all=全部",
    )
    parser.add_argument(
        "--ce-scope",
        choices=["subset", "all"],
        default="subset",
        help="CE 打分范围：subset=分层抽 200 条（默认，历史口径）；"
             "all=全部 335 条有 golden 样本（使 [X] 与 [Y] 同分母可比）。"
             "scope=all 时汇总写 summary_ce_all.json，不覆盖原 summary.json。",
    )
    parser.add_argument("--bm25-k", type=int, default=BM25_TOP_K,
                        help="送进 RRF 的 BM25 召回深度（默认=生产配置）")
    parser.add_argument("--vector-k", type=int, default=VECTOR_TOP_K,
                        help="送进 RRF 的向量召回深度（默认=生产配置）")
    parser.add_argument("--ce-candidate-k", type=int, default=CE_CANDIDATE_K,
                        help="RRF 后送 CE 的候选截断数（默认=生产配置）")
    parser.add_argument("--rrf-store-top", type=int, default=RRF_STORE_TOP,
                        help="RRF 落盘保留深度（默认 30；做候选深度实验时应 >= ce-candidate-k）")
    parser.add_argument("--out-dir", default=None,
                        help="产物目录（默认 experiments/baseline_ladder）。"
                             "实验请另指定（如 experiments/baseline_ladder/depth50），避免覆盖既有产物。")
    parser.add_argument("--tag", default=None, metavar="NAME",
                        help="实验标签：产物落到 experiments/runs/<NAME>/，"
                             "**不覆盖** experiments/baseline_ladder/（冻结产物的来源）。"
                             "示例：--tag k25test → experiments/runs/k25test/")
    args = parser.parse_args()

    # 实验参数生效（只改本进程的全局，不动 config.py）
    if args.out_dir:
        p = Path(args.out_dir)
        OUT_DIR = p if p.is_absolute() else (ROOT / p)
    elif args.tag:
        OUT_DIR = ROOT / "experiments" / "runs" / args.tag
    if args.out_dir or args.tag:
        RETRIEVAL_RECORDS = OUT_DIR / "retrieval_375.jsonl"
        CE_RECORDS = OUT_DIR / "ce_scores.jsonl"
        SUMMARY_FILE = OUT_DIR / "summary.json"
    BM25_RECALL = args.bm25_k
    VECTOR_RECALL = args.vector_k
    CE_CANDIDATE = args.ce_candidate_k
    RRF_STORE_TOP = max(args.rrf_store_top, CE_CANDIDATE)
    if args.ce_scope == "all":
        SUMMARY_FILE = OUT_DIR / "summary_ce_all.json"

    rows = load_eval_set()
    log(f"评测集：{EVAL_SET.name} 共 {len(rows)} 条  ｜ CE 范围={args.ce_scope}")
    log(f"检索参数：BM25 召回 {BM25_RECALL} + 向量召回 {VECTOR_RECALL} → RRF 落盘 {RRF_STORE_TOP} "
        f"→ CE 候选截断 {CE_CANDIDATE} ｜ 产物目录 {OUT_DIR.relative_to(ROOT)}")
    ce_ids, ce_subset_ids, ce_meta = build_ce_ids(rows, scope=args.ce_scope)
    log(f"CE 计划：子集 {ce_meta['subset_size']} + 拒答判定集（去重后总计 {ce_meta['ce_total']}）"
        f"  分层配额={ce_meta['quota_by_source']}")
    log(f"  超出子集的额外条目 {len(ce_meta['extra_beyond_subset'])} 条：{ce_meta['extra_beyond_subset']}")

    if args.stage == "plan":
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "ce_plan.json").write_text(
            json.dumps(ce_meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        log(f"抽样计划已落盘 {(OUT_DIR / 'ce_plan.json').relative_to(ROOT)}")
        return

    if args.stage in ("all", "retrieval"):
        run_retrieval(rows)

    if args.stage in ("all", "ce"):
        run_ce(rows, ce_ids, ce_subset_ids)

    if args.stage in ("all", "report", "retrieval", "ce"):
        summary = summarize(rows, ce_scope=args.ce_scope)
        log("===== 汇总 =====")
        for name, block in summary["stages"].items():
            if name.startswith("5_"):
                continue
            log(f"  档 {name}: hit@5={block.get('hit@5')} mrr@10={block.get('mrr@10')} "
                f"evaluated={block.get('evaluated')}")
        if "5_reject_sweep" in summary["stages"]:
            for srow in summary["stages"]["5_reject_sweep"]:
                log(f"  t={srow['threshold']}  误杀率={srow['reject_false_kill_rate']} "
                    f"({srow['reject_false_kill_n']})  正确拒答={srow['reject_correct_rate']} "
                    f"({srow['reject_correct_n']})  净命中={srow['net_hit_rate']} ({srow['net_hit_n']})")
        log(f"汇总已写入 {(SUMMARY_FILE).relative_to(ROOT)}")


if __name__ == "__main__":
    main()
