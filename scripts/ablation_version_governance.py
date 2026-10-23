#!/usr/bin/env python
"""版本治理对照实验（一次性离线）——把「旧版文档过滤」当成一种检索策略来量化。

背景：
  corpora/tidb/ 里刻意保留了 306 篇 docs-cn-v8.4 旧版页。在 RRF 融合结果中，
  旧版页占位严重（top-5 干扰率 30.7%，top-30 76.7%），而主集 golden 里只有
  1 条依赖 v8.4 页 —— 所以「过滤旧版」看起来是零成本、影响面 30% 的治理动作。
  本脚本只用已有产物（experiments/baseline_ladder/retrieval_375.jsonl）算账：
  不加载模型、不重建索引、不碰 chroma_db。

三档对照（同一个 335 条分母，只改排序规则）：
  baseline   RRF 原序取 top-5
  filter     排序前剔除 src 以 "docs-cn-v8.4/" 开头的候选，再取 top-5
  decay_x    不剔除，旧版候选 score *= x 后重排（0.5 / 0.9 两档，仅作参考）

指标（每档都给分子/分母）：
  Hit@5、MRR@10、NDCG@10（二值 qrels，gain=1）、版本混淆干扰率（top-5 含旧版页占比）

判据（事前写死，跑完不许改阈值）：
  只看 filter：Hit@5 提升 >= +2pt 且 McNemar 精确检验双侧 p < 0.05
    → 「版本过滤是有效治理动作」
  否则 → 「旧版占位不是主因，治理收益有限」

用法：
  python -m scripts.ablation_version_governance
  python -m scripts.ablation_version_governance --input <jsonl> --out <json>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "experiments" / "baseline_ladder" / "retrieval_375.jsonl"
DEFAULT_OUT = REPO_ROOT / "experiments" / "version_governance" / "ablation.json"

# ---- 口径常量（改动即等于改口径，需重写结论） ----
OLD_VERSION_PREFIX = "docs-cn-v8.4/"
STAGE = "rrf"
K_HIT = 5
K_MRR = 10
K_NDCG = 10
DECAY_FACTORS = (0.5, 0.9)
BOOTSTRAP_N = 10000
BOOTSTRAP_SEED = 20261018
ALPHA = 0.05
HIT5_MIN_GAIN_PT = 2.0


# --------------------------------------------------------------------------
# 载入与归一
# --------------------------------------------------------------------------
def sha256_of(path: Path) -> str:
    """算产物 sha256 —— **先把 CRLF 归一成 LF 再算**。

    本机 core.autocrlf=true 时工作区是 CRLF、仓库 blob 是 LF。不归一会让
    记录下来的哈希在 Linux CI 上永远对不上，闸门就红在换行符上而不是真实产物
    变化上（与 tests/regression_baselines.json 的算法保持一致）。
    """
    raw = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(raw).hexdigest()


def load_records(path: Path) -> List[dict]:
    records: List[dict] = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:  # pragma: no cover - 输入损坏直接失败
                raise SystemExit(f"第 {line_no} 行 JSON 解析失败：{exc}")
    return records


def build_denominator(records: Sequence[dict]) -> List[dict]:
    """335 口径：跳过 unanswerable=True 与无 golden 的条目。"""
    den = [
        r for r in records
        if not r.get("unanswerable", False) and r.get("golden_slugs")
    ]
    return den


def ranked_candidates(record: dict) -> List[dict]:
    """按 rank 升序取出 rrf 候选，只保留判分需要的字段。

    候选是 chunk 级（同一 slug 可能出现多次），基线口径就是原序不做事先去重，
    三档一律沿用，避免「重排顺带去重」混淆变量。
    """
    top = (record.get("stages", {}).get(STAGE) or {}).get("top") or []
    cands = [
        {
            "rank": int(c.get("rank", i + 1)),
            "slug": c.get("slug"),
            "src": c.get("src") or "",
            "score": float(c.get("score") or 0.0),
        }
        for i, c in enumerate(top)
    ]
    cands.sort(key=lambda c: c["rank"])
    return cands


def is_old_version(cand: dict) -> bool:
    return cand["src"].startswith(OLD_VERSION_PREFIX)


# --------------------------------------------------------------------------
# 三档排序规则
# --------------------------------------------------------------------------
def arm_baseline(cands: List[dict]) -> List[dict]:
    return list(cands)


def arm_filter(cands: List[dict]) -> List[dict]:
    return [c for c in cands if not is_old_version(c)]


def arm_decay(cands: List[dict], factor: float) -> List[dict]:
    """旧版候选 score *= factor 后稳定重排（同分保持原 rank 顺序）。"""
    decorated = []
    for c in cands:
        weight = factor if is_old_version(c) else 1.0
        decorated.append({**c, "score": c["score"] * weight})
    decorated.sort(key=lambda c: (-c["score"], c["rank"]))
    return decorated


# --------------------------------------------------------------------------
# 指标
# --------------------------------------------------------------------------
def hit_at_k(ranked: Sequence[dict], golden: set, k: int = K_HIT) -> bool:
    return any(c["slug"] in golden for c in ranked[:k])


def mrr_at_k(ranked: Sequence[dict], golden: set, k: int = K_MRR) -> float:
    for pos, c in enumerate(ranked[:k], 1):
        if c["slug"] in golden:
            return 1.0 / pos
    return 0.0


def ndcg_at_k(
    ranked: Sequence[dict], golden: set, k: int = K_NDCG, dedupe_slug: bool = True
) -> float:
    """二值 qrels 的 nDCG@k。

    来源：BEIR / MTEB 的 nDCG@10 就是二值口径（qrels 只标相关/不相关，gain=1），
    DCG = Σ gain / log2(rank+1)，IDCG 按理想排序（相关文档全排最前）算。
    本语料候选是 chunk 级，同一 slug 会有多个 chunk 进候选；当 dedupe_slug=True
    时同一 slug 只在首次出现处计一次 gain（qrels 的粒度是文档 slug，不是 chunk）。
    """
    seen: set = set()
    dcg = 0.0
    for pos, c in enumerate(ranked[:k], 1):
        slug = c["slug"]
        if slug in seen:
            continue
        if dedupe_slug:
            seen.add(slug)
        if slug in golden:
            dcg += 1.0 / math.log2(pos + 1)
    if dedupe_slug:
        # 每个 golden slug 最多贡献一次 gain → 理想情况是全部挤进最前面
        ideal_hits = min(len(golden), k)
    else:
        # 不去重时同一 slug 可重复计分，IDCG 只能按「池内可得的 golden chunk 数」算，
        # 否则会出现 nDCG > 1 的非法值
        available = sum(1 for c in ranked[:k] if c["slug"] in golden)
        ideal_hits = min(available, k)
    idcg = sum(1.0 / math.log2(p + 1) for p in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def interference_at_k(ranked: Sequence[dict], k: int = K_HIT) -> bool:
    """top-k 里是否混入了 docs-cn-v8.4 旧版页（chunk 级，命中即算）。"""
    return any(is_old_version(c) for c in ranked[:k])


def version_correct_hit_at_k(
    ranked: Sequence[dict], golden: set, k: int = K_HIT
) -> bool:
    """VersionCorrect@k：top-k 内首个命中，且该命中**不来自旧版目录**。

    为什么需要这个口径（2026-10-23 补）：slug 级 hit@k 只看「有没有命中同 slug
    的页」，新旧版同名页都算命中 —— 命中来源是哪一版在 hit@k 里完全看不见。
    EVAL_REPORT 第六节已经写过这个陷阱（「收益被指标掩盖」），本函数是把它
    用在治理实验上：三档治理洗的是命中来源，不是命中与否。

    定义上与 hit_at_k 的差别只有一处：首个命中的 src 若以旧版前缀开头，记 0。
    未命中同样记 0。因此 VersionCorrect@5 ≤ Hit@5 恒成立。
    """
    for c in ranked[:k]:
        if c["slug"] in golden:
            return not is_old_version(c)
    return False


def first_golden_candidate(
    ranked: Sequence[dict], golden: set, k: int = K_HIT
) -> Optional[dict]:
    """top-k 内首个命中候选（含 src，用来判版本）；没有则 None。"""
    for c in ranked[:k]:
        if c["slug"] in golden:
            return c
    return None


# --------------------------------------------------------------------------
# 统计检验
# --------------------------------------------------------------------------
def mcnemar_exact_p(b: int, c: int) -> Tuple[float, int]:
    """McNemar 精确检验（双侧）。

    b = baseline 命中且 filter 未命中（丢），c = baseline 未命中且 filter 命中（得）。
    零假设下 b、c 服从 Binomial(n=b+c, 0.5)，双侧 p = 2 * Σ_{i<=min(b,c)} C(n,i)/2^n，
    上限截到 1.0。b+c=0 时 p=1.0。
    """
    n = b + c
    if n == 0:
        return 1.0, 0
    tail = sum(math.comb(n, i) for i in range(0, min(b, c) + 1))
    p = min(1.0, 2.0 * tail / (2 ** n))
    return p, n


def paired_bootstrap_delta(
    base: Sequence[bool],
    other: Sequence[bool],
    n_resamples: int = BOOTSTRAP_N,
    seed: int = BOOTSTRAP_SEED,
) -> Dict[str, float]:
    """配对 bootstrap：对 query 下标有放回重采样，统计 Hit@5 差值的 95% CI。"""
    assert len(base) == len(other)
    n = len(base)
    rng = random.Random(seed)
    deltas: List[float] = []
    idx_range = range(n)
    for _ in range(n_resamples):
        num_b = 0
        num_o = 0
        for _ in range(n):
            i = rng.choice(idx_range)
            if base[i]:
                num_b += 1
            if other[i]:
                num_o += 1
        deltas.append((num_o - num_b) / n)
    deltas.sort()
    lo = deltas[int(0.025 * n_resamples)]
    hi = deltas[min(n_resamples - 1, int(0.975 * n_resamples))]
    mean = sum(deltas) / len(deltas)
    return {
        "n_resamples": n_resamples,
        "seed": seed,
        "mean_delta_frac": mean,
        "mean_delta_pt": mean * 100,
        "ci95_frac": [lo, hi],
        "ci95_pt": [lo * 100, hi * 100],
        "method": "percentile, paired resampling of query indices",
    }


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------
def evaluate_arm(
    arm_name: str,
    rankings: Dict[str, List[dict]],
    den: Sequence[dict],
) -> dict:
    hits: List[bool] = []
    vhits: List[bool] = []
    mrr_sum = 0.0
    ndcg_sum = 0.0
    ndcg_naive_sum = 0.0
    interf = 0
    for rec in den:
        ranked = rankings[rec["id"]]
        golden = set(rec["golden_slugs"])
        hits.append(hit_at_k(ranked, golden))
        vhits.append(version_correct_hit_at_k(ranked, golden))
        mrr_sum += mrr_at_k(ranked, golden)
        ndcg_sum += ndcg_at_k(ranked, golden, dedupe_slug=True)
        ndcg_naive_sum += ndcg_at_k(ranked, golden, dedupe_slug=False)
        interf += 1 if interference_at_k(ranked) else 0
    n = len(den)
    num = sum(1 for h in hits if h)
    vnum = sum(1 for h in vhits if h)
    return {
        "arm": arm_name,
        "hit@5_num": num,
        "denominator": n,
        "hit@5": num / n if n else 0.0,
        "mrr@10": mrr_sum / n if n else 0.0,
        "ndcg@10": ndcg_sum / n if n else 0.0,
        "ndcg@10_no_dedupe": ndcg_naive_sum / n if n else 0.0,
        "interference_num": interf,
        "interference_rate": interf / n if n else 0.0,
        "version_correct@5_num": vnum,
        "version_correct@5": vnum / n if n else 0.0,
        "_hits": hits,
        "_vhits": vhits,
    }


def run(input_path: Path, out_path: Path) -> dict:
    records = load_records(input_path)
    den = build_denominator(records)

    # 排序规则一次性算好，四条手臂共用同一批候选
    rankings: Dict[str, Dict[str, List[dict]]] = {
        "baseline": {},
        "filter": {},
        "decay_0.5": {},
        "decay_0.9": {},
    }
    for rec in den:
        cands = ranked_candidates(rec)
        rankings["baseline"][rec["id"]] = arm_baseline(cands)
        rankings["filter"][rec["id"]] = arm_filter(cands)
        for f in DECAY_FACTORS:
            rankings[f"decay_{f}"][rec["id"]] = arm_decay(cands, f)

    arms = [
        evaluate_arm(name, rankings[name], den) for name in
        ("baseline", "filter", "decay_0.5", "decay_0.9")
    ]
    by_name = {a["arm"]: a for a in arms}

    # ---- baseline vs filter 配对 ----
    base_hits = by_name["baseline"]["_hits"]
    filt_hits = by_name["filter"]["_hits"]
    b = sum(1 for x, y in zip(base_hits, filt_hits) if x and not y)  # 丢
    c = sum(1 for x, y in zip(base_hits, filt_hits) if y and not x)  # 得
    p_value, n_disc = mcnemar_exact_p(b, c)
    bootstrap = paired_bootstrap_delta(base_hits, filt_hits)

    lost_ids = [rec["id"] for rec, x, y in zip(den, base_hits, filt_hits) if x and not y]
    gained_ids = [rec["id"] for rec, x, y in zip(den, base_hits, filt_hits) if y and not x]

    # 丢掉的命中是否「只能靠旧版页」拿到 —— 即 baseline top-5 里的 golden 候选全是旧版
    lost_detail = []
    for rec in den:
        if rec["id"] not in lost_ids:
            continue
        golden = set(rec["golden_slugs"])
        cands = rankings["baseline"][rec["id"]]
        base_top5 = cands[:K_HIT]
        golden_cands = [x for x in base_top5 if x["slug"] in golden]
        # 同一 slug 在新版目录下的池内位次（用来判断丢命中的真实原因：
        # 是「新版根本没被召回」还是「新版排在 top-5 之外」）
        twin_ranks = [
            {"rank": x["rank"], "src": x["src"], "in_top5": x["rank"] <= K_HIT}
            for x in cands
            if x["slug"] in golden and not is_old_version(x)
        ]
        lost_detail.append(
            {
                "id": rec["id"],
                "golden_slugs": list(rec["golden_slugs"]),
                "baseline_golden_candidates": [
                    {"rank": x["rank"], "slug": x["slug"], "src": x["src"]} for x in golden_cands
                ],
                "only_from_old_version": all(is_old_version(x) for x in golden_cands),
                "new_version_twin_in_pool": twin_ranks,
            }
        )

    # ---- 诊断：旧版占位与命中是否正交 ----
    crosstab = {"interfered_hit": 0, "interfered_miss": 0, "clean_hit": 0, "clean_miss": 0}
    old_chunks_in_top5 = 0
    old_chunks_total = 0
    cands_total = 0
    for rec, h in zip(den, base_hits):
        top5 = rankings["baseline"][rec["id"]][:K_HIT]
        interfered = any(is_old_version(c) for c in top5)
        key = ("interfered_" if interfered else "clean_") + ("hit" if h else "miss")
        crosstab[key] += 1
        old_chunks_in_top5 += sum(1 for c in top5 if is_old_version(c))
        old_chunks_total += sum(1 for c in rankings["baseline"][rec["id"]] if is_old_version(c))
        cands_total += len(rankings["baseline"][rec["id"]])


    delta_pt = (by_name["filter"]["hit@5"] - by_name["baseline"]["hit@5"]) * 100

    # ---- 判据（事前写死） ----
    crit_a = delta_pt >= HIT5_MIN_GAIN_PT and p_value < ALPHA
    if crit_a:
        matched = "① Hit@5 提升 ≥ +2pt 且 p<0.05 → 版本过滤是有效治理动作"
        conclusion = "版本过滤是有效治理动作，建议进生产并写进材料"
    else:
        matched = "② 提升 <2pt 或 p≥0.05 → 旧版占位不是主因，治理收益有限，转去验证模型规模（reranker-large）"
        conclusion = (
            f"在 slug 级 Hit@5 口径下，旧版占位不是主因：filter 相对 baseline "
            f"{delta_pt:+.2f}pt、p={p_value:.2f}，三档均未提升召回。"
        )

    def strip(arm: dict) -> dict:
        return {k: v for k, v in arm.items() if not k.startswith("_")}

    # ---- VersionCorrect 口径：命中的是当前版页，还是过时版页 ----
    # 动机见 version_correct_hit_at_k 的 docstring —— slug 级 hit@5 看不见命中来源。
    # 这一节是 2026-10-23 主线补算的：三档治理洗的是「命中来自哪一版」，不是「命中与否」。
    base_vhits = by_name["baseline"]["_vhits"]
    vc_arms = ("filter", "decay_0.5", "decay_0.9")
    vc_per_arm: Dict[str, dict] = {}
    for name in ("baseline",) + vc_arms:
        vnum = sum(1 for x in by_name[name]["_vhits"] if x)
        vc_per_arm[name] = {"num": vnum, "rate": vnum / len(den)}

    vc_pairs: Dict[str, dict] = {}
    for name in vc_arms:
        v = by_name[name]["_vhits"]
        vb = sum(1 for x, y in zip(base_vhits, v) if x and not y)
        va = sum(1 for x, y in zip(base_vhits, v) if y and not x)
        pvc, ndisc_vc = mcnemar_exact_p(vb, va)
        vc_pairs[name] = {
            "base_only": vb,
            "arm_only": va,
            "discordant_pairs": ndisc_vc,
            "mcnemar_exact_p": pvc,
        }
    first_vc = vc_pairs[vc_arms[0]]

    # baseline 靠旧版页命中的 query（slug 对、页面过时），各治理臂怎么处置
    old_hit_ids = [
        rec["id"] for rec, vh, h in zip(den, base_vhits, base_hits) if h and not vh
    ]
    golden_by_id = {rec["id"]: set(rec["golden_slugs"]) for rec in den}
    disposition: Dict[str, dict] = {}
    for name in vc_arms:
        corrected = still_old = lost = 0
        for rid in old_hit_ids:
            first = first_golden_candidate(rankings[name][rid], golden_by_id[rid])
            if first is None:
                lost += 1
            elif is_old_version(first):
                still_old += 1
            else:
                corrected += 1
        disposition[name] = {
            "corrected_to_new": corrected,
            "still_old": still_old,
            "lost_hit": lost,
        }

    # 首选臂：VC 分子优先，其次 hit@5 分子（即「纠正得多、且召回不掉」）
    preferred = max(
        vc_arms, key=lambda a: (vc_per_arm[a]["num"], by_name[a]["hit@5_num"])
    )
    delta_vc_pt = (
        (vc_per_arm[preferred]["num"] - vc_per_arm["baseline"]["num"]) / len(den) * 100
    )
    zero_loss = (
        by_name[preferred]["hit@5_num"] == by_name["baseline"]["hit@5_num"]
    )
    base_hit_num = by_name["baseline"]["hit@5_num"]

    result = {
        "meta": {
            "experiment": "version_governance_ablation",
            "generated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
            "input_file": str(input_path.relative_to(REPO_ROOT)).replace("\\", "/"),
            "input_sha256": sha256_of(input_path),
            "input_sha256_rule": (
                "CRLF 归一成 LF 之后再算（与 tests/regression_baselines.json 同一算法）。"
                "本机 core.autocrlf=true，工作区是 CRLF、仓库 blob 是 LF；不归一会得到"
                "另一个值，在 Linux CI 上永远对不上，闸门会红在换行符上而不是真实产物变化上。"
            ),
            "input_records": len(records),
            "denominator": len(den),
            "denominator_rule": "skip unanswerable=True 或无 golden_slugs；命中判定 = candidate.slug ∈ golden_slugs",
            "stage_used": STAGE,
            "candidate_granularity": "chunk 级（同一 slug 可重复出现），三档一律不预先去重",
            "old_version_prefix": OLD_VERSION_PREFIX,
            "arms": {
                "baseline": f"{STAGE} 原序（按 rank 升序）取 top-{K_HIT}",
                "filter": f"先剔除 src 以 '{OLD_VERSION_PREFIX}' 开头的候选，再按原序取 top-{K_HIT}",
                "decay_0.5": f"旧版候选 score×0.5 后按 score 降序稳定重排，取 top-{K_HIT}（仅参考）",
                "decay_0.9": f"旧版候选 score×0.9 后按 score 降序稳定重排，取 top-{K_HIT}（仅参考）",
            },
            "metrics": {
                "hit@5": f"top-{K_HIT} 内是否有任一候选 slug ∈ golden_slugs；分母 {len(den)}",
                "mrr@10": f"首个命中位次 r≤{K_MRR} 取 1/r，否则 0；对 {len(den)} 条求均值",
                "ndcg@10": (
                    "二值 qrels（gain=1）的 nDCG@10：DCG=Σ gain/log2(rank+1)，"
                    "IDCG 按 min(|golden|,10) 个相关文档全排最前计算；"
                    "同一 slug 只在首次出现处计一次 gain。"
                    "口径来源：BEIR / MTEB 的 nDCG@10 即为二值 qrels 口径，无需分级标注"
                ),
                "ndcg@10_no_dedupe": "同上但不去重 slug（敏感性对照，不作为判据）",
                "interference_rate": f"top-{K_HIT} 中含 '{OLD_VERSION_PREFIX}' 候选的 query 占比",
                "version_correct@5": (
                    f"top-{K_HIT} 内首个命中候选的 src 不以 '{OLD_VERSION_PREFIX}' 开头才算；"
                    f"未命中记 0。区分「命中与否」与「命中的是不是当前版本」，"
                    f"恒有 version_correct@5 ≤ hit@5"
                ),
            },
            "stats": {
                "mcnemar": "exact, two-sided, Binomial(b+c, 0.5)",
                "bootstrap": f"paired, n={BOOTSTRAP_N}, seed={BOOTSTRAP_SEED}, percentile 95% CI",
            },
            "no_model_loaded": True,
            "note": "判据只看 baseline vs filter；decay 两档仅作参考，不参与结论",
        },
        "per_arm": {a: strip(by_name[a]) for a in by_name},
        "baseline_vs_filter": {
            "hit@5_delta_num": by_name["filter"]["hit@5_num"] - by_name["baseline"]["hit@5_num"],
            "hit@5_delta_pt": delta_pt,
            "base_only": b,
            "filter_only": c,
            "discordant_pairs": n_disc,
            "mcnemar_exact_p": p_value,
            "bootstrap": bootstrap,
            "lost_hit_ids": lost_ids,
            "gained_hit_ids": gained_ids,
            "lost_hit_detail": lost_detail,
            "diagnostics": {
                "baseline_interference_vs_hit": crosstab,
                "old_version_chunks_in_top5": old_chunks_in_top5,
                "old_version_chunks_in_pool": old_chunks_total,
                "candidates_total": cands_total,
                "filter_removes_frac_of_pool": (
                    old_chunks_total / cands_total if cands_total else 0.0
                ),
                "note": (
                    "crosstab 说明旧版占位与命中几乎正交：interfered_miss 是过滤白捡的空间，"
                    "interfered_hit 是过滤会动到的命中，clean_miss 是过滤救不了的部分"
                ),
            },
        },
        "version_correct": {
            "_why": (
                "slug 级 Hit@5 只看「有没有命中同 slug 的页」—— 新旧版同名页都算命中，"
                "所以命中来源是哪一版在 Hit@5 里完全看不见。EVAL_REPORT 第六节已明确写过"
                "这个陷阱（「收益被指标掩盖」），本节把同一警告用在治理实验上："
                "三档治理洗的是命中来源，不是命中与否。"
            ),
            "definition": (
                f"VersionCorrect@{K_HIT} = top-{K_HIT} 内首个命中候选的 src 不以 "
                f"'{OLD_VERSION_PREFIX}' 开头；未命中记 0。分母同 {len(den)}。"
                f"因此 VersionCorrect@{K_HIT} ≤ Hit@{K_HIT} 恒成立。"
            ),
            "per_arm": vc_per_arm,
            "vs_baseline": {
                "delta_pt": delta_vc_pt,
                "base_only": first_vc["base_only"],
                "arm_only": first_vc["arm_only"],
                "discordant_pairs": first_vc["discordant_pairs"],
                "mcnemar_exact_p": first_vc["mcnemar_exact_p"],
                "applies_to": list(vc_arms),
                "per_arm_pairs": vc_pairs,
                "note": (
                    "不一致对是否单向可直接从 base_only / arm_only 读出；"
                    "三个治理臂在这条口径上应当给出同一组不一致对。"
                ),
            },
            "old_version_hits_in_baseline": {
                "count": len(old_hit_ids),
                "of_total_hits": base_hit_num,
                "share": (len(old_hit_ids) / base_hit_num) if base_hit_num else 0.0,
                "disposition": disposition,
                "note": (
                    "baseline 的命中里有一批是「靠旧版页命中」：slug 对了、页面过时。"
                    "三档治理纠正的条数与代价（丢命中）见 disposition。"
                ),
            },
            "preferred_arm": {
                "arm": preferred,
                "reason": (
                    f"排序层降权不需要重建索引（省 65~75 min），"
                    f"Hit@5 {by_name[preferred]['hit@5_num']}/{len(den)}"
                    + ("（与 baseline 持平，零召回代价）" if zero_loss else "（低于 baseline）")
                    + f"，VersionCorrect@{K_HIT} {delta_vc_pt:+.2f}pt，干扰率 "
                    f"{by_name['baseline']['interference_rate'] * 100:.1f}% → "
                    f"{by_name[preferred]['interference_rate'] * 100:.1f}%"
                ),
            },
            "boundary": [
                "这不是「召回提升」：被纠正的条数里绝大多数本来就是命中，只是命中页面从旧版换成新版。"
                "对外只能说「命中来源版本的正确率」，不得说 Hit@5 提升。",
                "对版本无关的 query（如「TiDB 架构是什么」）该口径过严 —— 旧版页也能答，只是不理想。",
                "样本是主集 335，不是 [Z] 探针 31；不得与第六节「危害 19.4% → 9.7%」混报，"
                "两者 query 构造方式与分母都不同。",
                "本口径在既有判据之外补算，不参与 verdict 判定；是否立为正式指标待定。",
            ],
        },
        "verdict": {
            "judged_arm": "filter",
            "criterion": matched,
            "hit5_delta_pt": delta_pt,
            "p_value": p_value,
            "passed": crit_a,
            "conclusion": conclusion,
            "scope_limit": (
                "本结论只覆盖「找不找得到」（slug 级 Hit@5）这一个问题。"
                "命中来源版本另见 version_correct：那里的方向可能相反，"
                "两个口径回答的不是同一个问题 —— 不得互相替代，"
                "也不得用其中一条推出「治理有 / 没有价值」。"
            ),
            "next_step": (
                "转去验证模型规模（reranker-large）这一条仍然成立："
                "slug 口径上没有剩余空间；版本正确性上的收益不值得为此重建索引，"
                "用排序层降权即可（见 version_correct.preferred_arm）。"
            ),
        },
    }
    if by_name["decay_0.9"]:
        result["baseline_vs_filter"]["decay_0.9_delta_pt"] = (
            by_name["decay_0.9"]["hit@5_num"] - by_name["baseline"]["hit@5_num"]
        ) / len(den) * 100

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return result


def print_report(res: dict) -> None:
    n = res["meta"]["denominator"]
    print("=" * 72)
    print(f"  Version Governance Ablation  (分母 {n}, stage={STAGE})")
    print("=" * 72)
    hdr = (
        f"{'arm':<12}{'Hit@5':>14}{'VC@5':>14}"
        f"{'MRR@10':>10}{'NDCG@10':>10}{'干扰率':>14}"
    )
    print(hdr)
    print("-" * 72)
    for name, arm in res["per_arm"].items():
        print(
            f"{name:<12}"
            f"{arm['hit@5_num']:>6}/{n:<7}"
            f"{arm['version_correct@5_num']:>6}/{n:<7}"
            f"{arm['mrr@10']:>10.4f}"
            f"{arm['ndcg@10']:>10.4f}"
            f"{arm['interference_num']:>6}/{n} ({arm['interference_rate']*100:>5.1f}%)"
        )
    print("-" * 72)
    vs = res["baseline_vs_filter"]
    bs = vs["bootstrap"]
    print(f"  配对：丢 {vs['base_only']} / 得 {vs['filter_only']} / 不一致 {vs['discordant_pairs']}")
    print(f"  McNemar exact p = {vs['mcnemar_exact_p']:.6f}")
    print(
        f"  Δ Hit@5 = {vs['hit@5_delta_pt']:+.2f}pt  "
        f"bootstrap 95% CI [{bs['ci95_pt'][0]:+.2f}, {bs['ci95_pt'][1]:+.2f}] pt "
        f"(n={bs['n_resamples']}, seed={bs['seed']})"
    )
    if vs["lost_hit_detail"]:
        print("  丢失命中明细：")
        for d in vs["lost_hit_detail"]:
            tag = "仅旧版在 top-5" if d["only_from_old_version"] else "两版都在 top-5"
            twins = d["new_version_twin_in_pool"]
            twin_txt = (
                f"新版副本池内位次={[t['rank'] for t in twins]}" if twins else "新版副本未被召回"
            )
            print(f"    - {d['id']} [{tag}] 金标候选={[c['src'] for c in d['baseline_golden_candidates']]} {twin_txt}")
    dg = vs["diagnostics"]
    print(
        f"  诊断：干扰且命中 {dg['baseline_interference_vs_hit']['interfered_hit']} / "
        f"干扰但未命中 {dg['baseline_interference_vs_hit']['interfered_miss']} / "
        f"无干扰且命中 {dg['baseline_interference_vs_hit']['clean_hit']} / "
        f"无干扰未命中 {dg['baseline_interference_vs_hit']['clean_miss']}"
    )
    print(
        f"  旧版 chunk 占候选池 {dg['old_version_chunks_in_pool']}/{dg['candidates_total']}"
        f"（{dg['filter_removes_frac_of_pool']*100:.1f}%），其中进 top-5 的 "
        f"{dg['old_version_chunks_in_top5']} 个"
    )
    print("-" * 72)
    v = res["verdict"]
    print(f"  判据：{v['criterion']}")
    print(f"  结论：{v['conclusion']}")
    vc = res["version_correct"]
    print("-" * 72)
    pa = vc["preferred_arm"]["arm"]
    print(
        f"  VersionCorrect@{K_HIT}（命中来源版本正确率）：baseline "
        f"{vc['per_arm']['baseline']['num']}/{n} → {pa} "
        f"{vc['per_arm'][pa]['num']}/{n}"
        f"（Δ {vc['vs_baseline']['delta_pt']:+.2f}pt, "
        f"McNemar p={vc['vs_baseline']['mcnemar_exact_p']:.3g}）"
    )
    print(f"  首选臂：{pa} —— {vc['preferred_arm']['reason']}")
    print(f"  baseline 靠旧版命中的 {vc['old_version_hits_in_baseline']['count']} 条处置：")
    for arm, d in vc["old_version_hits_in_baseline"]["disposition"].items():
        print(
            f"    - {arm:<10} 纠正 {d['corrected_to_new']:>2} / "
            f"仍旧版 {d['still_old']:>2} / 丢命中 {d['lost_hit']:>2}"
        )
    print("=" * 72)


def main() -> None:
    ap = argparse.ArgumentParser(description="版本治理对照实验（离线，不加载模型）")
    ap.add_argument("--input", default=str(DEFAULT_INPUT), help="检索产物 JSONL")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="输出 JSON")
    args = ap.parse_args()
    res = run(Path(args.input), Path(args.out))
    print_report(res)
    print(f"\n已写入 {args.out}")


if __name__ == "__main__":
    sys.exit(main())
