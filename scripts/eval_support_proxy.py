"""路径一：用 bge-reranker 句级支持度做「忠实度」的客观代理指标，并做人工校准。

## 为什么是这个做法

7B 裁判两版 prompt 都过不了人工 Kappa（κ≈0，见 `EVAL_REPORT.md` 第九节）。要救
忠实度这一维，只有两条路：换更大的裁判模型，或换**不依赖裁判**的客观代理。

本脚本实现后者：把答案按中文标点拆句，逐句用 Cross-Encoder 打「这句是否被检索
上下文支持」的分，聚合到条目级，再拿 30 条人工标注做一致性检验。

## 事前定死的口径（看到分数之后不许改）

1. **切分**：30 条冻结样本 → 12 条留出（`HOLDOUT_SEED` 分层各 6 正 6 负）+
   18 条扫阈值。**留出集在扫阈值阶段完全不参与**，只用来报最终 κ。
   在 30 条上直接扫阈值再报 κ 等于在测试集上调参。
2. **句级聚合用 top-3 均值，不用 top-1 max**：max 会让「同主题但不支持」的片段
   占便宜（它只需要蒙中一句）。条目级 = 句级得分的均值。
3. **判据**：留出集 **κ ≥ 0.5 且 CI 95% 下限 > 0.2** 才算「有信号」；
   否则如实报「无信号」，不换阈值、不换聚合方式重试。
4. **必须同时报人工正率与代理正率**：只看 κ 会被宽严口径掩盖
   （人工正率 faithful 0.633，本身偏宽）。

## 已知的方法论缺陷（结论里必须带）

**循环依赖**：这里用的 `bge-reranker-base` 就是生产链路的 Cross-Encoder，而被判的
上下文正是它自己挑出来的 top-3。所以代理分数**系统性偏高**，且这种偏高人工校准
**发现不了**（校准比的是与人的一致性，不是绝对水平）。
→ **口径只能是「相对信号」，不得当绝对忠实度报**。

用法：
    python -m scripts.eval_support_proxy            # 跑全流程并落盘
    python -m scripts.eval_support_proxy --dry-run  # 只跑前 3 条，看分数区间
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import CROSS_ENCODER_MODEL  # noqa: E402  （需先补 sys.path）

CAL_DIR = PROJECT_ROOT / "experiments" / "judge_calibration"
FROZEN = CAL_DIR / "frozen_sample.json"
HUMAN = CAL_DIR / "human_labels.json"
GEN = PROJECT_ROOT / "experiments" / "generation_eval" / "gen_100.jsonl"
OUT = CAL_DIR / "support_proxy.json"

# ---- 事前定死的常量（改动等于改试验设计，会在 _doc 里留痕） ----
HOLDOUT_SEED = 20261017
HOLDOUT_N_EACH = 6            # 正向/负向各 6 条 → 留出 12 条
THRESHOLD_GRID = [round(0.10 + 0.05 * i, 2) for i in range(17)]   # 0.10 ~ 0.90
PASS_KAPPA = 0.5
PASS_CI_LOWER = 0.2
TOP_CHUNKS = 3
MAX_SENTENCES = 8
MIN_SENTENCE_CHARS = 6
BOOTSTRAP_ROUNDS = 2000

SPLIT_RE = re.compile(r"(?=\[来源\d+:)")
SENT_RE = re.compile(r"[。！？；\n]+")


# ============================================================
# 数据
# ============================================================


def load_inputs() -> tuple[list[dict], dict]:
    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
    human = json.loads(HUMAN.read_text(encoding="utf-8"))

    gen_rows = [json.loads(line) for line in GEN.read_text(encoding="utf-8").splitlines() if line.strip()]
    gen_by_id = {row["id"]: row for row in gen_rows}

    # 人工标签以 by_order 里的 order 为键，需要按 order 对齐到 id
    labels: dict[str, str] = {}
    for item in human["by_order"].values():
        labels[item["id"]] = item["faith"]

    rows = []
    for sel in frozen["selected"]:
        item_id = sel["id"]
        gen = gen_by_id.get(item_id)
        if gen is None or item_id not in labels:
            raise SystemExit(f"缺数据：{item_id}")
        rows.append(
            {
                "id": item_id,
                "stratum": sel["stratum"],
                "human": 1 if labels[item_id] == "忠实" else 0,
                "answer": gen.get("answer") or "",
                "context": gen.get("context") or "",
            }
        )
    return rows, frozen


def split_sentences(answer: str) -> list[str]:
    parts = [p.strip() for p in SENT_RE.split(answer)]
    kept = [p for p in parts if len(p) >= MIN_SENTENCE_CHARS]
    return kept[:MAX_SENTENCES]


def split_chunks(context: str) -> list[str]:
    pieces = [p.strip() for p in SPLIT_RE.split(context) if p.strip()]
    return pieces[:TOP_CHUNKS]


# ============================================================
# 支持度打分
# ============================================================


def score_items(rows: list[dict], dry_run: bool = False) -> list[dict]:
    """逐条打句级支持度。返回每条的句数、chunk 数与条目级得分。"""
    from src.fusion import CrossEncoderReranker

    reranker = CrossEncoderReranker()

    batch_pairs: list[tuple[str, str]] = []
    layout: list[dict] = []
    for idx, row in enumerate(rows):
        sentences = split_sentences(row["answer"])
        chunks = split_chunks(row["context"])
        start = len(batch_pairs)
        for sentence in sentences:
            for chunk in chunks:
                batch_pairs.append((sentence, chunk))
        layout.append({"index": idx, "n_sent": len(sentences), "n_chunk": len(chunks), "start": start,
                       "end": len(batch_pairs)})
        if dry_run and idx >= 2:
            break

    if not batch_pairs:
        raise SystemExit("没有可打分的句-片段对（检查 answer / context 是否为空）")

    scores = reranker.model.predict(batch_pairs, batch_size=16, show_progress_bar=False)
    scores = [float(s) for s in scores]
    print(f"打分完成：{len(scores)} 对，分数区间 [{min(scores):.4f}, {max(scores):.4f}]")
    if min(scores) < -0.01 or max(scores) > 1.01:
        print("⚠️ 分数不在 [0,1]，疑似未过 sigmoid —— 结果需重新审查", file=sys.stderr)

    for item in layout:
        n_pairs = item["end"] - item["start"]
        if item["n_sent"] == 0 or item["n_chunk"] == 0 or n_pairs == 0:
            item["support"] = None
            continue
        block = scores[item["start"]: item["end"]]
        # 句级 = 该句对 top-3 chunk 的均值；条目级 = 句级均值
        per_sentence = [sum(block[i:i + item["n_chunk"]]) / item["n_chunk"]
                        for i in range(0, len(block), item["n_chunk"])]
        item["support"] = sum(per_sentence) / len(per_sentence)

    out = []
    for item in layout:
        row = rows[item["index"]]
        out.append({**{k: row[k] for k in ("id", "stratum", "human")},
                    "n_sent": item["n_sent"], "n_chunk": item["n_chunk"],
                    "support": item["support"]})
    return out


# ============================================================
# Kappa
# ============================================================


def cohen_kappa(pred: list[int], human: list[int]) -> float:
    n = len(pred)
    if n == 0:
        return float("nan")
    po = sum(1 for p, h in zip(pred, human) if p == h) / n
    p_pos = sum(pred) / n
    h_pos = sum(human) / n
    pe = p_pos * h_pos + (1 - p_pos) * (1 - h_pos)
    if pe >= 1.0:
        return 0.0
    return (po - pe) / (1 - pe)


def bootstrap_ci(pred: list[int], human: list[int], rounds: int = BOOTSTRAP_ROUNDS) -> tuple[float, float]:
    rng = random.Random(HOLDOUT_SEED)
    n = len(pred)
    if n < 3:
        return (float("nan"), float("nan"))
    vals = []
    for _ in range(rounds):
        idx = [rng.randrange(n) for _ in range(n)]
        vals.append(cohen_kappa([pred[i] for i in idx], [human[i] for i in idx]))
    vals = [v for v in vals if v == v]
    vals.sort()
    lo = vals[int(0.025 * len(vals))]
    hi = vals[int(0.975 * len(vals)) - 1]
    return (lo, hi)


def auc(scores: list[float], labels: list[int]) -> float:
    """ROC-AUC（Mann-Whitney），只用于看排序能力，与阈值无关。"""
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return float("nan")
    wins = sum(1 for p in pos for q in neg if p > q) + 0.5 * sum(1 for p in pos for q in neg if p == q)
    return wins / (len(pos) * len(neg))


# ============================================================
# 主流程
# ============================================================


def main() -> int:
    ap = argparse.ArgumentParser(description="句级支持度代理指标的校准（路径一）")
    ap.add_argument("--dry-run", action="store_true", help="只跑前 3 条，检查分数区间，不落盘")
    args = ap.parse_args()

    rows, frozen = load_inputs()
    print(f"样本 {len(rows)} 条（人工正 {sum(r['human'] for r in rows)}/{len(rows)}）")

    scored = score_items(rows, dry_run=args.dry_run)
    valid = [s for s in scored if s["support"] is not None]
    if not valid:
        raise SystemExit("没有有效样本（answer 或 context 为空？）")
    print(f"有效 {len(valid)} 条，支持度区间 "
          f"[{min(s['support'] for s in valid):.4f}, {max(s['support'] for s in valid):.4f}]")

    if args.dry_run:
        for s in valid:
            print(f"  {s['id']} 人工={s['human']} 句{s['n_sent']}×块{s['n_chunk']} support={s['support']:.4f}")
        return 0

    # ---- 事前定死的切分：按人工标签分层，留出 12 条 ----
    rng = random.Random(HOLDOUT_SEED)
    pos = sorted([s["id"] for s in valid if s["human"] == 1])
    neg = sorted([s["id"] for s in valid if s["human"] == 0])
    holdout_ids = set(rng.sample(pos, min(HOLDOUT_N_EACH, len(pos)))) | set(
        rng.sample(neg, min(HOLDOUT_N_EACH, len(neg)))
    )
    scan = [s for s in valid if s["id"] not in holdout_ids]
    holdout = [s for s in valid if s["id"] in holdout_ids]
    print(f"切分：扫阈值 {len(scan)} 条 / 留出 {len(holdout)} 条（seed={HOLDOUT_SEED}）")

    # ---- 在 scan 上扫阈值 ----
    best = None
    scan_human = [s["human"] for s in scan]
    for t in THRESHOLD_GRID:
        pred = [1 if s["support"] >= t else 0 for s in scan]
        k = cohen_kappa(pred, scan_human)
        # tie-break：κ 相同时取更高阈值（判正更严格 = 更保守）
        if best is None or (k, t) > (best["kappa"], best["threshold"]):
            best = {"threshold": t, "kappa": k, "positive_rate": sum(pred) / len(pred)}

    # ---- 留出集只报一次 ----
    t = best["threshold"]
    h_pred = [1 if s["support"] >= t else 0 for s in holdout]
    h_human = [s["human"] for s in holdout]
    h_kappa = cohen_kappa(h_pred, h_human)
    ci = bootstrap_ci(h_pred, h_human)
    h_auc = auc([s["support"] for s in holdout], h_human)
    passed = (h_kappa >= PASS_KAPPA) and (ci[0] > PASS_CI_LOWER)

    result = {
        "_doc": (
            "路径一：bge-reranker 句级支持度作为忠实度的客观代理指标。"
            "设计在打分前定死：12 条留出（分层 6 正 6 负，seed 见 holdout_seed）+ 18 条扫阈值；"
            "句级取 top-3 chunk 均值（不用 top-1 max）；判据 = 留出 κ≥0.5 且 CI 下限>0.2。"
            "🔴 循环依赖：所用 reranker 即生产链路 CE，被判上下文也是它自己挑的 → 分数系统性偏高，"
            "只能作相对信号，不得当绝对忠实度报。人工标注为单标注者（系统作者）。"
        ),
        "created": "2026-10-17",
        "reranker_model": CROSS_ENCODER_MODEL,
        "holdout_seed": HOLDOUT_SEED,
        "n_total": len(valid),
        "n_scan": len(scan),
        "n_holdout": len(holdout),
        "threshold_grid": THRESHOLD_GRID,
        "scan_best": {"threshold": best["threshold"], "kappa": round(best["kappa"], 4),
                      "positive_rate": round(best["positive_rate"], 4)},
        "holdout": {
            "kappa": round(h_kappa, 4),
            "kappa_ci95": [round(ci[0], 4), round(ci[1], 4)],
            "auc": round(h_auc, 4),
            "human_positive_rate": round(sum(h_human) / len(h_human), 4),
            "proxy_positive_rate": round(sum(h_pred) / len(h_pred), 4),
            "confusion": {
                "TP": sum(1 for p, h in zip(h_pred, h_human) if p == 1 and h == 1),
                "FP": sum(1 for p, h in zip(h_pred, h_human) if p == 1 and h == 0),
                "FN": sum(1 for p, h in zip(h_pred, h_human) if p == 0 and h == 1),
                "TN": sum(1 for p, h in zip(h_pred, h_human) if p == 0 and h == 0),
            },
            "ids": sorted(s["id"] for s in holdout),
        },
        "pass_criteria": {"kappa_min": PASS_KAPPA, "ci_lower_min": PASS_CI_LOWER},
        "passes": passed,
        "verdict": (
            "有信号（可作相对信号使用，仍不得当已校准的绝对忠实度）"
            if passed else
            "无信号：句级支持度代理在留出集上未达判据，不换阈值/不换聚合方式重试"
        ),
        "items": [
            {"id": s["id"], "human": s["human"], "split": "holdout" if s["id"] in holdout_ids else "scan",
             "support": round(s["support"], 4), "n_sent": s["n_sent"], "n_chunk": s["n_chunk"]}
            for s in valid
        ],
        "frozen_sample_ref": str(FROZEN.relative_to(PROJECT_ROOT)).replace("\\", "/"),
    }

    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"→ 已写入 {OUT.relative_to(PROJECT_ROOT)}")
    print(f"scan 最优阈值 {t}（scan κ={best['kappa']:.3f}, 正率 {best['positive_rate']:.3f}）")
    print(f"留出：κ={h_kappa:.4f} CI=[{ci[0]:.3f}, {ci[1]:.3f}] AUC={h_auc:.3f} "
          f"人工正率={sum(h_human)/len(h_human):.3f} 代理正率={sum(h_pred)/len(h_pred):.3f}")
    print("结论：" + ("✅ 有信号" if passed else "❌ 无信号"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
