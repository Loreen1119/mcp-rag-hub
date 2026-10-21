#!/usr/bin/env python
"""baseline tag diff 门禁 —— 对比两份评测产物，检测指标退化。

用途（2026-10-21）：
  现有 tests/regression_baselines.json + test_regression_replay.py 是防篡改
  （产物一变 sha256 不匹配就红）。缺的是防退化：新配置跑出来 vs 旧配置，指标掉了多少。
  本脚本填补这个缺口。

输入：两份逐条 JSONL（格式同 experiments/baseline_ladder/retrieval_375.jsonl）
  每条含 id / golden_slugs / unanswerable / stages[].hit@5

输出：
  - per-metric 对比：Hit@5 的分子/分母（不要只给百分比）
  - 新增命中 id 列表、丢失命中 id 列表
  - 结论行：REGRESSION / OK（退出码 1 / 0）

阈值（两条都要打印出来）：
  - Hit@5 下降 > 1.0pt → REGRESSION
  - 分子减少 ≥ 3 条 → REGRESSION
  （335 分母下 1.0pt ≈ 3.35 条，所以两个条件都要让人看到）

用法：
  python -m scripts.diff_eval_run old.jsonl new.jsonl
  python -m scripts.diff_eval_run old.jsonl new.jsonl --stage rrf
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

# 阈值常量
REGRESSION_THRESHOLD_PT = 1.0   # Hit@5 下降超过 1.0pt 判退化
REGRESSION_THRESHOLD_N = 3      # 分子减少 ≥ 3 条判退化


def load_jsonl(path: str | Path) -> List[dict]:
    """加载 JSONL 评测产物。"""
    p = Path(path)
    if not p.exists():
        print(f"错误：文件不存在：{p}", file=sys.stderr)
        sys.exit(2)
    records = []
    with open(p, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"错误：{p} 第 {line_no} 行 JSON 解析失败：{e}", file=sys.stderr)
                sys.exit(2)
    return records


def extract_hit5(records: List[dict], stage: str) -> Dict[str, bool]:
    """提取每条记录在指定 stage 的 hit@5。

    只取 answerable（unanswerable=false）的记录 —— 不可答类无 golden，不计入 Hit@5 分母。
    """
    result: Dict[str, bool] = {}
    for r in records:
        rid = r.get("id")
        if rid is None:
            continue
        if r.get("unanswerable", False):
            continue
        stages = r.get("stages", {})
        if stage not in stages:
            continue
        hit5 = stages[stage].get("hit@5", False)
        result[rid] = bool(hit5)
    return result


def compute_hit5_numerator_denominator(hit_map: Dict[str, bool]) -> Tuple[int, int]:
    """计算 Hit@5 的分子（命中数）和分母（answerable 总数）。"""
    denominator = len(hit_map)
    numerator = sum(1 for v in hit_map.values() if v)
    return numerator, denominator


def diff_runs(
    old_path: str | Path,
    new_path: str | Path,
    stage: str = "rrf",
) -> int:
    """对比两份评测产物，输出 diff 报告。

    Returns:
        0 = OK, 1 = REGRESSION, 2 = 错误（分母不同 / 文件缺失）
    """
    old_records = load_jsonl(old_path)
    new_records = load_jsonl(new_path)

    old_hits = extract_hit5(old_records, stage)
    new_hits = extract_hit5(new_records, stage)

    old_num, old_den = compute_hit5_numerator_denominator(old_hits)
    new_num, new_den = compute_hit5_numerator_denominator(new_hits)

    # 分母校验：两份必须同口径
    if old_den != new_den:
        print(
            f"错误：分母不同，禁止跨口径比较。\n"
            f"  {old_path}: 分母 = {old_den}\n"
            f"  {new_path}: 分母 = {new_den}",
            file=sys.stderr,
        )
        return 2

    # 逐 id 对比
    old_hit_ids: Set[str] = {rid for rid, v in old_hits.items() if v}
    new_hit_ids: Set[str] = {rid for rid, v in new_hits.items() if v}
    gained_ids = sorted(new_hit_ids - old_hit_ids)
    lost_ids = sorted(old_hit_ids - new_hit_ids)

    # 变化量
    delta_num = new_num - old_num
    delta_pt = (new_num / new_den * 100 - old_num / old_den * 100) if new_den > 0 else 0.0

    # 判定
    regression_reasons: List[str] = []
    if delta_pt < -REGRESSION_THRESHOLD_PT:
        regression_reasons.append(
            f"Hit@5 下降 {abs(delta_pt):.2f}pt > {REGRESSION_THRESHOLD_PT}pt 阈值"
        )
    if delta_num <= -REGRESSION_THRESHOLD_N:
        regression_reasons.append(
            f"分子减少 {abs(delta_num)} 条 ≥ {REGRESSION_THRESHOLD_N} 条阈值"
        )

    is_regression = len(regression_reasons) > 0

    # ---- 输出报告 ----
    print(f"{'='*60}")
    print(f"  Eval Diff Report: {stage}")
    print(f"{'='*60}")
    print(f"  旧产物: {old_path}")
    print(f"  新产物: {new_path}")
    print(f"  Stage:  {stage}")
    print()

    print(f"  {'指标':<12} {'旧':>12} {'新':>12} {'变化':>12}")
    print(f"  {'-'*48}")
    print(
        f"  {'Hit@5 分子':<12} {old_num:>12} {new_num:>12} {delta_num:>+12}"
    )
    print(
        f"  {'Hit@5 分母':<12} {old_den:>12} {new_den:>12} {0:>+12}"
    )
    print(
        f"  {'Hit@5 率':<12} {old_num/old_den*100 if old_den else 0:>11.2f}% "
        f"{new_num/new_den*100 if new_den else 0:>11.2f}% {delta_pt:>+11.2f}pt"
    )
    print()

    if gained_ids:
        print(f"  新增命中（{len(gained_ids)} 条）：")
        for rid in gained_ids[:20]:
            print(f"    + {rid}")
        if len(gained_ids) > 20:
            print(f"    … 还有 {len(gained_ids) - 20} 条")
    else:
        print(f"  新增命中：0 条")

    if lost_ids:
        print(f"  丢失命中（{len(lost_ids)} 条）：")
        for rid in lost_ids[:20]:
            print(f"    - {rid}")
        if len(lost_ids) > 20:
            print(f"    … 还有 {len(lost_ids) - 20} 条")
    else:
        print(f"  丢失命中：0 条")

    print()

    # 判定条件详情（两条都打印）
    print(f"  判定条件：")
    pt_judge = "🔴 触发" if delta_pt < -REGRESSION_THRESHOLD_PT else "🟢 通过"
    n_judge = "🔴 触发" if delta_num <= -REGRESSION_THRESHOLD_N else "🟢 通过"
    print(f"    ① Hit@5 下降 {abs(delta_pt):.2f}pt "
          f"{'>' if delta_pt < -REGRESSION_THRESHOLD_PT else '≤'} "
          f"{REGRESSION_THRESHOLD_PT}pt → {pt_judge}")
    print(f"    ② 分子减少 {abs(delta_num)} 条 "
          f"{'≥' if delta_num <= -REGRESSION_THRESHOLD_N else '<'} "
          f"{REGRESSION_THRESHOLD_N} 条 → {n_judge}")
    print()

    if is_regression:
        print(f"  🔴 结论：REGRESSION")
        for reason in regression_reasons:
            print(f"     - {reason}")
        print()
        return 1
    else:
        print(f"  🟢 结论：OK")
        print()
        return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="baseline tag diff 门禁 —— 对比两份评测产物检测退化",
    )
    parser.add_argument("old", help="旧产物 JSONL 路径")
    parser.add_argument("new", help="新产物 JSONL 路径")
    parser.add_argument(
        "--stage",
        default="rrf",
        help="对比的 stage（默认 rrf；可选 bm25 / vector / rrf）",
    )
    args = parser.parse_args()

    exit_code = diff_runs(args.old, args.new, stage=args.stage)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
