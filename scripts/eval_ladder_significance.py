#!/usr/bin/env python
"""检索阶梯（BM25 / 向量 / RRF / CE）的配对显著性检验 —— 落盘 + 可 `--check` 对拍。

为什么有这个脚本：
  `EVAL_REPORT.md` 第十一节报的 p 值与 CI 此前只有散文、没有可复算的产物，而
  「配对检验只在 ablation_version_governance.py 里实现过、且只服务版本治理那一个实验」
  正是这套评测最容易被追问的缺口。本脚本把同一批 335 条 query 的配对检验
  固定成 `experiments/baseline_ladder/significance.json`，并由
  `tests/test_ladder_significance.py` 在 CI 回放层复算对拍。

口径（改任何一条都等于改结论，必须同步改测试）：
  - 分母 335 = 375 条主集里「非 unanswerable 且有 golden」的样本。
  - 配对以 **query 的 `id` 为键**对齐，不按文件行序 zip —— 见 `_load_hit_vectors()`
    里的注释：两个产物的行序本来就不同，按行号 zip 会算出 65/65 这种假的不一致对。
  - Δ 一律是「arm − base」的百分点，正数表示 arm 更高。
  - lost = base 命中且 arm 未命中；gained = arm 命中且 base 未命中。
  - significant = p < alpha **且** 95% CI 不含 0（两个条件同时成立才允许说「显著」）。

统计函数**只 import 复用**，不复制实现：
  `scripts/ablation_version_governance.mcnemar_exact_p / paired_bootstrap_delta`
  —— 项目已经踩过「同一逻辑两处实现，修 bug 只修一处」的坑。

用法：
  python -m scripts.eval_ladder_significance            # 重算并写入 significance.json
  python -m scripts.eval_ladder_significance --check    # 重算并与已存产物对拍（忽略 generated_at）
  python -m scripts.eval_ladder_significance --check --out <json>
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
# 支持 `python scripts/eval_ladder_significance.py`（此时 sys.path[0] 是 scripts/，
# `scripts.xxx` 这种包内绝对导入会失败；与 src/mcp_server.py 的兜底同理）。
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.ablation_version_governance import (  # noqa: E402  (必须在 sys.path 兜底之后)
    ALPHA,
    BOOTSTRAP_N,
    BOOTSTRAP_SEED,
    build_denominator,
    load_records,
    mcnemar_exact_p,
    paired_bootstrap_delta,
    sha256_of,
)

# ---- 路径 ----
RETRIEVAL_PATH = REPO_ROOT / "experiments" / "baseline_ladder" / "retrieval_375.jsonl"
CE_PATH = REPO_ROOT / "experiments" / "baseline_ladder" / "ce_scores.jsonl"
DEFAULT_OUT = REPO_ROOT / "experiments" / "baseline_ladder" / "significance.json"

# ---- 口径常量 ----
STAGES = ("bm25", "vector", "rrf")
# 输出顺序 = EVAL_REPORT 第十一节表格顺序；元组是 (base, arm)，Δ 按 arm − base。
COMPARISONS: Sequence[Tuple[str, str]] = (
    ("vector", "rrf"),
    ("bm25", "rrf"),
    ("bm25", "vector"),
    ("rrf", "ce"),
)
# CE 产物的命中字段名按候选列表**自动探测**：不同批次曾经用过不同名字
# （生产批是 `hit@5`，历史批次写过 `ce_top_hit@5`），硬编码会在换批时静默 KeyError
# 或更糟 —— 静默取到别的字段。探测失败一律 fail-closed。
CE_HIT_FIELD_CANDIDATES = ("hit@5", "ce_top_hit@5", "ce_hit@5", "hit5", "hit_5")

_ROUND_DELTA = 4
_ROUND_CI = 4
_ROUND_P = 6


# ============================================================
# 载入与配对
# ============================================================
def detect_hit_field(record: dict, candidates: Sequence[str] = CE_HIT_FIELD_CANDIDATES) -> str:
    """在一条记录里探测命中字段名；探测不到就 fail-closed。"""
    for name in candidates:
        if name in record:
            return name
    raise SystemExit(
        "CE 产物里找不到命中字段："
        f"试过 {list(candidates)}，实际字段 {sorted(record)}"
    )


def _stage_hit(record: dict, stage: str) -> bool:
    """取某一路的 hit@5 布尔值；缺字段直接失败，不默认 False。"""
    try:
        value = record["stages"][stage]["hit@5"]
    except KeyError as exc:
        raise SystemExit(f"记录 {record.get('id')!r} 缺少 stages.{stage}.hit@5（{exc}）")
    if value is None:
        raise SystemExit(f"记录 {record.get('id')!r} 的 stages.{stage}.hit@5 为 None")
    return bool(value)


def _load_hit_vectors() -> Tuple[Dict[str, List[bool]], Dict[str, object]]:
    """构造四条臂的逐条命中向量，**同序**（以 retrieval_375.jsonl 的 id 顺序为基准）。

    为什么必须按 id 对齐而不是按文件行序 zip：
      `ce_scores.jsonl` 的行序是「mk_* / ask_* / doc_* 交错」的产物顺序，
      `retrieval_375.jsonl` 是「asktug → doc_fact → unanswerable → miss_kill」的
      评测集顺序。两者不是同一序（实测 `row_order_identical=False`）。
      按行号 zip 会把这 335 条配成 335 组假配对，算出 **65 得 / 65 丢** 这种
      纯属错配的不一致对；按 id 对齐后是 **27 得 / 27 丢**，与
      `tests/regression_baselines.json` 里 ce_hit@5 的 note「与 RRF 逐位相同
      （+27/−27 净 0）」一致。错配的那一版曾出现在任务书里，已按本注释纠正。
    """
    retrieval = load_records(RETRIEVAL_PATH)
    ce_rows = load_records(CE_PATH)

    den = build_denominator(retrieval)
    ce_den = build_denominator(ce_rows)
    if not den:
        raise SystemExit(f"分母为空：{RETRIEVAL_PATH}")
    if not ce_den:
        raise SystemExit(f"分母为空：{CE_PATH}")

    ce_field = detect_hit_field(ce_den[0])
    ce_by_id: Dict[str, dict] = {r["id"]: r for r in ce_den}
    ids = [r["id"] for r in den]

    missing = [i for i in ids if i not in ce_by_id]
    if missing:
        raise SystemExit(
            f"CE 产物缺 {len(missing)} 条主集样本（前 5 条：{missing[:5]}）—— 两条臂不是同一批 query，禁止配对"
        )
    extra = [i for i in ce_by_id if i not in set(ids)]
    if extra:
        raise SystemExit(
            f"CE 产物多出 {len(extra)} 条不属于主集分母的样本（前 5 条：{extra[:5]}）"
        )

    hits: Dict[str, List[bool]] = {
        stage: [_stage_hit(r, stage) for r in den] for stage in STAGES
    }
    ce_vector: List[bool] = []
    none_ids: List[str] = []
    for i in ids:
        raw = ce_by_id[i][ce_field]
        if raw is None:
            none_ids.append(i)
        ce_vector.append(bool(raw))
    if none_ids:
        raise SystemExit(
            f"CE 字段 {ce_field!r} 有 {len(none_ids)} 条为 None（前 5 条：{none_ids[:5]}）"
        )
    hits["ce"] = ce_vector

    meta: Dict[str, object] = {
        "pairing_key": "id",
        "row_order_identical": ids == [r["id"] for r in ce_den],
        "ce_hit_field": ce_field,
    }
    return hits, meta


# ============================================================
# 计算
# ============================================================
def compute() -> dict:
    """跑完四组对比，返回待落盘的结构（不含 generated_at）。"""
    hits, meta = _load_hit_vectors()
    denominator = len(hits["bm25"])
    comparisons: List[dict] = []

    for base, arm in COMPARISONS:
        base_vec = hits[base]
        arm_vec = hits[arm]
        if len(base_vec) != len(arm_vec):  # 理论上不可能，写出来防未来改动
            raise SystemExit(f"{base}/{arm} 配对长度不一致：{len(base_vec)} vs {len(arm_vec)}")

        lost = sum(1 for b, a in zip(base_vec, arm_vec) if b and not a)
        gained = sum(1 for b, a in zip(base_vec, arm_vec) if a and not b)
        base_hits = sum(1 for b in base_vec if b)
        arm_hits = sum(1 for a in arm_vec if a)

        p_value, discordant = mcnemar_exact_p(lost, gained)
        boot = paired_bootstrap_delta(base_vec, arm_vec)  # 内部用 BOOTSTRAP_N / BOOTSTRAP_SEED
        lo, hi = boot["ci95_pt"]
        crosses_zero = bool(lo <= 0.0 <= hi)
        delta_pt = (arm_hits - base_hits) / denominator * 100.0

        comparisons.append(
            {
                "comparison": f"{arm}_vs_{base}",
                "base": base,
                "arm": arm,
                "base_hits": base_hits,
                "arm_hits": arm_hits,
                "delta_pt": round(delta_pt, _ROUND_DELTA),
                "ci95_pt": [round(lo, _ROUND_CI), round(hi, _ROUND_CI)],
                "mcnemar_p": round(p_value, _ROUND_P),
                "mcnemar_discordant": discordant,
                "lost": lost,
                "gained": gained,
                # 两个条件同时成立才叫显著：p 过线、且 CI 不含 0。只报 p 的话，
                # 会出现「p<0.05 但 CI 跨 0」的自相矛盾表述（本次四组两者一致）。
                "significant": bool(p_value < ALPHA and not crosses_zero),
                "ci_crosses_zero": crosses_zero,
            }
        )

    return {
        "denominator": denominator,
        "bootstrap": {
            "n_resamples": BOOTSTRAP_N,
            "seed": BOOTSTRAP_SEED,
            "method": "percentile, paired resampling of query indices",
        },
        "alpha": ALPHA,
        "pairing": meta,
        "inputs": [
            {
                "path": str(RETRIEVAL_PATH.relative_to(REPO_ROOT)).replace("\\", "/"),
                "sha256": sha256_of(RETRIEVAL_PATH),
                "denominator": denominator,
            },
            {
                "path": str(CE_PATH.relative_to(REPO_ROOT)).replace("\\", "/"),
                "sha256": sha256_of(CE_PATH),
                "denominator": denominator,
            },
        ],
        "comparisons": comparisons,
    }


# ============================================================
# 对拍
# ============================================================
_IGNORED_KEYS = frozenset({"generated_at"})


def _diff_nodes(stored, fresh, path: str = "$") -> List[str]:
    """逐字段比对，返回人类可读的差异行（空列表=一致）。"""
    if isinstance(stored, dict) and isinstance(fresh, dict):
        diffs: List[str] = []
        for key in sorted(set(stored) | set(fresh)):
            if not path.count(".") and key in _IGNORED_KEYS:  # 仅跳过顶层 generated_at
                continue
            sub = f"{path}.{key}"
            if key not in stored:
                diffs.append(f"{sub}: 已存产物缺失，新算得 {fresh[key]!r}")
            elif key not in fresh:
                diffs.append(f"{sub}: 新算结果缺失，已存为 {stored[key]!r}")
            else:
                diffs.extend(_diff_nodes(stored[key], fresh[key], sub))
        return diffs
    if isinstance(stored, list) and isinstance(fresh, list):
        if len(stored) != len(fresh):
            return [f"{path}: 长度不一致，已存 {len(stored)} / 新算 {len(fresh)}"]
        diffs = []
        for i, (s, f) in enumerate(zip(stored, fresh)):
            diffs.extend(_diff_nodes(s, f, f"{path}[{i}]"))
        return diffs
    if isinstance(stored, float) and isinstance(fresh, float):
        if abs(stored - fresh) > 1e-9:
            return [f"{path}: 已存 {stored!r} ≠ 新算 {fresh!r}"]
        return []
    if stored != fresh:
        return [f"{path}: 已存 {stored!r} ≠ 新算 {fresh!r}"]
    return []


def compare_with_stored(fresh: dict, out_path: Path) -> List[str]:
    if not out_path.exists():
        return [f"产物不存在：{out_path}（先不带 --check 跑一次生成）"]
    try:
        stored = json.loads(out_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [f"产物不是合法 JSON：{out_path}（{exc}）"]
    return _diff_nodes(stored, fresh)


# ============================================================
# 输出
# ============================================================
def write_result(result: dict, out_path: Path) -> dict:
    payload = {"generated_at": datetime.now().astimezone().isoformat(timespec="seconds")}
    payload.update(result)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return payload


def print_report(payload: dict) -> None:
    print("=" * 92)
    print(f"检索阶梯配对显著性（分母 {payload['denominator']}，"
          f"bootstrap n={payload['bootstrap']['n_resamples']} seed={payload['bootstrap']['seed']}）")
    print(f"配对键：{payload['pairing']['pairing_key']}　"
          f"两产物行序一致：{payload['pairing']['row_order_identical']}　"
          f"CE 命中字段：{payload['pairing']['ce_hit_field']!r}")
    print("-" * 92)
    header = f"{'对比':<16}{'base':>6}{'arm':>6}{'Δpt':>9}{'95% CI':>20}{'p':>10}{'丢/得':>10}  显著"
    print(header)
    for c in payload["comparisons"]:
        lo, hi = c["ci95_pt"]
        label = f"{c['arm']} vs {c['base']}"
        ci_text = f"[{lo:+.2f}, {hi:+.2f}]"
        disc_text = f"{c['lost']}/{c['gained']}"
        flag_text = "是" if c["significant"] else "否"
        print(
            f"{label:<16}{c['base_hits']:>6}{c['arm_hits']:>6}{c['delta_pt']:>+9.2f}"
            f"{ci_text:>20}{c['mcnemar_p']:>10.4f}{disc_text:>10}  {flag_text}"
        )
    print("=" * 92)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="检索阶梯配对显著性检验（离线，只读产物，不加载模型）"
    )
    parser.add_argument("--check", action="store_true",
                        help="重算并与已存 significance.json 对拍（忽略 generated_at）")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="产物 JSON 路径")
    args = parser.parse_args(argv)

    out_path = Path(args.out)
    fresh = compute()

    if args.check:
        diffs = compare_with_stored(fresh, out_path)
        if diffs:
            print(f"❌ 与已存产物不一致（{len(diffs)} 处）：{out_path}")
            for line in diffs:
                print(f"  - {line}")
            return 1
        print(f"✅ 与已存产物一致：{out_path}")
        print_report({"generated_at": "-", **fresh})
        return 0

    payload = write_result(fresh, out_path)
    print_report(payload)
    print(f"\n已写入 {out_path}")
    print("下一步：python -m scripts.eval_ladder_significance --check  应 exit 0")
    return 0


if __name__ == "__main__":
    sys.exit(main())
