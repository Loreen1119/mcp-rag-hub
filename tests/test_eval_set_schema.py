"""评测集 schema 校验 —— CI 里能跑的那层（不读语料本体）。

为什么要单独一层：
- `corpora/tidb/` 整目录不入库（CC BY-SA 传染 + UGC，无再分发授权），**CI 里没有语料**，
  所以「golden 源文件是否真的存在」这类校验只能在本地跑，进不了 CI。
- CI 里能守的是**形状**：条数、分母、字段、slug 形态、primary 分布、冻结清单自洽。
  这些恰好是「改评测集却忘了记录」最容易出问题的地方 —— 与 `data/failure_mode_spec.md`
  第五节的红线（改冻结字段必须走解冻流程 + 版本 +0.1 + 记 corrections）配套。

依赖纪律：只用标准库 + pytest，不读 corpora/、不加载模型。
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BASELINE_FILE = ROOT / "tests" / "regression_baselines.json"

MAIN_SET = ROOT / "data" / "eval_set.jsonl"
VERSION_SET = ROOT / "data" / "eval_set_version.jsonl"
JUDGE_GOLD = ROOT / "data" / "judge_gold_100.jsonl"
FROZEN = ROOT / "data" / "eval_set_FROZEN.json"

# 与 data/failure_mode_spec.md 第三节「标注基准配置」对齐
EXPECTED_SOURCE_COUNTS = {"asktug": 148, "doc_fact": 167, "unanswerable": 40, "miss_kill": 20}
EXPECTED_TOTAL = 375
EXPECTED_HIT_DENOMINATOR = 335
EXPECTED_UNANSWERABLE = 40
ALLOWED_PRIMARY = {
    "召回缺失",
    "排序压制",
    "术语鸿沟",
    "应拒答未拒答",
    "拒答误杀",
    "none",
    "unclassified",
}


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        pytest.fail(f"评测集文件不存在：{path.relative_to(ROOT)}")
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


@pytest.fixture(scope="module")
def main_rows() -> list[dict]:
    return _read_jsonl(MAIN_SET)


@pytest.fixture(scope="module")
def version_rows() -> list[dict]:
    return _read_jsonl(VERSION_SET)


# ============================================================
# 主集 375 的形状
# ============================================================


def test_主集条数与来源分布(main_rows):
    assert len(main_rows) == EXPECTED_TOTAL, f"主集应为 {EXPECTED_TOTAL} 条，实际 {len(main_rows)}"
    assert dict(Counter(r["source"] for r in main_rows)) == EXPECTED_SOURCE_COUNTS, (
        "来源分布与口径不符（AskTUG 148 + 文档事实 167 + 不可答 40 + 误杀探针 20）"
    )


def test_id_唯一且非空(main_rows):
    ids = [r["id"] for r in main_rows]
    assert all(isinstance(i, str) and i.strip() for i in ids), "存在空 id"
    dupes = [i for i, c in Counter(ids).items() if c > 1]
    assert not dupes, f"id 重复：{dupes}"


def test_必填字段齐全(main_rows):
    required = ("id", "query", "source", "golden_slugs", "unanswerable")
    missing = [
        (r.get("id"), field)
        for r in main_rows
        for field in required
        if field not in r
    ]
    assert not missing, f"缺字段：{missing[:5]}"


def test_Hit5_分母与不可答样本(main_rows):
    with_golden = [r for r in main_rows if r.get("golden_slugs")]
    unanswerable = [r for r in main_rows if r.get("unanswerable")]

    assert len(with_golden) == EXPECTED_HIT_DENOMINATOR, (
        f"Hit@5 分母应为 {EXPECTED_HIT_DENOMINATOR}（有 golden 的非不可答样本），实际 {len(with_golden)}"
    )
    assert len(unanswerable) == EXPECTED_UNANSWERABLE, (
        f"不可答样本应为 {EXPECTED_UNANSWERABLE}，实际 {len(unanswerable)}"
    )
    # 不可答 = 无 golden；两者必须严格互补，否则分母口径会被悄悄改掉
    assert all(not r.get("golden_slugs") for r in unanswerable), "不可答样本不应带 golden"
    assert len(with_golden) + len(unanswerable) == EXPECTED_TOTAL


def test_golden_slug_形态(main_rows):
    """slug 是「文档页级」标识：非空、无空白、不带 .md 后缀。"""
    bad = []
    for row in main_rows:
        for slug in row.get("golden_slugs") or []:
            if not isinstance(slug, str) or not slug.strip() or any(ch.isspace() for ch in slug) or slug.endswith(".md"):
                bad.append((row["id"], slug))
    assert not bad, f"slug 形态可疑：{bad[:5]}"


def test_primary_分布与冻结值一致(main_rows):
    """375 条 failure_mode.primary 的分布是直接对外报的数字，冻结在 regression_baselines.json。"""
    expected = json.loads(BASELINE_FILE.read_text(encoding="utf-8"))["extra_checks"]["primary_distribution"]["expected"]

    primaries = []
    for row in main_rows:
        fm = row.get("failure_mode") or {}
        primary = fm.get("primary")
        assert primary in ALLOWED_PRIMARY, f"{row['id']} 的 primary 非法：{primary!r}"
        assert primary, f"{row['id']} 缺 failure_mode.primary（空 ≠ none；未标注要写 unclassified）"
        primaries.append(primary)

    actual = dict(Counter(primaries))
    assert actual == expected, f"primary 分布变了：基线 {expected}，实际 {actual}"
    assert sum(actual.values()) == EXPECTED_TOTAL


# ============================================================
# 独立探针集 31（不与主集混报）
# ============================================================


def test_版本探针集条数与字段(version_rows):
    assert len(version_rows) == 31, f"版本混淆独立探针集应为 31 条，实际 {len(version_rows)}"
    for row in version_rows:
        assert row.get("golden_slug"), f"{row.get('id')} 缺 golden_slug"
        assert row.get("old_version") and row.get("new_version"), f"{row.get('id')} 缺版本字段"


# ============================================================
# Judge 金标 100 与冻结清单自洽
# ============================================================


def test_judge金标100条且都带参考答():
    rows = _read_jsonl(JUDGE_GOLD)
    assert len(rows) == 100, f"Judge 金标应为 100 条，实际 {len(rows)}"
    missing = [r.get("id") for r in rows if not r.get("reference_answer")]
    assert not missing, f"缺 reference_answer（Context Recall 需要它）：{missing[:5]}"


def test_冻结清单自洽():
    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
    assert frozen.get("version") == "0.1", f"冻结版本应为 0.1，实际 {frozen.get('version')}"
    assert frozen.get("total_main") == EXPECTED_TOTAL
    assert frozen.get("total_version_probe") == 31
    corrections = frozen.get("corrections_v0_1") or []
    assert len(corrections) == 3, f"v0.1 三条标注修正应记录在案，实际 {len(corrections)}"
    assert "freeze_scope" in frozen and "supersedes" in frozen, "冻结清单缺 freeze_scope / supersedes"
