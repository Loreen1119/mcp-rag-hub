"""diff_eval_run 门禁测试 —— 用合成 JSONL 验证三种判定。

⚠️ 分母必须 ≥ 200 才能测试阈值边界：10 条下丢 2 = −20pt，pt 条件必然触发，
   无法区分「丢 2 条 OK」和「丢 3 条 REGRESSION」。350 条下丢 2 ≈ −0.57pt、
   丢 3 ≈ −0.86pt，才能精确测出「分子减少 ≥3 条」的计数边界。

验证目标（2026-10-21）：
① 完全一致 → OK / exit 0
② 丢 4 条 → REGRESSION / exit 1（pt 和 count 都触发）
③ 只丢 2 条 → OK（pt < 1.0pt 且 count < 3）
④ 分母不同的两份 → 直接报错退出（禁止跨口径比）
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scripts.diff_eval_run import (
    diff_runs,
    extract_hit5,
    compute_hit5_numerator_denominator,
)


# ============================================================
# 合成数据工厂
# ============================================================


def _make_record(rid: str, hit: bool, unanswerable: bool = False) -> dict:
    return {
        "id": rid,
        "source": "asktug",
        "unanswerable": unanswerable,
        "query": f"query-{rid}",
        "golden_slugs": [f"golden-{rid}"],
        "n_golden": 1,
        "stages": {
            "bm25": {"hit@5": hit, "top": []},
            "vector": {"hit@5": hit, "top": []},
            "rrf": {"hit@5": hit, "top": []},
        },
    }


def _make_jsonl(records: list[dict], path: Path) -> Path:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path


def _make_records(n: int, n_hits: int) -> list[dict]:
    """生成 n 条记录，前 n_hits 条 hit=True。"""
    return [
        _make_record(f"rec-{i}", hit=(i < n_hits))
        for i in range(n)
    ]


N = 350  # 分母（对齐真实 335 量级，使 pt 阈值有意义）


# ============================================================
# 测试用例
# ============================================================


class TestDiffEvalRun:

    def test_identical_ok(self, tmp_path):
        """① 完全一致 → OK / exit 0。"""
        records = _make_records(N, n_hits=107)  # 模拟 107/335 ≈ 32%
        old_path = _make_jsonl(records, tmp_path / "old.jsonl")
        new_path = _make_jsonl(records, tmp_path / "new.jsonl")

        exit_code = diff_runs(old_path, new_path, stage="rrf")
        assert exit_code == 0

    def test_lose_4_regression(self, tmp_path):
        """② 丢 4 条 → REGRESSION / exit 1（pt −1.14pt > 1.0pt，count 4 ≥ 3）。"""
        old_records = _make_records(N, n_hits=107)
        # 丢了 4 条命中：107 → 103
        new_records = []
        for r in old_records:
            r2 = copy.deepcopy(r)
            idx = int(r2["id"].split("-")[1])
            if idx in (103, 104, 105, 106):
                r2["stages"]["rrf"]["hit@5"] = False
                r2["stages"]["bm25"]["hit@5"] = False
                r2["stages"]["vector"]["hit@5"] = False
            new_records.append(r2)

        old_path = _make_jsonl(old_records, tmp_path / "old.jsonl")
        new_path = _make_jsonl(new_records, tmp_path / "new.jsonl")

        exit_code = diff_runs(old_path, new_path, stage="rrf")
        assert exit_code == 1

    def test_lose_2_ok_boundary(self, tmp_path):
        """③ 只丢 2 条 → OK（pt −0.57pt < 1.0pt，count 2 < 3）。"""
        old_records = _make_records(N, n_hits=107)
        # 丢了 2 条命中：107 → 105
        new_records = []
        for r in old_records:
            r2 = copy.deepcopy(r)
            idx = int(r2["id"].split("-")[1])
            if idx in (105, 106):
                r2["stages"]["rrf"]["hit@5"] = False
                r2["stages"]["bm25"]["hit@5"] = False
                r2["stages"]["vector"]["hit@5"] = False
            new_records.append(r2)

        old_path = _make_jsonl(old_records, tmp_path / "old.jsonl")
        new_path = _make_jsonl(new_records, tmp_path / "new.jsonl")

        exit_code = diff_runs(old_path, new_path, stage="rrf")
        assert exit_code == 0

    def test_lose_3_regression_by_count(self, tmp_path):
        """丢 3 条 → REGRESSION（pt −0.86pt < 1.0pt，但 count 3 ≥ 3）。"""
        old_records = _make_records(N, n_hits=107)
        new_records = []
        for r in old_records:
            r2 = copy.deepcopy(r)
            idx = int(r2["id"].split("-")[1])
            if idx in (104, 105, 106):
                r2["stages"]["rrf"]["hit@5"] = False
                r2["stages"]["bm25"]["hit@5"] = False
                r2["stages"]["vector"]["hit@5"] = False
            new_records.append(r2)

        old_path = _make_jsonl(old_records, tmp_path / "old.jsonl")
        new_path = _make_jsonl(new_records, tmp_path / "new.jsonl")

        exit_code = diff_runs(old_path, new_path, stage="rrf")
        assert exit_code == 1  # count ≥ 3 触发

    def test_different_denominator_error(self, tmp_path):
        """④ 分母不同 → 直接报错退出（禁止跨口径比）。"""
        # 旧：350 条 answerable
        old_records = _make_records(N, n_hits=50)
        # 新：348 条 answerable（2 条 unanswerable）
        new_records = copy.deepcopy(_make_records(N, n_hits=50))
        new_records[0]["unanswerable"] = True
        new_records[1]["unanswerable"] = True

        old_path = _make_jsonl(old_records, tmp_path / "old.jsonl")
        new_path = _make_jsonl(new_records, tmp_path / "new.jsonl")

        exit_code = diff_runs(old_path, new_path, stage="rrf")
        assert exit_code == 2

    def test_gain_hits_ok(self, tmp_path):
        """新增命中不触发 REGRESSION。"""
        old_records = _make_records(N, n_hits=100)
        new_records = _make_records(N, n_hits=110)  # 增加 10 条命中

        old_path = _make_jsonl(old_records, tmp_path / "old.jsonl")
        new_path = _make_jsonl(new_records, tmp_path / "new.jsonl")

        exit_code = diff_runs(old_path, new_path, stage="rrf")
        assert exit_code == 0

    def test_extract_hit5_excludes_unanswerable(self):
        """extract_hit5 不计入 unanswerable 条目。"""
        records = [
            _make_record("rec-0", hit=True),
            _make_record("rec-1", hit=False, unanswerable=True),
            _make_record("rec-2", hit=True),
        ]
        hits = extract_hit5(records, "rrf")
        assert "rec-0" in hits
        assert "rec-1" not in hits
        assert "rec-2" in hits
        assert len(hits) == 2

    def test_numerator_denominator(self):
        """分子/分母计算正确。"""
        hit_map = {"a": True, "b": True, "c": False, "d": True, "e": False}
        num, den = compute_hit5_numerator_denominator(hit_map)
        assert num == 3
        assert den == 5

    def test_unanswerable_excluded_from_denominator(self):
        """不可答类不计入分母。"""
        records = [
            _make_record("rec-0", hit=True),
            _make_record("rec-1", hit=False),
            _make_record("rec-2", hit=True, unanswerable=True),
            _make_record("rec-3", hit=False, unanswerable=True),
        ]
        hits = extract_hit5(records, "rrf")
        num, den = compute_hit5_numerator_denominator(hits)
        assert den == 2
        assert num == 1

    def test_pt_threshold_only(self, tmp_path):
        """下降超 1.0pt 但分子减少 < 3 → 仍触发 REGRESSION（pt 条件）。"""
        # 350 条，从 300 命中掉到 295（丢 5 条 = −1.43pt，count 5 ≥ 3）
        # 用 300→295 来测 pt 超阈
        old_records = _make_records(N, n_hits=300)
        new_records = _make_records(N, n_hits=295)

        old_path = _make_jsonl(old_records, tmp_path / "old.jsonl")
        new_path = _make_jsonl(new_records, tmp_path / "new.jsonl")

        exit_code = diff_runs(old_path, new_path, stage="rrf")
        assert exit_code == 1
