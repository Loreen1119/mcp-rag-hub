"""`experiments.generate_report` 的结论正确性测试（C10 回归）。

为什么专门测这个：报告里的结论段曾经是**写死的文案**，不随数据变化 ——
于是实测 ΔMRR=0 时，报告照样印出"图检索为双路召回提供了增量价值"；
还印过"BM25 在 exact_match 上的优势"，而实测 BM25 与 Vector 在该类均为 0.8000(**持平**)。
这类错误的危害是：**它看起来像数据，其实是断言**，拿去讲会被追问穿。

所以这里的测试分两类：
1. `_delta_phrase` 的纯逻辑（三档、阈值边界）
2. **回归断言**：给定数据 → 报告里"必须出现什么"和"绝不允许出现什么"

全部不依赖模型与网络；`EXPERIMENTS_DIR` 被重定向到 tmp_path，
避免测试写坏真实的 `experiments/report.md`。
"""

from __future__ import annotations

import src.evaluation.experiments as exp_mod
from src.evaluation.experiments import (
    _MIN_MEANINGFUL_GAIN,
    _delta_phrase,
    generate_report,
)


# ============================================================
# 构造最小可用的实验产物
# ============================================================


def _ablation(triple_mrr: float, full_mrr: float = 0.7889) -> dict:
    """只填 generate_report 会读到的字段。"""
    def m(mrr: float) -> dict:
        return {"mrr": mrr, "hit@5": 1.0, "precision@5": 0.5}

    return {
        "results": {
            "bm25_only": m(0.5469),
            "vector_only": m(0.7246),
            "graph_only": m(0.0958),
            "rrf_fusion": m(0.7833),
            "full_pipeline": m(full_mrr),
            "triple_fusion": m(triple_mrr),
            "concat_naive": m(0.5770),
        }
    }


def _category(exact_bm25: float, exact_vec: float,
              sem_bm25: float, sem_vec: float) -> dict:
    def row(b: float, v: float) -> dict:
        return {
            "bm25_only": {"mrr": b},
            "vector_only": {"mrr": v},
            "full_pipeline": {"mrr": max(b, v)},
        }

    return {"categories": {
        "exact_match": row(exact_bm25, exact_vec),
        "semantic": row(sem_bm25, sem_vec),
    }}


# ============================================================
# 1. 纯逻辑：三档 + 阈值边界
# ============================================================


def test_delta_phrase_positive():
    assert "A 更高" in _delta_phrase(0.05, "A", "B")


def test_delta_phrase_negative():
    """负差值时必须说 B 更高 —— 不能因为参数顺序就默认 A 好。"""
    assert "B 更高" in _delta_phrase(-0.05, "A", "B")


def test_delta_phrase_zero_is_tie():
    assert _delta_phrase(0.0, "A", "B") == "两者基本持平"


def test_delta_phrase_below_threshold_is_tie():
    """低于阈值的差异不算"更好"（浮点噪声，不构成结论）。"""
    tiny = _MIN_MEANINGFUL_GAIN / 2
    assert _delta_phrase(tiny, "A", "B") == "两者基本持平"
    assert _delta_phrase(-tiny, "A", "B") == "两者基本持平"


# ============================================================
# 2. 回归：GraphRAG 增量（C10 的主案发现场）
# ============================================================


def test_zero_gain_must_not_claim_gain(tmp_path, monkeypatch):
    """ΔMRR = 0 时，报告**不得**出现"增量价值/带来了增量"。

    这正是修复前的 bug：`if triple_mrr >= full_mrr` 在相等时成立，
    于是零增量也印"提供了增量价值"。
    """
    monkeypatch.setattr(exp_mod, "EXPERIMENTS_DIR", tmp_path)
    rep = generate_report(ablation=_ablation(triple_mrr=0.7889, full_mrr=0.7889))

    assert "未观察到增量" in rep
    assert "带来了增量" not in rep
    assert "增量价值" not in rep


def test_positive_gain_is_claimed(tmp_path, monkeypatch):
    monkeypatch.setattr(exp_mod, "EXPERIMENTS_DIR", tmp_path)
    rep = generate_report(ablation=_ablation(triple_mrr=0.85, full_mrr=0.7889))

    assert "带来了增量" in rep
    assert "未观察到增量" not in rep


def test_negative_gain_is_reported_as_regression(tmp_path, monkeypatch):
    """三路比双路差时，必须明说"拉低了效果" —— 不能沉默，也不能说成收益。"""
    monkeypatch.setattr(exp_mod, "EXPERIMENTS_DIR", tmp_path)
    rep = generate_report(ablation=_ablation(triple_mrr=0.70, full_mrr=0.7889))

    assert "拉低了效果" in rep
    assert "带来了增量" not in rep


# ============================================================
# 3. 回归：两路互补的结论（按实测数据下，不预设）
# ============================================================


def test_complementarity_verified_only_when_both_hold(tmp_path, monkeypatch):
    monkeypatch.setattr(exp_mod, "EXPERIMENTS_DIR", tmp_path)
    # BM25 在 exact 更高、Vector 在 semantic 更高 —— 这才叫"互补"
    rep = generate_report(category=_category(
        exact_bm25=0.90, exact_vec=0.70, sem_bm25=0.30, sem_vec=0.60))

    assert "两路互补得到验证" in rep


def test_no_presupposed_advantage_when_tied(tmp_path, monkeypatch):
    """exact_match 类两路持平时**不得**说"BM25 的优势"。

    修复前这里印的是 `**BM25 在 exact_match 上的优势**` —— 完全没检查就断言；
    实测该类别 BM25 = Vector = 0.8000，是持平，不是优势。
    """
    monkeypatch.setattr(exp_mod, "EXPERIMENTS_DIR", tmp_path)
    rep = generate_report(category=_category(
        exact_bm25=0.80, exact_vec=0.80, sem_bm25=0.2375, sem_vec=0.50))

    assert "在 exact_match 上的优势" not in rep
    assert "两者基本持平" in rep
    assert "只验证了一半" in rep


def test_no_complementarity_claim_when_data_contradicts(tmp_path, monkeypatch):
    """两路都不符预期时，必须如实说"未从数据得到支持"，不能硬说互补。"""
    monkeypatch.setattr(exp_mod, "EXPERIMENTS_DIR", tmp_path)
    rep = generate_report(category=_category(
        exact_bm25=0.50, exact_vec=0.70, sem_bm25=0.60, sem_vec=0.30))

    assert "未从数据得到支持" in rep
    assert "两路互补得到验证" not in rep


# ============================================================
# 4. 兼容性：空/缺字段不该炸
# ============================================================


def test_report_with_no_data_does_not_crash(tmp_path, monkeypatch):
    monkeypatch.setattr(exp_mod, "EXPERIMENTS_DIR", tmp_path)
    rep = generate_report()
    assert "分析报告" in rep


def test_report_file_is_written(tmp_path, monkeypatch):
    monkeypatch.setattr(exp_mod, "EXPERIMENTS_DIR", tmp_path)
    generate_report(ablation=_ablation(triple_mrr=0.7889))
    assert (tmp_path / "report.md").exists()
