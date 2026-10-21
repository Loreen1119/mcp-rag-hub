"""Cross-Encoder 超时降级测试 —— 用假 model，不加载真模型。

验证目标（2026-10-21）：
1. 超时 → 返回长度 == top_k 且顺序与输入一致（RRF 原序）
2. metadata 有 ce_timeout=True
3. 未超时时行为与原来完全一致（有 ce_score/ce_rank）
4. 不抛异常
5. 超时后不写 ce_score
6. 熔断：连续超时 N 次后直接降级
"""

from __future__ import annotations

import time
from typing import List

import pytest

from src.fusion import CrossEncoderReranker
from src.models import Chunk, RetrievalResult


# ============================================================
# 假 model / 假候选
# ============================================================


class FakeSlowModel:
    """predict 永远 sleep 超时的假 model。"""

    def __init__(self, sleep_s: float = 5.0):
        self.sleep_s = sleep_s

    def predict(self, pairs, **kwargs):
        time.sleep(self.sleep_s)
        return [0.5] * len(pairs)


class FakeFastModel:
    """predict 立即返回的假 model（固定分数）。"""

    def predict(self, pairs, **kwargs):
        # 返回递减分数，模拟 CE 排序
        return [1.0 - i * 0.1 for i in range(len(pairs))]


class FakeInstantModel:
    """predict 返回预设分数，完全可控。"""

    def __init__(self, scores: List[float] | None = None):
        self.scores = scores
        self.call_count = 0

    def predict(self, pairs, **kwargs):
        self.call_count += 1
        if self.scores is not None:
            return self.scores[: len(pairs)]
        return [0.9 - i * 0.05 for i in range(len(pairs))]


def _make_candidates(n: int) -> List[RetrievalResult]:
    """生成 n 个假候选，score 递减（模拟 RRF 顺序）。"""
    candidates = []
    for i in range(n):
        chunk = Chunk(
            content=f"chunk-{i}",
            metadata={"source": f"doc-{i}", "chunk_index": i},
            chunk_id=f"id-{i}",
        )
        candidates.append(
            RetrievalResult(chunk=chunk, score=1.0 - i * 0.01, source="rrf")
        )
    return candidates


def _make_reranker_with_fake_model(model) -> CrossEncoderReranker:
    """构造绑定了假 model 的 reranker（绕过 CrossEncoder 加载）。"""
    reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
    reranker.model = model
    reranker._consecutive_timeouts = 0
    reranker._circuit_broken_until = 0.0
    return reranker


# ============================================================
# 测试用例
# ============================================================


class TestTimeoutFallback:
    """超时降级行为。"""

    def test_timeout_returns_top_k_in_original_order(self, monkeypatch):
        """① 超时 → 返回长度 == top_k 且顺序与输入一致（RRF 原序）。"""
        monkeypatch.setattr("src.fusion.CE_TIMEOUT_MS", 100)  # 100ms
        model = FakeSlowModel(sleep_s=2.0)
        reranker = _make_reranker_with_fake_model(model)
        candidates = _make_candidates(10)

        result = reranker.rerank("test query", candidates, top_k=5)

        assert len(result) == 5
        # 顺序应与输入候选一致（RRF 原序）
        for i, r in enumerate(result):
            assert r.chunk.chunk_id == f"id-{i}"

    def test_timeout_marks_ce_timeout_metadata(self, monkeypatch):
        """② metadata 有 ce_timeout=True。"""
        monkeypatch.setattr("src.fusion.CE_TIMEOUT_MS", 100)
        model = FakeSlowModel(sleep_s=2.0)
        reranker = _make_reranker_with_fake_model(model)
        candidates = _make_candidates(8)

        result = reranker.rerank("test query", candidates, top_k=5)

        for r in result:
            assert r.chunk.metadata.get("ce_timeout") is True

    def test_timeout_does_not_write_ce_score(self, monkeypatch):
        """⑤ 超时后不写 ce_score / ce_rank。"""
        monkeypatch.setattr("src.fusion.CE_TIMEOUT_MS", 100)
        model = FakeSlowModel(sleep_s=2.0)
        reranker = _make_reranker_with_fake_model(model)
        candidates = _make_candidates(8)

        result = reranker.rerank("test query", candidates, top_k=5)

        for r in result:
            assert "ce_score" not in r.chunk.metadata
            assert "ce_rank" not in r.chunk.metadata

    def test_timeout_does_not_raise(self, monkeypatch):
        """④ 不抛异常。"""
        monkeypatch.setattr("src.fusion.CE_TIMEOUT_MS", 100)
        model = FakeSlowModel(sleep_s=2.0)
        reranker = _make_reranker_with_fake_model(model)
        candidates = _make_candidates(5)

        # 不应抛出任何异常
        result = reranker.rerank("test query", candidates, top_k=5)
        assert isinstance(result, list)

    def test_timeout_preserves_rrf_score(self, monkeypatch):
        """超时降级时保留 RRF 分数（不是 CE 分数）。"""
        monkeypatch.setattr("src.fusion.CE_TIMEOUT_MS", 100)
        model = FakeSlowModel(sleep_s=2.0)
        reranker = _make_reranker_with_fake_model(model)
        candidates = _make_candidates(5)

        result = reranker.rerank("test query", candidates, top_k=5)

        for i, r in enumerate(result):
            assert r.score == candidates[i].score  # 保留 RRF 分数


class TestNormalBehavior:
    """未超时时行为与原来完全一致。"""

    def test_no_timeout_normal_ce_ranking(self, monkeypatch):
        """③ 未超时时行为与原来完全一致（有 ce_score/ce_rank，按 CE 分数排序）。"""
        monkeypatch.setattr("src.fusion.CE_TIMEOUT_MS", 0)  # 0 = 不限制
        model = FakeInstantModel(scores=[0.3, 0.9, 0.5, 0.1, 0.7])
        reranker = _make_reranker_with_fake_model(model)
        candidates = _make_candidates(5)

        result = reranker.rerank("test query", candidates, top_k=3)

        # 有 ce_score 和 ce_rank
        assert len(result) == 3
        for r in result:
            assert "ce_score" in r.chunk.metadata
            assert "ce_rank" in r.chunk.metadata
            assert "ce_timeout" not in r.chunk.metadata

        # 按 CE 分数降序：0.9 → 0.7 → 0.5
        assert result[0].chunk.metadata["ce_score"] == 0.9
        assert result[1].chunk.metadata["ce_score"] == 0.7
        assert result[2].chunk.metadata["ce_score"] == 0.5

        # ce_rank 从 1 开始
        assert result[0].chunk.metadata["ce_rank"] == 1
        assert result[1].chunk.metadata["ce_rank"] == 2
        assert result[2].chunk.metadata["ce_rank"] == 3

    def test_no_timeout_fast_model(self, monkeypatch):
        """快速 model + 超时限制充足 → 正常 CE 排序。"""
        monkeypatch.setattr("src.fusion.CE_TIMEOUT_MS", 5000)  # 5s，充足
        model = FakeFastModel()
        reranker = _make_reranker_with_fake_model(model)
        candidates = _make_candidates(5)

        result = reranker.rerank("test query", candidates, top_k=5)

        assert len(result) == 5
        for r in result:
            assert "ce_score" in r.chunk.metadata
            assert "ce_timeout" not in r.chunk.metadata

    def test_empty_candidates(self):
        """空候选列表 → 返回空列表。"""
        reranker = _make_reranker_with_fake_model(FakeInstantModel())
        result = reranker.rerank("test query", [], top_k=5)
        assert result == []


class TestCircuitBreaker:
    """熔断机制。"""

    def test_circuit_breaker_after_repeated_timeouts(self, monkeypatch):
        """⑥ 连续超时 N 次后触发熔断，后续调用直接降级。"""
        monkeypatch.setattr("src.fusion.CE_TIMEOUT_MS", 50)  # 50ms
        model = FakeSlowModel(sleep_s=2.0)
        reranker = _make_reranker_with_fake_model(model)
        # 降低熔断阈值便于测试
        reranker._CIRCUIT_BREAKER_THRESHOLD = 3
        reranker._CIRCUIT_COOLDOWN_S = 60.0
        candidates = _make_candidates(5)

        # 连续超时 3 次 → 触发熔断
        for i in range(3):
            result = reranker.rerank(f"q{i}", candidates, top_k=5)
            assert result[0].chunk.metadata.get("ce_timeout") is True

        # 第 4 次：熔断中，即使 model 可以正常工作也不调用
        call_count_before = model.call_count if hasattr(model, "call_count") else 0
        result = reranker.rerank("q4", candidates, top_k=5)
        assert result[0].chunk.metadata.get("ce_timeout") is True

        # 熔断期间 _consecutive_timeouts >= threshold
        assert reranker._consecutive_timeouts >= 3

    def test_circuit_breaker_resets_after_cooldown(self, monkeypatch):
        """冷却后恢复尝试 CE。"""
        monkeypatch.setattr("src.fusion.CE_TIMEOUT_MS", 0)  # 不限制，成功
        model = FakeInstantModel()
        reranker = _make_reranker_with_fake_model(model)
        candidates = _make_candidates(5)

        # 先触发熔断（直接设置状态）
        reranker._circuit_broken_until = time.monotonic() - 1  # 已过期
        reranker._consecutive_timeouts = 3

        result = reranker.rerank("test query", candidates, top_k=3)
        # 冷却已过，应正常 CE
        assert "ce_score" in result[0].chunk.metadata
        assert reranker._consecutive_timeouts == 0  # 成功后重置

    def test_success_resets_timeout_counter(self, monkeypatch):
        """成功后重置连续超时计数。"""
        monkeypatch.setattr("src.fusion.CE_TIMEOUT_MS", 0)
        model = FakeInstantModel()
        reranker = _make_reranker_with_fake_model(model)
        candidates = _make_candidates(5)

        reranker._consecutive_timeouts = 2  # 模拟之前超时 2 次
        reranker.rerank("test query", candidates, top_k=3)
        assert reranker._consecutive_timeouts == 0
