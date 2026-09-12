"""AnswerValidator 的引用编号归一化测试。

背景：qwen2.5:3b 实测会把 used_source_ids 输出成字符串 `["1","2"]`
而不是 `[1,2]`。旧实现用 `isinstance(item, int)` 严格判断，导致每一个
答案都被判 invalid_citation —— 应用层必须容错，而不是指望模型永远输出
正确类型。这里锁住归一化行为。
"""

from __future__ import annotations

import pytest

from src.answer_validator import AnswerStatus, AnswerValidator, RefusalReason


@pytest.fixture
def validator() -> AnswerValidator:
    return AnswerValidator(evidence_count=3, min_evidence_count=1)


def _payload(ids, status: str = AnswerStatus.ANSWERED.value) -> dict:
    return {"status": status, "answer": "RRF 是倒数排名融合。", "used_source_ids": ids, "reason": ""}


@pytest.mark.parametrize(
    "ids,expected",
    [
        (["1", "2"], [1, 2]),      # 字符串编号：3b 实测会输出这种
        ([1, 2], [1, 2]),          # 正常整数
        ([1.0, 2.0], [1, 2]),      # 浮点
        ([2, 1], [2, 1]),          # 保持原顺序，不排序
        ([1, 1, 2], [1, 2]),       # 去重
    ],
)
def test_citation_ids_are_normalized(validator, ids, expected):
    result = validator.validate(_payload(ids))
    assert result["status"] == AnswerStatus.ANSWERED.value
    assert result["used_source_ids"] == expected


@pytest.mark.parametrize(
    "ids",
    [
        [1, 9],     # 越界：只有 3 条证据
        [0],        # 编号从 1 开始
        [],         # 空引用
        [True],     # bool 是 int 子类，不能当编号
        ["a"],      # 非数字字符串
        "1",        # 整体不是列表
        [None],
    ],
)
def test_invalid_citations_are_rejected(validator, ids):
    result = validator.validate(_payload(ids))
    assert result["status"] == AnswerStatus.INVALID_OUTPUT.value
    assert result["reason"] == RefusalReason.INVALID_CITATION.value


def test_malformed_json_is_rejected(validator):
    result = validator.validate("这不是 JSON")
    assert result["status"] == AnswerStatus.INVALID_OUTPUT.value
    assert result["reason"] == RefusalReason.INVALID_JSON.value


def test_unknown_status_is_rejected(validator):
    result = validator.validate(_payload([1], status="maybe"))
    assert result["status"] == AnswerStatus.INVALID_OUTPUT.value
    assert result["reason"] == RefusalReason.INVALID_OUTPUT.value


def test_insufficient_evidence_keeps_status_without_citation_check(validator):
    """不可答时不做引用校验，保留 LLM 给出的 status 与 reason。"""
    result = validator.validate(
        {
            "status": AnswerStatus.INSUFFICIENT_EVIDENCE.value,
            "answer": "",
            "used_source_ids": [],
            "reason": RefusalReason.INSUFFICIENT_CONTEXT.value,
        }
    )
    assert result["status"] == AnswerStatus.INSUFFICIENT_EVIDENCE.value
    assert result["reason"] == RefusalReason.INSUFFICIENT_CONTEXT.value
