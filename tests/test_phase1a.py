"""Contract tests for structured answer validation."""

from src.answer_validator import AnswerStatus, AnswerValidator, RefusalReason


def test_valid_answer_with_citations():
    result = AnswerValidator(3).validate({"status": "answered", "answer": "RRF 融合多路排序。", "used_source_ids": [1, 2]})
    assert result["status"] == AnswerStatus.ANSWERED.value


def test_citation_out_of_range_is_rejected():
    result = AnswerValidator(2).validate({"status": "answered", "answer": "answer", "used_source_ids": [3]})
    assert result["reason"] == RefusalReason.INVALID_CITATION.value


def test_empty_citations_are_rejected():
    result = AnswerValidator(2).validate({"status": "answered", "answer": "answer", "used_source_ids": []})
    assert result["reason"] == RefusalReason.INVALID_CITATION.value


def test_insufficient_evidence_is_preserved():
    result = AnswerValidator(2).validate({"status": "insufficient_evidence", "answer": "", "reason": "insufficient_context"})
    assert result["status"] == AnswerStatus.INSUFFICIENT_EVIDENCE.value


def test_generation_unavailable_is_preserved():
    result = AnswerValidator(2).validate({"status": "generation_unavailable", "answer": "", "reason": "timeout"})
    assert result["status"] == AnswerStatus.GENERATION_UNAVAILABLE.value


def test_invalid_json_is_rejected():
    result = AnswerValidator(1).validate("not json")
    assert result["reason"] == RefusalReason.INVALID_JSON.value
