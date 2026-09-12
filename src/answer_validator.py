"""Validation for structured RAG answers and their source citations."""

from __future__ import annotations

import json
from enum import Enum
from typing import Any


class AnswerStatus(str, Enum):
    ANSWERED = "answered"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    GENERATION_UNAVAILABLE = "generation_unavailable"
    INVALID_OUTPUT = "invalid_output"


class RefusalReason(str, Enum):
    LOW_RETRIEVAL_SCORE = "low_retrieval_score"
    INSUFFICIENT_CONTEXT = "insufficient_context"
    TIMEOUT = "timeout"
    CONNECTION_ERROR = "connection_error"
    SERVICE_ERROR = "service_error"     # 服务端明确报错：模型不存在、参数非法等
    INVALID_JSON = "invalid_json"
    INVALID_OUTPUT = "invalid_output"   # 未知 status / 非 dict 结构等无法解析的输出
    INVALID_CITATION = "invalid_citation"


class AnswerValidator:
    """Validate the answer contract and citations against retrieved evidence."""

    def __init__(self, evidence_count: int, min_evidence_count: int = 1):
        self.evidence_count = evidence_count
        self.min_evidence_count = min_evidence_count

    def validate(self, output: str | dict[str, Any]) -> dict[str, Any]:
        if isinstance(output, str):
            try:
                output = json.loads(output)
            except (TypeError, json.JSONDecodeError):
                return self._invalid(RefusalReason.INVALID_JSON.value)
        if not isinstance(output, dict):
            return self._invalid(RefusalReason.INVALID_OUTPUT.value)

        status = output.get("status")
        allowed = {item.value for item in AnswerStatus}
        if status not in allowed:
            return self._invalid(RefusalReason.INVALID_OUTPUT.value)

        result = {
            "status": status,
            "answer": str(output.get("answer", "")),
            "used_source_ids": output.get("used_source_ids", []),
            "reason": output.get("reason"),
        }
        if status == AnswerStatus.ANSWERED.value:
            citations = self._normalize_citations(
                result["used_source_ids"], self.evidence_count
            )
            if citations is None or len(citations) < self.min_evidence_count:
                return self._invalid(RefusalReason.INVALID_CITATION.value)
            result["used_source_ids"] = citations
            if not result["answer"].strip():
                return self._invalid(RefusalReason.INVALID_OUTPUT.value)
        return result

    @staticmethod
    def _normalize_citations(raw: Any, evidence_count: int) -> list[int] | None:
        """把 LLM 返回的各种"编号"统一成去重的 int 列表，无法归一则返回 None。

        小模型（如 qwen2.5:3b）实测会把 used_source_ids 输出成字符串
        `["1","2"]` 而不是 `[1,2]`。此前直接 `isinstance(item, int)` 判断，
        会导致**每一个答案**都被判 invalid_citation —— 应用层必须容错，
        而不是指望模型永远输出正确类型。
        """
        if not isinstance(raw, list):
            return None

        normalized: list[int] = []
        for item in raw:
            # bool 是 int 的子类，先排除，避免 True 被当成证据编号 1
            if isinstance(item, bool):
                return None
            if isinstance(item, int):
                number = item
            elif isinstance(item, float) and item.is_integer():
                number = int(item)
            elif isinstance(item, str) and item.strip().isdigit():
                number = int(item.strip())
            else:
                return None
            if number < 1 or number > evidence_count:
                return None
            normalized.append(number)

        # 去重并保持原始顺序
        return list(dict.fromkeys(normalized))

    @staticmethod
    def _invalid(reason: str) -> dict[str, Any]:
        return {"status": AnswerStatus.INVALID_OUTPUT.value, "answer": "", "used_source_ids": [], "reason": reason}


def parse_and_validate(output: str | dict[str, Any], evidence_count: int, min_evidence_count: int = 1) -> dict[str, Any]:
    return AnswerValidator(evidence_count, min_evidence_count).validate(output)
