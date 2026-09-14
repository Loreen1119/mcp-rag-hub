"""acceptance_eval 的纯逻辑测试。

只测不需要 Ollama、不需要加载模型的函数 —— 因为指标算错的话，跑完一圈评测
得到的数字全是错的，而且很难肉眼发现（尤其"引用编号 1-based 映射到来源文件"这一步）。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.evaluation.acceptance_eval import (
    MULTI_FILE,
    UNANS_FILE,
    _cited_sources,
    _join_context,
    _keywords_hit,
    _load_cases,
    _pct,
)


# ============================================================
# 验收集结构（防止改题时字段漏掉，脚本静默跑出空指标）
# ============================================================


def test_refusal_set_shape():
    cases = _load_cases(UNANS_FILE)
    assert len(cases) == 10, f"拒答集应为 10 条，实际 {len(cases)}"
    for c in cases:
        assert c.get("query_id"), "缺 query_id"
        assert c.get("query"), f"{c.get('query_id')} 缺 query"
        assert c.get("category"), f"{c.get('query_id')} 缺 category"
        assert c.get("expected_status") == "insufficient_evidence", (
            f"{c.get('query_id')} 的 expected_status 应为 insufficient_evidence"
        )


def test_multi_evidence_set_shape():
    cases = _load_cases(MULTI_FILE)
    assert len(cases) == 3, f"多证据集应为 3 条，实际 {len(cases)}"
    for c in cases:
        qid = c.get("query_id")
        assert c.get("expected_status") == "answered", f"{qid} 的 expected_status 应为 answered"
        reqs = c.get("required_sources") or []
        assert len(reqs) == 2, f"{qid} 应有 2 个 required_sources（跨文档题的必要条件）"
        for r in reqs:
            assert r.get("keywords"), f"{qid} 的某个 required_source 缺 keywords"
        golden = c.get("golden_chunk_sources") or []
        assert len(golden) == 2, f"{qid} 应有 2 个 golden_chunk_sources"
        assert len(set(golden)) == 2, f"{qid} 的 golden 来源重复"


# ============================================================
# 引用编号映射（最容易错的一步：used_source_ids 是 1-based）
# ============================================================


def test_cited_sources_is_one_based():
    chunks = [{"source_doc": "a.md"}, {"source_doc": "b.md"}, {"source_doc": "c.md"}]
    assert _cited_sources(chunks, [1, 3]) == ["a.md", "c.md"]
    # 0 是非法编号（会被 validator 拦），这里也不该返回任何东西
    assert _cited_sources(chunks, [0]) == []


def test_cited_sources_ignores_out_of_range_and_dupes():
    chunks = [{"source_doc": "a.md"}, {"source_doc": "b.md"}]
    assert _cited_sources(chunks, [2, 2, 99]) == ["b.md"]
    assert _cited_sources(chunks, []) == []
    assert _cited_sources(chunks, None) == []


def test_cited_sources_tolerates_string_ids():
    """小模型偶尔把 used_source_ids 写成字符串数组，validator 会尽量归一，这里也别崩。"""
    chunks = [{"source_doc": "a.md"}, {"source_doc": "b.md"}]
    assert _cited_sources(chunks, ["1", "2"]) == ["a.md", "b.md"]
    assert _cited_sources(chunks, ["x"]) == []


# ============================================================
# 关键词命中
# ============================================================


def test_keywords_hit_case_insensitive():
    assert _keywords_hit("用 SQLModel 定义表", ["sqlmodel"]) == ["sqlmodel"]
    assert _keywords_hit("用 sqlmodel 定义表", ["SQLModel"]) == ["SQLModel"]


def test_keywords_hit_chinese_and_miss():
    text = "在应用启动前执行 lifespan，关闭时释放连接"
    hit = _keywords_hit(text, ["lifespan", "Session", "启动"])
    assert hit == ["lifespan", "启动"]
    assert _keywords_hit("", ["a"]) == []
    assert _keywords_hit("abc", []) == []


# ============================================================
# 小工具
# ============================================================


def test_join_context_labels_sources():
    ctx = _join_context([{"source_doc": "x.md", "content": "内容一"}, {"source_doc": "y.md", "content": "内容二"}])
    assert "[来源1: x.md]" in ctx and "[来源2: y.md]" in ctx and "内容二" in ctx


def test_pct_handles_none():
    assert _pct(None) == "—"
    assert _pct(0.0) == "0.0%"
    assert _pct(0.75) == "75.0%"
