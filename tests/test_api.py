"""D2 验收：HTTP 服务层的接口契约。

三条设计约束（都是为了避免「假绿」）：

1. **不加载模型、不起 Ollama。** `agent` / `src.pipeline` 在 `api.py` 里是惰性 import，
   测试用 `app.dependency_overrides` 注入假实现 —— 否则这个文件会从 0.3 秒变成
   几分钟（建索引 + 加载 1.1GB reranker），CI 也跟着变脆。
2. **`/health` 必须能证明它没碰管线。** 只断言「返回 200」是假绿：一个调用
   `get_pipeline()` 的实现慢两分钟也照样 200。所以有一个用例把
   `src.pipeline.get_pipeline` 换成抛异常的函数，健康检查仍须通过。
3. **`/query` 的三态映射要真断言。** 不启 Ollama 时它恒返 `generation_unavailable`，
   这在真实服务上是合法状态 —— 用 curl 验会「通过」但什么都没验到。这里用假
   runner 把三种 status 一次性覆盖。
"""

from __future__ import annotations

import types

import pytest
from fastapi.testclient import TestClient

import api as api_module

client = TestClient(api_module.app)


# ============================================================
# 假实现
# ============================================================


class _FakeResult:
    """模拟 src.fusion 的检索结果对象（只用到这 3 个属性）。"""

    def __init__(self, content: str, source: str, score: float, chunk_id: str = "cid"):
        self.chunk = types.SimpleNamespace(
            content=content,
            metadata={"source": source, "heading_breadcrumb": "一级 / 二级"},
            chunk_id=chunk_id,
        )
        self.score = score


class _FakeRetriever:
    def __init__(self, results):
        self._results = results
        self.calls = 0

    def search(self, query: str):
        self.calls += 1
        return list(self._results)


class _FakePipeline:
    """融合层：把传入的两路结果按顺序当成 rrf / ce 两档返回，便于区分。"""

    def __init__(self):
        self.kwargs: list[dict] = []

    def run(self, bm25_results, vector_results, query, **kwargs):
        self.kwargs.append(kwargs)
        return {
            "rrf": list(bm25_results),
            "cross_encoder": list(vector_results),
        }


def _fake_ctx(with_graph: bool = False):
    rrf_pool = [
        _FakeResult("a" * 400, "docs-cn/a.md", 0.91, "c1"),
        _FakeResult("短正文", "asktug/b.md", 0.42, "c2"),
    ]
    ce_pool = [_FakeResult("精排结果", "docs-cn/c.md", 0.77, "c3")]
    return types.SimpleNamespace(
        chunks=[],
        bm25=_FakeRetriever(rrf_pool),
        vector=_FakeRetriever(ce_pool),
        graph=_FakeRetriever([_FakeResult("图路", "tidb-net/d.md", 0.11, "c4")]) if with_graph else None,
        pipeline=_FakePipeline(),
    )


@pytest.fixture(autouse=True)
def _clean_caches():
    """清掉模块级缓存，并保证单例不串味（api 与 agent 共用同一个 get_pipeline）。

    teardown 只在 `src.pipeline` **已经被导入**时才 reset —— 用 `sys.modules` 判断而不是
    直接 import，否则每个用例都会拉起 torch（本文件 18 个用例就是 18 次）。
    """
    api_module._chunk_count_cache.update(at=0.0, value=None, status="missing")
    api_module._source_chunk_cache.update(at=0.0, value=None)
    api_module._scan_cache.update(at=0.0, value=None)
    yield
    api_module.app.dependency_overrides.clear()
    import sys

    pipeline_mod = sys.modules.get("src.pipeline")
    if pipeline_mod is not None:
        pipeline_mod.reset_pipeline()


@pytest.fixture
def fake_ctx():
    ctx = _fake_ctx()
    api_module.app.dependency_overrides[api_module.get_ctx] = lambda: ctx
    return ctx


# ============================================================
# /health
# ============================================================


def test_health_reports_config_without_index(monkeypatch, tmp_path):
    """索引缺失时降级为 degraded，但配置字段仍完整、HTTP 仍 200。"""
    monkeypatch.setattr(api_module, "CHROMA_PERSIST_DIR", tmp_path / "no-such-dir")

    resp = client.get("/health")

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "degraded"
    assert body["index_status"] == "missing"
    assert body["chunk_count"] is None
    # 配置透传：这几个开关决定了对外行为，漏报会让人误判部署状态
    assert body["corpus"]
    assert isinstance(body["kg_enabled"], bool)
    assert isinstance(body["reranker_enabled"], bool)
    assert body["llm_backend"] in ("ollama", "deepseek")
    assert body["retrieval"]["ce_candidate_k"] == api_module.CE_CANDIDATE_K


def test_health_never_builds_pipeline(monkeypatch):
    """`/health` 调 get_pipeline() 会从 200 变成「两分钟没响应」——这里钉死它。"""
    import src.pipeline as pipeline_mod

    def _boom(*_args, **_kwargs):
        raise AssertionError("/health 不允许构建管线（冷启动约 2 分钟）")

    monkeypatch.setattr(pipeline_mod, "get_pipeline", _boom)

    resp = client.get("/health")

    assert resp.status_code == 200


# ============================================================
# /documents
# ============================================================


def test_documents_aggregates_and_paginates(monkeypatch):
    """9 千个文件不能一次吐出来：默认给聚合，清单分页。"""
    files = [f"docs-cn/{i:04d}.md" for i in range(230)]
    files += [f"asktug/{i:04d}.md" for i in range(65)]
    files += ["README.md"]  # 根目录文件要落到 (root)，不能并进某个来源
    monkeypatch.setattr(api_module, "_scan_corpus", lambda: sorted(files))
    monkeypatch.setattr(api_module, "_index_status", lambda: (104192, "ready"))

    body = client.get("/documents").json()

    assert body["total_docs"] == 296
    assert body["total_chunks"] == 104192
    by_source = {s["source"]: s for s in body["sources"]}
    assert by_source["docs-cn"]["doc_count"] == 230
    assert by_source["asktug"]["doc_count"] == 65
    assert by_source["(root)"]["doc_count"] == 1
    # 默认不算 chunk 数（要扫 10 万行 metadata）
    assert all(s["chunk_count"] is None for s in body["sources"])
    # 默认单页 50 条，且 has_more 正确
    assert len(body["documents"]) == 50
    assert body["has_more"] is True


def test_documents_limit_is_capped(monkeypatch):
    """单页上限 200 —— 放开就等于允许一次吐近 MB 的 JSON。"""
    monkeypatch.setattr(api_module, "_scan_corpus", lambda: ["docs-cn/x.md"])
    monkeypatch.setattr(api_module, "_index_status", lambda: (1, "ready"))

    assert client.get("/documents", params={"limit": 500}).status_code == 422
    assert client.get("/documents", params={"limit": 200}).status_code == 200


def test_documents_with_chunk_counts(monkeypatch):
    monkeypatch.setattr(
        api_module,
        "_scan_corpus",
        lambda: ["docs-cn/a.md", "docs-cn/b.md", "asktug/c.md"],
    )
    monkeypatch.setattr(api_module, "_index_status", lambda: (900, "ready"))
    monkeypatch.setattr(api_module, "_source_chunk_counts", lambda: {"docs-cn": 800, "asktug": 100})

    body = client.get("/documents", params={"with_chunk_counts": "true"}).json()

    by_source = {s["source"]: s for s in body["sources"]}
    assert by_source["docs-cn"]["chunk_count"] == 800
    assert by_source["asktug"]["chunk_count"] == 100


# ============================================================
# /search
# ============================================================


def test_search_default_mode_is_rrf(fake_ctx):
    """默认必须是 rrf —— ce 会加载 1.1GB reranker，不能做默认档。"""
    body = client.post("/search", json={"query": "如何优化"}).json()

    assert body["mode"] == "rrf"
    assert body["count"] == 2
    assert [h["rank"] for h in body["hits"]] == [1, 2]
    # rrf 档不应触发 reranker 的粗筛参数
    assert fake_ctx.pipeline.kwargs[0]["ce_top_k"] == api_module.CE_TOP_K


def test_search_modes_map_to_expected_stage(fake_ctx):
    bm25 = client.post("/search", json={"query": "q", "mode": "bm25"}).json()
    vector = client.post("/search", json={"query": "q", "mode": "vector"}).json()
    ce = client.post("/search", json={"query": "q", "mode": "ce"}).json()

    assert bm25["hits"][0]["source"] == "docs-cn/a.md"
    assert vector["hits"][0]["source"] == "docs-cn/c.md"
    assert ce["hits"][0]["source"] == "docs-cn/c.md"


def test_search_truncates_long_content(fake_ctx):
    body = client.post("/search", json={"query": "q", "mode": "bm25"}).json()

    long_hit, short_hit = body["hits"][0], body["hits"][1]
    assert long_hit["truncated"] is True and len(long_hit["content"]) == 300
    assert short_hit["truncated"] is False and short_hit["content"] == "短正文"


def test_search_respects_top_k(fake_ctx):
    body = client.post("/search", json={"query": "q", "mode": "bm25", "top_k": 1}).json()
    assert len(body["hits"]) == 1


def test_search_graph_is_optional(fake_ctx):
    """KG 路关闭时不得把 graph 传进融合层（否则 rrf_weights 长度不匹配）。"""
    assert fake_ctx.graph is None
    body = client.post("/search", json={"query": "q"}).json()
    assert body["kg_enabled"] is False
    assert fake_ctx.pipeline.kwargs[0]["graph_results"] is None
    assert fake_ctx.pipeline.kwargs[0]["rrf_weights"] is None


# ============================================================
# /query
# ============================================================


def _runner_returning(answer: dict, chunks=None):
    def _run(query: str) -> dict:
        return {
            "query": query,
            "answer": answer,
            "last_retrieved_chunks": chunks or [],
            "attempt": 1,
            "rewritten_queries": [],
        }

    return _run


@pytest.mark.parametrize(
    "status",
    ["answered", "insufficient_evidence", "generation_unavailable"],
)
def test_query_maps_all_three_states(status):
    """三态一次性覆盖 —— 靠 curl 验的话，不起 Ollama 只能看到 generation_unavailable。"""
    answer = {"status": status, "answer": "答案正文", "used_source_ids": [1], "reason": None}
    api_module.app.dependency_overrides[api_module.query_runner] = lambda: _runner_returning(answer)

    body = client.post("/query", json={"query": "q"}).json()

    assert body["status"] == status
    assert body["answer"] == "答案正文"
    assert body["used_source_ids"] == [1]


def test_query_returns_evidence_and_trace():
    chunks = [
        {"rank": 1, "content": "证据一", "score": 0.8123, "source_doc": "docs-cn/a.md", "headings": "h", "chunk_id": "c1"},
        {"rank": 2, "content": "证据二", "score": 0.4511, "source_doc": "asktug/b.md", "headings": "h", "chunk_id": "c2"},
    ]
    answer = {"status": "answered", "answer": "x", "used_source_ids": [1], "reason": None}
    api_module.app.dependency_overrides[api_module.query_runner] = lambda: _runner_returning(answer, chunks)

    body = client.post("/query", json={"query": "q"}).json()

    assert [e["source"] for e in body["evidence"]] == ["docs-cn/a.md", "asktug/b.md"]
    assert body["evidence"][0]["score"] == pytest.approx(0.8123)
    assert body["attempt"] == 1


def test_query_falls_back_when_answer_is_not_a_dict():
    """模型偶尔吐出非 dict（历史上有 2 条 invalid_output），接口不能因此 500。"""
    api_module.app.dependency_overrides[api_module.query_runner] = lambda: _runner_returning("oops")  # type: ignore[arg-type]

    body = client.post("/query", json={"query": "q"}).json()

    assert body["status"] == "invalid_output"
    assert body["answer"] == ""


def test_query_returns_503_when_pipeline_unavailable():
    def _explode(_query: str) -> dict:
        raise RuntimeError("知识库为空")

    api_module.app.dependency_overrides[api_module.query_runner] = lambda: _explode

    resp = client.post("/query", json={"query": "q"})

    assert resp.status_code == 503
    assert "知识库为空" in resp.json()["detail"]


# ============================================================
# 契约
# ============================================================


def test_openapi_exposes_the_four_paths():
    spec = client.get("/openapi.json").json()

    assert set(spec["paths"]) == {"/health", "/documents", "/search", "/query"}
    # 四个端点的入参模型都在 schema 里 —— 接口契约得是机器可读的
    assert "SearchRequest" in spec["components"]["schemas"]
    assert "QueryResponse" in spec["components"]["schemas"]


def test_request_validation_rejects_empty_query():
    assert client.post("/search", json={"query": ""}).status_code == 422
    assert client.post("/query", json={}).status_code == 422
