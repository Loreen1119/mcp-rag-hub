"""
Streamlit 前端 — RAG 知识检索系统交互界面。

运行方式: streamlit run app.py
"""

from __future__ import annotations

import streamlit as st
from pathlib import Path

from src.pipeline import get_pipeline
from agent import run_query

# ============================================================
# 页面配置
# ============================================================

st.set_page_config(
    page_title="RAG 知识检索系统",
    page_icon="",
    layout="wide",
)


def _render_results(results, source_label):
    """渲染检索结果列表。"""
    if not results:
        st.warning("无结果")
        return

    for i, r in enumerate(results[:10]):
        meta = r.chunk.metadata
        heading = meta.get("heading_breadcrumb", meta.get("source", "?"))

        with st.container():
            # 排名 + 分数 + 来源
            cols = st.columns([0.05, 0.15, 0.8])
            with cols[0]:
                st.markdown(f"**#{i+1}**")
            with cols[1]:
                if source_label in ("bm25", "vector"):
                    st.metric("Score", f"{r.score:.4f}")
                elif source_label == "rrf":
                    st.metric("RRF", f"{r.score:.6f}")
                else:
                    st.metric("CE", f"{r.score:.4f}")
            with cols[2]:
                st.markdown(f"**{heading}**")
                st.text(r.chunk.content[:300] + ("..." if len(r.chunk.content) > 300 else ""))

            st.divider()

# ============================================================
# 缓存：文档加载 + 索引构建（只跑一次，后续从缓存读）
# ============================================================


@st.cache_resource
def load_pipeline():
    """获取共享 RAG 管线上下文（懒加载单例）。全程缓存。"""
    with st.spinner("正在加载文档并构建索引..."):
        ctx = get_pipeline()
    return ctx


# ============================================================
# 侧边栏
# ============================================================

with st.sidebar:
    st.title("配置")
    st.divider()

    with st.spinner("加载中..."):
        ctx = load_pipeline()
    chunks, bm25, vector, pipeline = ctx.chunks, ctx.bm25, ctx.vector, ctx.pipeline

    st.metric("已索引 Chunk 数", len(chunks))

    st.divider()
    st.caption("文档列表")
    sources = sorted(set(c.metadata.get("source", "?") for c in chunks))
    for s in sources:
        count = sum(1 for c in chunks if c.metadata.get("source") == s)
        st.caption(f"  {s} ({count} chunks)")

    st.divider()
    st.caption("技术栈: BM25 + ChromaDB + RRF + Cross-Encoder")

# ============================================================
# 主区域
# ============================================================


def _render_evidence_cards(chunks, used_source_ids=None):
    """Render numbered evidence cards matching the citation IDs in the answer."""
    used_source_ids = set(used_source_ids or [])
    for index, chunk in enumerate(chunks[:3], start=1):
        marker = "已引用" if index in used_source_ids else "证据"
        with st.container(border=True):
            st.markdown(f"**[{index}] {marker} · {chunk['source_doc']}**")
            if chunk.get("headings"):
                st.caption(chunk["headings"])
            st.write(chunk["content"])
            st.caption(f"CE Score: {chunk['score']:.4f}")


def _render_answer(answer_data, chunks):
    status = answer_data.get("status")
    reason = answer_data.get("reason") or "未提供"
    if status == "answered":
        st.success("回答已生成")
        st.markdown(answer_data.get("answer", ""))
        cited = answer_data.get("used_source_ids", [])
        if cited:
            st.caption("引用来源：" + " ".join(f"[{item}]" for item in cited))
    elif status == "insufficient_evidence":
        st.warning("知识库信息不足，未生成答案")
        st.caption(f"拒答原因：{reason}")
    elif status == "generation_unavailable":
        st.error("回答生成服务暂不可用，但已找到检索证据")
        st.caption(f"故障原因：{reason}")
        fallback = answer_data.get("answer", "")
        if fallback:
            st.markdown(fallback)
    else:
        st.error("回答处理出错")
        st.caption(f"处理原因：{reason}")

    if chunks:
        st.subheader("检索证据")
        _render_evidence_cards(chunks, answer_data.get("used_source_ids", []))


st.title("企业知识库问答 Agent")
st.caption("混合检索 + 证据门控 + 可追溯引用")
query = st.text_input("输入查询", placeholder="例如：RRF 如何工作？")

if query:
    with st.spinner("正在检索并生成回答..."):
        result = run_query(query)

    answer_data = result.get("answer", {})
    if not isinstance(answer_data, dict):
        answer_data = {"status": "invalid_output", "answer": "", "used_source_ids": [], "reason": "invalid_output"}
    chunks = result.get("last_retrieved_chunks", [])

    answer_tab, debug_tab = st.tabs(["问答模式", "调试模式"])
    with answer_tab:
        _render_answer(answer_data, chunks)

    with debug_tab:
        st.subheader("检索结果对比")
        bm25_results = bm25.search(query)
        vector_results = vector.search(query)
        output = pipeline.run(bm25_results, vector_results, query)
        tab1, tab2, tab3, tab4 = st.tabs([
            f"BM25 关键词 ({len(bm25_results)})",
            f"向量语义 ({len(vector_results)})",
            f"RRF 融合 ({len(output['rrf'])})",
            f"Cross-Encoder 精排 ({len(output['cross_encoder'])})",
        ])
        with tab1:
            _render_results(bm25_results, "bm25")
        with tab2:
            _render_results(vector_results, "vector")
        with tab3:
            _render_results(output["rrf"], "rrf")
        with tab4:
            _render_results(output["cross_encoder"], "cross_encoder")

        st.subheader("Agent 迭代历史")
        for item in result.get("retrieval_history", []):
            st.json(item)
        st.subheader("查询改写历史")
        if result.get("rewritten_queries"):
            for item in result["rewritten_queries"]:
                st.code(item)
        else:
            st.caption("本次查询未触发改写")
