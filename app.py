"""
Streamlit 前端 — RAG 知识检索系统交互界面。

运行方式: streamlit run app.py
"""

from __future__ import annotations

import streamlit as st
from pathlib import Path

from src.pipeline import get_pipeline

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

st.title("RAG 智能知识检索系统")
st.caption("BM25 关键词 + 向量语义 → RRF 融合 → Cross-Encoder 精排")

query = st.text_input("输入查询", placeholder="例如：混合检索策略、RAG 的核心优化方向...")

if query:
    # 执行检索
    bm25_results = bm25.search(query)
    vector_results = vector.search(query)
    output = pipeline.run(bm25_results, vector_results, query)

    # ---- 四阶段结果对比 ----
    st.divider()
    st.subheader("检索结果对比")

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

    # ---- 最终答案区域 ----
    st.divider()
    st.subheader("Top 答案")

    if output["cross_encoder"]:
        best = output["cross_encoder"][0]
        st.success(f"**来源**: {best.chunk.metadata.get('heading_breadcrumb', best.chunk.metadata.get('source', '?'))}")

        # 展示最佳匹配段落
        st.markdown("**最佳匹配段落**")
        st.info(best.chunk.content[:800] + ("..." if len(best.chunk.content) > 800 else ""))

        # 分数明细
        score_meta = best.chunk.metadata
        st.caption(
            f"CE Score: {best.score:.4f}  |  "
            f"RRF Score: {score_meta.get('rrf_score', 'N/A')}  |  "
            f"原始来源: {score_meta.get('original_source', 'N/A')}"
        )
