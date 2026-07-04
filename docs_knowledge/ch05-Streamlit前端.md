# 第 5 章：Streamlit 前端

## 知识点

### 1. Streamlit 独特的执行模型与"重复计算灾难"

在传统 Web 开发（Vue/React）中，页面更新是组件级局部刷新。Streamlit 则采用完全不同的"脚本全刷新执行模型"：

- **核心机制**：网页上发生任何用户交互（输入框打字、点击按钮、切换 Tab），Streamlit 都会将整个 Python 脚本从第 1 行到最后一行彻底重新执行一遍。
- **RAG 场景的真实代价**：如果不做拦截防护，用户每次切换 Tab 浏览日志，系统都会重新加载 `all-MiniLM-L6-v2` 模型（~90MB）、重新扫描本地文档目录、重新执行滑窗切片与索引构建。每次交互延迟轻松达到数秒，CPU 空转。
- **全局变量失效**：由于每次交互脚本从头重跑，Python 全局变量在每次刷新时都会被重置，无法用来跨交互常驻大模型对象。

### 2. `@st.cache_resource` 的"免死金牌"机制

为了化解重复计算灾难，Streamlit 提供了缓存装饰器。但在 RAG 场景下，装饰器选型有严格的底层约束。

#### 缓存拦截的底层执行链路：

```
第一次交互 → 缓存库为空 → 真正执行函数体（加载模型、解析文档、构建索引）→ 产出对象锁定在内存中
                                                                              │
后续交互   → 缓存命中   → 跳过整个函数体，直接返回内存中的对象引用 → 0 毫秒通过
```

#### 为什么必须用 `@st.cache_resource` 而不能用 `@st.cache_data`？

| | `@st.cache_data` | `@st.cache_resource` |
|---|---|---|
| **定位** | 缓存纯数据（dict、list、DataFrame） | 缓存不可序列化的活资源（模型、DB 连接、线程池） |
| **底层原理** | 将返回值通过 `pickle` 序列化为二进制字节流存储，使用时反序列化解压 | 不序列化，直接保持对象的内存引用 |
| **RAG 场景适用性** | 不适用：`SentenceTransformer` 和 ChromaDB client 底层包含 C++ 指针和网络连接，**在数学上无法被 pickle 序列化**，错用会直接抛序列化崩溃异常 | 必须使用：锁死活对象的内存全局引用 |

### 3. 四 Tab 联动：算法全链路透明的"白盒调试看板"

相比于普通 RAG 系统只有一个黑盒对话框，本项目利用 Streamlit 的 `st.tabs` 组件，将 FusionPipeline 暴露的 Dict 复合中间结果（各阶段的融合与重排序列）完整呈现在四个并列 Tab 中：

- **Tab 1（BM25 稀疏召回）**：展示基于词频统计捞出的精准关键词匹配结果
- **Tab 2（Vector 稠密召回）**：展示基于语义向量相似度召回的语义相关但字面不同的内容
- **Tab 3（RRF 排名融合）**：呈现 RRF 算法利用 `1/(k+rank)` 公式对两路结果进行平权融合后的排名大洗牌，观察"双路共识"如何打破单路垄断
- **Tab 4（Cross-Encoder 精排）**：展示精排模型如何通过自注意力机制对候选集重新打分，将最相关的 Top-5 内容排到最顶端，作为喂给 LLM 的最终上下文

**工程价值**：研发期是排查"哪一步召回了噪声"的全链路听诊器；演示期是向团队和面试官展示模块级贡献与算法可解释性的可视化工具。

### 4. `_render_results` 的多态渲染引擎

实现四 Tab 诊断的核心功臣是 `_render_results` 函数：

- **异构数据多态兼容**：不同检索阶段返回的分数维度完全不同（BM25 是词频加权分、Vector 是余弦相似度、RRF 是倒数排名和、Cross-Encoder 是 Logit 分）。该函数通过 `source_label` 参数动态路由渲染策略，一套代码兼容四种完全不同的算法层展示。
- **RRF 精度微调的数学原因**：在渲染 RRF 分数时，代码刻意将格式化精度拓展至**小数点后六位（`.6f`）**。这是因为引入平滑常数 k=60 后，顶端文档的 RRF 分差被极度压缩（例如 `0.032258` vs `0.022643`）。若用 `.4f`，微弱的排名逆袭差距将被四舍五入抹平，`.6f` 高精度才能清晰呈现双路共识引发的名次变化。
- **非等宽栅格布局**：`st.columns([0.05, 0.15, 0.8])` 手写三列非等宽水平栅格——左侧 5% 锁死排名序号，中间 15% 用 `st.metric` 指标卡片大字号突出核心得分，右侧 80% 黄金区域渲染标题面包屑和正文快照。

### 5. Streamlit vs Gradio

- **Streamlit**：Python 脚本风格，`st.xxx` API 搭积木，适合数据应用和内部工具
- **Gradio**：专为 ML 模型 demo 设计，`gr.Interface` 一行代码启动，适合对外分享模型能力
- 本项目选 Streamlit 的原因是交互以"查询输入 → 结果展示"为主，不需要 Gradio 的模型托管功能

### 面试速记

- **Streamlit 执行模型**：每次交互重新执行整个脚本，`@st.cache_resource` 拦截重复计算
- **cache_resource vs cache_data**：resource 缓存不可序列化对象（模型、DB 连接），data 缓存可序列化数据（dict、DataFrame）。`SentenceTransformer` 含 C++ 指针，无法 pickle，必须用 resource
- **四 Tab 设计**：BM25 → Vector → RRF → Cross-Encoder，全链路白盒可观测
- **`_render_results`**：多态渲染引擎，`.6f` 高精度保留 RRF 名次逆袭细节
- **和 Gradio 区别**：Streamlit 更接近写 Python 脚本，Gradio 更接近搭 ML 模型 demo

---

## 关键实现与代码走读

> 以下代码节选自实际 `app.py`，注释为讲解用。

### ① 缓存管线加载

```python
@st.cache_resource
def load_pipeline():
    with st.spinner("正在加载文档并构建索引..."):
        chunks = process_directory()
        bm25 = BM25Retriever(chunks)
        vector = VectorRetriever(chunks, rebuild=True)
        pipeline = FusionPipeline()
    return chunks, bm25, vector, pipeline
```

**追问应对**："为什么不用全局变量？"— Streamlit 每次交互重新运行脚本，全局变量会被重置。`@st.cache_resource` 让 Streamlit 跳过函数体直接返回缓存对象，避免反复加载模型（SentenceTransformer 冷加载一次约 3 秒）。

**追问应对**："为什么不用 `@st.cache_data`？"— `cache_data` 底层用 pickle 序列化返回值。`SentenceTransformer` 和 ChromaDB client 内部含 C++ 指针引用和网络连接，无法被 pickle 序列化，强行使用会直接抛异常。`cache_resource` 不序列化，直接保持对象引用。

### ② 四阶段检索与分轨展示

```python
if query:
    bm25_results = bm25.search(query)
    vector_results = vector.search(query)
    output = pipeline.run(bm25_results, vector_results, query)

    # 四 Tab 并排对比
    tab1, tab2, tab3, tab4 = st.tabs([
        f"BM25 关键词 ({len(bm25_results)})",
        f"向量语义 ({len(vector_results)})",
        f"RRF 融合 ({len(output['rrf'])})",
        f"Cross-Encoder 精排 ({len(output['cross_encoder'])})",
    ])

    with tab1: _render_results(bm25_results, "bm25")
    with tab2: _render_results(vector_results, "vector")
    with tab3: _render_results(output["rrf"], "rrf")
    with tab4: _render_results(output["cross_encoder"], "cross_encoder")

    # 最终答案区
    if output["cross_encoder"]:
        best = output["cross_encoder"][0]
        st.success(
            f"**来源**: {best.chunk.metadata.get('heading_breadcrumb', best.chunk.metadata.get('source', '?'))}"
        )
        st.markdown("**最佳匹配段落**")
        st.info(best.chunk.content[:800] + ("..." if len(best.chunk.content) > 800 else ""))

        # 分数明细 — 展示 Cross-Encoder 分、RRF 分、原始来源三位一体
        score_meta = best.chunk.metadata
        st.caption(
            f"CE Score: {best.score:.4f}  |  "
            f"RRF Score: {score_meta.get('rrf_score', 'N/A')}  |  "
            f"原始来源: {score_meta.get('original_source', 'N/A')}"
        )
```

**追问应对**："为什么展示所有中间阶段而不是只展示最终结果？"— 一方面方便调试和验证各阶段行为是否正确，另一方面面试演示时可以直观看到 Cross-Encoder 如何改变了排序。对于用户来说，这个设计让他们信任系统的排序逻辑。

### ③ `_render_results` 渲染引擎

```python
def _render_results(results, source_label):
    if not results:
        st.warning("无结果")
        return

    for i, r in enumerate(results[:10]):    # 仅展示 Top-10，控制视觉噪声
        meta = r.chunk.metadata
        heading = meta.get("heading_breadcrumb", meta.get("source", "?"))

        with st.container():
            cols = st.columns([0.05, 0.15, 0.8])   # 非等宽三列栅格

            with cols[0]:
                st.markdown(f"**#{i+1}**")          # 排名 Rank

            with cols[1]:                            # 按数据源路由分数展示
                if source_label in ("bm25", "vector"):
                    st.metric("Score", f"{r.score:.4f}")
                elif source_label == "rrf":
                    st.metric("RRF", f"{r.score:.6f}")   # 高精度展示双路共识
                else:
                    st.metric("CE", f"{r.score:.4f}")

            with cols[2]:
                st.markdown(f"**{heading}**")        # 标题面包屑
                st.text(r.chunk.content[:300] + ("..." if len(r.chunk.content) > 300 else ""))

            st.divider()
```

---

## 面试话术

**面试官**："我看你用 Streamlit 写了个前端，这东西不就是调几个 API 搭积木吗，有什么技术含量？"

**回答**："我选择 Streamlit 的核心目的不是'搭个界面'，而是**构建一个全链路透明的算法白盒调试看板**，解决传统 RAG 对话框黑盒无法量化各模块贡献的痛点。

落地时需要解决 Streamlit 独特的执行模型问题：**每次用户交互（切换 Tab、输入文字）都会强制重跑整个 Python 脚本**。如果不加拦截，底层 SentenceTransformer 模型和 ChromaDB 客户端会被反复灾难性加载，单次交互延迟飙升至数秒。

我用 `@st.cache_resource` 对整个管线的加载和索引构建做了内存级拦截。这里有一个关键选型：为什么不用 `@st.cache_data`？因为 `cache_data` 底层依赖 pickle 序列化，而大模型对象和 ChromaDB client 包含 C++ 指针和网络连接，在工程上**无法被 pickle 序列化**——强行使用会直接崩溃。`cache_resource` 不序列化，直接保持内存引用，将后续交互延迟打到 0 毫秒。

前端呈现上，通过四 Tab 联动将 Pipeline 暴露的 RRF 融合列表和 Cross-Encoder 重排列表等中间结果完整展示，打造了一个从召回、融合到精排的全链路可视化日志流。"

---

**面试官**："你具体是怎么做渲染的？"

**回答**："`_render_results` 是一个**接收多态数据源标签 `source_label` 的通用排版引擎**。四阶段返回的分数维度完全不同——BM25 是词频分、Vector 是余弦相似度、RRF 是倒数排名和、Cross-Encoder 是 Logit 分——通过 source_label 动态路由，一套代码兼容四种数据结构。

布局上，我用 `st.columns([0.05, 0.15, 0.8])` 手写了非等宽水平栅格：左侧 5% 极窄区展示排名，中间 15% 用 `st.metric` 大字号突出核心分数，右侧 80% 渲染标题面包屑和 300 字正文快照。

有一个重要细节：在渲染 RRF 分数时，我刻意将精度设为 `.6f`（六位小数）。原因是 RRF 的平滑常数 k=60 会极度压缩顶端文档之间的分数差异，比如 `0.032258` vs `0.022643`——如果只用 `.4f`，这种由双路共识引发的微妙名次逆袭就会被四舍五入抹平，`.6f` 高精度保证了排名的可观测性。"

---

## 产出文件

- `app.py` — Streamlit 前端（根目录，Streamlit 默认入口）

## 相关章节

- [[ch04-RRF融合与重排序]] — FusionPipeline 在这里被 Streamlit 调用
- [[ch06-Ragas评测]] — 评测脚本也调用同一个 FusionPipeline
