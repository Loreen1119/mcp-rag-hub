# 第 5 章：Streamlit 前端

## 知识点

### 1. Streamlit 执行模型

- **每次交互重新执行整个脚本**：用户输入、点击按钮、切换 Tab 都会触发从头到尾的重新执行
- `@st.cache_resource`：装饰器，缓存函数返回值。模型加载、索引构建等重操作只跑一次，后续交互直接从缓存读
- `st.session_state`：字典，跨交互保持状态。本次没用到（因为每次交互的逻辑完全由 query 驱动，不需要跨交互记忆）

### 2. 为什么用 st.cache_resource 而不是 st.cache_data

- `@st.cache_data`：缓存数据（DataFrame、list、dict），用 pickle 序列化
- `@st.cache_resource`：缓存资源（模型、数据库连接），不序列化，直接保持对象引用
- SentenceTransformer、ChromaDB client 等对象不可 pickle，必须用 `@st.cache_resource`

### 3. 四 Tab 对比设计

四个 Tab 分别展示 BM25 / Vector / RRF / Cross-Encoder 的结果，面试时可以直接演示：
- BM25 Tab：哪些词被精确匹配了
- Vector Tab：哪些语义相关但字面不同的内容被召回了
- RRF Tab：两路融合后的排序变化
- CE Tab：Cross-Encoder 如何进一步纠偏排序

### 4. Streamlit vs Gradio

- **Streamlit**：Python 脚本风格，`st.xxx` API 搭积木，适合数据应用和内部工具
- **Gradio**：专为 ML 模型 demo 设计，`gr.Interface` 一行代码启动，适合对外分享模型能力
- 本项目选 Streamlit 的原因是交互以"查询输入 → 结果展示"为主，不需要 Gradio 的模型托管功能

### 面试速记

- **Streamlit 执行模型**：每次交互重新执行（保证计算一直是正确的），cache 装饰器防止重计算
- **cache_resource vs cache_data**：resource 缓存不可序列化对象（模型），data 缓存可序列化数据
- **和 Gradio 区别**：Streamlit 更接近写 Python 脚本，Gradio 更接近搭 ML 模型 demo

## 产出文件

- `app.py` — Streamlit 前端（根目录，Streamlit 默认入口）

## 关键实现

### ① 缓存管线加载

```python
@st.cache_resource
def load_pipeline():
    chunks = process_directory()
    bm25 = BM25Retriever(chunks)
    vector = VectorRetriever(chunks, rebuild=True)
    pipeline = FusionPipeline()
    return chunks, bm25, vector, pipeline
```

**追问应对**：「为什么不用全局变量？」— Streamlit 每次交互重新运行脚本，全局变量会被重置。`@st.cache_resource` 让 Streamlit 跳过函数体直接返回缓存对象，避免反复加载模型（SentenceTransformer 加载一次 ~3 秒）。

### ② 四阶段结果同时展示

```python
output = pipeline.run(bm25_results, vector_results, query)
# output = {"rrf": [...], "cross_encoder": [...]}
```

**追问应对**：「为什么展示所有中间阶段而不是只展示最终结果？」— 一方面方便调试和验证各阶段行为是否正确，另一方面面试演示时可以直观看到 Cross-Encoder 如何改变了排序。对于用户来说，这个设计让他们信任系统的排序逻辑。

---

## 相关章节

- [[ch04-RRF融合与重排序]] — FusionPipeline 在这里被 Streamlit 调用
- [[ch06-Ragas评测]] — 评测脚本也调用同一个 FusionPipeline
