# MCP-RAG-Hub

从底层原理出发、全手工实现的 RAG 知识检索系统。覆盖文档解析、Token 级切块、BM25+向量+实体共现图三路召回、RRF 融合、Cross-Encoder 重排序、LangGraph 代理编排、FastMCP 工具封装、以及完整评测体系的端到端管线。

**[项目详解](docs_knowledge/项目详解.md)** · **[技术视角详解](docs_knowledge/技术视角详解.md)** · **[章节笔记](docs_knowledge/chapters/)**

## 功能

- **文档消化** — PDF / Markdown / TXT 自动加载，编码自检测，Token 级滑动窗口切块
- **三路检索** — BM25 关键词 + 向量语义 + 知识图谱（LLM 抽取三元组）三路并行召回
- **知识图谱** — DeepSeek LLM 抽取三元组，NetworkX 有向图索引，路径搜索与 Chunk 关联
- **RRF 融合** — 基于排名的多路结果融合，消除量纲差异
- **CE 精排** — Cross-Encoder 联合编码重排序，两阶段检索（Bi-Encoder 粗筛 → CE 精排）
- **Streamlit 交互** — 四标签页展示 BM25/向量/RRF/CE 各阶段检索结果
- **LangGraph 代理** — 五节点状态机，条件路由，查询改写与自我纠错
- **MCP 工具** — FastMCP 封装四个工具接口，可接入任何 MCP 客户端
- **完整评测** — MRR/Hit@K/Precision@K/Recall@K + LLM-as-Judge（Ragas）+ 消融实验

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 确保 ChromaDB 模型可用（首次运行自动下载）
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# 启动 Streamlit 界面
streamlit run app.py

# 或启动 MCP 服务
python src/mcp_server.py

# 运行检索评测
python src/evaluation/retrieval_eval.py

# 运行消融实验（含 GraphRAG）
python src/evaluation/experiments.py

# 运行图检索演示
python src/graph_retriever.py
```

## 项目结构

```
mcp-rag-hub/
├── app.py                     # Streamlit 前端入口
├── agent.py                   # LangGraph Agent 入口
├── config.py                  # 全局配置中心
├── requirements.txt
│
├── src/
│   ├── models.py              # Chunk / RetrievalResult 数据结构
│   ├── data_pipeline.py       # 文档加载 + Token 级滑动窗口切块
│   ├── retrievers.py          # BM25 + 向量双路召回
│   ├── graph_retriever.py       # 实体共现图检索（GraphRAG）
│   ├── fusion.py              # RRF 融合 + Cross-Encoder 重排序
│   ├── mcp_server.py          # FastMCP 工具封装
│   └── evaluation/            # 评测子包
│       ├── metrics.py         # 共享评测指标
│       ├── retrieval_eval.py  # 检索质量评测
│       ├── llm_eval.py        # LLM-as-Judge 生成评测
│       ├── agent_eval.py      # Agent 改写评测
│       └── experiments.py     # 消融实验与数据分析
│
├── data/
│   ├── test_queries_all.json  # 全部 36 条（E01-E08, S01-S08, M01-M08, G01-G12）
│   ├── train_queries.json     # 训练集 18 条（按类别难度平衡分配）
│   └── test_queries.json      # 测试集 18 条（与 train 互补，不参与调参）
│
├── docs/                      # 测试文档
├── docs_knowledge/            # 项目文档与章节笔记
├── experiments/               # 实验结果 JSON
└── chroma_db/                 # ChromaDB 持久化向量库
```

## 关键数据

> 基于 36 组四层 Golden Test Set（exact_match / semantic / mixed / graph 四类），train/test 分离，test 全程不参与调参

**最终指标（Test 集）：** BM25+向量双路 RRF 融合 MRR **0.78**，经 Cross-Encoder 精排后完整管线 MRR **0.81**，Hit@5 **100%**，Recall@5 **0.90**。

**Train vs Test 揭示的规律：**

| 分集 | RRF MRR | Hybrid MRR | CE MRR |
|------|---------|-----------|--------|
| Train | 0.6075 | 0.6662 | 0.5602 |
| Test | 0.7824 | 0.5919 | **0.8148** |

- Train 上 Hybrid > RRF：说明 KG 在部分关系型查询上确实有用
- Test 上 Hybrid < RRF：说明 KG 泛化性不够，在新 queries 上引入了噪声
- CE 精排救了场：Test CE 0.81 >> Hybrid 0.59，说明两阶段设计是稳健的

**知识图谱的价值定位：** 展示 LLM 三元组抽取 + 有向知识图谱构建 + 三路融合设计，针对多实体关系型查询作为第三路召回补充。

### Test 集（18 条）

| 指标 | BM25 | Vector | RRF | Graph | Hybrid | CE（全管线） |
|------|------|--------|-----|-------|--------|------|
| MRR | 0.7913 | 0.6769 | 0.7824 | 0.5000 | 0.5919 | **0.8148** |
| Hit@5 | 0.8333 | 0.7778 | 0.9444 | 0.5000 | 0.7778 | 0.9444 |
| Prec@5 | 0.5111 | 0.4333 | 0.4444 | 0.4778 | 0.4444 | 0.5333 |
| Recall@5 | 0.7130 | 0.6111 | 0.7685 | 0.3426 | 0.6296 | 0.7963 |

### Train 集（18 条）

| 指标 | BM25 | Vector | RRF | Graph | Hybrid | CE（全管线） |
|------|------|--------|-----|-------|--------|------|
| MRR | 0.6685 | 0.5836 | 0.6075 | 0.5000 | 0.6662 | 0.5602 |
| Hit@5 | 0.8889 | 0.8889 | 0.8333 | 0.5000 | 0.8333 | 0.8889 |
| Prec@5 | 0.4556 | 0.3667 | 0.4556 | 0.5000 | 0.4778 | 0.4444 |
| Recall@5 | 0.8889 | 0.8611 | 0.8056 | 0.4167 | 0.7500 | 0.8889 |

## 技术栈

| 层次 | 技术 |
|------|------|
| 语言 | Python 3.11+ |
| Embedding | sentence-transformers/all-MiniLM-L6-v2（384 维） |
| Cross-Encoder | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| 向量库 | ChromaDB（HNSW 索引, cosine 距离） |
| 关键词检索 | rank-bm25 + jieba 分词 |
| 图检索 | NetworkX 实体共现图 + 子图遍历 |
| 融合 | RRF（Reciprocal Rank Fusion, k=60） |
| 代理 | LangGraph（声明式状态机, 条件路由） |
| LLM | Ollama + qwen2.5:7b |
| 评测 | Ragas（Faithfulness / Answer Relevancy / Context Recall） |
| MCP | FastMCP 2.0（stdio 传输） |
| UI | Streamlit |
