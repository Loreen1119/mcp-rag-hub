# MCP-RAG-Hub

从底层原理出发、全手工实现的 RAG 知识检索系统。覆盖文档解析、Token 级切块、BM25+向量混合召回、RRF 融合、Cross-Encoder 重排序、LangGraph 代理编排、FastMCP 工具封装、以及完整评测体系的端到端管线。知识图谱路作为可选实验功能默认关闭。

**[项目详解](docs_knowledge/项目详解.md)** · **[技术视角详解](docs_knowledge/技术视角详解.md)** · **[章节笔记](docs_knowledge/chapters/)**

## 功能

- **文档消化** — PDF / Markdown / TXT / Python 自动加载，编码自检测；Markdown 标题面包屑 + Token 级滑动窗口切块，Python 源码按 AST 函数/类边界语义分块
- **混合检索** — BM25 关键词 + 向量语义双路召回，知识图谱作为可选实验功能（`ENABLE_KG` 开关）
- **RRF 融合** — 基于排名的多路结果融合，消除量纲差异
- **CE 精排** — Cross-Encoder 联合编码重排序，两阶段检索（Bi-Encoder 粗筛 → CE 精排）
- **Streamlit 交互** — 四标签页展示 BM25/向量/RRF/CE 各阶段检索结果
- **LangGraph 代理** — 五节点状态机，条件路由，查询改写与自我纠错
- **MCP 工具** — FastMCP 封装四个工具接口，可接入任何 MCP 客户端
- **完整评测** — MRR/Hit@K/Precision@K/Recall@K + 自实现 LLM-as-Judge + 消融实验

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
│   ├── data_pipeline.py       # 文档加载 + 按文件类型路由分块（Markdown 标题滑窗 / Python AST / 通用滑窗）
│   ├── retrievers.py          # BM25 + 向量双路召回
│   ├── graph_retriever.py     # 实体共现图检索（可选实验功能）
│   ├── kg_retriever.py        # LLM 三元组知识图谱检索（可选实验功能）
│   ├── kg_builder.py          # DeepSeek LLM 抽取三元组、构建知识图谱缓存
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
├── docs/                      # 知识库数据源
├── journal/                   # 踩坑日志与学习笔记
├── docs_knowledge/            # 项目文档与章节笔记
├── experiments/               # 实验结果 JSON
└── chroma_db/                 # ChromaDB 持久化向量库
```

## 关键数据

> 基于 36 条四类分层 Golden Test Set（exact_match / semantic / mixed / graph），train/test 严格分离，test 集不参与调参。
> 以下为 2026-08-04 清理知识库（过程笔记迁出 docs/）并重写测试集后的**最终基线**，实验过程详见 [KG 消融实验笔记](journal/2026-08-04-kg-ablation-notes.md)。

### Test 集（18 条）各阶段指标

| 阶段 | MRR@5 | Hit@5 | Prec@5 | Recall@5 |
|------|-------|-------|--------|----------|
| BM25 | **0.8704** | 0.9444 | 0.4667 | 0.8611 |
| Vector | 0.7384 | 0.9444 | 0.4889 | 0.8981 |
| RRF 融合 | 0.8565 | **1.0000** | 0.4889 | **0.9722** |
| CE 精排（全管线） | 0.7546 | 0.9444 | **0.5556** | 0.8796 |

**怎么读这张表：**

- **RRF 融合阶段 Hit@5 = 100%**：召回侧能力充足，正确文档几乎必然进入候选集，后续只需排序
- **CE 精排把 Precision@5 从 0.49 提升到 0.56**：精排的价值在于把最相关的文档顶到最前；MRR 略低于 RRF 系个别 case 被重排出 Top-5 所致，属已知权衡而非普遍问题
- **延迟剖析（本地 CPU）**：全链路均值 395ms，其中 CE 精排 373ms 占 94%——这正是采用两阶段架构、将精排限制在 Top-20 候选内的原因（详见 `experiments/latency_profile.json`）

**知识图谱的价值定位：** 作为可选实验功能（`ENABLE_KG` 开关），展示 LLM 三元组抽取 + 有向知识图谱构建 + 多路融合设计。通过对照实验发现：在小规模同主题语料上，KG 路因实体区分度不足而不提升指标（清理噪声文档后 KG Hit@5 从 33% 升至 100%，但 MRR 天花板仍低），故默认关闭——这是数据驱动的架构决策，不是妥协。完整归因过程见 [KG 消融实验笔记](journal/2026-08-04-kg-ablation-notes.md)。

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
| 评测 | 自实现 LLM-as-Judge（Faithfulness / Answer Relevancy / Context Recall，Ollama qwen2.5:7b 评分） |
| MCP | FastMCP 2.0（stdio 传输） |
| UI | Streamlit |
