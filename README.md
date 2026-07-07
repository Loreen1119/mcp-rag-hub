# MCP-RAG-Hub

从底层原理出发、全手工实现的 RAG 知识检索系统。覆盖文档解析、Token 级切块、BM25+向量双路召回、RRF 融合、Cross-Encoder 重排序、LangGraph 代理编排、FastMCP 工具封装、以及完整评测体系的端到端管线。

**[项目详解](docs_knowledge/项目详解.md)** · **[技术视角详解](docs_knowledge/技术视角详解.md)** · **[章节笔记](docs_knowledge/chapters/)**

## 功能

- **文档消化** — PDF / Markdown / TXT 自动加载，编码自检测，Token 级滑动窗口切块
- **双路检索** — BM25 关键词（jieba 分词）+ 向量语义（all-MiniLM-L6-v2, ChromaDB）
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

# 运行消融实验
python src/evaluation/experiments.py
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
│   ├── retrievers.py          # BM25 + ChromaDB 双路召回
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
│   └── test_queries.json      # GoldenTestSet（15 组三层分类）
│
├── docs/                      # 测试文档
├── docs_knowledge/            # 项目文档与章节笔记
├── experiments/               # 实验结果 JSON
└── chroma_db/                 # ChromaDB 持久化向量库
```

## 关键数据

| 指标 | 值 |
|------|-----|
| 全管线 MRR | **1.00**（15/15 完美命中） |
| BM25→全管线 MRR 提升 | +0.10（0.90 → 1.00） |
| 语义类 MRR 提升 | +0.27（0.80 → 1.00） |
| 全管线延迟 | ~232 ms（CE 占 92%） |
| LLM Faithfulness | **0.95**（qwen2.5:7b） |

## 技术栈

| 层次 | 技术 |
|------|------|
| 语言 | Python 3.11+ |
| Embedding | sentence-transformers/all-MiniLM-L6-v2（384 维） |
| Cross-Encoder | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| 向量库 | ChromaDB（HNSW 索引, cosine 距离） |
| 关键词检索 | rank-bm25 + jieba 分词 |
| 融合 | RRF（Reciprocal Rank Fusion, k=60） |
| 代理 | LangGraph（声明式状态机, 条件路由） |
| LLM | Ollama + qwen2.5:7b |
| 评测 | Ragas（Faithfulness / Answer Relevancy / Context Recall） |
| MCP | FastMCP 2.0（stdio 传输） |
| UI | Streamlit |
