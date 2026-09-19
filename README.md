# MCP-RAG-Hub

一个可本地运行的 RAG 知识库 MCP 服务：把文档喂给它，Claude 等 AI 客户端就能通过标准接口检索你的本地 PDF / Markdown / TXT / Python 文档。

<div align="center">
  <img src="screenshots/ui-query-results.png" width="720" alt="RAG 检索界面：输入查询后展示 BM25/向量/RRF/Cross-Encoder 四阶段结果"/>
  <br/>
  <em>输入一条查询，界面同时展示关键词、语义、融合、精排四个阶段的检索结果</em>
</div>

全链路自研（非调现成 RAG 框架）：文档解析 → BM25+向量混合召回 → RRF 融合 → Cross-Encoder 精排 → LangGraph 代理编排 → FastMCP 工具封装，配 18 条可答 + 10 条拒答 + 3 条多证据共三套验收集。

## 功能

- **全手工 RAG 管线** — 文档解析（PDF/Markdown/TXT/Python，按类型分块）→ BM25+向量双路召回 → RRF 融合 → Cross-Encoder 精排，全链路自研，不依赖现成 RAG 框架
- **Streamlit 交互** — 四标签页逐阶段展示 BM25/向量/RRF/CE 检索结果，输入一条查询即可看到每路召回与最终排序
- **LangGraph 代理** — 五节点状态机，条件路由，查询改写与自我纠错
- **MCP 工具** — FastMCP 封装四个工具接口，可接入 Claude Desktop 等任何 MCP 客户端
- **完整评测** — 18 条 Golden Test Set（E/S/M/G 四类分层），MRR/Hit@K/Precision@K/Recall@K + 消融实验 + 自实现 LLM-as-Judge；另有 10 条拒答集与 3 条多证据集用于端到端验收
- **知识图谱检索**（可选实验功能，默认关闭）— LLM 三元组抽取 + 有向图构建，用 `ENABLE_KG` 开关启用

**四阶段检索界面实拍**：

| BM25 关键词召回 | 向量语义召回 |
|:---:|:---:|
| <img src="screenshots/ui-tab-BM25.png" width="330" alt="BM25 关键词召回阶段截图"/> | <img src="screenshots/ui-tab-vector.png" width="330" alt="向量语义召回阶段截图"/> |
| **RRF 融合** | **Cross-Encoder 精排** |
| <img src="screenshots/ui-tab-RRF.png" width="330" alt="RRF 融合阶段截图"/> | <img src="screenshots/ui-tab-Cross-Encoder.png" width="330" alt="Cross-Encoder 精排阶段截图"/> |

## 快速开始

**前提**：Python 3.11+。知识库语料放在 `corpora/<语料名>/` 目录（默认 `corpora/fastapi-zh`，仓库内置一份真实的 FastAPI 中文文档，122 个文件 / 930K），首次运行会自动解析并构建索引。切换语料用环境变量 `MCP_RAG_CORPUS`（每份语料使用独立的 Chroma 集合，互不覆盖；详见 [corpora/README.md](corpora/README.md)）。

```bash
# 1. 创建虚拟环境（Windows）
python -m venv .venv
.venv\Scripts\activate

# 2. 安装依赖
pip install -r requirements.txt

# 3. Windows 下额外安装 PyTorch（CPU 版即可，requirements.txt 不自动带）
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 4. 启动 Streamlit 网页界面，浏览器打开 http://localhost:8501
streamlit run app.py
```

> 也可跳过界面，直接作为 MCP 服务使用（见下文）。Embedding 模型首次运行会自动下载到本地缓存，之后可设 `HF_HUB_OFFLINE=1` 离线启动。
>
> 纯检索演示无需 Ollama；仅 **LangGraph 代理 / LLM 评测（LLM-as-Judge）** 需要本机安装 Ollama。
> 模型分工：**生成用 `qwen2.5:3b`**（纯 CPU 可跑的窄任务模型）、**评测裁判用 `qwen2.5:7b`**（3b 当裁判无区分度）。
> 7b 需约 5.6 GB 物理可用内存，本机开发环境全开时加载不了，属正常现象。

### 方式二：Docker（一条命令，不依赖本机 Python 环境）

```powershell
$env:HF_CACHE = "$env:USERPROFILE\.cache\huggingface"   # 指向宿主已缓存的模型；不设会直接报错
docker compose up --build                               # 浏览器打开 http://localhost:8501
```

模型不烧进镜像（国内直连 huggingface.co 不稳），而是挂载宿主已有缓存；Ollama 仍在宿主运行，
容器经 `host.docker.internal` 访问。**前置条件、验证命令、已知限制**见
[`docs/reference/Docker部署说明.md`](docs/reference/Docker部署说明.md)。

## 接入 MCP 客户端（Claude Desktop / Cursor 等）

把本服务作为知识库工具接入任何 MCP 客户端。以 Claude Desktop 为例，在 `claude_desktop_config.json` 中加：

```json
{
  "mcpServers": {
    "rag-knowledge": {
      "command": "python",
      "args": ["D:/path/to/mcp-rag-hub/src/mcp_server.py"]
    }
  }
}
```

启动后暴露四个工具：

| 工具 | 用途 |
|------|------|
| `search_knowledge` | 混合检索（BM25+向量→RRF→CE）并返回精排结果 |
| `list_documents` | 查看当前知识库已索引的文档 |
| `get_chunk` | 按编号获取指定切片的完整内容 |
| `get_chunk_count` | 查看知识库切片总数 |

## 项目结构

<details>
<summary>点开看完整目录树（含各模块职责）</summary>

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
├── corpora/                   # 知识库语料（每份语料一个子目录，见 corpora/README.md）
│   └── fastapi-zh/            # 当前生效语料：FastAPI 中文文档 122 篇 / 1445 chunk
│
├── data/
│   ├── test_queries.json          # 可答 Golden Test Set 18 条（E/S/M/G 四类分层）
│   ├── unanswerable_queries.json  # 拒答集 10 条（near_miss / out_of_domain / missing_*）
│   ├── multi_evidence_queries.json# 多证据集 3 条（每条需跨 2 篇文档才能答全）
│   └── knowledge_triples.jsonl    # KG 三元组缓存（可选实验功能 ENABLE_KG）
│
├── docs/                      # 项目文档（人读，不参与索引）
│   ├── 项目详解.md / 技术视角详解.md / 开发过程中遇到的问题.md
│   ├── chapters/              #   ch01～ch10 逐层实现笔记
│   └── reference/             #   外围知识参考（RAG核心概念 / Embedding选型指南 /
│                              #     切片策略 / AST分块方案 / Docker部署说明）
├── journal/                   # 过程笔记（踩坑日志 / 实验记录 / 进度清单，强时间序）
├── Dockerfile                 # 单容器镜像（CPU 版 torch + 构建期从 ModelScope 烧入模型）
├── docker-compose.yml         # 一条命令拉起，含卷挂载与健康检查
├── experiments/               # 实验结果 JSON + 评测隔离向量库
└── chroma_db/                 # ChromaDB 持久化向量库（主）
```

</details>

## 关键数据

> 基于 18 条四类分层 Golden Test Set（exact_match / semantic / mixed / graph），每条 `golden_answer` 都能在 `golden_chunk_sources` 指定的文件正文里回查到原文依据（`python scripts/validate_golden.py` 校验，当前 0 问题）。
> 以下为 2026-09-12 语料迁移至 `corpora/fastapi-zh` 后的**当前基线**。语料更换后旧基线（基于项目自身 `docs/`）已失效，实验过程详见 [KG 消融实验笔记](journal/KG消融实验-2026-08-04.md)。

### 检索侧：Test 集（18 条）各阶段指标

| 阶段 | MRR@5 | Hit@5 | Prec@5 | Recall@5 |
|------|-------|-------|--------|----------|
| BM25 | 0.5469 | 0.8889 | 0.4333 | 0.8889 |
| Vector | 0.7246 | 0.8333 | 0.3889 | 0.8333 |
| RRF 融合 | 0.7833 | 0.9444 | 0.5000 | 0.9444 |
| CE 精排（全管线） | **0.7889** | **1.0000** | **0.5444** | **1.0000** |

**怎么读这张表：**

- **两路的失败集合几乎不重叠**：向量把 S04 从 MRR 0.000 拉到 1.000，BM25 则在 Vector 掉到 0.000 的 S02/S06 上仍有召回——这是"混合召回"最硬的证据，不是感觉上需要两路，而是数据上两路的盲区互补
- **RRF 补短板，不加上限**：单路 Hit@5 88.9% / 83.3% → 融合 94.4%，它是唯一把两路各自漏掉的单条同时补回的环节
- **CE 的价值在 Hit@5，不在 MRR**：Hit@5 补到 **100%**，但净 MRR 只 +0.006（救回 E02 0.33→1.0，也把 G08/M04 从 1.0 挪到 0.25）——精排的定位是"兜底 + 排序"，不是"提分"
- **延迟剖析（2026-09-15 用新语料重测；2026-09-18 CE 截断优化后复测）**：**CE 精排是绝对瓶颈** ——
  改前基线（全量候选）单条 query 全链路 **12.8 秒**，其中 CE **12.76 秒**（占 99.8%）；
  BM25 / 向量 / 图检索 / RRF 四项合计仅 **56 ms**。
  改后（RRF 之后截断到 30 候选送 CE，2026-09-18 复测）CE 降至 **7.3 秒**（降约 43%），Hit@5 仍为 100%。
  ⚠️ 旧记录（全链路 395ms、CE 占 94%）**不可沿用** —— 那是旧语料（3 个 chunk）+ 英文
  `cross-encoder/ms-marco-MiniLM-L-6-v2` 的产物，两个变量同时变了。
  ⚠️ 另发现：文档所称"CE 限制在 Top-20 候选内"**与代码不符** —— `rerank` 的 `top_k`
  只控制**输出条数**，`pairs` 是对**全部候选**做前向（`src/fusion.py:122-123`）。
  优化方向：RRF 之后先截断再送 CE —— 已于 2026-09-18 落地（`CE_CANDIDATE_K=30`，`src/fusion.py` 的 `rerank()` 在 pairs 构造前截断候选），复测 CE 从 12.8s 降至 7.3s（降约 43%），top-5 精度零回归。详见 [A 组验收结果](journal/端到端验收结果-2026-09-15.md) §6

> 详细归因与单条 case 追踪见 [ch09 消融实验与数据分析](docs/chapters/ch09-消融实验与数据分析.md)。

### 生成侧：端到端验收（2026-09-15）

| 验收 | 结果 |
|---|---|
| 18 条可答 · **生成率** | **100%（18/18）** |
| 18 条可答 · 三维分数（**7b 裁判**） | Faithfulness **0.856** / Answer Relevancy **0.794** / Context Recall **0.600** |
| 10 条拒答 · **拒答率** | **90%**（near_miss 类 5/5 全对） |
| 误答率（幻觉风险） | 10% —— 唯一失手是**状态误判**：答案正文已写"未在文档中找到"，但 `status` 填了 `answered` |
| 3 条多证据 · 跨文档引用率 | **0%** |
| 关键词覆盖：答案 / 检索证据 | **50% / 83.3%** ← 检索没漏，是**生成层**没把两处整合起来 |

> 三维分数**以 7b 裁判版为准** —— 同一批答案用 3b 当裁判会打安全分
> （18 条里 13 条 Faithfulness 都是整 0.70，无区分度）。
> 完整结果、两个评测坑（3b 裁判无区分度 / `num_predict=512` 截断导致静默 0 分）与面试口径，
> 见 [A 组验收结果](journal/端到端验收结果-2026-09-15.md)。

**知识图谱检索（可选实验）**：用 LLM 抽取实体三元组、构建有向知识图谱，作为混合检索的一路补充。`ENABLE_KG` 开关控制。当前在**小规模、同主题语料**上未带来检索提升（独立评估见 [KG 消融实验笔记](journal/KG消融实验-2026-08-04.md)），因此默认关闭——它展示的是「多路召回设计」的扩展能力，而非当前主推的检索路线。

## 技术栈

| 层次 | 技术 |
|------|------|
| 语言 | Python 3.11+ |
| Embedding | BAAI/bge-small-zh-v1.5（512 维，中文优化） |
| Cross-Encoder | BAAI/bge-reranker-base（中文优化） |
| 向量库 | ChromaDB（HNSW 索引, cosine 距离） |
| 关键词检索 | rank-bm25 + jieba 分词 |
| 图检索 | NetworkX 实体共现图 + 子图遍历 |
| 融合 | RRF（Reciprocal Rank Fusion, k=60） |
| 代理 | LangGraph（声明式状态机, 条件路由） |
| LLM（生成） | Ollama + qwen2.5:3b（纯 CPU 推理；7b 在本机开发环境会 OOM） |
| LLM（裁判） | Ollama + qwen2.5:7b（`JUDGE_MODEL`，仅跑评测时加载） |
| 评测 | 自实现 LLM-as-Judge（Faithfulness / Answer Relevancy / Context Recall） |
| MCP | FastMCP 2.0（stdio 传输） |
| UI | Streamlit |

## 文档导航

| 我 想... | 读这篇 |
|---------|--------|
| 了解这个项目做了什么、怎么用的 | [docs/项目详解.md](docs/项目详解.md) |
| 深入技术细节和架构决策 | [docs/技术视角详解.md](docs/技术视角详解.md) |
| 系统学习每一层的实现笔记 | [docs/chapters/](docs/chapters/) |
