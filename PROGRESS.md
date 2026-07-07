# RAG 智能知识检索系统 — 项目进度

## 项目概述

搭建高性能 RAG 智能知识检索系统，支持 BM25 + ChromaDB 双路召回、RRF 融合、Cross-Encoder 重排，Ragas 评测，FastMCP 封装，LangGraph Agent 编排。

## 关键决策

- **Embedding 方式**：手动 sentence-transformers 做 embedding 再存 ChromaDB（不用 ChromaDB 内置），便于面试讲清向量生成过程
- **LLM 选择**：优先本地 Ollama 模型，LangGraph Agent 章节可先出代码框架 + 状态图，不强求调通
- **PDF 解析**：pdfplumber 提取文本，不处理表格/图片
- **Chunk 策略**：512 token / 128 overlap 滑窗切片，使用 tiktoken (cl100k_base) 做精确 token 计数
- **分词**：中文用 jieba 分词后送入 BM25（比字符级 n-gram 更准确）
- **Markdown 增强**：解析标题层级，为每个 Chunk 附加上下文面包屑（继承自旧版 MetaFetch-RAG 的设计）
- **编码兼容**：自动检测 UTF-8/GBK/GB2312，兼容中文文档（继承自旧版设计）
- **旧版参考**：`../MetaFetch-RAG/` 包含 v0.1 手写原型（HashEmbedder + InMemoryVectorStore），用于面试讲述"从原理到生产"的演进路径
- **Cross-Encoder 模型**：`cross-encoder/ms-marco-MiniLM-L-6-v2`
- **Embedding 模型**：`all-MiniLM-L6-v2`（384 维）
- **环境注意事项**：
  - PyTorch 通过 conda 安装（`conda install pytorch cpuonly -c pytorch`），Windows DLL 兼容性更好
  - `config.py` 开头处理了 OpenMP 重复加载问题（`KMP_DUPLICATE_LIB_OK=TRUE`）
  - 已验证的版本组合：PyTorch 2.5.1 + transformers 4.44.2 + sentence-transformers 2.7.0

## 文件结构

```
mcp-rag-hub/
├── app.py                     # Streamlit 前端入口（第5章）
├── agent.py                   # LangGraph Agent 入口（第8章）
├── config.py                  # 全局配置中心
├── requirements.txt
├── PROGRESS.md                # 本文件
│
├── src/
│   ├── __init__.py
│   ├── models.py              # Chunk / RetrievalResult 数据结构
│   ├── data_pipeline.py       # 文档加载 + 切片（第2章）
│   ├── retrievers.py          # BM25 + ChromaDB 双路召回（第3章）
│   ├── fusion.py              # RRF 融合 + Cross-Encoder 重排（第4章）
│   ├── mcp_server.py          # FastMCP 工具封装（第7章）
│   └── evaluation/            # 评测子包
│       ├── __init__.py
│       ├── metrics.py         # 共享评测指标
│       ├── retrieval_eval.py  # 检索质量评测（第6章）
│       ├── llm_eval.py        # LLM-as-Judge 生成评测
│       ├── agent_eval.py      # Agent 改写评测
│       └── experiments.py     # 消融实验与数据分析（第9章）
│
├── data/
│   └── test_queries.json      # GoldenTestSet（第6章）
│
├── docs/                      # 测试文档
│   ├── sample_rag_paper.md    # RAG 综述（Markdown, 含标题层级）
│   └── sample_notes.txt       # 项目笔记（纯文本）
│
├── docs_knowledge/            # 项目文档与章节笔记
│   ├── README.md              # 文档导航
│   ├── 项目详解.md             # 面向非技术读者的项目介绍
│   ├── 技术视角详解.md         # 面向技术读者的架构详解
│   └── chapters/              # 面试复习笔记（10章）
│
├── experiments/               # 消融实验数据（JSON）
└── chroma_db/                 # ChromaDB 持久化向量库（gitignore）
```

## 章节目录与进度

| 章 | 内容 | 状态 | 产出 |
|----|------|------|------|
| 1 | 项目骨架与数据模型 | ✅ 完成 | models.py / config.py / requirements.txt |
| 2 | 文档加载与切片管线 | ✅ 完成 | data_pipeline.py |
| 3 | BM25 + ChromaDB 双路召回 | ✅ 完成 | retrievers.py |
| 4 | RRF 融合 + Cross-Encoder 重排 | ✅ 完成 | fusion.py |
| 5 | Streamlit 前端 | ✅ 完成 | app.py |
| 6 | 检索评测（MRR/Hit@K/Precision@K） | ✅ 完成 | evaluation/retrieval_eval.py / data/test_queries.json |
| 7 | FastMCP 工具封装 | ✅ 完成 | mcp_server.py |
| 8 | LangGraph Agent 编排 | ✅ 完成 | agent.py |
| 9 | 消融实验与数据分析 | ✅ 完成 | experiments/ + evaluation/experiments.py |
| 10 | 面试复盘 | ✅ 完成 | docs_knowledge/chapters/ch10-面试复盘.md |
| — | **Ollama LLM 生成评测（第 6 章增强）** | ✅ 完成 | evaluation/llm_eval.py / qwen2.5:7b |

## 当前进度

- **1~10 章全部完成**，核心检索链路 + 前端 + 评测 + MCP + Agent + 消融实验 + LLM 生成评测 + 面试复盘均已落地
- LLM 评测终版：qwen2.5:7b v1（F=0.95 / AR=0.86 / CR=0.79）
- docs_knowledge/ Ch02~Ch07、Ch10 已改写，Ch01/Ch08/Ch09 保持原始技术风格
- **唯一待办**：重跑 `python src/llm_evaluate.py` 覆盖 `experiments/llm_evaluation_results.json`（当前为 v2 废弃数据）

## LLM 生成评测结果（qwen2.5:7b v1 终版）

| 指标 | Mean | Min | Max |
|------|:----:|:---:|:---:|
| Faithfulness | **0.9533** | 0.7 | 1.0 |
| Answer Relevancy | **0.86** | 0.7 | 1.0 |
| Context Recall | 0.7933 | 0.7 | 1.0 |

| 类别 | Faith | Relev | Recall |
|------|:-----:|:-----:|:------:|
| exact_match | **1.0** | 0.76 | 0.76 |
| mixed | 0.9 | **0.98** | 0.74 |
| semantic | 0.96 | 0.84 | **0.88** |

> v2（Prompt 收紧至 200 字/禁止推论）已废弃——E01 Faithfulness 暴跌至 0.0，三个指标全降。终版为 v1 Prompt（300 字/允许自然关联技术说明）。

## 下一步任务

### ① ~~跑 LLM 生成评测 3B→7B 三代迭代~~ ✅ 已完成

- 3B 区分度差 → 7B 区分度良好 → v2 Prompt 收紧负优化 → 回退 v1
- 终版模型：qwen2.5:**7b**，纯 CPU 推理，全量 15 条约 1 小时
- 环境：Ollama 绿色版 `D:\ollama`，serve 端口 127.0.0.1:11434

### ② ~~第 10 章面试复盘~~ ✅ 已完成

产出 `docs_knowledge/ch10-面试复盘.md`，内容：
- 三段核心代码默写（RRF / BM25 / Cross-Encoder）+ 面试边写边说旁白
- 三个必问题逐字稿（BM25 vs 向量 / RRF vs 加权 / Bi-Encoder vs Cross-Encoder）
- 30 秒项目电梯演讲
- 10 个追问方向（原理层 3 + 实现层 3 + 工程层 4）+ 速查清单

### ③ docs_knowledge/ 补齐

| 章 | 状态 |
|----|------|
| Ch01 | 原始技术风格（不改写） |
| Ch02 | ✅ Gemini 改写 |
| Ch03 | ✅ Gemini 改写 |
| Ch04 | ✅ Gemini 改写 |
| Ch05 | ✅ Claude 改写（融合 Gemini + 真实代码校对 + 面试话术） |
| Ch06 | ✅ Claude 改写（补全源码走读 + S03 降级与修复叙事 + 面试话术） |
| Ch07 | ✅ Claude 改写（修正虚构函数/假数字 + 微服务积木叙事 + 面试话术） |
| Ch08 | 原始技术风格（不改写） |
| Ch09 | 原始技术风格（不改写） |
| Ch10 | ✅ Claude 全量产出（面试复盘） |

### ④ ~~LLM 评测 v2 尝试~~ 已废弃

> Prompt 收紧效果负面（F:0.95→0.91, E01→0.0），已回退 v1。v2 数据留档 `experiments/llm_evaluation_results.json`，需重跑覆盖。

## 当前待办（2026-07-05）

- [ ] 重跑 `python src/evaluation/llm_eval.py` 覆盖 v2 废弃数据
