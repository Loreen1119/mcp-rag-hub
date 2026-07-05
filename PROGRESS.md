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
├── src/
│   ├── __init__.py
│   ├── models.py          # Chunk / RetrievalResult 数据结构
│   ├── data_pipeline.py   # 文档加载 + 切片（第2章）
│   ├── retrievers.py      # BM25 + ChromaDB 双路召回（第3章）
│   ├── fusion.py          # RRF 融合 + Cross-Encoder 重排（第4章）
│   ├── evaluate.py        # Ragas 评测（第6章）
│   └── mcp_server.py      # FastMCP 工具封装（第7章）
├── docs/                  # 测试文档
│   ├── sample_rag_paper.md    # RAG 综述（Markdown, 含标题层级）
│   └── sample_notes.txt       # 项目笔记（纯文本）
├── docs_knowledge/        # 每章知识点汇总
├── experiments/           # 消融实验数据
├── app.py                 # Streamlit 前端（第5章）
├── agent.py               # LangGraph Agent（第8章）
├── test_queries.json      # GoldenTestSet（第6章）
├── config.py              # 全局配置
├── requirements.txt
└── PROGRESS.md            # 本文件
```

## 章节目录与进度

| 章 | 内容 | 状态 | 产出 |
|----|------|------|------|
| 1 | 项目骨架与数据模型 | ✅ 完成 | models.py / config.py / requirements.txt |
| 2 | 文档加载与切片管线 | ✅ 完成 | data_pipeline.py |
| 3 | BM25 + ChromaDB 双路召回 | ✅ 完成 | retrievers.py |
| 4 | RRF 融合 + Cross-Encoder 重排 | ✅ 完成 | fusion.py |
| 5 | Streamlit 前端 | ✅ 完成 | app.py |
| 6 | 检索评测（MRR/Hit@K/Precision@K） | ✅ 完成 | evaluate.py / test_queries.json |
| 7 | FastMCP 工具封装 | ✅ 完成 | mcp_server.py |
| 8 | LangGraph Agent 编排 | ✅ 完成 | agent.py |
| 9 | 消融实验与数据分析 | ✅ 完成 | experiments/ + src/experiments.py |
| 10 | 面试复盘 | ✅ 完成 | docs_knowledge/ch10-面试复盘.md |
| — | **Ollama LLM 生成评测（第 6 章增强）** | ✅ 完成 | llm_evaluate.py / qwen2.5:7b |

## 当前进度

- **1~10 章全部完成**，核心检索链路 + 前端 + 评测 + MCP + Agent + 消融实验 + LLM 生成评测 + 面试复盘均已落地
- **已跑完**：Ollama LLM 评测 v1（qwen2.5:7b，全量 15 条），结果见 `experiments/llm_evaluation_results.json`
- **待跑**：LLM 评测 v1 重跑（当前 `experiments/llm_evaluation_results.json` 为 v2 废弃数据，需跑一次覆盖）

## LLM 生成评测结果（qwen2.5:7b v1）

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

> 关键发现：7B 区分度远超 3B；Faithfulness 0.95 说明系统几乎不编造；Context Recall ~0.7-0.8 是因为语料中确实缺少 BM25 公式、RRF 公式等细节。

## LLM 生成评测 v2 尝试（已废弃）

> v1 结果分析后尝试收紧 Prompt，但效果负面——E01 Faithfulness 暴跌至 0.0。
> 已回退至 v1，此为最终版本。v2 数据留档 `experiments/llm_evaluation_results.json`。

| # | 尝试 | 结果 | 结论 |
|---|------|------|------|
| 1 | 生成 Prompt 收紧（200字/禁止推论） | Faithfulness 0.95→0.91, E01 崩至 0.0 | 过度约束导致 7B 回答过于简短偏离原文 |
| 2 | golden_answer 从公式级收紧到概念级 | Context Recall 无变化 | 公式缺失本质是语料问题，改 golden_answer 无意义 |

> 终版结论：**v1 Prompt（300字/允许自然关联技术说明）** 是当前 7B + 3 Chunk 小语料下的最优平衡。

## 下一步任务（按优先级）

### ① ~~跑 LLM 生成评测~~ ✅ 已完成

- 模型：qwen2.5:**7b**（3B 区分度太差，已删除）
- 环境：Ollama 绿色版 `D:\ollama`，serve 端口 127.0.0.1:11434
- 全量 15 条耗时约 2 小时（纯 CPU 推理）

### ② ~~第 10 章面试复盘~~ ✅ 已完成

产出 [interview_notes.md](interview_notes.md)，内容：
- ✅ 三段核心代码默写（RRF / BM25 检索流程 / Cross-Encoder 精排流程）—— 从实际源码精简，白板手写级别
- ✅ 三个必问题逐字稿（BM25 vs 向量 / RRF vs 加权 / Bi-Encoder vs Cross-Encoder）—— 均含"原理→实现→数据"三段论
- ✅ 30 秒项目电梯演讲
- ✅ 追问应对逻辑（原理层 3 问 + 实现层 3 问 + 工程层 4 问）
- ✅ 面试前一天速查清单

产出见 `docs_knowledge/ch10-面试复盘.md`

### ③ docs_knowledge/ 补齐

| 章 | 状态 | 方式 |
|----|------|------|
| Ch01 | 原始技术风格 | — |
| Ch02 | ✅ Gemini 改写 | 大白话 + 面试话术双层风格 |
| Ch03 | ✅ Gemini 改写 | 大白话 + 面试话术双层风格 |
| Ch04 | ✅ Gemini 改写 | 大白话 + 面试话术双层风格 |
| Ch05 | ✅ Claude 改写 | 融合 Gemini 精华 + 真实代码校对 + 面试话术升级 |
| Ch06 | ✅ Claude 改写 | 融合 Gemini 精华（S03 降级与修复叙事）+ 补全源码走读 + 面试话术升级 |
| Ch07 | ✅ Claude 改写 | 融合 Gemini 精华（微服务积木/精装房vs裸接口）+ 修正虚构函数与假数字 + 面试话术升级 |
| Ch08 | 待改写 | 原始技术风格 |
| Ch09 | 待改写 | 原始技术风格 |

### ④ ~~LLM 评测 v2 重跑~~ → 已废弃，回退至 v1

> v2 Prompt 收紧效果负面，已回退。终版为 v1 结果（F=0.95 / AR=0.86 / CR=0.79）。

## 简历定稿（基于 9 章真实落地，无假数字）

```
项目一：RAG 智能知识检索系统
开发技术：Python / ChromaDB / BM25 / Cross-Encoder / Streamlit / LangGraph / FastMCP

项目介绍：
针对企业知识库专有名词匹配不准、语义召回缺失的痛点，搭建 BM25+向量双路召回、
RRF 融合、Cross-Encoder 重排的 RAG 检索系统。设计 Token 级切片管线，自实现
检索评测与消融实验体系，并通过 FastMCP+LangGraph 封装为 Agent 调用的检索微服务。

核心职责：
1. 实现 BM25 + ChromaDB 双路召回，手动实现 RRF 算法消除两路得分量纲差异；
   集成 Cross-Encoder 两阶段精排，消融实验验证 CE 在语义查询上从 MRR=0 修复至
   MRR=1.0，同时占总延迟 92%，体现召回-精度-延迟的工程权衡。
2. 设计 Token 级滑窗切片管线（tiktoken cl100k_base），集成 Markdown 标题面包屑
   解析与 UTF-8/GBK 编码降级检测链，基于贪心倒退算法保证重叠区语义完整。
3. 自实现 MRR/Hit@K/Precision@K 评测模块与 15 组分层 GoldenTestSet；跑通五维度消融
   实验（模块隔离/分类别/参数扫描/延迟剖析/Query 追踪），以数据驱动架构决策。
4. 通过 FastMCP 将检索链路封装为标准化 Tool 接口，基于 LangGraph 构建 5 节点条件
   路由 Agent（analyze→retrieve→check→rewrite/generate），支持多轮检索决策与查询改写。
```

## 消融实验核心数据（面试弹药库）

| 证据 | 数据 | 面试时讲什么 |
|------|------|-------------|
| S03 语义查询 | BM25 MRR=0 → Vector 1.0 → RRF 0.5 → CE 1.0 | "BM25 对语义查询完全盲视，CE 修正了 RRF 退化" |
| CE 延迟占比 | 214ms / 232ms = 92.3% | "CE 是最贵的模块，所以必须两阶段粗筛+精排" |
| BM25 vs Vector 分类别 | exact_match: 持平 / semantic: Vector 碾压 | "两者各有所长，数据证明混合召回必要性" |
| RRF 退化 | Vector MRR=0.967 → RRF MRR=0.933 | "RRF 在小语料下可能负优化，CE 是纠错器" |
| 参数不敏感 | RRF_K 30/60/120 全部 MRR=1.0 | "参数在合理范围内系统表现稳定，降低运维成本" |

## 环境配置速查

| 配置项 | 值 | 位置 |
|--------|-----|------|
| HF 模型缓存 | `D:/huggingface_cache` | `src/__init__.py` |
| pip 缓存 | `D:/pip_cache` | 启动时 export |
| ChromaDB 持久化 | `chroma_db/`（项目根目录） | `config.py` |
| C 盘剩余空间 | ~968KB（严禁写入大文件） | — |

## 已知问题

- **Ragas 不可用**：Windows Anaconda SSL 证书冲突，已自实现评测模块替代（`src/evaluate.py`）
- **Ollama 评测模型**：当前使用 qwen2.5:**7b**（3B 区分度太差已删除），纯 CPU 推理，全量 15 条约 1 小时
- **HuggingFace 离线模式**：`src/__init__.py` 已设 `HF_HUB_OFFLINE=1`，避免每次启动连接 huggingface.co 超时重试
- **C 盘空间不足**：所有大文件（模型缓存、pip 缓存、临时文件）已迁至 D 盘

## 新窗口启动指令

说：**"继续 PROGRESS.md"**

必要的初始化命令：
```bash
cd "d:/1base/computer/Agent/DevRoot/mcp-rag-hub"
export KMP_DUPLICATE_LIB_OK=TRUE
export PIP_CACHE_DIR=/d/pip_cache
export TMPDIR=/d/tmp
```

## 当前待办（2026-07-05）

- [ ] 重跑 LLM 评测 v1（`ollama serve` + `python src/llm_evaluate.py`）—— `experiments/llm_evaluation_results.json` 当前为 v2 废弃数据，需重跑覆盖为 v1 真实结果
- [x] 第 10 章面试复盘（docs_knowledge/ch10-面试复盘.md）
- [x] docs_knowledge/ Ch05、Ch06、Ch07 改写
- [x] docs_knowledge/ Ch02、Ch03、Ch04 Gemini 改写
- [ ] Ch08、Ch09 不改写（用户决定保持原始技术风格）
