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
| 6 | Ragas 自动化评测 | ✅ 完成 | evaluate.py / test_queries.json |
| 7 | FastMCP 工具封装 | ✅ 完成 | mcp_server.py |
| 8 | LangGraph Agent 编排 | ✅ 完成 | agent.py |
| 9 | 消融实验与数据分析 | ✅ 完成 | experiments/ + src/experiments.py |
| 10 | 面试复盘 | ⬜ 未开始 | interview_notes.md |

## 当前进度

- **当前章节**：第 9 章 ✅ 完成
- **下一章**：第 10 章（面试复盘）
- **上一次产出**：src/experiments.py（五维度消融实验 + 6 份实验数据文件）

## 重启指南

新窗口启动后，说：**"继续 PROGRESS.md 里的 RAG 项目，从第 10 章开始"**

必要的初始化命令（新窗口需执行）：
```bash
cd "d:/1base/computer/Agent/DevRoot/mcp-rag-hub"
export KMP_DUPLICATE_LIB_OK=TRUE
export PIP_CACHE_DIR=/d/pip_cache
export TMPDIR=/d/tmp
```

## 环境配置速查

| 配置项 | 值 | 位置 |
|--------|-----|------|
| HF 模型缓存 | `D:/huggingface_cache` | `src/__init__.py` |
| pip 缓存 | `D:/pip_cache` | 启动时 export |
| ChromaDB 持久化 | `chroma_db/`（项目根目录） | `config.py` |
| C 盘剩余空间 | ~968KB（严禁写入大文件） | — |

## 已知问题

- **Ragas 不可用**：Windows Anaconda SSL 证书冲突（`aiohttp` → `ssl.load_default_certs`）。已自实现评测模块替代，见 `src/evaluate.py`
- **Ollama 未安装**：LangGraph Agent 章节（第 8 章）需先 `ollama pull qwen2.5:7b`

## 变更日志

| 日期 | 内容 |
|------|------|
| 2026-07-01 | 初始化项目，确定架构方案，创建进度追踪文件 |
| 2026-07-01 | 第 1 章完成：项目骨架、数据结构定义、环境搭建（含 Windows 兼容修复） |
| 2026-07-01 | 从 MetaFetch-RAG 迁移：测试数据文件 + 吸收设计模式 |
| 2026-07-01 | 第 2 章完成：文档加载与切片管线（tiktoken token 级切片 / 编码检测 / Markdown 面包屑） |
| 2026-07-01 | 修复 OMP 重复加载 + HF 缓存：环境变量统一移至 src/__init__.py |
| 2026-07-01 | 第 3 章完成：BM25 + ChromaDB 双路召回（手动 embedding / 原始分数不归一化） |
| 2026-07-01 | 第 4 章完成：RRF 融合 + Cross-Encoder 重排序（5 行 RRF / Bi vs Cross 两阶段策略） |
| 2026-07-01 | 第 5 章完成：Streamlit 前端（四 Tab 对比 / cache_resource 缓存管线） |
| 2026-07-01 | 第 6 章完成：自实现评测 MRR/Hit@K/Precision@K + 15 组分层 GoldenTestSet |
| 2026-07-01 | 评测基线数据：BM25 MRR=0.90 → Vector 0.97 → CE 1.00（S03 为关键证据） |
| 2026-07-01 | 第 7 章完成：FastMCP 工具封装（4 个 Tool / 懒加载管线 / stdio transport） |
| 2026-07-01 | 第 8 章完成：LangGraph Agent 编排（5 节点 / 条件边 / 查询改写 / Ollama fallback） |
| 2026-07-01 | 第 9 章完成：五维度消融实验（模块隔离/分类别/参数扫描/延迟剖析/Query追踪） |
