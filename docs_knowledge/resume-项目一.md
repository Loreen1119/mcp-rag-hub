# 项目简历稿：RAG 智能知识检索系统

> 2026-09-12 更新：检索基线已随语料迁移到真实第三方文档（`corpora/fastapi-zh`，FastAPI 中文文档）后重跑，旧基线（基于项目自身 `docs/` 的 2 文档 3 chunk）作废，本稿所有指标均为当前代码实测。
> 数据来源：`data/test_queries.json`（18 条）+ `src/evaluation/retrieval_eval.py` 实测输出；归因与单条 case 追踪见 `docs_knowledge/chapters/ch09-消融实验与数据分析.md`。

**时间：2025.12 - 至今 | GitHub 开源**

**技术栈：** Python / ChromaDB / LangGraph / BM25 / RRF / Cross-Encoder / FastMCP / Ollama / Streamlit

## 项目介绍

面向企业知识库中关键词匹配与语义搜索难以兼顾的痛点，不依赖现成 RAG 框架，手工实现 "BM25 + 向量双路召回 → RRF 融合 → Cross-Encoder 精排" 的两阶段检索系统，并通过 LangGraph + FastMCP 封装为 Agent 可调用的检索微服务。

## 核心职责

1. **双路召回与融合排序：** 实现 jieba + BM25Okapi 与 bge-small-zh-v1.5 + ChromaDB（HNSW / cosine）双路召回（各返回 Top-20），手写 RRF 算法（k=60，支持分路权重）以排名替代得分、消除两路量纲差异；两阶段架构只在 RRF 融合候选内做 Cross-Encoder（bge-reranker-base）精排并返回 Top-5，避免对全库逐对打分——实测两路失败集合几乎不重叠：向量补回 BM25 完全漏掉的纯语义查询，BM25 则覆盖向量漏掉的专有名词查询。
2. **多格式文档解析管线：** 支持 PDF（pdfplumber）/ Markdown / TXT / Python 源码；Markdown 标题栈面包屑注入章节溯源，tiktoken Token 级贪心滑窗切块，Python AST 函数/类边界语义分块，编码自动检测降级链兼容遗留文档。
3. **LangGraph 自纠错 Agent：** 构建 analyze → retrieve → check → rewrite → generate 五节点状态机，按 CE 分数阈值（0.3，reranker sigmoid 口径）条件路由，低置信结果触发查询改写重检（上限 2 次）；本地 Ollama qwen2.5:3b 生成（纯 CPU 可跑的窄任务模型），LLM 不可用时自动降级为检索证据拼接。
4. **评测体系与数据驱动决策：** 自建 18 条四类分层 Golden Test Set（每条 golden 依据可回溯到源文件，脚本自动校验），手写 MRR / Hit@K / Precision@K / Recall@K 四级累积指标与 LLM-as-Judge 三维生成评测；并以一次真实的"设计被数据推翻"完成迭代——原以 CE 绝对分数做拒答门控，实测可答与不可答两组的分数分布几乎完全重叠，遂删除该常量、把拒答交给 LLM 语义判断；另通过对照实验定位知识图谱路在同质语料上无增量收益，据此将 KG 降级为 `ENABLE_KG` 可选开关（默认关闭）。
5. **MCP 微服务化：** 基于 FastMCP 2.0 封装 4 个标准工具（search_knowledge / list_documents / get_chunk / get_chunk_count），stdio 传输，可接入任意 MCP 客户端。

## 项目成效（语料 `corpora/fastapi-zh` 122 篇 / 1445 chunk，18 条四类分层测试集）

- 四级累积 MRR：BM25 0.5469 → 向量 0.7246 → RRF 0.7833 → CE **0.7889**，每一级都在加分
- **RRF 补召回**：把单路 Hit@5（88.9% / 83.3%）补到 **94.4%**，是唯一同时补回两路各自漏掉 case 的环节
- **CE 补兜底**：Hit@5 由 94.4% 补到 **100%**、Precision@5 由 0.5000 升到 **0.5444**；但净 MRR 仅 +0.0056（救回 E02 0.33→1.0，也把 G08/M04 挪后），据此把它的定位写成"排序 + 兜底"而非"提分"
- 全链路延迟：⚠️ 待重测（旧值基于英文 CE 模型，已换为 bge-reranker-base，不可沿用）
