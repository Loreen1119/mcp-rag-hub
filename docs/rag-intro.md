# RAG 核心概念速览

> 面向 mcp-rag-hub 项目的关键知识索引。更系统的入门学习请移步 [docs_knowledge/](../docs_knowledge/README.md)。

## Retriever 与 Generator 的配合

Retriever（检索器）和 Generator（生成器）是 RAG 的两大支柱，二者的协作质量直接决定系统效果。

**Retriever 负责"找对东西"**。常用策略：

- **稀疏检索（Sparse）**：如 BM25，基于词频/逆文档频率，适合精确术语匹配。
- **稠密检索（Dense）**：如 BGE-M3 向量检索，捕捉语义相关性。
- **混合检索（Hybrid）**：Sparse + Dense 双路召回，通过 RRF 融合结果。

**Generator 负责"理解并回答"**，表现受上下文窗口大小、指令遵循能力、对齐信息的能力影响。如果检索返回了无关片段，模型可能被误导——因此现代 RAG 会引入 **Cross-Encoder 重排序** 对检索结果二次筛选。

mcp-rag-hub 已实现：BM25 + 向量 + 实体共现图三路召回 → RRF 融合 → Cross-Encoder 重排。

## Naive RAG 的局限

基础 RAG 的"检索 → 拼接 → 生成"流程存在明显短板：

1. **语义鸿沟**：查询与文档表述差异大时，向量检索可能漏掉相关内容。
2. **上下文稀释**：Top-K 过大时，关键信息被无关内容挤占。
3. **多跳推理无力**：单次检索无法回答需要综合多个片段推理的对比性问题。
4. **缺乏纠错**：生成错误后无反馈回路。

## 进阶策略

mcp-rag-hub 对标的主要技术方向：

| 策略 | mcp-rag-hub 对应 |
|------|------------------|
| 查询改写（Query Rewriting） | LangGraph Agent 中的查询改写节点 |
| 子问题拆分（Decomposition） | Agent 编排中的条件路由 |
| 重排序（Re-ranking） | Cross-Encoder 精排 |
| 混合检索（Hybrid Retrieval） | BM25 + Dense + RRF 融合 |
| 图增强（Graph RAG） | 实体共现图三路召回 |
| 迭代检索（Iterative RAG） | LangGraph 状态机驱动的多轮检索 |
