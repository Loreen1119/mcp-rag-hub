# 项目简历稿：RAG 智能知识检索系统

> 2026-08-06 重写：旧版（三路混合检索 / MRR 0.967）与当前代码和实验数据不符，已全部替换为代码核实口径。
> 数据来源：`journal/2026-08-04-kg-ablation-notes.md` 附录二（最终基线）、`experiments/` 实验文件。

**时间：2025.12 - 至今 | GitHub 开源**

**技术栈：** Python / ChromaDB / LangGraph / BM25 / RRF / Cross-Encoder / FastMCP / Ollama / Streamlit

## 项目介绍

面向企业知识库中关键词匹配与语义搜索难以兼顾的痛点，不依赖现成 RAG 框架，手工实现 "BM25 + 向量双路召回 → RRF 融合 → Cross-Encoder 精排" 的两阶段检索系统，并通过 LangGraph + FastMCP 封装为 Agent 可调用的检索微服务。

## 核心职责

1. **双路召回与融合排序：** 实现 jieba + BM25Okapi 与 all-MiniLM-L6-v2 + ChromaDB（HNSW / cosine）双路召回，手写 RRF 算法（k=60，支持分路权重）以排名替代得分消除量纲差异；两阶段架构将 Cross-Encoder 精排限制在 Top-20 候选内（参数网格搜索调优确定），平衡精度与其占全链路 94% 的延迟开销。
2. **多格式文档解析管线：** 支持 PDF（pdfplumber）/ Markdown / TXT / Python 源码；Markdown 标题栈面包屑注入章节溯源，tiktoken Token 级贪心滑窗切块，Python AST 函数/类边界语义分块，编码自动检测降级链兼容遗留文档。
3. **LangGraph 自纠错 Agent：** 构建 analyze → retrieve → check → rewrite → generate 五节点状态机，按 CE 分数阈值条件路由，低置信结果触发查询改写重检（上限 2 次）；本地 Ollama qwen2.5:7b 生成，LLM 不可用时自动降级为规则式回答。
4. **评测体系与数据驱动决策：** 自建 36 条四类分层 Golden Test Set（train/test 严格分离），手写 MRR / Hit@K / Precision@K / Recall@K 指标与 LLM-as-Judge 三维生成评测；通过对照实验定位知识图谱路在同质语料上的失效根因（清理噪声文档后 Hit@5 从 33% 升至 100%，但 MRR 天花板仍低），据此将 KG 降级为 `ENABLE_KG` 可选开关。
5. **MCP 微服务化：** 基于 FastMCP 2.0 封装 4 个标准工具（search_knowledge / list_documents / get_chunk / get_chunk_count），stdio 传输，可接入任意 MCP 客户端。

## 项目成效（Test 集 18 条，2026-08-04 最终基线）

- RRF 融合阶段 **Hit@5 = 100%**、**Recall@5 = 0.97**、MRR = 0.86；BM25 单路 MRR = 0.87
- CE 精排将 Precision@5 从 0.49 提升至 **0.56**
- 全链路延迟 395ms（本地 CPU），CE 精排占 94%（373ms）；BM25 0.38ms、RRF 0.06ms 近乎免费
