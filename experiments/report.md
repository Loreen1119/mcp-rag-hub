============================================================
  RAG 系统消融实验 — 分析报告
============================================================

## 1. 模块贡献分析

| 配置 | MRR | Hit@5 | Prec@5 | Δ MRR |
|------|-----|-------|--------|-------|
| bm25_only | 0.5469 | 0.8889 | 0.4333 | +0.0000 |
| vector_only | 0.7246 | 0.8333 | 0.3889 | +0.1777 |
| graph_only | 0.0958 | 0.1111 | 0.0222 | -0.6288 |
| concat_naive | 0.5770 | 0.8333 | 0.4333 | +0.4812 |
| rrf_fusion | 0.7833 | 0.9444 | 0.5000 | +0.2063 |
| full_pipeline | 0.7889 | 1.0000 | 0.5444 | +0.0056 |
| triple_fusion | 0.7889 | 1.0000 | 0.5333 | +0.0000 |

**BM25 → Full Pipeline MRR 提升: +0.2420**
**Vector → Full Pipeline MRR 提升: +0.0643**
**Full (双路) → Triple (三路+GraphRAG) MRR 提升: +0.0000**
**GraphRAG 独立 MRR: 0.0958** (vs BM25=0.5469, Vector=0.7246)

[WARN] 三路混合检索（+GraphRAG）MRR=0.7889，相对双路 +0.0000，**未观察到增量**。图检索在此语料上不构成收益，ENABLE_KG 默认关闭是合理的。

## 2. 分类表现

| 类别 | BM25 MRR | Vector MRR | Graph MRR | Full MRR | Triple MRR | Full→Triple Δ |
|------|----------|------------|-----------|----------|------------|---------------|
| exact_match | 0.8000 | 0.8000 | 0.0312 | 1.0000 | 1.0000 | +0.0000 |
| graph | 0.7083 | 0.8571 | 0.0833 | 0.7917 | 0.7917 | +0.0000 |
| mixed | 0.3611 | 0.6750 | 0.2500 | 0.8125 | 0.8125 | +0.0000 |
| semantic | 0.2375 | 0.5000 | 0.0250 | 0.5500 | 0.5500 | +0.0000 |

**exact_match 类**: BM25=0.8000 vs Vector=0.8000 → 两者基本持平
**semantic 类**: Vector=0.5000 vs BM25=0.2375 → Vector 更高（+0.2625）

[WARN] 向量检索在语义类更高（符合预期），但 BM25 未在精确匹配类占优 —— 两路互补只验证了一半。

## 3. 延迟分析

| 阶段 | 平均延迟 (ms) |
|------|-------------|
| Bm25 Search | 5.23 |
| Vector Search | 18.25 |
| Graph Search | 7.08 |
| Rrf Fusion | 0.18 |
| Ce Rerank | 12763.11 |

**Pipeline 总延迟: 12793.85 ms**
瓶颈阶段: ce_rerank_ms (12763.11 ms)

============================================================
  实验数据文件:
    D:\1base\computer\Agent\DevRoot\mcp-rag-hub\experiments\ablation_results.json
    D:\1base\computer\Agent\DevRoot\mcp-rag-hub\experiments\category_breakdown.json
    D:\1base\computer\Agent\DevRoot\mcp-rag-hub\experiments\parameter_sweep.json
    D:\1base\computer\Agent\DevRoot\mcp-rag-hub\experiments\latency_profile.json
    D:\1base\computer\Agent\DevRoot\mcp-rag-hub\experiments\query_deep_dive.json
============================================================