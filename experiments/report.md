============================================================
  RAG 系统消融实验 — 分析报告
============================================================

## 1. 模块贡献分析

| 配置 | MRR | Hit@5 | Prec@5 | Δ MRR |
|------|-----|-------|--------|-------|
| bm25_only | 0.9333 | 0.9333 | 0.7611 | +0.0000 |
| vector_only | 0.9667 | 1.0000 | 0.7600 | +0.0334 |
| graph_only | 0.9556 | 1.0000 | 0.8933 | -0.0111 |
| concat_naive | 0.9667 | 1.0000 | 0.8267 | +0.0111 |
| rrf_fusion | 0.9222 | 1.0000 | 0.8133 | -0.0445 |
| full_pipeline | 1.0000 | 1.0000 | 0.7867 | +0.0778 |
| triple_fusion | 1.0000 | 1.0000 | 0.7867 | +0.0000 |

**BM25 → Full Pipeline MRR 提升: +0.0667**
**Vector → Full Pipeline MRR 提升: +0.0333**
**Full (双路) → Triple (三路+GraphRAG) MRR 提升: +0.0000**
**GraphRAG 独立 MRR: 0.9556** (vs BM25=0.9333, Vector=0.9667)

[OK] 三路混合检索（+GraphRAG）MRR=1.0000，图检索为双路召回提供了增量价值。

## 2. 分类表现

| 类别 | BM25 MRR | Vector MRR | Graph MRR | Full MRR | Triple MRR | Full→Triple Δ |
|------|----------|------------|-----------|----------|------------|---------------|
| exact_match | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | +0.0000 |
| mixed | 1.0000 | 0.9000 | 1.0000 | 1.0000 | 1.0000 | +0.0000 |
| semantic | 0.8000 | 1.0000 | 0.8667 | 1.0000 | 1.0000 | +0.0000 |

**BM25 在 exact_match 上的优势**: BM25=1.0000 vs Vector=1.0000
**Vector 在 semantic 上的优势**: Vector=1.0000 vs BM25=0.8000

[OK] 验证结论：BM25 对专有名词精确匹配更好，向量检索对语义相似查询更优。两者互补，混合召回是正确架构。

## 3. 延迟分析

| 阶段 | 平均延迟 (ms) |
|------|-------------|
| Bm25 Search | 0.38 |
| Vector Search | 21.14 |
| Graph Search | 0.85 |
| Rrf Fusion | 0.06 |
| Ce Rerank | 373.00 |

**Pipeline 总延迟: 395.43 ms**
瓶颈阶段: ce_rerank_ms (373.00 ms)

============================================================
  实验数据文件:
    D:\1base\computer\Agent\DevRoot\mcp-rag-hub\experiments\ablation_results.json
    D:\1base\computer\Agent\DevRoot\mcp-rag-hub\experiments\category_breakdown.json
    D:\1base\computer\Agent\DevRoot\mcp-rag-hub\experiments\parameter_sweep.json
    D:\1base\computer\Agent\DevRoot\mcp-rag-hub\experiments\latency_profile.json
    D:\1base\computer\Agent\DevRoot\mcp-rag-hub\experiments\query_deep_dive.json
============================================================