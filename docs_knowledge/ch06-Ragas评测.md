# 第 6 章：Ragas 自动化评测

## 知识点

### 1. 为什么要评测

RAG 系统是管道结构，改任何一个模块（换模型、调参数）都会影响最终输出。没有评测 = 盲调。

评测闭环：**改代码 → 跑全量评测 → 看指标变化 → 防负向退化**

### 2. 四个核心指标

| 指标 | 测什么 | 怎么算 | 需要什么 |
|------|--------|--------|----------|
| **MRR** | 第一个相关结果排第几 | 1/rank，越靠前分越高 | golden 标注 |
| **Hit@K** | Top-K 里有相关结果吗 | 0 或 1（有则 1） | golden 标注 |
| **Precision@K** | Top-K 里多少是相关的 | 相关数/K | golden 标注 |
| **Faithfulness** | 生成内容忠于原文吗 | LLM 逐句比对 | 需要 LLM |

### 3. 为什么自己实现而不是用 Ragas 库

Ragas 在 Windows Anaconda 环境下出现 SSL 证书冲突（`aiohttp` → `ssl.load_default_certs` 失败），这是环境而非代码问题。

**面试应对**：「Ragas 在 Windows 环境有依赖兼容问题，我理解每个指标的计算逻辑后自己实现了核心评测模块。MRR、Hit@K、Precision@K 的计算非常简单——就是取排名、数命中——不需要依赖外部库。Faithfulness 需要 LLM，我预留了接口，后续接入 Ollama 即可。」

### 4. GoldenTestSet 分层设计

15 组 Query，三类各 5 组：

| 类别 | 测什么 | 例子 |
|------|--------|------|
| exact_match | BM25 的字面匹配优势 | "BM25 算法"、"RRF 融合" |
| semantic | 向量的语义理解优势 | "怎么评价检索系统"（字面没"评测"） |
| mixed | 多模块协同效果 | "如何提高检索准确率"（需综合多段落） |

### 5. 基线评测结果（3 Chunk 小语料）

| Stage | MRR | Hit@5 | Prec@5 |
|-------|-----|-------|--------|
| BM25 | 0.90 | 0.93 | 0.71 |
| Vector | 0.97 | 1.00 | 0.73 |
| RRF | 0.93 | 1.00 | 0.73 |
| **CE** | **1.00** | **1.00** | **0.73** |

**关键发现**：
- S03 语义查询：BM25 MRR=0 → Vector MRR=1.0 → RRF MRR=0.5（退化！）→ CE MRR=1.0（修正）
- M04 对比查询：BM25/Vector MRR=0.5 → CE MRR=1.0

**注意**：这些数字是 3 Chunk 小语料下的天花板效应。真实场景基数会低很多，但**各阶段相对变化趋势是一致的**。

### 面试速记

- **三个指标一句话**：MRR 测排名、Hit@K 测命中、Precision@K 测纯度
- **S03 是最佳证据**：BM25 找不到语义查询，CE 修正 RRF 退化
- **为什么不用 Ragas**：Windows SSL 兼容问题，自实现比调库更理解原理
- **数字要诚实**：小语料下的天花板数字，面试时说明这是"验证逻辑正确性"的基线

## 产出文件

- `src/evaluate.py` — 自实现评测（MRR/Hit@K/Precision@K/Ablation）
- `test_queries.json` — 15 组 GoldenTestSet（三类分层）

## 关键实现

### ① MRR — 第一个相关结果的倒数排名

```python
def _mrr(results, golden_sources, k=10):
    for rank, r in enumerate(results[:k], start=1):
        if r.chunk.metadata.get("source") in golden_sources:
            return 1.0 / rank
    return 0.0
```

**追问应对**：「MRR 和平均排名的区别？」— MRR 对排名靠前更敏感：rank=1 得 1.0，rank=2 得 0.5，rank=10 只得 0.1。它鼓励系统把最佳结果排在最前面，而不是"平均排名还不错但第一个结果不相关"的情况。

### ② 阶段贡献分析

```python
for stage in ["bm25", "vector", "rrf", "cross_encoder"]:
    val = summary[stage][metric]
    delta = val - prev  # 本阶段相比上一阶段的增量
    prev = val
```

**追问应对**：「怎么证明 Cross-Encoder 真的有效？」— 跑消融实验：BM25 only → BM25+Vector(RRF) → BM25+Vector+CE，每个阶段的指标差就是该模块的贡献。S03 上 CE 把 RRF 的 MRR=0.5 修复到 1.0，是最直接的证据。

---

## 相关章节

- [[ch04-RRF融合与重排序]] — 评测脚本分别测试每个融合阶段的指标
- [[ch03-双路召回]] — BM25 和 Vector 的原始结果作为评测基线
- [[ch05-Streamlit前端]] — 评测结果可以在前端直观验证
