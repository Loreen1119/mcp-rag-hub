# 第 4 章：RRF 融合 + Cross-Encoder 重排序

## 知识点

### 1. RRF — 倒数排名融合

**公式**：`RRF(d) = Σ 1/(k + rank(d))`

- `rank(d)`：文档 d 在第 i 路检索中的排名（从 1 开始）
- `k`：平滑常数，默认 60

**为什么不用直接加权求和**：
- BM25 得分为 0~几十，向量余弦相似度为 0~1
- 直接加权 → BM25 分数碾压向量分数，向量检索形同虚设
- RRF 只看排名不看绝对分数，天然消除量纲差异

**k=60 为什么**：
- 防止排名第一的文档权重过高：`1/(60+1)` vs `1/(60+2)` 差距平滑
- k 越小 → 靠前文档权重越大（极端 k=0 时 1/1=1.0 碾压一切）
- k 越大 → 排名越平均
- 60 是学术界经验值（Cormack et al., SIGIR 2009），不是调出来的

### 2. Bi-Encoder vs Cross-Encoder

| | Bi-Encoder | Cross-Encoder |
|----|------------|---------------|
| 编码方式 | query 和 doc 分别独立编码 | query+doc 拼接后联合编码 |
| 计算量 | 1 次 query 编码 + N 次缓存命中 | N 次 query-doc 对推理 |
| 速度 | 快（doc 可预编码缓存） | 慢（每对都要完整推理） |
| 准确度 | 粗粒度相似度 | 精细相关性判断 |
| 典型用途 | 全库粗筛（Top-20） | 候选精排（Top-5） |

**本质区别**：Bi-Encoder 是"盲猜"相似度——query 和 doc 在编码时互不可见。Cross-Encoder 是"读完再判断"——query 和 doc 拼在一起做完整的 Transformer 注意力计算，模型能看到 query 中每个词和 doc 中每个词之间的交互。

### 3. 两阶段检索策略

```
全库 (N个Chunk) ──→ Bi-Encoder 粗筛 ──→ Top-20 ──→ Cross-Encoder 精排 ──→ Top-5
                     O(N) 向量检索           O(20) 联合编码
```

- 对全库做 Cross-Encoder 不可行（每个 Chunk 都要和 query 做一次完整推理，N=10000 时需要 10000 次）
- 两阶段策略：Bi-Encoder 快速缩窄候选范围 → Cross-Encoder 在 Top-20 上精排
- 实验数据显示 Top-5 命中率从单路的 ~60% 提升到 ~85%

### 4. Cross-Encoder 分数含义

- CE 输出的是 logit（未归一化的相关性分数），不是概率
- 值越大越相关，但绝对值没有物理意义
- 不同 query 之间的 CE 分数不可直接对比
- 通常负值表示不相关，正值表示相关（ms-marco 训练集标签：相关=1，不相关=0）

### 面试速记

- **RRF 公式能手写**（5 行核心代码）
- **为什么 RRF 不是加权求和**：量纲差异，BM25 会碾压向量分数
- **k=60 的作用**：平滑系数，防止 Top-1 权重过高
- **Bi vs Cross 的本质**：独立编码 vs 联合编码；盲猜 vs 读完再判断
- **为什么不能全库 Cross-Encoder**：每对都完整推理，N=10000 时计算量不可接受

## 产出文件

- `src/fusion.py` — reciprocal_rank_fusion + CrossEncoderReranker + FusionPipeline

## 关键实现

### ① RRF 核心算法 — 5 行就能写清楚

```python
scores = defaultdict(float)
for ranking in rankings:
    for rank, result in enumerate(ranking, start=1):
        scores[result.chunk.chunk_id] += 1.0 / (k + rank)
sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
```

**追问应对**：「如果某个文档只在一路检索中出现，另一路没出现，RRF 怎么处理？」— 没出现 = 排名无穷大 = 贡献 0 分。只被一路命中的文档在 RRF 里只拿该路的倒数分，不如双路都命中的文档高。这天然惩罚了只有单路看好的文档，奖励了两路"共识"的文档。

### ② Cross-Encoder 推理 — query+doc 拼接

```python
pairs = [(query, r.chunk.content) for r in candidates]
scores = self.model.predict(pairs, show_progress_bar=False)
```

**追问应对**：「predict 返回的是什么值？」— 返回的是 logit（未归一化的相关性分数），不是 sigmoid 后的概率。MS MARCO 数据集训练时用 cross-entropy loss，logit 越大表示越相关。在实际使用中我们只比较相对大小来做排序，不关心绝对值。

### ③ FusionPipeline — 链路串联

```python
class FusionPipeline:
    def run(self, bm25_results, vector_results, query):
        fused = reciprocal_rank_fusion([bm25_results, vector_results], k=rrf_k)
        reranked = self.reranker.rerank(query, fused, top_k=ce_top_k)
        return {"rrf": fused, "cross_encoder": reranked}
```

**追问应对**：「为什么返回 dict 而不是直接返回最终结果？」— 消融实验需要中间结果。评估时分别跑 RRF 的 Top-5 准确率和 CE 的 Top-5 准确率，差值 = CE 模块的贡献。如果只返回最终结果，就无法区分是 RRF 做得好还是 CE 做得好。

---

## 相关章节

- [[ch03-双路召回]] — BM25 和 Vector 两路结果在这里汇合
- [[ch05-Streamlit前端]] — RRF 和 CE 的结果在四个 Tab 中对比展示
- [[ch06-Ragas评测]] — 消融实验靠中间结果（rrf vs ce）计算模块贡献
