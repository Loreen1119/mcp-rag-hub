# 第 4 章：RRF 融合 + Cross-Encoder 重排序

## 知识点

### 1. 两阶段漏斗模型

在工业级架构中，如果让最准确的重排模型去逐一通读全库几万张纸条，延迟不可接受。业界统一采用两阶段漏斗：

```
全库 (10000个Chunk)
  │
  ▼ 第一阶段：双路粗筛 (Bi-Encoder + BM25)
  │ O(N) 毫秒级，瞬间过滤 99.8% 噪音
  │
Top-20 候选集
  │
  ▼ 第二阶段：Cross-Encoder 精排
  │ O(20) 全注意力联合编码
  │
Top-5 最终结果
```

- 第一阶段：Bi-Encoder 的 doc 向量可提前缓存，检索时只编码 query 一次 + HNSW 近似搜索，成本极低
- 第二阶段：Cross-Encoder 对每个 (query, doc) 对做完整推理，每对约 200ms，只能在 20 条候选上精排
- 工程成效：Top-5 命中率从单路的 ~60% 提升到 ~85%，全库延迟仍保持在 250ms 以内

### 2. RRF — 倒数排名融合（不看分数看排名）

**核心矛盾**：BM25 得分 0~几十，向量余弦相似度 0~1。直接加权求和 → BM25 断层式碾压，向量召回形同虚设。

**RRF 解法**：彻底剥夺原始分数的发言权，只考量各路中的**名次**。

```
RRF_Score(d) = Σ 1 / (k + rank_i(d))
```

- `rank_i(d)`：文档 d 在第 i 路检索中的排名，从 1 开始
- `k`：平滑常数，默认 60

**k=60 为什么？（核心物理作用）**

k 的设计目标是**压制单路 Top-1 的霸权，奖励双路共识**。

| | k=0（无平滑） | k=60（标准） |
|---|---|---|
| 第 1 名得分 | 1/1 = 1.0000 | 1/61 ≈ 0.01639 |
| 第 2 名得分 | 1/2 = 0.5000 | 1/62 ≈ 0.01612 |
| 第 1 名 vs 第 2 名 | 2 倍碾压 | 差距极细微 |

k=60 下，单路排名之间的分数差距被故意锁得极其细微。单路拿第一不再有特权。**只有那些在双路都拿到前排、达成"双路共识"的纸条，两份分数叠加后才能像重磅炸弹一样瞬间超越单路偏见的孤僻纸条。RRF 天然通过数学公式奖励共识、惩罚偏见。**

k=60 是学术界经验值（Cormack et al., SIGIR 2009），不是本项目调出来的。

**追问应对**：「如果某个文档只在一路出现，另一路没出现？」— 没出现 = 排名无穷大 = 贡献分 0。孤僻文档只拿单路的细微倒数分，总分会瞬间被双路都命中的共识文档超越。这在工程上天然起到了抑制单路偶发性噪声的效果。

### 3. Bi-Encoder vs Cross-Encoder

| 维度 | Bi-Encoder（第 3 章向量检索） | Cross-Encoder（本章精排） |
|------|---------------------------|-------------------------|
| 编码方式 | Query 和 Doc **独立编码，互不可见** | Query 和 Doc **拼接后联合编码** |
| 底层数学 | 各自生成特征向量 → 空间点积/余弦 | 全自注意力矩阵两两交叉计算 |
| 大白话 | **看表格相亲**：双方各自填表，不让见面，拿两张纸算匹配度 | **关小黑屋聊天**：拼在一起，逐字对照，实时注入上下文 |
| 速度 | 毫秒级（Doc 向量可预计算缓存） | 每对 ~200ms（每对都要完整推理） |
| 用途 | 全库粗筛 | 候选精排 |

**Cross-Encoder 为什么能理解指代关系**：不是因为逻辑推理，而是因为它的"出厂训练"中见识过几亿条 (query, doc, label) 标准答案。特定文本结构（如 query 中的代词 + doc 中的实体名称）撞上矩阵乘法时产生高能共振——query 的特征在注意力层被物理级加权缝合到 doc 的 token 上，从而给出精细的相关性判断。这种能力刻在了数千万权重参数中。

### 4. Cross-Encoder 分数含义（logit）

- CE 输出的是 **logit（未归一化的相关性分数）**，不是 sigmoid 后的概率
- **绝对不能跨 query 对比绝对值**：不同 query 的 CE 绝对分数在数学上不可横向对比
- 工程中只提取其相对大小关系用于排序，不看绝对值
- MS MARCO 数据集训练（相关=1，不相关=0），通常正值表示相关，负值表示不相关

### 面试速记

- **RRF 公式能手写**：`score += 1.0 / (k + rank)`，5 行代码
- **k=60 一句话**：压制单路霸权，奖励双路共识——不存在单路战神
- **Bi vs Cross 类比**：看表格相亲 vs 关小黑屋聊天；盲猜匹配 vs 通读再判断
- **漏斗模型**：10000 → 20 → 5，粗筛保召回，精排保精度
- **logit ≠ 概率**：只比较相对大小做排序，不看绝对值
- **为什么返回 dict**：为消融实验埋伏笔——分别计算 RRF 和 CE 的准确率，差值 = CE 模块的 ROI

## 产出文件

- `src/fusion.py` — reciprocal_rank_fusion + CrossEncoderReranker + FusionPipeline

## 关键实现

### ① RRF 核心算法 — 5 行纯名次逻辑

```python
scores = defaultdict(float)
for ranking in rankings:
    for rank, result in enumerate(ranking, start=1):     # start=1 极其重要！
        scores[result.chunk.chunk_id] += 1.0 / (k + rank) # 多路反复命中的纸条在此累加
sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
```

**追问应对**：「`enumerate(..., start=1)` 为什么不能从 0 开始？」— 排名从第 1 名起算，若从 0 开始，k=60 时第一名得分 = 1/(60+0) = 1/60，而第二名 = 1/(60+1) = 1/61，第一名反而有优势大于第二名——不符合排名语义。从 1 开始才能正确映射"第 n 名"的物理意义。

### ② Cross-Encoder 推理 — Query 与 Doc 物理拼接

```python
pairs = [(query, r.chunk.content) for r in candidates]  # (query, doc) 元组对
scores = self.model.predict(pairs, show_progress_bar=False)
```

**追问应对**：「predict 返回的值能当概率用吗？」— 绝对不能。CE 输出的是 logit，不是 sigmoid 后的概率。不同 query 之间的 CE 绝对分数不可横向对比。在工程中只提取相对大小关系用于重新排序，绝不看绝对值。更不能用 threshold=0.5 来判定相关/不相关——这是 logit，不是概率。

### ③ FusionPipeline — 显式暴露中间结果

```python
class FusionPipeline:
    def run(self, bm25_results, vector_results, query):
        fused = reciprocal_rank_fusion([bm25_results, vector_results], k=rrf_k)
        reranked = self.reranker.rerank(query, fused, top_k=ce_top_k)
        return {"rrf": fused, "cross_encoder": reranked}  # 暴露两阶段中间产物
```

**追问应对**：「为什么返回 dict 而不是直接返回 Top-5？」— 为消融实验埋下伏笔。在工业级 AI 研发中，每引入一个重模型都必须用数据证明其算力消耗的合理性。返回两阶段中间结果后，评测脚本能分别计算 RRF 的准确率 和 CE 精排后的准确率，差值 = CE 模块的 ROI。如果 CE 只带来 1% 的提升却增加 200ms 延迟，那就不值得加——这个决策依赖 dict 暴露的中间数据。

---

## 相关章节

- [[ch03-双路召回]] — BM25 和 Vector 两路结果在这里汇合
- [[ch05-Streamlit前端]] — RRF 和 CE 的结果在四个 Tab 中对比展示
- [[ch06-Ragas评测]] — 消融实验靠中间结果（rrf vs ce）计算模块贡献
