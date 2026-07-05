# 第 6 章：RAG 自动化评测

## 知识点

### 1. 为什么要评测 —— 没有度量就没有优化

RAG 系统是管道结构，改任何一个模块（换模型、调参数）都会影响最终输出。没有评测 = 盲调。

评测闭环：**改代码 → 跑全量评测 → 看指标变化 → 防负向退化**。

本项目评测分两层：
- **检索层评测**（`evaluate.py`）：MRR / Hit@K / Precision@K，测"找没找到"
- **生成层评测**（`llm_evaluate.py`）：Faithfulness / Answer Relevancy / Context Recall，测"答没答好"

### 2. 检索层评测：三个硬指标

| 指标 | 测什么 | 计算方式 | 一句话 |
|------|--------|----------|--------|
| **MRR** | 第一个相关结果排第几 | 1/第一个相关结果的排名 | "正确答案排多靠前" |
| **Hit@K** | Top-K 里有正确结果吗 | 有则 1，无则 0 | "至少找到了吗" |
| **Precision@K** | Top-K 里多少是正确的 | 正确数/K | "找到的里面对了多少" |

**MRR 和平均排名的区别**：MRR 惩罚后排结果。rank=1 得 1.0，rank=2 得 0.5，rank=10 只得 0.1。它鼓励系统把最佳结果顶在最前面，而不是"平均排名不错但第一页全是噪声"。

### 3. GoldenTestSet 分层设计

15 组 Query，三类各 5 组，故意不对称地覆盖各检索器强弱项：

| 类别 | 测什么 | 例子 | 预期谁赢 |
|------|--------|------|----------|
| exact_match (E01~E05) | 专有名词精确匹配 | "BM25 算法"、"RRF 融合" | BM25 优势 |
| semantic (S01~S05) | 语义相似但字面不同 | "怎么评价检索系统好坏"（字面没"评测"） | Vector 优势 |
| mixed (M01~M05) | 多模块协同综合 | "如何提高检索准确率"（需综合多段落） | 看 RRF+CE 效果 |

每个 test case 含三个字段：`golden_chunk_sources`（检索评测用）、`golden_answer`（LLM 生成评测用）、`notes`（标注预期行为）。

### 4. 为什么自己实现而不是用 Ragas 库

Ragas 在 Windows Anaconda 环境下出现 SSL 证书冲突（`aiohttp` → `ssl.load_default_certs` 失败），属环境问题而非代码问题。

但更重要的是能力考量：MRR、Hit@K、Precision@K 的计算逻辑极其简单——就是取排名、数命中——为了这几行数学引入一个第三方框架的依赖链（aiohttp、datasets、langchain…），得不偿失。自己实现不到 100 行，完全可控。

**面试时怎么说**：「Ragas 在 Windows 有环境兼容问题，但我理解每个指标的计算原理后自己实现了核心评测模块。MRR、Hit@K、Precision@K 的计算就是取排名和数命中，不需要依赖外部库。Faithfulness 等需要 LLM，我通过 Ollama 接入 qwen2.5:7b 做了完整的 LLM-as-Judge 三维评测。」

### 5. 检索层基线结果（3 Chunk 小语料）

| Stage | MRR | Hit@5 | Prec@5 |
|-------|-----|-------|--------|
| BM25 | 0.90 | 0.93 | 0.71 |
| Vector | 0.97 | 1.00 | 0.73 |
| RRF | 0.93 | 1.00 | 0.73 |
| **CE** | **1.00** | **1.00** | **0.73** |

三个关键发现：
- **S03 是最佳证据——"降级与修复"全链路**：语义查询 "怎么评价检索系统"（字面全无"评测"二字）→ BM25 MRR=0（彻底盲视，张嘴吃零蛋）→ Vector MRR=1.0（语义理解一剑封喉）→ **RRF MRR=0.5（负优化退化！因为 BM25 捞回的"系统"字面噪声被 RRF 无脑融合，真答案被挤到第二名）** → CE MRR=1.0（精排模型逐字通读，识破噪声伪装，将真答案重新顶回 #1）。这一条 query 的四阶段数据，直接证明了"双路粗筛 + 后置精排"管线中 Cross-Encoder 不是锦上添花，而是**对抗 RRF 退化的最后纠错器**
- **Prec@5 恒定为 0.73**：小语料天花板。总共就 3 个相关 Chunk，Top-5 最多命中 3 个，但总有文档只有 1 个 Chunk 被召回
- **CE 在多个 query 上修补了 RRF 退化**：RRF 不是银弹，小语料下可能负优化，Cross-Encoder 是最后的纠错器

> 免责：这些数字是 3 Chunk 小语料下的天花板效应。真实场景绝对值会低很多，但**各阶段相对变化趋势一致**——面试时诚实说明这是"验证逻辑正确性"的基线即可。

### 6. 生成层评测：LLM-as-Judge 三维打分

检索只是"找没找到"，最终用户的体验取决于"答没答好"。因此引入 LLM-as-Judge，用更强的 LLM 对 RAG 生成结果做自动化质量仲裁。

#### 三个生成指标

| 指标 | 测什么 | 比对对象 |
|------|--------|----------|
| **Faithfulness** | 答案是否忠于检索上下文（检测幻觉/编造） | 生成答案 vs 检索上下文 |
| **Answer Relevancy** | 答案是否紧扣用户问题（检测跑题） | 生成答案 vs 用户问题 |
| **Context Recall** | 检索上下文是否覆盖了参考答案的关键信息（检测检索遗漏） | 检索上下文 vs 黄金答案 |

#### 三代演进：从 3B 到 7B

| 轮次 | 模型 | 结果 | 教训 |
|------|------|------|------|
| 第一轮 | qwen2.5:3b | Faithfulness/Relevancy/Recall 全部 0.7，无区分度 | 3B 太小，做评判不准，只会打安全分 |
| 第二轮 | qwen2.5:7b | Faithfulness 0.95 / Relevancy 0.86 / Recall 0.79，有真实方差 | 7B 具备足够评判能力 |

**面试时怎么说**：「评测 LLM 的选择本身就是一条工程经验——3B 模型太小，只会给安全分。用 7B 之后打分才有了区分度，Faithfulness 从 0.23 直接拉到了 0.95。」

#### 全量 15 条 7B 评测结果（v1 最终版）

| 指标 | Mean | Min | Max | 解读 |
|------|:----:|:---:|:---:|------|
| Faithfulness | **0.9533** | 0.7 | 1.0 | 系统几乎不编造，RAG 防幻觉目标达成 |
| Answer Relevancy | **0.86** | 0.7 | 1.0 | 回答基本扣题，mixed 类查询最高（0.98） |
| Context Recall | 0.7933 | 0.7 | 1.0 | 检索语料中确实缺少公式等细节 |

| 类别 | Faith | Relev | Recall | 解读 |
|------|:-----:|:-----:|:------:|------|
| exact_match | **1.0** | 0.76 | 0.76 | 字面匹配忠实度满分 |
| mixed | 0.9 | **0.98** | 0.74 | 综合分析题扣题度最高 |
| semantic | 0.96 | 0.84 | **0.88** | 语义类召回最完整 |

> v1 是现阶段最优结果。后续曾尝试收紧生成 Prompt（200 字/禁止推论），反而导致 E01 Faithfulness 暴跌至 0.0——生成过于简短偏离原文内容。最终结论：**300 字/允许自然关联技术说明** 是当前 7B + 3 Chunk 小语料下的最佳平衡点。

#### Context Recall 偏低的关键发现

Context Recall 稳定在 0.7~0.8，不是检索失败，而是语料覆盖不足——BM25 公式推导、RRF 数学公式、Cross-Encoder 自注意力机制细节、MCP 接口封装等知识点在检索语料中确实不存在。**这是一个值得在面试中主动展示的工程洞察**：评测不仅告诉你"好"，还告诉你"缺什么"。

#### 平均生成耗时

全量 15 条，每条 4 次 LLM 调用（1 次生成 + 3 次评分），平均每条 **233 秒（~4 分钟）**，7B 在纯 CPU 上总计约 1 小时。

### 7. 逐条增量写入：一个工程踩坑

原始代码是全部跑完一次性 `json.dump`。第一轮 3B 跑时 E02/E03 的 Ollama 调用超时返回空内容，Ctrl+C 中断后已跑完的数据全丢了。

修复：改为每跑完一条立刻写 checkpoint 到磁盘。这个改动看似 trivial，但体现了"长时间运行的评测脚本必须防丢失"的工程意识——评测脚本跑一次几十分钟到一小时，丢一次数据的代价远大于写 checkpoint 的几毫秒 IO。

---

## 关键实现与代码走读

> 以下代码节选自实际 `src/evaluate.py`，注释为讲解用。

### ① 三个核心指标：加起来不到 30 行

```python
def _mrr(results, golden_sources, k=10):
    """MRR@K — 第一个相关结果的倒数排名。无命中返回 0。"""
    for rank, r in enumerate(results[:k], start=1):
        if _is_relevant(r, golden_sources):
            return 1.0 / rank       # rank=1→1.0, rank=2→0.5, rank=10→0.1
    return 0.0


def _hit_at_k(results, golden_sources, k=5):
    """Hit@K — Top-K 中是否至少有一个相关结果。"""
    for r in results[:k]:
        if _is_relevant(r, golden_sources):
            return 1.0               # 找到一个就 1
    return 0.0                       # 一个没找到就 0


def _precision_at_k(results, golden_sources, k=5):
    """Precision@K — Top-K 中相关结果的比例。"""
    if not results[:k]:
        return 0.0
    hits = sum(1 for r in results[:k] if _is_relevant(r, golden_sources))
    return hits / min(k, len(results[:k]))
```

三个函数的逻辑就是"取排名、数命中、算比例"。**MRR 的核心洞察**：rank=1 得 1.0，rank=2 得 0.5，rank=10 只得 0.1——它惩罚后排结果，鼓励系统把最佳答案顶在第一位，而不是"平均排名不错但前几页全是噪声"。

### ② 四阶段全量评测主循环

```python
for tc in test_cases:
    query = tc["query"]
    golden_sources = tc["golden_chunk_sources"]

    # 执行各阶段检索
    bm25_results = bm25.search(query, top_k=BM25_TOP_K)
    vector_results = vector.search(query, top_k=BM25_TOP_K)
    rrf_results = reciprocal_rank_fusion([bm25_results, vector_results])
    ce_results = pipeline.reranker.rerank(query, rrf_results, top_k=CE_TOP_K)

    # 每个阶段独立计算相同的三个指标
    for stage, results in [("bm25", bm25_results), ...]:
        mrr = _mrr(results, golden_sources)
        hit = _hit_at_k(results, golden_sources, k=5)
        prec = _precision_at_k(results, golden_sources, k=5)
```

**面试时强调**：同一个 query 的四阶段结果用相同的 golden_sources 和相同的三个指标计算，保证对比的公平性。每个模块的增量贡献（delta）一览无余。

### ③ 阶段贡献消融分析

```python
def ablation_analysis(summary):
    for metric in ["mrr", "hit@5", "precision@5"]:
        prev = 0.0
        for stage in ["bm25", "vector", "rrf", "cross_encoder"]:
            val = summary[stage][metric]
            delta = val - prev          # 本阶段的增量贡献
            print(f"{stage}: {val:.4f}  (+{delta:+.4f})")
            prev = val
```

BM25 → +Vector → +RRF → +CE 逐级累进，每个模块的净贡献一目了然。比如 S03 语义查询：BM25 MRR=0 → Vector +1.0 → RRF -0.5 → CE +0.5。**数字会说话**。

### ④ `_is_relevant` 判定逻辑

```python
def _is_relevant(result, golden_sources):
    """判定检索结果是否相关：Chunk 的 source 文件在 golden_sources 中。"""
    source = result.chunk.metadata.get("source", "")
    return source in golden_sources
```

基于文件级 source 匹配——简单但有效。3 Chunk 小语料下每个文档就是一块，文件级判定足够。大语料下可升级为 Chunk 级 ID 匹配。

---

## 面试话术

**面试官**："你怎么评测你们的 RAG 系统？"

**回答**："评测分两层。**检索层**用 MRR、Hit@K、Precision@K 三个硬指标，测'找没找到'。关键不需要第三方库——MRR 就是取第一个相关结果的排名取倒数，Hit@K 就是数命中，三个公式加起来不到十行代码。我用 15 组分层 GoldenTestSet 跑四阶段消融，同一把裁判尺子量四个阶段的输出，保证对比公平。

我手里最硬核的证据是 S03 这条语义查询。用户问'怎么评价检索系统'——字面一个'评测'都没有。BM25 直接 MRR=0，完全盲视。Vector 语义理解一剑封喉，MRR=1.0 满分。但到了 RRF 融合阶段，因为 BM25 捞回的'系统'字面噪声被无脑融合，真答案被挤到第二——**MRR 直接退化到 0.5**。最后 Cross-Encoder 逐字通读，识破噪声伪装，把 MRR 逆势修复回 1.0。

这条数据直接证明了'双路粗筛 + 后置精排'管线中，Cross-Encoder 不是锦上添花，而是对抗 RRF 退化的最后纠错器。"

**生成层**用 LLM-as-Judge 做三维打分：Faithfulness 测幻觉、Answer Relevancy 测跑题、Context Recall 测检索遗漏。这里有一条经验——我最初用 qwen2.5 的 3B 版本，结果三个指标全部打 0.7，完全没有区分度。换到 7B 之后打分才真实可用，Faithfulness 跑到了 0.95。

评测的核心价值不仅是打分，更是告诉你'缺什么'。Context Recall 稳定在 0.7~0.8，不是因为检索烂，而是检索语料里确实没有 BM25 公式推导这些细节——评测帮你定位知识库的盲区。"

---

**面试官**："为什么不用 Ragas 库？"

**回答**："两个原因。技术上，Ragas 在 Windows Anaconda 环境有 SSL 证书冲突，底层 aiohttp 的 SSL 握手会失败。但更重要的是能力考量——MRR、Hit@K、Precision@K 的计算逻辑极其简单，就是为了这几行数学引入一个包含 datasets、langchain 依赖链的第三方框架，不合理。自己实现不到 100 行，完全可控。

对于 Faithfulness 等需要 LLM 的能力，我写了一套独立的 LLM-as-Judge 评测脚本，用 Ollama 接入本地 qwen2.5:7b 跑三维打分，Prompt 模板参考了 Ragas 的思路。这样做的好处是完全掌握评测链路，不依赖黑盒库。"

---

## 产出文件

- `src/evaluate.py` — 自实现检索评测（MRR/Hit@K/Precision@K + 消融分析）
- `src/llm_evaluate.py` — LLM-as-Judge 生成评测（Faithfulness/Relevancy/ContextRecall）
- `test_queries.json` — 15 组 GoldenTestSet（三类分层，含 golden_answer）
- `experiments/llm_evaluation_results.json` — 全量 15 条 7B 评测结果

## 相关章节

- [[ch04-RRF融合与重排序]] — 评测脚本分别测试每个融合阶段的指标
- [[ch03-双路召回]] — BM25 和 Vector 的原始结果作为评测基线
- [[ch05-Streamlit前端]] — 评测结果可以在前端直观验证
