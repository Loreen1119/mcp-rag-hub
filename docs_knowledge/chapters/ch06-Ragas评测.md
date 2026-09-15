# 第 6 章：RAG 自动化评测

## 知识点

### 1. 为什么要评测 —— 没有度量就没有优化

RAG 系统是管道结构，改任何一个模块（换模型、调参数）都会影响最终输出。没有评测 = 盲调。

评测闭环：**改代码 → 跑全量评测 → 看指标变化 → 防负向退化**。

本项目评测分四层（脚本都在 `src/evaluation/` 子包下）：
- **检索层评测**（`retrieval_eval.py`）：MRR / Hit@K / Precision@K / Recall@K，测"找没找到"
- **生成层评测**（`llm_eval.py`）：Faithfulness / Answer Relevancy / Context Recall，测"答没答好"
- **改写评测**（`agent_eval.py`）：改写效果 A/B 对比 + 语义保真度 + CE 阈值校准，测"改写有没有让检索变好"
- **端到端验收**（`acceptance_eval.py`）：拒答集（`--refusal`）、多证据集（`--multi`）、裁判（`--judge`），测"该不该答 / 引用全不全"

### 2. 检索层评测：四个硬指标

| 指标 | 测什么 | 计算方式 | 一句话 |
|------|--------|----------|--------|
| **MRR** | 第一个相关结果排第几 | 1/第一个相关结果的排名 | "正确答案排多靠前" |
| **Hit@K** | Top-K 里有正确结果吗 | 有则 1，无则 0 | "至少找到了吗" |
| **Precision@K** | Top-K 里多少是正确的 | 正确数/K | "找到的里面对了多少" |
| **Recall@K** | 所有 goldens 被覆盖了多少 | 去重后的来源文件命中数 / total_golden_sources | "应该找到的都找到了吗" |

注意：Recall@K 的分子按 unique source 去重计数，而非 chunk 级计数。因为 golden_sources 是文档名列表，一个文档可能拆成多个 chunk，按 chunk 计数会导致 recall > 1.0 的 bug。

**MRR 和平均排名的区别**：MRR 惩罚后排结果。rank=1 得 1.0，rank=2 得 0.5，rank=10 只得 0.1。它鼓励系统把最佳结果顶在最前面，而不是"平均排名不错但第一页全是噪声"。

### 3. GoldenTestSet 分层设计

现在 **18 条**，四类，故意不对称地覆盖各检索器强弱项（语料 = FastAPI 中文文档 122 篇 / 1445 chunk）：

| 类别 | 条数 | 测什么 | 例子 | 预期谁赢 |
|------|------|--------|------|----------|
| exact_match (E) | 4 | 专有名词精确匹配 | "路径操作装饰器" | BM25 优势 |
| semantic (S) | 4 | 语义相似但字面不同 | "应用启动只想执行一次初始化、关闭时收尾，该用什么机制？"（→lifespan，字面全无） | Vector 优势 |
| mixed (M) | 4 | 多模块协同综合 | "FastAPI 应用怎么做测试？" | 看 RRF+CE 效果 |
| graph (G) | 6 | 需要把多个概念串起来 | "SQLModel 与 SQLAlchemy、Pydantic 是什么关系？" | 看全链路 |

每个 test case 含四个字段：`query`、`golden_chunk_sources`（检索评测用）、`golden_answer`（生成评测用）、`notes`（标注预期行为）。
**每条 `golden_answer` 都能在 `golden_chunk_sources` 指定的文件正文里回查到原文**，由 `scripts/validate_golden.py` 自动校验。

另外两套验收集：**10 条拒答集**（测"该不该说不知道"）+ **3 条多证据集**（测跨文档引用覆盖率）。

> **不做 train/test 拆分**。拆分原本是给「用 CE 分数卡阈值自动拒答」调参用的，那方案被自己的数据证伪后
> 拆分就失去了意义，2026-09-12 起收敛为单一 golden set。

### 4. 为什么自己实现而不是用 Ragas 库

Ragas 在 Windows Anaconda 环境下出现 SSL 证书冲突（`aiohttp` → `ssl.load_default_certs` 失败），属环境问题而非代码问题。

但更重要的是能力考量：MRR、Hit@K、Precision@K 的计算逻辑极其简单——就是取排名、数命中——为了这几行数学引入一个第三方框架的依赖链（aiohttp、datasets、langchain…），得不偿失。自己实现不到 100 行，完全可控。

**面试时怎么说**：「Ragas 在 Windows 有环境兼容问题，但我理解每个指标的计算原理后自己实现了核心评测模块。MRR、Hit@K、Precision@K 的计算就是取排名和数命中，不需要依赖外部库。Faithfulness 等需要 LLM，我通过 Ollama 接入 qwen2.5:7b 做了完整的 LLM-as-Judge 三维评测。」

### 5. 检索层基线结果（18 条，语料 1445 Chunk，2026-09-12）

| Stage | MRR | Hit@5 | Prec@5 | Recall@5 |
|-------|-----|-------|--------|----------|
| BM25 | 0.5469 | 0.8889 | 0.4333 | 0.8889 |
| Vector | 0.7246 | 0.8333 | 0.3889 | 0.8333 |
| RRF | 0.7833 | 0.9444 | 0.5000 | 0.9444 |
| **CE（全管线）** | **0.7889** | **1.0000** | **0.5444** | **1.0000** |

四个关键发现：

- **两路的失败集合几乎不重叠**（这是"混合召回"最硬的证据）：向量把 S04 从 MRR 0.000 拉到 1.000，
  BM25 则在 Vector 掉到 0.000 的 S02 / S06 上仍有召回。不是感觉上需要两路，是数据上两路的盲区互补。
- **S06 是"CE 不可省"的最佳证据**：S06（配置放代码外、运行时读取 → settings）的轨迹是
  BM25 0.2000 → Vector **0.0000** → RRF **0.1000（融合后 Hit@5 反而掉到 0，真答案被挤掉）** → CE 0.2000（救回）。
  **RRF 不是银弹，它会引入退化；Cross-Encoder 是对抗这种退化的最后纠错器。**
- **CE 的价值在 Hit@5，不在 MRR**：Hit@5 补到 100%，但 MRR 只 +0.006——因为它**有得有失**：
  救回 E02（0.33→1.00）、S04（0.50→1.00），也把 G08 / M04 从 1.0 弄到 0.25。
  精排的定位是"兜底 + 排序"，不是"提分"。
- **一条对自己预期的修正**：上面四类表里的"预期谁赢"是**设计意图**，实测有一处不符 ——
  exact_match 类预期"BM25 优势"，而实测 BM25 与 Vector 的 **MRR 完全相同**（均 0.8000）。
  所以 BM25 的价值不在"这类题更强"，而在**逐条失败集合不重叠**（见上一条）。
  > 这条是修 `src/evaluation/experiments.py` 的写死结论时发现的：原实现把"预期"直接印成了
  > "结论"（`**BM25 在 exact_match 上的优势**`），且没检查它是否真的更大。
  > 已在 C10 中改为按实测判断，并加了回归测试（`tests/test_experiments_report.py`）。

> **别再引用旧稿里的 BM25 0.90 / CE 1.00 那张表**——那是 3 Chunk 小语料下的天花板效应（总共就 3 个相关 chunk，
> 各配置轻松满分，等于没测出区分度）。**旧表的问题是它证明不了任何事**，新表才有真实的失败案例可以讲。

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
| 第一轮 | qwen2.5:3b | 三项指标全部 0.7，无区分度；后来还出现过自相矛盾的判词（同一句里说"大部分信息来自上下文"却按"大量编造"档打 0.10 分） | 3B 太小，**做裁判**不准，只会打安全分 |
| 第二轮 | qwen2.5:7b | 有真实方差，Faithfulness 从 0.23 拉到 0.95 | 7B 才具备评判能力 |

**结论（已写进架构）**：生成用 **3b**、裁判用 **7b**（`config.JUDGE_MODEL`）。
生成是"给 3 条证据做抽取归纳 + 标引用 + 输出 JSON"的窄任务，3b 够用且 CPU 可跑；
裁判要"逐句比对答案与上下文并按档打分"，3b 做不到。**这两件事的模型需求不同，不该共用一个。**

> ⚠️ **上表的 0.95 / 0.86 / 0.79 是历史数字，绑定的是旧语料 + 旧 15 条评测集，不要照着讲。**
> 换语料后评测集全换，旧数字自动失效。

#### ⚠️ 当前生成层评测的真实状态（2026-09-14）

| 项 | 状态 |
|---|---|
| `experiments/llm_evaluation_results.json` | 只有一次**旧语料、3 条、3b 自评**的快跑：F 0.60 / AR 0.50 / CR 0.50，单条生成 14.6~43.8 秒 |
| 新语料（`corpora/fastapi-zh`）× 18 条 × 7b 裁判 | **还没跑** |

也就是说：**项目目前只有"检索准"的完整证据（18 条四阶段指标），还没有一条可信的"回答质量好"的证据。**
面试时被问到生成质量，正确说法是"检索侧我有 18 条分层数据；生成侧的自实现 Judge 已就位、
3b 端到端能出真答案，但 18 条 × 7b 裁判的完整评测正在跑"——**而不是报一个跑不出来的 0.95**。

#### Context Recall 这个指标要会解释

Context Recall 稳定偏低（历史 0.7~0.8）**不一定是检索失败，可能是语料覆盖不足**——
参考答案里有些细节在知识库里根本不存在。**这是一个值得在面试中主动展示的工程洞察**：
评测的价值不仅是打分，更是告诉你"知识库缺什么"。

#### 生成耗时（实测，纯 CPU）

- 3b 生成单条：**14.6 ~ 43.8 秒**（比早期估的 15~25 秒慢，现场演示要留足时间）
- 7b 裁判更慢，且需 5.0~5.5 GB 内存 —— 跑之前先关掉占内存的应用
- 所以 `LLM_TIMEOUT_SECONDS` 从 20 提到 **120**，否则纯 CPU 推理必然超时

### 7. 逐条增量写入：一个工程踩坑

原始代码是全部跑完一次性 `json.dump`。第一轮 3B 跑时 E02/E03 的 Ollama 调用超时返回空内容，Ctrl+C 中断后已跑完的数据全丢了。

修复：改为每跑完一条立刻写 checkpoint 到磁盘。这个改动看似 trivial，但体现了"长时间运行的评测脚本必须防丢失"的工程意识——评测脚本跑一次几十分钟到一小时，丢一次数据的代价远大于写 checkpoint 的几毫秒 IO。

---

## 关键实现与代码走读

> 以下代码节选自实际 `src/evaluation/retrieval_eval.py` 与 `src/evaluation/metrics.py`，注释为讲解用。

### ① 四个核心指标：来自 `src/evaluation/metrics.py`

```python
def mrr(results, golden_sources, k=10):
    """MRR@K — 第一个相关结果的倒数排名。无命中返回 0。"""
    for rank, r in enumerate(results[:k], start=1):
        if is_relevant(r, golden_sources):
            return 1.0 / rank       # rank=1→1.0, rank=2→0.5, rank=10→0.1
    return 0.0


def hit_at_k(results, golden_sources, k=5):
    """Hit@K — Top-K 中是否至少有一个相关结果。"""
    for r in results[:k]:
        if is_relevant(r, golden_sources):
            return 1.0               # 找到一个就 1
    return 0.0                       # 一个没找到就 0


def precision_at_k(results, golden_sources, k=5):
    """Precision@K — Top-K 中相关结果的比例。"""
    if not results[:k]:
        return 0.0
    hits = sum(1 for r in results[:k] if is_relevant(r, golden_sources))
    return hits / min(k, len(results[:k]))


def recall_at_k(results, golden_sources, k=10):
    """Recall@K — 所有 goldens 中被检索到的比例。"""
    total_golden = len(golden_sources)
    if total_golden == 0:
        return 0.0
    unique_sources = set()
    for r in results[:k]:
        if is_relevant(r, golden_sources):
            unique_sources.add(r.chunk.metadata.get("source", ""))
    return len(unique_sources) / total_golden
```

四个函数来自共享模块 `src/evaluation/metrics.py`——`retrieval_eval.py` 和 `experiments.py` 之前各自复制粘贴了这些函数，重构后统一引用 metrics.py，消除了重复代码。

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

**面试时强调**：同一个 query 的四阶段结果用相同的 golden_sources 和相同的四个指标计算，保证对比的公平性。每个模块的增量贡献（delta）一览无余。

### ③ 阶段贡献消融分析

```python
def ablation_analysis(summary):
    for metric in ["mrr", "hit@5", "precision@5", "recall@5"]:
        prev = 0.0
        for stage in ["bm25", "vector", "rrf", "cross_encoder"]:
            val = summary[stage][metric]
            delta = val - prev          # 本阶段的增量贡献
            print(f"{stage}: {val:.4f}  (+{delta:+.4f})")
            prev = val
```

BM25 → +Vector → +RRF → +CE 逐级累进，每个模块的净贡献一目了然。比如 S04（中间件）语义查询：
BM25 MRR=0.0000 → Vector 1.0000 → RRF 0.5000 → CE 1.0000。
再看 S06：BM25 0.2000 → Vector 0.0000 → RRF 0.1000（**负贡献**）→ CE 0.2000（救回）。**数字会说话。**

### ④ `is_relevant` 判定逻辑（from `src/evaluation/metrics.py`）

```python
def is_relevant(result, golden_sources):
    """判定检索结果是否相关：Chunk 的 source 文件在 golden_sources 中。"""
    source = result.chunk.metadata.get("source", "")
    return source in golden_sources
```

基于文件级 source 匹配——简单但有效。当前语料 1445 个 chunk、122 个源文件，文件级判定仍然够用（一个文件可能被拆成多个 chunk，所以 Recall 的分子按 unique source 去重计数，避免 >1.0 的 bug）。更细的粒度可升级为 Chunk 级 ID 匹配。

注意：`source` 现在是**相对语料根的路径**而非文件名（`CHUNK_ID_RULE_VERSION=3`）——嵌套语料里有 11 个 `index.md`，按文件名做 source 会撞 ID。

### ⑤ 改写评测：A/B 对比 + 语义保真度 + CE 阈值校准

`src/evaluation/agent_eval.py` 补上了原始评测体系最大的盲区——Agent 在做 `rewrite_query`，但 `retrieval_eval.py` 只测原始 query。

`AgentEvaluator` 类的核心逻辑：

```python
class AgentEvaluator:
    def evaluate(self):
        for tc in self.test_cases:
            rewritten = _rewrite_query_standalone(tc["query"])  # 一次 LLM 改写
            # 原始检索 vs 改写检索（共享管线）
            mrr_orig = mrr(ce_orig, goldens)
            mrr_rw = mrr(ce_rw, goldens)
            delta_mrr = mrr_rw - mrr_orig           # 改写效果
            fidelity = self._check_fidelity(query, rewritten)  # 语义保真度
```

**改写效果判定**：对每组 test case，用原始 query 和改写 query 各跑一次全链路检索，对比 CE 阶段的 MRR/Hit@5/Precision@5/Recall@5。Δ > 0 → 改写有效，Δ < 0 → 改写退化。汇总输出 `helpful_rate`（改写有效比例）和 `avg_delta_mrr`（平均检索增益）。

**语义保真度检查**：用 LLM-as-Judge 判断改写后的 query 是否保留了原始信息需求。关键区分——评"信息需求"而非"字面相似"。"怎么报销"→"差旅费报销审批流程"是好的改写（字面不同但意图一致）。

**CE 阈值校准**：Agent 用 `CE top-1 分数 < 阈值` 决定是否触发改写。当前阈值 **0.3**（`config.CE_THRESHOLD`）——
注意**量纲跟着 Cross-Encoder 模型走**：`bge-reranker-base` 经 sigmoid 输出 [0,1]；
早期英文 `ms-marco-MiniLM` 输出无界 logits，对应经验值是 3.0。换模型不改阈值 = 判定恒定偏一侧。

`calibrate_threshold()` 用"改写是否实际提升了 MRR"作为 ground truth，扫阈值找 F1 最优点，输出含 confidence 字段（小样本下标记 "low"）。

> **踩过的坑（已修）**：校准原本硬编码扫描区间 `1.0~6.0`（旧 logits 量纲）。换 bge-reranker 后忘了改，
> 后果是所有 [0,1] 分数恒小于 1.0 → 每个阈值都判定"全部需要改写" → F1 恒定不变，**校准静默空跑且不报错**。
> 现在改为**按本次实测分数的 min/max 自适应扫描**，换任何模型都不会再失效。

---

## 面试话术

**面试官**："你怎么评测你们的 RAG 系统？"

**回答**："评测分四层。**检索层**用 MRR、Hit@K、Precision@K、Recall@K 四个硬指标，测'找没找到'。关键不需要第三方库——MRR 就是取第一个相关结果的排名取倒数，Hit@K 就是数命中，四个公式都在共享模块 `src/evaluation/metrics.py` 里。我用 **18 条四类分层**的 GoldenTestSet 跑四阶段消融，同一把裁判尺子量四个阶段的输出，保证对比公平。语料是真实的 FastAPI 中文文档，122 篇、1445 个切片。

检索侧的完整数据：BM25 MRR 0.5469 / Hit@5 88.9%，Vector 0.7246 / 83.3%，RRF 融合 0.7833 / 94.4%，加 CE 后 0.7889 / **100%**。

我最硬的两条证据是 S04 和 S06。S04 问'请求前后插入通用逻辑'（答案是中间件），字面一个字都没提到——BM25 MRR=0.0000 完全盲视，Vector 满分 1.0000。**但反过来也有**：S02、S06 这两条是 Vector 掉到 0.0000，BM25 还能捞到——所以我说的'互补'是数据上的失败集合不重叠，不是套话。

S06 更能说明精排的价值：它 BM25 0.2 → Vector 0.0 → **RRF 0.1，融合之后反而更差了**，最后只有 Cross-Encoder 把它救回来。所以 CE 在我的管线里不是锦上添花，是对抗融合退化的最后纠错器。不过我也如实讲：CE 的 MRR 净收益只有 +0.006，它真正的贡献是 Hit@5 从 94.4% 补到 100%——**它是兜底和排序，不是提分**。

**生成层**用自实现的 LLM-as-Judge 做三维打分：Faithfulness 测幻觉、Answer Relevancy 测跑题、Context Recall 测检索遗漏。这里有一条实打实的经验——**生成和裁判的模型需求不同，不能共用一个**：3B 当裁判时三项指标全打 0.7 毫无区分度，还出现过自相矛盾的判词；换 7B 才有真实方差。所以我的架构是生成用 3b（窄任务、CPU 可跑）、裁判用 7b。

生成侧我要**如实说明进度**：18 条 × 7B 裁判的完整评测还在跑，现在文件里只有旧语料 3 条、3b 自评的小样本结果。**检索质量的证据是完整的，回答质量的证据我不想拿没跑完的数字充数。**

**改写评测层**——我做了改写效果 A/B 对比，对每组 test case 分别跑原始和改写 query 的检索，对比 MRR delta 判断改写是否真的变好；同时用 LLM-as-Judge 检查改写有没有偏离原始意图（评'信息需求'而非'字面相似'）。CE 触发阈值也从拍脑袋的值变成了有校准实验支撑的 0.3。

**端到端验收层**——除了可答集，我还有 10 条拒答集（测'该不该说不知道'）和 3 条多证据集（测跨文档引用覆盖率）。因为一个只会答、不会说'不知道'的 RAG，在生产上是不能用的。

评测的核心价值不仅是打分，更是告诉你'缺什么'。Context Recall 偏低不一定是检索烂，可能是知识库里根本没有那部分内容——评测帮你定位知识库的盲区。"

---

**面试官**："为什么不用 Ragas 库？"

**回答**："两个原因。技术上，Ragas 在 Windows Anaconda 环境有 SSL 证书冲突，底层 aiohttp 的 SSL 握手会失败。但更重要的是能力考量——MRR、Hit@K、Precision@K 的计算逻辑极其简单，就是为了这几行数学引入一个包含 datasets、langchain 依赖链的第三方框架，不合理。自己实现不到 100 行，完全可控。

对于 Faithfulness 等需要 LLM 的能力，我写了一套独立的 LLM-as-Judge 评测脚本，用 Ollama 接入本地 qwen2.5:7b 跑三维打分，Prompt 模板参考了 Ragas 的思路。这样做的好处是完全掌握评测链路，不依赖黑盒库。"

---

## 产出文件

- `src/evaluation/metrics.py` — 共享指标函数（mrr/hit_at_k/precision_at_k/recall_at_k/is_relevant/load_test_cases）
- `src/evaluation/retrieval_eval.py` — 自实现检索评测（四指标 + 消融分析，引用 metrics.py）
- `src/evaluation/llm_eval.py` — LLM-as-Judge 生成评测（Faithfulness/Relevancy/ContextRecall）
- `src/evaluation/agent_eval.py` — 改写评测（A/B 效果对比 + 语义保真度 + CE 阈值校准）
- `src/evaluation/acceptance_eval.py` — 端到端验收（拒答集 / 多证据集 / 裁判）
- `src/evaluation/experiments.py` — 消融 + 分类别 + 参数扫描 + 延迟剖析 + 单 query 深挖
- `data/test_queries.json` — 18 条 GoldenTestSet（四类分层，含 golden_answer）
- `data/unanswerable_queries.json` — 10 条拒答集
- `data/multi_evidence_queries.json` — 3 条多证据集
- `experiments/llm_evaluation_results.json` — ⚠️ 仍是旧语料 3 条 / 3b 自评，**新语料 × 18 条 × 7b 裁判待跑**
- `experiments/agent_evaluation_results.json` — 改写评测结果（改写效果 + 阈值校准）

## 相关章节

- [[ch04-RRF融合与重排序]] — 评测脚本分别测试每个融合阶段的指标
- [[ch03-双路召回]] — BM25 和 Vector 的原始结果作为评测基线
- [[ch05-Streamlit前端]] — 评测结果可以在前端直观验证
