# RAG 知识检索系统 — 简易版方案

## 目标

两周内完成一个**面试能讲清楚原理、核心代码能手写伪代码**的 RAG 检索系统。AI 负责写代码，你负责理解原理 + 跑实验拿真实数字 + 准备面试问答。

## 砍完后保留的核心链路

```
PDF/TXT → 切片 → BM25 + ChromaDB 双路召回 → RRF 融合 → Cross-Encoder 重排 → Streamlit → Ragas 评测
```

砍掉的：FastMCP / Metadata 增强管线 / 多格式高级解析 / 30 组 Query（砍到 10~15 组）。

---

## 模块拆分（5 天 × 5 模块）

### 模块 1：文档加载与切片管线

**要写的文件**：`data_pipeline.py`

| 输入 | 处理 | 输出 |
|------|------|------|
| `docs/` 下若干 `.txt` 文件 | 读文件 → 按 512 字符切片，128 字符重叠 → 每片带文件名 + 位置元数据 | `List[Chunk]` |

**面试必问原理**：
- 为什么 512/128 这个切法？（太长→检索精度降，太短→语义不完整；128 重叠是为了防止关键信息恰好落在两片交界处丢失）
- `Chunk` 数据结构里 Metadata 存什么？（source 文件名、chunk_index、前后文指针，用于检索命中后溯源）
- 为什么不用 LangChain 的 RecursiveCharacterTextSplitter？（可以用，但要理解它底层也是滑窗 + 分隔符优先级，面试时能说清楚就行）

---

### 模块 2：BM25 + ChromaDB 双路召回

**要写的文件**：`retrievers.py`

| 路径 | 库 | 原理 |
|------|-----|------|
| 关键词召回 | `rank_bm25` | 统计 TF-IDF 词频权重，对专有名词/编号匹配极强 |
| 语义召回 | `ChromaDB` | 默认 all-MiniLM-L6-v2 做 embedding，捕获同义词/语义相似 |

**面试必问原理**：
- BM25 和 TF-IDF 有什么区别？（BM25 加了长度归一化 + TF 饱和曲线，长文档不会因为词频高就霸占高分）
- 为什么用 all-MiniLM-L6-v2？（384 维，轻量，本地 CPU 毫秒级推理。和你的 Low-Code 项目里手写 384 维向量库形成知识闭环——面试官会注意到两个项目的向量维度一样，你能解释为什么）
- ChromaDB 这次为什么能用了？（Windows 环境之前因为 onnxruntime DLL 问题你手写了向量库。ChromaDB 包的解决方法是 `pip install chromadb` 时指定 `onnxruntime` 版本，或者直接用 sentence-transformers 独立做 embedding 再存入 ChromaDB。面试时被问"这次怎么不手写了"——答："ChromaDB 自带持久化和增量管理，RAG 场景文档多、更新频繁，自己维护 npy 文件不如用成熟的向量数据库"）

---

### 模块 3：RRF 融合 + Cross-Encoder 重排序

**要写的文件**：`fusion.py`

**RRF 公式**：

$$\text{RRF}(d) = \sum_{r \in R} \frac{1}{k + r(d)}$$

其中 $r(d)$ 是文档 $d$ 在第 $r$ 路检索中排第几名，$k=60$ 是平滑常数。

**面试必问原理**：
- 为什么 RRF 而不是直接加权求和？（BM25 得分量级是 0~几十，向量余弦相似度是 0~1，直接加权 BM25 的分会碾压向量分。RRF 只看排名不看绝对分数，天然消除量纲差异）
- k=60 为什么？（避免排第一的文档权重过高——1/(60+1) vs 1/(60+2) 差距平滑；k 越小排名靠前的文档权重越大，k 越大越平均。60 是学术界经验值，不是调出来的）
- Cross-Encoder 和 Bi-Encoder 区别？（Bi-Encoder 把 query 和 doc 分别编码再算相似度，快但粗糙；Cross-Encoder 把 query+doc 拼接在一起送进模型，做一次完整的注意力计算，准但慢。所以先 Bi-Encoder 粗筛 Top-20，再 Cross-Encoder 精排 Top-5）

**核心代码你要能手写**：
```python
def reciprocal_rank_fusion(rankings, k=60):
    """rankings: List[List[str]], 每路排序好的文档ID列表"""
    scores = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
```
这段就 5 行，面试写出来就是满分级回答。

---

### 模块 4：Streamlit 前端

**要写的文件**：`app.py`

功能极简：文本框输入查询 → 展示 Top-5 检索结果（来源文档 + 片段 + 相似度分数 + 重排序前后对比）。

**面试必问**：Streamlit 和 Gradio 的区别？答："Streamlit 更接近写 Python 脚本，状态管理简单；Gradio 更适合 ML 模型 demo。这个场景用户交互就是输入查询看结果，Streamlit 够用。"

---

### 模块 5：Ragas 自动化评测

**要写的文件**：`evaluate.py` + `test_queries.json`

| 指标 | 测什么 | 怎么算 |
|------|--------|--------|
| Faithfulness | 生成内容是否忠于检索到的原文？ | LLM 逐句比对 |
| Context Relevancy | 检索到的片段和问题是否相关？ | 检索出的句子中无关句占比 |
| Context Recall | 标准答案中应被检索到的信息，实际检索到了吗？ | 漏检率 |

**面试必问原理**：
- 这三个指标的 Gold Standard 怎么来？（手动标注 10~15 组 (query, 理想文档片段, 标准答案) 三元组。不需要海量数据，Ragas 的设计就是用少量高质标注做迭代评测）
- Faithfulness 从 0.62 到 0.81 是在哪一轮优化里涨的？（这组数字删掉。跑你自己的实验，拿到你自己真实的数字。前后对比至少要有基线 vs Cross-Encoder 加入后 vs RRF 调参后的三轮数据）

---

## 排期（5 天，每天 4~6 小时）

| 天 | 模块 | 产出 |
|----|------|------|
| 1 | 文档加载 + BM25 召回 | `data_pipeline.py` + `retrievers.py` 第一版，BM25 能跑通 |
| 2 | ChromaDB 向量召回 + RRF 融合 | 双路都通，RRF 能输出排序结果 |
| 3 | Cross-Encoder 重排 + Streamlit | `fusion.py` 完整 + `app.py` 界面可交互 |
| 4 | Ragas 评测 + GoldenTestSet | `evaluate.py` + 10 组手写测试用例 + 跑出基线数字 |
| 5 | 调参 + 笔记 + 面试准备 | 跑 RRF k 值和 Cross-Encoder Top-N 的消融实验，记录真实数字 |

---

## 最终简历写法（等跑完实验再填数字）

```
项目一：RAG 智能知识检索系统
开发技术：Python / ChromaDB / BM25 / Cross-Encoder / Streamlit / Ragas
项目介绍：
针对传统检索语义理解不足、关键词匹配僵化的问题，搭建融合稀疏/稠密双路召回与重排序的高性能 RAG 检索系统。
核心职责：
1. 实现 BM25 关键词检索与 ChromaDB 向量检索双路召回，通过 RRF 算法融合排序结果，消除两路得分量纲差异，配合 Cross-Encoder 对 Top-K 候选精排，Top-5 命中率从 XX% 提升至 XX%。
2. 设计 512/128 滑窗切片管线，对文档进行语义片段提取并附带 Metadata 增强，降低长文档的检索噪声。
3. 基于 Ragas 构建 15 组场景 Query 的 GoldenTestSet，引入 Faithfulness / Context Relevancy / Context Recall 三维自动化评测，建立迭代评测闭环，将 Faithfulness 从 XX 提升至 XX%。
```

**注意**：XX 跑完实验再填。绝不在面试前写自己没测过的数字。

---

## 你需要亲自动手练的 3 段核心代码

1. **RRF 融合算法**（5 行）— 上面有
2. **BM25 检索流程**（伪代码级）— 分词→算 IDF→算 TF→算 BM25 分数→Top-K 排序
3. **Cross-Encoder 推理流程**（伪代码级）— query+doc 拼接→tokenize→model forward→取 logit→排序

---

## 面试 3 个必问题 + 回答要点

1. **"BM25 和向量检索各有什么优势？为什么两路融合？"**
   BM25 对专有名词、数字、编号敏感，向量对同义词、语义相近的表达敏感。比如搜"Q3营收"，BM25 能精确命中包含字面"Q3营收"的文档；向量能匹配到"第三季度财务收入"这种字面不相关但语义一致的段落。两者互补。

2. **"RRF 和直接加权平均有什么区别？"**
   直接加权的前提是两路得分在同一量纲——但 BM25 得分为 0~几十，向量余弦相似度为 0~1，直接加权 BM25 会碾压。RRF 只比较排名不比较绝对分，天然归一化。k=60 是平滑系数，防止排名第一的文档权重过高。

3. **"Cross-Encoder 比 Bi-Encoder 好在哪里？代价是什么？"**
   Bi-Encoder 独立编码 query 和 doc，是"盲猜"相似度；Cross-Encoder 把 query+doc 拼在一起做全注意力计算，是"读完再判断"，准确率显著更高。代价是计算量——每个 (query, doc) 对都要完整推理一次，所以不能对全库做，只能对 Top-20 精排。
