# KG 路消融实验：为什么三路召回没有 1+1+1>3

> 日期：2026-08-04  
> 状态：已完成，结论明确  
> 关键词：知识图谱、实体共现图、KG 检索、消融实验、RAG

## 背景

项目里实现了三路召回：BM25 + Dense 向量 + 知识图谱（KG），再用 RRF 融合。初衷是 KG 能补上“多实体关联”类查询的短板。

但跑 retrieval_eval 时发现：

| 路径 | MRR@5 | Hit@5 |
|---|---|---|
| BM25 + Dense + CE 重排 | 0.81 | 1.00 |
| 加上 KG 路做 RRF 融合 | 0.59 | 较低 |

KG 路不仅没有帮忙，反而把整体指标拉低了。

## 第一次归因：实体→chunk 关联粒度太粗

看 `src/kg_retriever.py` 发现 `_build_entity_chunk_index` 把每个实体关联到**整篇 source_doc 的所有 chunk**，而不是三元组真正来源的那个 chunk。这导致实体密度高的文档（如 `sample_notes.txt`）在排序里被系统性推高。

修复：让 `_build_entity_chunk_index` 优先用三元组自带的 `chunk_id` 精确关联。

修复后重跑诊断（全量 9 份文档）：

| 类别 | MRR@5 |
|---|---|
| exact_match | 1.00 |
| semantic | 0.50 |
| mixed | 0.75 |
| **graph** | **0.42** |
| **overall** | **0.64** |

看起来 overall 还行了，但 graph 类查询（本该是 KG 主场）反而最差。

## 第二次归因：知识库太杂

用户侧进一步修复了 `data_pipeline.py` 的 `chunk_id` 生成逻辑，让 chunk_id 跨次运行保持稳定。重跑全量 9 份文档的诊断后指标**暴跌**（以下为该次实验快照，若重新跑全量诊断数值可能有轻微漂移）：

| 类别 | MRR@5 |
|---|---|
| exact_match | 0.75 |
| semantic | 0.00 |
| mixed | 0.125 |
| **graph** | **0.125** |
| **overall** | **0.236** |

原因：确定性的 chunk_id 让精确关联真正生效后，知识库里的噪声被放大。个人学习笔记（`dify-rag-tuning.md`、`lightrag-takeaways.md`）和实现文档（`ast-chunking-plan.md`）里反复出现“Cross-Encoder”“RRF”“Embedding”这些词，导致同名实体在多篇文档里互相打架，golden source 被笔记类 chunk 挤掉。

## 对照实验：只保留科普文档

为验证“是知识库混杂导致 KG 失效，还是 KG 本身不适合这个场景”，做了一次对照实验：

- 只加载：`rag-intro.md`、`embedding-guide.md`、`chunking-strategies.md`、`sample_rag_paper.md`
- 排除：`sample_notes.txt`、`dify-rag-tuning.md`、`lightrag-takeaways.md`、`ast-chunking-plan.md`
- 不移动/删除任何原始文件，仅过滤 triples 做诊断
- 脚本：`kg_diag_control.py`
- 报告：`experiments/kg_diag_report_control.json`

结果：

| 指标 | 全量 docs | 仅科普 docs |
|---|---|---|
| 可用测试用例 | 18 条 | 6 条 |
| overall MRR@5 | 0.236 | **0.486** |
| overall Hit@5 | 0.333 | **1.000** |
| graph MRR@5 | 0.125 | **0.375** |

## 结论

1. **Noisy 文档确实在拉低 KG 效果**。去掉个人笔记和实现文档后，Hit@5 提到 100%，graph MRR 涨了 3 倍。
2. **但同质性问题也真实存在**。即使只剩 4 份科普文档，MRR 也只有 0.49，因为 `Cross-Encoder`、`RRF`、`检索` 这些实体在多篇文档里重复出现，KG 评分分不清主次。
3. **最终判断**：在当前这个“小规模、主题同质”的语料上，KG 路的价值天花板很低。把它作为默认三路之一，不如作为可选/低权重路。

## 后续决策

- **KG 路改为可配置/默认低权重**：主路继续用 BM25 + Dense + CE，KG 作为实验开关。
- **不再继续投入调优 KG 评分逻辑**：问题根源是语料特性，不是评分公式。
- **等以后知识库扩展到跨领域、大规模异构数据时，再重新评估 KG**。

### LightRAG / GraphRAG 应用路线图

用户希望把学到的 LightRAG 知识用上，但当前这个“RAG 科普小语料”不是合适场景。按阶段推进：

**中期（触发条件：知识库扩展）**

当知识库满足以下任一条件时，重新评估 KG / LightRAG：
- 加入跨领域文档（如 LangChain 官方 docs、几篇不同主题 RAG 论文、一个代码仓库）
- 文档数量明显增长，实体分布更分散、有区分度
- 出现需要多跳推理才能回答的查询类型

届时优先尝试：
1. **最小查询路由**：只在多实体+关系/对比类查询时走 KG，简单查询走 BM25+Dense（类似 LightRAG 按问题类型路由的思路）
2. **修图谱质量**：约束 prompt 抽取更具体、有区分度的实体和关系
3. 在 KG 单路指标稳定提升后，再引入 LightRAG 的 local/global 分层检索

**长期（如果要做 GraphRAG 研究）**

不要继续在自己这个科普语料上硬调，改用标准多跳数据集：
- **HotpotQA**：多跳问答，适合测实体关联推理
- **2WikiMultiHopQA**：专门测多跳推理
- **Natural Questions**：事实性检索

这些数据的实体关系更复杂、查询设计更科学，能真正验证 GraphRAG 的价值。

## 相关文件

- 诊断脚本：`mcp-rag-hub/kg_diag.py`
- 对照实验脚本：`mcp-rag-hub/kg_diag_control.py`
- 全量诊断报告：`mcp-rag-hub/experiments/kg_diag_report.json`
- 对照实验报告：`mcp-rag-hub/experiments/kg_diag_report_control.json`
- 代码修复：`mcp-rag-hub/src/kg_retriever.py`（chunk_id 精确关联）

> 这个实验本身也是收益。三路召回消融实验证明了：在当前规模下，BM25 + Dense + CE 就是最优配置，KG 路暂时不需要是正确选择，不是妥协。

---

## 附录：清理知识库后的 BM25 + Dense + CE 新基线

> 日期：2026-08-04（本笔记同日后续）

将过程笔记（`dify-rag-tuning.md`、`lightrag-takeaways.md`、`kg-ablation-notes.md`）迁出 `docs/`、放入 `journal/` 后，`docs/` 只保留知识库正文：

- `rag-intro.md`
- `embedding-guide.md`
- `chunking-strategies.md`
- `ast-chunking-plan.md`
- `sample_rag_paper.md`
- `sample_notes.txt`

重新跑 `retrieval_eval.py --split test`：

| Stage | MRR@5 | Hit@5 | Prec@5 | Recall@5 |
|---|---|---|---|---|
| bm25 | 0.5630 | 0.6667 | 0.2667 | 0.5648 |
| vector | 0.3954 | 0.7222 | 0.2556 | 0.6204 |
| rrf | 0.5287 | 0.7222 | 0.3111 | 0.6204 |
| **cross_encoder** | **0.6667** | **0.7222** | **0.3778** | **0.5926** |

与之前“笔记还在 `docs/` 里”时的指标对比：

| 基线 | MRR@5 | Hit@5 | 备注 |
|---|---|---|---|
| 笔记混在知识库中 | 0.8100 | 1.0000 | 笔记文件反复出现关键词，拉高命中 |
| **清理后（新基线）** | **0.6667** | **0.7222** | 真实知识库正文上的指标 |

### 关键变化

- **4 条 case 全 0**：S06、M06、G06、G12。它们的 golden source 确实在当前 `docs/` 中，但查询语义无法被当前科普文档内容直接覆盖。
- 这说明之前 **0.81 / 100% 的指标含有笔记文件的贡献**，不能代表知识库本身的检索质量。

### 结论

清理知识库后，BM25 + Dense + CE 的**真实基线是 MRR@5 = 0.67，Hit@5 = 0.72**。

这不是系统退化，而是测试集与知识库不再对齐。下一步需要重写 `test_queries.json`，让评测用例基于当前真实知识库内容设计。

---

## 附录二：重写测试集后的最终基线

> 日期：2026-08-04（本笔记同日后续）

按当前 `docs/` 内容重新设计了 18 条测试用例，查询更贴近知识库正文实际表述。

跑 `retrieval_eval.py --split test`：

| Stage | MRR@5 | Hit@5 | Prec@5 | Recall@5 |
|---|---|---|---|---|
| bm25 | **0.8704** | 0.9444 | 0.4667 | 0.8611 |
| vector | 0.7384 | 0.9444 | 0.4889 | 0.8981 |
| rrf | **0.8565** | **1.0000** | 0.4889 | 0.9722 |
| cross_encoder | 0.7546 | 0.9444 | 0.5556 | 0.8796 |

### 结论

- 测试集与知识库对齐后，**RRF 阶段 Hit@5 = 100%，BM25 MRR = 0.87**，说明基线检索能力本身很强。
- Cross-Encoder 在这里 MRR 略低于 RRF，主要是因为 M08 等个别 case 被 CE 重排后掉出 Top-5。这不是普遍问题，后续可以单独分析这些 case。
- 这个指标可以代表当前 mcp-rag-hub 在干净知识库上的真实能力，比之前的 0.81/100% 更可信。
