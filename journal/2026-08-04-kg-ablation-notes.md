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

## 相关文件

- 诊断脚本：`mcp-rag-hub/kg_diag.py`
- 对照实验脚本：`mcp-rag-hub/kg_diag_control.py`
- 全量诊断报告：`mcp-rag-hub/experiments/kg_diag_report.json`
- 对照实验报告：`mcp-rag-hub/experiments/kg_diag_report_control.json`
- 代码修复：`mcp-rag-hub/src/kg_retriever.py`（chunk_id 精确关联）

> 这个实验本身也是收益。三路召回消融实验证明了：在当前规模下，BM25 + Dense + CE 就是最优配置，KG 路暂时不需要是正确选择，不是妥协。
