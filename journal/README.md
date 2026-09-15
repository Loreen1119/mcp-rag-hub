# journal/ — mcp-rag-hub 踩坑日志与工程笔记

> 个人实战记录，非知识库数据源，**不参与 RAG 索引和检索评测**。

## 你关心什么？

| 关心的问题 | 看这篇 |
|-----------|--------|
| **还剩什么没做、下一项做什么** | [未完成清单.md](./未完成清单.md) — **可勾选**，做完一项划掉一项 |
| 下一步做什么、为什么这个优先级 | [ROADMAP.md](./ROADMAP.md) — 优先级判断与理由 |
| 想按步骤把 Ollama 跑起来、做四类验收 | [P0-演练手册.md](./P0-演练手册.md) — 内存预算 / 启动姿势 / 五步命令 / 会丢数据的坑 |
| **端到端验收跑出什么了** | [A组验收结果-2026-09-15.md](./A组验收结果-2026-09-15.md) — 生成率 100% / 拒答率 90% / 瓶颈在生成层 |
| 出题（golden）时查语料里到底有哪些原句 | `python scripts/corpus_grep.py <输出文件>` — 一次性摘录，用完即弃 |
| Dify 上踩了什么坑 | [2026-07-31-dify-rag-tuning.md](./2026-07-31-dify-rag-tuning.md) — 踩坑日志 |
| 三路召回 KG 路为什么不 work | [2026-08-04-kg-ablation-notes.md](./2026-08-04-kg-ablation-notes.md) — 消融实验 |
| LightRAG 源码有什么可借鉴的 | [docs/lightrag-takeaways.md](../docs/lightrag-takeaways.md) — ⚠️ 已于 2026-09-15 **归档到 `docs/`**（改进方向经实测否决，源码分析部分仍有效） |

## 文件清单

```
journal/
├── README.md                              # 本文件
├── 未完成清单.md                           # 可勾选的待办清单（做完一项划掉一项）
├── ROADMAP.md                             # 路线图：已完成 / P0~P4 未完成清单 + 优先级理由
├── P0-演练手册.md                          # P0 可执行 runbook（内存/启动/验收/避坑）
├── A组验收结果-2026-09-15.md               # ★ 首轮端到端验收：生成率/拒答率/引用覆盖 + 两条异常根因
├── 2026-07-31-dify-rag-tuning.md          # Dify 实战踩坑日志
├── 2026-08-04-kg-ablation-notes.md        # KG 路消融实验
├── kg_diag.py / kg_diag_control.py        # 诊断脚本：KG 路对照实验
└── phase0_answerability_analysis_v1.py    # 历史：Phase 0 分析脚本 v1（已被
                                           #   src/evaluation/phase0_answerability_analysis_v2.py 取代，
                                           #   保留是为了记录"CE 阈值门控"这条被证伪的路线）
```

知识库语料在 [`corpora/`](../corpora/README.md)，项目自身文档在 [`docs/`](../docs/README.md)，
系统学习资料在 [`docs_knowledge/`](../docs_knowledge/README.md)。

## 已清理 / 已迁出的文件（2026-09-14、09-15）

| 文件 | 处理 | 原内容 | 为什么 |
|---|---|---|---|
| `PROGRESS.md` | 09-14 删除 | 7/27 阶段进度（7.5KB） | 含旧模型名（`all-MiniLM-L6-v2`、`ms-marco-MiniLM-L-6-v2`）与 "Ragas 评测" 口径；还指向一个**已不存在的目录** `../MetaFetch-RAG/`。功能已被 `ROADMAP.md` 取代 |
| `corpus_facts.md` | 09-14 删除 | 9/12 语料摘录 dump（23KB） | 是 `scripts/corpus_grep.py` 的一次性输出，**出题完成后即失去用途**，且一条命令可重生 |
| `PHASE0_RECOVERY.md` | 09-14 删除 | 9/8 环境排障交接（4.5KB） | 纯环境修复步骤（重建 venv 等）；且引用的 `multi_evidence_queries_v2.json` **已不存在** —— 一份指向空目标的交接文档 |
| `EXECUTION_PLAN.md` | **09-15 删除** | 9/8 的 Phase 0-1B 执行方案（11.9KB） | 计划**已全部执行完**（Phase 0/1A/1B 都落地了），而残留内容全部过期：`CE_ANSWER_THRESHOLD=2.5`（机制已废弃）、内嵌的两个验收集示例是**旧版**、写"36 条"、且它的"面试亮点"第一条正是**已被证伪的"用混淆矩阵选拒答阈值"**。零独占信息，被 ch09 / 问题集 / ROADMAP 完全覆盖 |
| `lightrag-takeaways.md` | **09-15 迁出 → `docs/`** | LightRAG 源码笔记（见 [docs/](../docs/lightrag-takeaways.md)） | 它 8/04 被迁出 `docs/` 的理由是"当时 `docs/` 还当知识库语料、笔记会污染检索"；**09-12 语料换成 `corpora/` 后该约束失效**。且它的落点（改进 KG 路）已被 09-15 消融实测否决（三路 vs 双路 ΔMRR = 0），故归档到 `docs/` 并加警示，保留其源码分析部分 |

前四份**已用 `git rm` 删除**（历史里可随时取回：`git show <commit>^:journal/<文件名>`）；
09-14 那三份另在 `%TEMP%\journal_backup_20260914\` 留过副本，09-15 两份在
`%TEMP%\journal_backup_20260915\`。删的理由都不是"看着旧"，而是
**内容是"旧口径"的第二个源头**、且**全仓库零引用**。

**保留 `kg_diag*.py` 是有意的**：它们是 `experiments/kg_diag_report*.json` 的**原始生成脚本** ——
删了那两份报告就变成无源数据（见 [experiments/README.md](../experiments/README.md)）。
`phase0_answerability_analysis_v1.py` 同理保留：它记录了「CE 阈值门控」这条被数据证伪的路线。

> ⚠️ 注：`experiments/kg_diag_report.json` 与 `2026-08-04-kg-ablation-notes.md` 里出现的
> `lightrag-takeaways.md` **是历史叙述**（记的是"当时它是噪声来源文档"），不是超链接 ——
> 文件已迁到 `docs/`，但那两处**刻意不改**，改了就是篡改历史事实。

> 环境踩坑（沙箱、编码、Ollama、数据恢复通道等）统一记在
> [`docs_knowledge/开发过程中遇到的问题.md`](../docs_knowledge/开发过程中遇到的问题.md)。
