# journal/ — mcp-rag-hub 踩坑日志与工程笔记

> 个人实战记录，非知识库数据源，**不参与 RAG 索引和检索评测**。

## 你关心什么？

| 关心的问题 | 看这篇 |
|-----------|--------|
| 下一步做什么、哪些已经做完 | [ROADMAP.md](./ROADMAP.md) — **项目路线图（P0~P4 未完成清单）** |
| 想按步骤把 Ollama 跑起来、做四类验收 | [P0-演练手册.md](./P0-演练手册.md) — 内存预算 / 启动姿势 / 五步命令 / 会丢数据的坑 |
| 出题（golden）时查语料里到底有哪些原句 | [corpus_facts.md](./corpus_facts.md) — 关键语料逐句摘录 |
| Dify 上踩了什么坑 | [2026-07-31-dify-rag-tuning.md](./2026-07-31-dify-rag-tuning.md) — 踩坑日志 |
| 三路召回 KG 路为什么不 work | [2026-08-04-kg-ablation-notes.md](./2026-08-04-kg-ablation-notes.md) — 消融实验 |
| LightRAG 有什么可借鉴的 | [lightrag-takeaways.md](./lightrag-takeaways.md) — 概念评估 |
| 项目整体走到哪一步了 | [PROGRESS.md](./PROGRESS.md) — 阶段进度 |
| 早期执行计划 / 中断后的恢复记录 | [EXECUTION_PLAN.md](./EXECUTION_PLAN.md)、[PHASE0_RECOVERY.md](./PHASE0_RECOVERY.md) — 历史文档 |

## 文件清单

```
journal/
├── README.md                              # 本文件
├── ROADMAP.md                             # 路线图：已完成 / P0~P4 未完成清单
├── P0-演练手册.md                          # P0 可执行 runbook（内存/启动/验收/避坑）
├── PROGRESS.md                            # 阶段进度
├── corpus_facts.md                        # 语料关键句摘录（出题用）
├── 2026-07-31-dify-rag-tuning.md          # Dify 实战踩坑日志
├── 2026-08-04-kg-ablation-notes.md        # KG 路消融实验
├── lightrag-takeaways.md                  # LightRAG 学习笔记
├── EXECUTION_PLAN.md                      # 历史：早期执行计划
├── PHASE0_RECOVERY.md                     # 历史：Phase 0 中断恢复记录
├── kg_diag.py / kg_diag_control.py        # 诊断脚本：KG 路对照实验
└── phase0_answerability_analysis_v1.py    # 历史：Phase 0 分析脚本 v1（已被
                                           #   src/evaluation/phase0_answerability_analysis_v2.py 取代，
                                           #   保留是为了记录"CE 阈值门控"这条被证伪的路线）
```

知识库语料在 [`corpora/`](../corpora/README.md)，项目自身文档在 [`docs/`](../docs/README.md)，
系统学习资料在 [`docs_knowledge/`](../docs_knowledge/README.md)。

> 环境踩坑（沙箱、编码、Ollama、数据恢复通道等）统一记在
> [`docs_knowledge/开发过程中遇到的问题.md`](../docs_knowledge/开发过程中遇到的问题.md)。
