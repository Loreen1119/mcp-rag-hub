# docs/ — 项目自身的文档（人读）

> ⚠️ **这里已经不再是 RAG 的知识库语料。**
> 知识库语料已迁到 [`corpora/`](../corpora/README.md)（当前默认 `corpora/fastapi-zh`，真实第三方文档），
> 代码通过 `config.DOCS_DIR` 读取，可用环境变量 `MCP_RAG_CORPUS` 切换。
> 本目录只放项目自己的技术参考与速查笔记，供人阅读，不参与索引。
>
> 个人踩坑笔记在 [`journal/`](../journal/README.md)，系统学习资料在 [`docs_knowledge/`](../docs_knowledge/README.md)。

## 你关心什么？

| 关心的问题 | 看这篇 |
|-----------|--------|
| RAG 核心概念（检索/生成/进阶策略） | [rag-intro.md](./rag-intro.md) |
| Embedding 模型怎么选 | [embedding-guide.md](./embedding-guide.md) |
| 切片策略有哪些，代码文件怎么切 | [chunking-strategies.md](./chunking-strategies.md) |
| AST 分块方案怎么实现的 | [ast-chunking-plan.md](./ast-chunking-plan.md) — **已实现** |
| LightRAG 源码笔记 + **为什么没做图检索** | [lightrag-takeaways.md](./lightrag-takeaways.md) — ⚠️ 该方向已由实测否决，但源码分析仍可参考 |
| 怎么用 Docker 跑起来 | [docker-deployment.md](./docker-deployment.md) — 前置条件与已知限制 |
| RAG 技术综述（示例/模板） | [sample_rag_paper.md](./sample_rag_paper.md) |
| 项目技术栈早期草稿 | [sample_notes.txt](./sample_notes.txt) |

## 文件清单

```
docs/
├── README.md                  # 本文件
├── rag-intro.md               # RAG 核心概念速览
├── embedding-guide.md         # Embedding 选型参考
├── chunking-strategies.md     # 切片策略速览（含 AST 实践）
├── ast-chunking-plan.md       # AST 分块方案（已实现）
├── lightrag-takeaways.md      # LightRAG 源码笔记（原在 journal/，2026-09-15 归档到此）
├── docker-deployment.md       # Docker 部署说明
├── sample_rag_paper.md        # RAG 技术综述（示例/模板）
└── sample_notes.txt           # 项目技术栈早期草稿
```

> **关于 `lightrag-takeaways.md`**：它原本在 `journal/`（2026-08-04 从本目录迁出，理由是当时
> `docs/` 还在当知识库语料、笔记类文档会污染检索）。2026-09-12 语料换成 `corpora/` 后，
> `docs/` 不再参与索引，这个约束随之失效；且该文档的落点（改进 KG 检索）已被
> 2026-09-15 的模块消融实测**否决**（三路 vs 双路 ΔMRR = 0），故归档至此、保留其源码分析部分。
