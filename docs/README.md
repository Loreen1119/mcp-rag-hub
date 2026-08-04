# docs/ — mcp-rag-hub 知识库数据源

> 这里放技术参考文档，会被 RAG 管线加载和索引。个人踩坑笔记在 [`journal/`](../journal/README.md)，系统学习资料在 [`docs_knowledge/`](../docs_knowledge/README.md)。

## 你关心什么？

| 关心的问题 | 看这篇 |
|-----------|--------|
| RAG 核心概念（检索/生成/进阶策略） | [rag-intro.md](./rag-intro.md) |
| Embedding 模型怎么选 | [embedding-guide.md](./embedding-guide.md) |
| 切片策略有哪些，代码文件怎么切 | [chunking-strategies.md](./chunking-strategies.md) |
| AST 分块方案怎么实现的 | [ast-chunking-plan.md](./ast-chunking-plan.md) — **已实现** |
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
├── sample_rag_paper.md        # RAG 技术综述（示例/模板）
└── sample_notes.txt           # 项目技术栈早期草稿
```
