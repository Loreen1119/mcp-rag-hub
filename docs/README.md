# docs/ — 项目文档（人读）

> ⚠️ **本目录既不是知识库语料、也不参与索引。**
> 语料在 [`corpora/`](../corpora/README.md)（当前默认 `corpora/fastapi-zh`，真实第三方文档），
> 代码通过 `config.DOCS_DIR` 读取，可用环境变量 `MCP_RAG_CORPUS` 切换；
> 索引产物落在仓库根的 `chroma_db/`（主）与 `experiments/chroma_db/`（评测隔离）。

## 文档目录的分工

2026-09-19 按「内容性质」重新划界，**目录名即用途**：

| 目录 | 放什么 | 读者 |
|---|---|---|
| **`docs/`（本目录）** | **项目文档** —— 项目是什么、怎么用、为什么这么设计 | 外部读者 / 面试官 / 未来的自己 |
| [`journal/`](../journal/README.md) | **过程笔记** —— 踩坑日志、实验记录、进度清单，强时间序 | 自己 |

> 另有一类**求职材料**（简历、面试稿、部署手册）单独放在 `career/`：因含个人信息与服务器信息，
> 只在本机维护，**不进本仓库**。

## 想了解什么？

| 关心的问题 | 看这篇 |
|-----------|--------|
| 这个项目是什么、能做什么（零术语门槛） | [项目详解.md](./项目详解.md) |
| 系统设计、算法选择与关键数据（技术向） | [技术视角详解.md](./技术视角详解.md) |
| 踩过哪些坑、哪些结论被数据推翻 | [开发过程中遇到的问题.md](./开发过程中遇到的问题.md) |
| 按构建顺序逐层复习知识点 | [chapters/](./chapters/) — ch01～ch10 递进 |
| RAG 核心概念（检索/生成/进阶策略） | [RAG核心概念.md](./reference/RAG核心概念.md) |
| Embedding 模型怎么选 | [Embedding选型指南.md](./reference/Embedding选型指南.md) |
| 切片策略有哪些、代码文件怎么切 | [切片策略.md](./reference/切片策略.md) |
| AST 分块方案怎么实现的 | [AST分块方案.md](./reference/AST分块方案.md) — **已实现** |
| 怎么用 Docker 跑起来 | [Docker部署说明.md](./reference/Docker部署说明.md) |
| RAG 技术综述 / 技术栈早期草稿（示例） | [samples/](./samples/) |

## 推荐阅读路径

```
路人视角     → 项目详解（15 分钟）
                    ↓ 还想深入了解
               技术视角详解（20 分钟）

开发者视角   → 技术视角详解（20 分钟）
                    ↓ 想动手跑代码
               仓库根 README → 项目详解 → 跑起来 → 看源码

准备面试     → chapters/ch10-面试复盘.md（核心代码 + 必问题）
                    ↓ 某章不熟
               chapters/ch0X-xxx.md（回到对应章节补漏）
                    ↓ 想聊「踩过什么坑 / 怎么发现结论错了」
               开发过程中遇到的问题.md
```

## 章节学习笔记（chapters/）

| 章节 | 内容 |
|------|------|
| [ch01](./chapters/ch01-项目骨架与数据模型.md) | RAG 系统分层架构、核心数据类设计、项目工程规范 |
| [ch02](./chapters/ch02-文档加载与切片管线.md) | PDF/Markdown/TXT 文档加载、Token 级滑动窗口切块、编码自检测 |
| [ch03](./chapters/ch03-双路召回.md) | BM25 关键词检索 + ChromaDB 向量语义检索双路召回 |
| [ch04](./chapters/ch04-RRF融合与重排序.md) | RRF 倒数排名融合、Cross-Encoder 精排、两阶段排序策略 |
| [ch05](./chapters/ch05-Streamlit前端.md) | Streamlit 交互界面、两 Tab（问答/调试）+ 四阶段调试看板 |
| [ch06](./chapters/ch06-Ragas评测.md) | 三层评测体系：检索层 MRR/Hit@K + 生成层 Ragas + 改写层 A/B |
| [ch07](./chapters/ch07-FastMCP工具封装.md) | MCP 协议、FastMCP 工具封装、MCP Inspector 调试 |
| [ch08](./chapters/ch08-LangGraph-Agent编排.md) | LangGraph 状态机、五节点 Agent 编排、条件路由与查询改写 |
| [ch09](./chapters/ch09-消融实验与数据分析.md) | 四级累积指标、两路召回互补、语料升级前后的结论修正 |
| [ch10](./chapters/ch10-面试复盘.md) | 三段核心代码默写、三个必问题逐字稿、六个进阶追问复盘 |

## 文件清单

```
docs/
├── README.md                        # 本文件 — 导航
├── 项目详解.md                       # 讲本项目：面向非技术读者
├── 技术视角详解.md                    # 讲本项目：面向技术读者
├── 开发过程中遇到的问题.md             # 讲本项目：踩坑与结论修正全集
├── chapters/                        # 逐层实现笔记（ch01～ch10）
├── reference/                       # 外围知识参考（不是本项目文档）
│   ├── RAG核心概念.md                #   RAG 核心概念速览
│   ├── Embedding选型指南.md          #   Embedding 选型参考
│   ├── 切片策略.md                   #   切片策略速览（含 AST 实践）
│   ├── AST分块方案.md                #   AST 分块方案（已实现）
│   └── Docker部署说明.md             #   Docker 部署说明
└── samples/                         # 示例 / 早期草稿（非活文档）
    ├── sample_rag_paper.md          #   RAG 技术综述（示例/模板）
    └── sample_notes.txt             #   项目技术栈早期草稿
```

> **为什么加 `reference/` 这一层**（2026-09-19）：划分依据是读者要看的是「**这个项目**」还是
> 「**相关的技术知识**」—— 前者留 `docs/` 根，后者进 `reference/`。顶层从 11 项降到 7 项。

## 关于几篇「搬过家」的文档

- **`reference/RAG核心概念.md` / `reference/Embedding选型指南.md` / `reference/切片策略.md`**：
  它们**曾经是知识库语料**（2026-09-12 之前，`docs/` 就是语料目录）。语料换成 `corpora/` 后，
  它们只作为技术参考留下；另有一份副本被 `journal/scripts/kg_diag_control.py` 当**对照实验素材**引用。
  ⚠️ 该脚本的 `CLEAN_DOCS` 是按「相对 DOCS_DIR 的路径」过滤的，所以 09-19 移入 `reference/` 时
  **同步改成了 `reference/xxx.md`** —— 不改的话对照实验会**静默漏掉**这几份文档（这个坑 09-15 移
  `sample_rag_paper.md` 时踩过一次）。
- **`lightrag-takeaways.md`**：2026-09-19 **移回 [`journal/`](../journal/README.md)**，
  并按该目录的命名约定更名为 `LightRAG源码笔记-2026-09-15.md`。
  它几经往返（原在 journal → 09-15 移到 docs → 09-19 移回 journal），最终判据是**内容性质**：
  它和 `journal/KG消融实验-2026-08-04.md` 是同一类东西（某条技术路线的评估笔记），
  应当同处一室；`docs/` 只留「讲项目本身」的文档。

## 文档口径同步记录

> ⚠️ 正文中的模型名与指标基线以 `config.py` 与 [ch09](./chapters/ch09-消融实验与数据分析.md) 为准。
> 项目在 2026-09-12 把语料从项目自身 `docs/` 换成了 `corpora/fastapi-zh`。
>
> **已对齐**：仓库根 `README.md`、面试背诵稿与简历段落（`career/`，本地维护）、
> `开发过程中遇到的问题.md`、ch09、`项目详解.md`、`技术视角详解.md`（2026-09-14 全篇重写：
> 模型名、切块参数、CE 阈值量纲、三套验收集、18 条基线、延迟标注、消融 7 配置、自实现 Judge 现状）。
> **已全部对齐**（2026-09-14）：ch02 / ch03 / ch05 / ch06 / ch08 / ch10 六篇章节笔记。
> —— 模型名（`bge-small-zh-v1.5` 512 维 / `bge-reranker-base`）、切块 256/38、CE 阈值 0.3 及其量纲陷阱、
> 18 条四类验收集（+10 拒答 / +3 多证据）、09-12 新基线、延迟口径、顶层两 Tab 界面。
> **保留了旧模型名（all-MiniLM / ms-marco）的位置，都是在解释「为什么换掉它」，是刻意保留的。**
