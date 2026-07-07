# MCP-RAG-Hub 文档导航

不知道该看哪个文件？

- **想快速了解这个项目是什么？** → [项目详解](./项目详解.md) — 零术语门槛，从用户视角讲清全貌
- **想深入理解架构和技术细节？** → [技术视角详解](./技术视角详解.md) — 完整的系统设计、算法选择与关键数据
- **想按章节复习知识点？** → [chapters/](./chapters/) — 从 ch01 到 ch10 按构建顺序递进

## 推荐阅读路径

```
你是路人 → 项目详解（15 分钟）
                ↓ 还想深入了解
           技术视角详解（20 分钟）

你是开发者 → 技术视角详解（20 分钟）
                ↓ 想动手跑代码
           README.md → 项目详解 → 跑起来 → 看源码

你在准备面试 → chapters/ch10-面试复盘.md（核心代码 + 必问题）
                    ↓ 某章不熟
               chapters/ch0X-xxx.md（回到对应章节补漏）
```

## 章节学习笔记

| 章节 | 内容 |
|------|------|
| [ch01](./chapters/ch01-项目骨架与数据模型.md) | RAG 系统分层架构、核心数据类设计、项目工程规范 |
| [ch02](./chapters/ch02-文档加载与切片管线.md) | PDF/Markdown/TXT 文档加载、Token 级滑动窗口切块、编码自检测 |
| [ch03](./chapters/ch03-双路召回.md) | BM25 关键词检索 + ChromaDB 向量语义检索双路召回 |
| [ch04](./chapters/ch04-RRF融合与重排序.md) | RRF 倒数排名融合、Cross-Encoder 精排、两阶段排序策略 |
| [ch05](./chapters/ch05-Streamlit前端.md) | Streamlit 交互界面、四标签页结果展示、session_state 状态管理 |
| [ch06](./chapters/ch06-Ragas评测.md) | 三层评测体系：检索层 MRR/Hit@K + 生成层 Ragas + 改写层 A/B 对比 |
| [ch07](./chapters/ch07-FastMCP工具封装.md) | MCP 协议、FastMCP 工具封装、MCP Inspector 调试 |
| [ch08](./chapters/ch08-LangGraph-Agent编排.md) | LangGraph 状态机、五节点 Agent 编排、条件路由与查询改写 |
| [ch09](./chapters/ch09-消融实验与数据分析.md) | 模块消融、参数扫描、延迟分析、类别细分、实验报告生成 |
| [ch10](./chapters/ch10-面试复盘.md) | 三段核心代码默写、三个必问题逐字稿、六个进阶追问深度复盘 |

## 文件结构

```
docs_knowledge/
├── README.md                  # 本文件 — 文档导航
├── 项目详解.md                 # 面向非技术读者
├── 技术视角详解.md             # 面向技术读者
└── chapters/                  # 面试复习笔记（10 章）
    ├── ch01 ~ ch10
```
