# 知识点汇总

每做完一章，生成对应 `ch0X-章节名.md`，记录该章核心知识点。面试前翻这 10 个文件即可完成复盘。

---

## [[ch01-项目骨架与数据模型|第 1 章：项目骨架与数据模型]]

**学什么**：
- RAG 系统的标准分层架构（数据层→检索层→融合层→应用层→评测层）
- `Chunk` 和 `RetrievalResult` 两个核心 dataclass 的设计思路——为什么字段要预留、为什么用 dataclass 而不是 dict
- Python 项目工程规范：`src/` 布局、`config.py` 集中管理参数、依赖管理

**面试关联**：被问"你的项目怎么组织的"时，能画出分层架构图并解释每层职责

---

## [[ch02-文档加载与切片管线|第 2 章：文档加载与切片管线]]

**学什么**：
- `pdfplumber` 提取 PDF 文本的原理（不是 OCR，是解析 PDF 内部文本流）
- 为什么需要切片：Embedding 模型有最大输入长度限制；过长文本语义被稀释
- 滑窗切片的三个参数：chunk_size（512）、overlap（128）、length_function（token 还是字符）
- Chunk Metadata 的设计：source、chunk_index、page_number——用于检索命中后溯源

**面试关联**："512/128 这个数字怎么定的？""overlap 的作用是什么？""切片太短和太长各有什么问题？"

---

## [[ch03-双路召回|第 3 章：BM25 + ChromaDB 双路召回]]

**学什么**：
- BM25 的数学原理：TF 饱和曲线 + IDF 逆文档频率 + 文档长度归一化
- BM25 和 TF-IDF 的区别：BM25 的 TF 分量有上限（饱和），长文档不会因词频高霸占高分
- 为什么 BM25 对专有名词/编号/代码片段匹配强？因为这些 token 的 IDF 极高
- Embedding 模型的工作原理：将自然语言映射到高维空间，语义相近的文本距离近
- `all-MiniLM-L6-v2` 的特点：384 维、轻量、本地 CPU 毫秒级推理
- ChromaDB 的持久化机制和增量管理
- **为什么手动做 embedding 而不是让 ChromaDB 内置**：控制 embedding 过程，面试能讲清向量怎么来的

**面试关联**："BM25 和向量检索各自什么场景下强？""为什么需要两路召回？""384 维是什么概念？"

---

## [[ch04-RRF融合与重排序|第 4 章：RRF 融合与 Cross-Encoder 重排序]]

**学什么**：
- RRF（倒数排名融合）的数学公式：`RRF(d) = Σ 1/(k + rank(d))`
- 为什么用 RRF 而不是直接加权求和：BM25 得分量级（0~几十）和余弦相似度（0~1）不在同一量纲
- k=60 的作用：平滑系数，防止排名第一的文档权重过大
- Bi-Encoder vs Cross-Encoder：
  - Bi-Encoder：query 和 doc 分别独立编码→算余弦相似度，快但粗糙
  - Cross-Encoder：query+doc 拼接→一次完整注意力计算→输出相关性分数，准但慢
- 两阶段检索策略：Bi-Encoder 粗筛 Top-20 → Cross-Encoder 精排 Top-5

**面试关联**：RRF 代码能手写、能解释为什么 k=60、能说清 Bi-Encoder 和 Cross-Encoder 的本质区别

---

## [[ch05-Streamlit前端|第 5 章：Streamlit 前端]]

**学什么**：
- Streamlit 的执行模型：每次交互重新执行整个脚本，`st.session_state` 管理跨交互状态
- 结果对比展示：同一 query 在 BM25/向量/RRF/重排四个阶段的 Top-K 差异
- 片段高亮、分数可视化的前端技巧

**面试关联**："Streamlit 和 Gradio 的区别？""session_state 是做什么的？"

---

## [[ch06-Ragas评测|第 6 章：Ragas 自动化评测]]

**学什么**：
- RAG 系统评测的三个核心指标：
  - **Faithfulness（忠实度）**：生成的答案是否忠于检索到的原文？LLM 逐句比对，检测幻觉
  - **Context Relevancy（上下文相关性）**：检索到的片段中有多少是真正和问题相关的？
  - **Context Recall（上下文召回率）**：标准答案中的关键信息，检索环节是否都覆盖到了？
- GoldenTestSet 的设计方法：分层抽样，覆盖精确匹配/语义泛化/混合查询三类场景
- Ragas 的工作原理：需要一个评测 LLM（本地 Ollama 或 API），对每个指标自动打分
- "每轮优化+回归测试"闭环：改代码→跑全量评测→看指标变化→防止负向退化

**面试关联**："你怎么量化评估你的检索系统？""这三个指标分别测什么？""Faithfulness 从 XX 到 XX 是在哪一轮优化提升的？"

---

## 第 7 章：FastMCP 工具封装（待完成）

**学什么**：
- MCP（Model Context Protocol）协议的核心概念：Server / Tool / Resource / Prompt
- FastMCP 框架如何把 Python 函数变成 MCP Tool
- `@mcp.tool()` 装饰器的用法：函数签名即 Tool 接口定义
- MCP Inspector 测试 Tool 可用性
- MCP 在 AI 应用开发中的生态位：让 LLM Agent 可以"插拔式"调用外部能力

**面试关联**："MCP 是什么？""为什么要把 RAG 封装成 Tool？""FastMCP 帮你做了什么？"

---

## 第 8 章：LangGraph Agent 编排（待完成）

**学什么**：
- LangGraph 的核心概念：StateGraph / Node / Edge / Conditional Edge
- Agent 状态机设计：定义 AgentState，每个节点修改状态的一部分
- 检索决策循环：search → judge（结果够不够）→ rewrite（不够就改写 query）→ search → generate
- Agent 如何通过 MCP Client 调用第 7 章的检索 Tool
- 和 LangChain Agent 的区别：LangGraph 是显式状态图，可控性强；LangChain Agent 是黑盒推理循环

**面试关联**："Agentic RAG 和普通 RAG 有什么区别？""LangGraph 的状态图怎么设计？""Agent 怎么决策是否需要二次检索？"

---

## 第 9 章：消融实验与数据分析（待完成）

**学什么**：
- 消融实验的设计方法：控制变量法，每次只改一个模块（on/off 或参数变化）
- 实验变量：RRF k 值（10/30/60/100）、Cross-Encoder Top-N（5/10/20）、各模块开关
- 如何用 matplotlib 画对比柱状图，提炼面试能引用的数字
- 如何从数据中形成结论："加入 Cross-Encoder 后 Top-5 命中率从 XX% 提升至 XX%"

**面试关联**：所有需要填数字的地方都来自这一章

---

## 第 10 章：面试复盘（待完成）

**学什么**：
- 三段核心代码默写：RRF 算法、BM25 检索流程、Cross-Encoder 推理流程
- 三个必问题逐字稿：BM25 vs 向量、RRF vs 加权、Cross-Encoder vs Bi-Encoder
- 30 秒项目电梯演讲：一句话说清楚做了什么、为什么这样做、效果怎么样
- 追问应对逻辑：面试官深度追问时从原理层→实现层→工程层逐层展开

**面试关联**：全部
