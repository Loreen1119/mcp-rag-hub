# 企业知识库问答 Agent — Phase 0-1B 完整执行方案

**目标**：将 RAG 系统从"检索过程演示"升级为"可回答、可拒答、可演示"的完整产品。  
**时间窗口**：3-4 天  
**验收标准**：四类用例通过、拒答门控数据驱动、故障降级可靠

---

## 📊 Phase 0：数据驱动阈值选择（Day 1）

### 0.1 数据集创建

#### `data/unanswerable_queries.json` （10 条）
```json
[
  {
    "query_id": "U01",
    "query": "公司今年的财务报表在哪里可以下载？",
    "category": "out_of_domain",
    "expected_status": "insufficient_evidence",
    "reasoning": "知识库仅包含 RAG 技术文档，完全不涉及公司财务信息"
  },
  {
    "query_id": "U02",
    "query": "如何为这个知识库配置 Milvus 集群？",
    "category": "near_miss",
    "expected_status": "insufficient_evidence",
    "reasoning": "文档讨论向量库选型与 ChromaDB 使用，未提及 Milvus 的特定配置步骤"
  },
  {
    "query_id": "U03",
    "query": "本项目生产环境的 SLA 和并发上限是多少？",
    "category": "missing_detail",
    "expected_status": "insufficient_evidence",
    "reasoning": "文档未涉及生产部署指标、SLA 承诺或性能规格"
  },
  {
    "query_id": "U04",
    "query": "如何通过 Kubernetes 部署该项目？",
    "category": "missing_feature",
    "expected_status": "insufficient_evidence",
    "reasoning": "文档未包含 K8s 部署指南，仅有本地运行与 Streamlit 启动说明"
  },
  {
    "query_id": "U05",
    "query": "项目是否支持 PostgreSQL 作为向量数据库后端？",
    "category": "near_miss",
    "expected_status": "insufficient_evidence",
    "reasoning": "文档明确使用 ChromaDB，提及过向量库选型但未包含 PostgreSQL pgvector 的集成方案"
  },
  {
    "query_id": "U06",
    "query": "如何配置 OpenAI GPT-4o 的 API Key 进行回答生成？",
    "category": "near_miss",
    "expected_status": "insufficient_evidence",
    "reasoning": "系统使用 Ollama 本地模型，文档未涵盖 OpenAI API 接入方法"
  },
  {
    "query_id": "U07",
    "query": "谁是本项目的主要负责人，联系方式是什么？",
    "category": "out_of_domain",
    "expected_status": "insufficient_evidence",
    "reasoning": "知识库不包含组织结构或人员信息"
  },
  {
    "query_id": "U08",
    "query": "知识库中有哪些员工的简历或个人信息？",
    "category": "out_of_domain",
    "expected_status": "insufficient_evidence",
    "reasoning": "知识库不包含员工隐私数据"
  },
  {
    "query_id": "U09",
    "query": "该系统是否支持扫描 PDF 的图像 OCR 和表格识别？",
    "category": "near_miss",
    "expected_status": "insufficient_evidence",
    "reasoning": "文档提及 PDF 解析为文本，但未涵盖 OCR、图像提取或表格结构化的实现"
  },
  {
    "query_id": "U10",
    "query": "Cross-Encoder 模型训练时使用了什么标注数据集？",
    "category": "near_miss",
    "expected_status": "insufficient_evidence",
    "reasoning": "文档说明使用预训练模型 ms-marco-MiniLM-L-6-v2，未提及其训练细节"
  }
]
```

#### `data/multi_evidence_queries.json` （3 条）
```json
[
  {
    "query_id": "M01",
    "query": "解释 RRF 融合和 Cross-Encoder 精排如何协同工作来改进检索效果",
    "expected_status": "answered",
    "required_sources": [
      {
        "description": "RRF 融合的原理与实现",
        "keywords": ["RRF", "rank fusion", "BM25", "vector", "分数归一化"]
      },
      {
        "description": "Cross-Encoder 精排的作用与局限",
        "keywords": ["Cross-Encoder", "重排序", "ms-marco", "MiniLM", "Top-K"]
      }
    ],
    "reasoning": "完整答案需要同时说明 RRF 如何融合多路召回、Cross-Encoder 如何在融合基础上进行精排"
  },
  {
    "query_id": "M02",
    "query": "为什么这个项目同时采用 BM25 和向量检索，而不是只用其中一种？",
    "expected_status": "answered",
    "required_sources": [
      {
        "description": "BM25 关键词检索的优势与局限",
        "keywords": ["BM25", "精确匹配", "词频", "jieba", "分词"]
      },
      {
        "description": "向量语义检索的优势与互补性",
        "keywords": ["embedding", "向量", "语义", "ChromaDB", "相似度"]
      }
    ],
    "reasoning": "需要对比两种方法的优劣，说明为什么混合检索优于单一方法"
  },
  {
    "query_id": "M03",
    "query": "LangGraph Agent 的状态机设计如何利用检索结果来改写查询和生成答案？",
    "expected_status": "answered",
    "required_sources": [
      {
        "description": "LangGraph 状态机的五节点设计与条件路由",
        "keywords": ["LangGraph", "state machine", "analyze", "retrieve", "check_results"]
      },
      {
        "description": "查询改写和答案生成的具体逻辑",
        "keywords": ["rewrite_query", "generate_answer", "Ollama", "LLM"]
      }
    ],
    "reasoning": "需要同时理解状态机的整体架构和每个关键节点的业务逻辑"
  }
]
```

### 0.2 数据质量审查清单

**unanswerable_queries.json 审查**
- [ ] U01-U10 在当前知识库（docs/）中确实无答案或信息明显不足
- [ ] 每条的 `reasoning` 都能解释为什么 expected_status 是 insufficient_evidence
- [ ] 10 条涵盖三个分类（out_of_domain / near_miss / missing），分布合理

**multi_evidence_queries.json 审查**
- [ ] M01-M03 的 golden source 确实分布在两个不同的文档片段（chunk）
- [ ] 不能仅用 Top-1 片段回答，必须综合两个来源
- [ ] `required_sources` 中的关键词足以精准定位 golden chunk

### 0.3 Phase 0 输出验收

运行脚本后检查三件产物：
```
experiments/
├── answerability_score_analysis.json    ← 原始数据 + 混淆矩阵
├── answerability_score_analysis.md      ← 可读报告 + 阈值建议
└── retrieval_miss_cases.json            ← 检索漏掉的 case
```

**验收标准**：
- [ ] 阈值推荐有清晰的约束式理由（FP 优先 < FN）
- [ ] 无答案问题确实被正确拒答（False Acceptance < 30%）
- [ ] retrieval_miss 的 case 都被记录并可解释
- [ ] 多证据覆盖率 ≥ 66%（至少 2/3 的多证据问题能在 Top-3 中找到）

---

## 🔧 Phase 1A：结构化答案 + 校验（Day 2）

### 1A.1 配置更新 config.py

```python
# ===== Phase 0 所得配置 =====
CE_ANSWER_THRESHOLD = 2.5          # 从 answerability_score_analysis.md 读取推荐值
ANSWER_TOP_K = 3                   # 生成答案用的证据数
RETRIEVAL_CHECK_TOP_K = 5          # 判定检索是否充分的阈值
MIN_EVIDENCE_COUNT = 1             # 全局最低证据数

# ===== 系统故障降级 =====
LLM_TIMEOUT_SECONDS = 20           # Ollama 调用超时
LLM_MAX_RETRIES = 1                # 仅对网络错误重试一次
```

### 1A.2 新增 answer_validator.py

关键类：
- `AnswerStatus`：answered | insufficient_evidence | generation_unavailable | invalid_output
- `RefusalReason`：按 status 分层的拒答原因枚举
- `AnswerValidator`：校验 LLM 返回的 JSON 和引用合法性

### 1A.3 改造 agent.py:generate_answer()

返回格式：
```json
{
  "status": "answered | insufficient_evidence | generation_unavailable | invalid_output",
  "answer": "...",
  "used_source_ids": [1, 2],
  "reason": "low_retrieval_score | insufficient_context | timeout | ..."
}
```

关键变化：
- Prompt 明确告诉 LLM 只能引用 [1][2][3]
- 编号从 1 开始（对应 Top-K 顺序）
- 异常处理分类（超时/连接错误 vs JSON 解析失败）

### 1A.4 单元测试

创建 `tests/test_phase1a.py`，验证：
- [ ] 正常回答的校验
- [ ] 源 ID 超出范围的拒绝
- [ ] 引用未声明源的拒绝
- [ ] 信息不足拒答
- [ ] 生成不可用的处理

运行：
```bash
python -m pytest tests/test_phase1a.py -v
```

### 1A.5 单元验收清单

- [ ] 所有单元测试通过
- [ ] generate_answer() 返回合法 JSON
- [ ] 校验逻辑正确判断引用有效性
- [ ] 异常处理（Ollama 不可用、超时）的降级正确

---

## 🎨 Phase 1B：UI 与完整验收（Day 3）

### 1B.1 改造 app.py

添加"问答模式"和"调试模式"双 Tab：

**问答模式**：
- status=answered → 显示答案 + [1][2][3] 引用 + Top-3 证据卡片
- status=insufficient_evidence → "知识库信息不足" + 已检索内容
- status=generation_unavailable → "服务暂不可用，但已找到证据" + 证据卡片
- status=invalid_output → "处理出错" + 证据卡片

**调试模式**：
- 保留现有四阶段检索展示
- 显示检索迭代历史
- 显示改写历史

### 1B.2 四类场景验收

运行 `streamlit run app.py`，测试：

**场景 1：有答案（36 条 answerable）**
- [ ] 问答模式展示 LLM 生成的答案
- [ ] 引用编号正确（[1][2] 等）
- [ ] 来源卡片与引用对应
- [ ] 调试模式显示四阶段检索结果

**场景 2：无答案（10 条 unanswerable）**
- [ ] 问答模式显示"知识库信息不足"
- [ ] 显示拒答原因（如"low_retrieval_score"）
- [ ] 下方展示已检索到的相关内容
- [ ] 不生成虚假答案

**场景 3：多证据问题（3 条 multi_evidence）**
- [ ] 答案同时引用多个来源（[1][2] 或 [1][2][3]）
- [ ] 引用覆盖所有必要信息点

**场景 4：LLM 不可用（系统故障）**
- [ ] 显示"回答生成服务暂不可用"
- [ ] 仍展示已检索的证据
- [ ] 不说"知识库信息不足"（避免混淆业务与技术问题）

---

## 📋 Day-by-Day 执行清单

### Day 1（Phase 0）

- [ ] 创建 `data/unanswerable_queries.json`（10 条）
- [ ] 创建 `data/multi_evidence_queries.json`（3 条）
- [ ] 人工审查两个数据集质量
- [ ] 创建 `src/evaluation/phase0_answerability_analysis.py`
- [ ] 运行脚本，生成三件产物：
  - `experiments/answerability_score_analysis.json`
  - `experiments/answerability_score_analysis.md`
  - `experiments/retrieval_miss_cases.json`
- [ ] 根据分析结果更新 `config.py` 的 `CE_ANSWER_THRESHOLD`

### Day 2（Phase 1A）

- [ ] 新增 `src/answer_validator.py`（校验逻辑）
- [ ] 更新 `config.py`（6 个新参数）
- [ ] 改造 `agent.py:generate_answer()`（结构化输出）
- [ ] 创建 `tests/test_phase1a.py`
- [ ] 运行单元测试全过：
  ```bash
  python -m pytest tests/test_phase1a.py -v
  ```
- [ ] 验证 agent.py 独立运行：
  ```bash
  python agent.py --query "RRF 如何工作？"
  ```

### Day 3（Phase 1B）

- [ ] 改造 `app.py`（问答 + 调试双 Tab）
- [ ] 测试四个场景（有答案 / 无答案 / 多证据 / LLM 不可用）
- [ ] 逐条验证验收清单
- [ ] 拍截图或录屏演示

### Day 4（可选）

- [ ] 整理演示脚本：3-5 个问题的 Q&A 示例
- [ ] 更新 README（增加"问答 Agent"相关说明）
- [ ] 提交这版代码

---

## ✅ 完成后的产品定位

**产品名称**：企业知识库问答 Agent

**核心能力**：
1. ✅ 混合检索（BM25 + 向量）+ RRF 融合 + Cross-Encoder 精排
2. ✅ 结构化 LLM 生成（带引用编号和来源溯源）
3. ✅ 数据驱动的拒答门控（不回答不确定的问题）
4. ✅ 可降级的故障处理（LLM 不可用时仍可展示证据）
5. ✅ 双模式 UI（问答模式 for 用户，调试模式 for 工程师）

**面试亮点**：
- "我实现了从 36 条评测集的数据分布出发，用混淆矩阵选择拒答阈值的方法"
- "LLM 故障时不会伪装成知识不足，而是明确告诉用户'服务暂不可用'"
- "所有引用都可机器校验，避免 LLM 编造文档名"
- "支持多阶段拒答门控：CE 分数 + 证据数量 + LLM 信息不足判断 + 引用校验"
