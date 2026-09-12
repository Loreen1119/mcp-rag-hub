# 项目路线图（2026-09-12 更新）

> 定位：面向企业内部技术文档的知识库问答 Agent。
> 暂不称"多模态"——尚未支持图片、表格、OCR / 扫描 PDF。等补上再改主标题。

---

## 一、已完成

### Phase 0：数据准备 + 检索分布分析 ✅ 全部完成

| 产物 | 位置 |
|---|---|
| 无答案集 10 条 | `data/unanswerable_queries.json` |
| 多证据集 3 条 | `data/multi_evidence_queries.json` |
| 混淆矩阵 + 候选阈值 | `experiments/answerability_score_analysis.json` |
| 阈值建议报告 | `experiments/answerability_score_analysis.md` |
| 检索漏召回 case | `experiments/retrieval_miss_cases.json` |
| 多证据分析 | `experiments/multi_evidence_analysis.json` |

### Phase 1A：结构化答案 + 引用校验 ✅ 已完成

- `config.py`：`ANSWER_TOP_K=3` / `RETRIEVAL_CHECK_TOP_K=5` / `MIN_EVIDENCE_COUNT=1` / `LLM_TIMEOUT_SECONDS=20` / `LLM_MAX_RETRIES=1` 全部就位
- `agent.py:generate_answer()` 返回 `{status, answer, used_source_ids, reason}`
- `src/answer_validator.py`：`AnswerStatus`（answered / insufficient_evidence / generation_unavailable / invalid_output）+ `RefusalReason`（low_retrieval_score / insufficient_context / timeout / connection_error）分层枚举齐备
- 引用编号 `[1][2][3]` 稳定化 + `used_source_ids ⊆ {1,2,3}` 校验

### Phase 1B：UI 双模式 + 四类状态 ✅ 已完成

- `app.py` 有 `st.tabs(["问答模式", "调试模式"])`
- 四类状态全部渲染：`answered`（答案 + 引用编号）/ `insufficient_evidence`（拒答原因）/ `generation_unavailable`（故障原因 + BM25 原文拼接）/ 兜底 `invalid_output`
- 证据卡片带"已引用"标记 + CE 分数

---

## 二、重大变更：CE 阈值拒答门控已废弃 ⚠️

**原 plan 的核心设计是「用 CE Top-1 分数卡阈值做拒答」——实测后已推翻并删除。**

实测数据（36 可答 / 10 不可答）：

| 阈值 | 放行可答 | 误拒可答 | 误放无答案 | 误接受率 |
|---|---|---|---|---|
| 7.81 | 12 | 24 | 2 | 20% |
| 7.0 | 27 | 9 | 7 | 70% |

可答均值 **7.4248**、不可答均值 **7.0910**，分布严重重叠 —— **CE Top-1 分数对「可答/不可答」没有区分能力**，卡任何一条线都是赌。

**当前实现**：`CE_ANSWER_THRESHOLD` 常量已从 `config.py` 删除；拒答交给 LLM + validator 判断，CE 分数只用于排序和展示。

**影响**：原 plan 的 Phase 1C（阈值精调）已无意义，不要再去找"最优阈值"。

---

## 三、未完成清单（按优先级）

### P0 — 阻塞面试演示，必须解决

- [ ] **Ollama 实际可用**。当前 `qwen2.5:7b` 需 5.6 GiB，本机内存不足加载失败；线上所有答案都走 BM25 降级拼接。
      → 面试现场若只能演示"服务暂不可用 + 原文拼接"，产品故事讲不完整。
      → 方向：换轻量模型（qwen2.5:3b / phi-3.5）或解决内存占用。

### P1 — 实现缺口，改动都很小

- [ ] `_call_llm` 未使用 `LLM_TIMEOUT_SECONDS` / `LLM_MAX_RETRIES`（配置悬空）。
      `agent.py:97` 的 `ollama.chat()` 没传 timeout、无重试；失败一律 `except: return ""`，reason 硬编码 `connection_error`，无法区分超时/连接错误/服务错误。
- [ ] 四类端到端验收未跑：18 条答案生成率/误拒率、10 条拒答率/幻觉率、3 条引用覆盖率 —— 全部依赖 Ollama 可用。

### P2 — Phase 2A：文档管理（原 plan 第 2 阶段）

- [ ] Streamlit 上传入口（`app.py` 目前**没有** `file_uploader`，只有 sidebar）
- [ ] 文档列表 / 删除
- [ ] 上传后触发全量重建（先不做真增量）

### P3 — Phase 2B：真正的增量索引

- [ ] 文件级 manifest（当前是 corpus-hash 语料集级判定，文档一变就整体重建）
- [ ] chunk_id 稳定化
- [ ] 向量库按文档删除/新增

### P4 — Phase 3：展示与交付（面试前）

- [ ] Docker / docker-compose（根目录无 Dockerfile）
- [ ] FastAPI 服务层 + OpenAPI
- [ ] 登录/权限
- [ ] 架构图、评测报告展示页、演示数据集

---

## 四、下一步建议

按依赖关系，**先解 P0，否则后面全都演示不出来**：

```text
1. 让 Ollama 真正跑起来（换小模型 / 腾内存）      ← 解锁一切
        ↓
2. _call_llm 接上 timeout + 重试 + 错误分类       ← 20 行以内
        ↓
3. 跑四类端到端验收，出评测数字                   ← 面试可讲的证据
        ↓
4. Phase 2A 文档上传                              ← 补"知识库"闭环
        ↓
5. Phase 3 包装（Docker / API / 架构图）
```

**一个判断修正**：此前认为"Ollama 有 BM25 降级兜底就够了，不必处理"。若目标是**面试演示**，这个判断不成立 —— 现场永远显示"回答生成服务暂不可用"，等于核心卖点（能回答问题）缺席。降级路径该保留，但不能是常态。
