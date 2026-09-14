# 项目路线图（2026-09-12 更新）

> 定位：面向企业内部技术文档的知识库问答 Agent。
> 暂不称"多模态"——尚未支持图片、表格、OCR / 扫描 PDF。等补上再改主标题。

---

## 一、已完成

### 语料与检索基线 ✅（2026-09-12 / 09-14）

- 索引语料已换成**真实第三方文档**，用 `MCP_RAG_CORPUS` 切换，各自独立 Chroma 集合：
  - `corpora/fastapi-zh/` —— FastAPI 中文文档 122 篇 / 677KB / **1445 chunk**（MIT）
  - `corpora/vue-zh/` —— Vue 3 官方中文文档 110 篇 / 941KB / **1804 chunk**（CC BY 4.0，图片除外）
  - 原 `docs/`（项目自身文档）**不再参与索引**；两份语料的来源与许可声明见 `corpora/README.md`
- 检索基线（fastapi-zh，18 条 golden）：
  BM25 MRR 0.5469 → Vector 0.7246 → RRF 0.7833 → **CE 0.7889 / Hit@5 100%**
- 三套验收集已按新语料重写并全部通过可考证性校验（`scripts/validate_golden.py`，问题数 0）
- ⚠️ `vue-zh` 的**向量索引与三套验收集尚未配齐**（换语料必然要重做）

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

实测数据（**旧语料**：36 条可答 / 10 条不可答。该语料及其测试集已于 2026-09-12 更换，
旧文件归档在 `data/_archive/`，此处保留的只有"阈值门控不可行"这个**结论**）：

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

- [ ] **Ollama 在新语料上跑通** —— 注意：这**不再是"换轻量模型"的问题**，模型侧已经解决。
      - ✅ 2026-09-12 已装 `qwen2.5:3b` 并在**旧语料**上实测跑通生成：3 条 E 类全部产出真答案，
        Faithfulness 1.0 / 0.1 / 0.7，单条生成 **14.5~43.8 秒**（纯 CPU，比早先估的 15~25s 慢）。
      - ❌ 之后语料换成 `corpora/fastapi-zh`，**一次都没再跑过**。
      - ⚠️ Ollama **不自启**，每次都要手动起；正确姿势见 `docs_knowledge/开发过程中遇到的问题.md`
        第 6、7 条（**脱离会话启动** + **不要覆盖 `OLLAMA_MODELS`**，否则命中空壳模型目录）。
      - 待办：起服务 → 在 fastapi-zh 上重跑生成与裁判评测 → 出四类验收数字。

### P1 — 实现缺口，改动都很小

- [x] ~~`_call_llm` 未使用 `LLM_TIMEOUT_SECONDS` / `LLM_MAX_RETRIES`（配置悬空）~~ —— **已完成**（2026-09-12，`agent.py`）：
      `L143` `ollama.Client(timeout=LLM_TIMEOUT_SECONDS)`、`L145` 重试循环、`L103` 把异常归一成
      `timeout` / `connection_error` / `service_error`；`LLM_TIMEOUT_SECONDS` 已从 20 提到 **120**（20s 对纯 CPU 必然超时）。
- [ ] 四类端到端验收未跑（**入口已齐，只差 Ollama 起着**）：
      | 验收 | 自动化入口 |
      |---|---|
      | 18 条可答 · 生成率 / 误拒率 | ✅ `src/evaluation/llm_eval.py` |
      | 10 条拒答 · 拒答率 / 幻觉率 | ✅ `src/evaluation/acceptance_eval.py --refusal`（2026-09-14 新增） |
      | 3 条多证据 · 引用覆盖率 | ✅ `src/evaluation/acceptance_eval.py --multi`（2026-09-14 新增） |
      | Ollama 不可用 · 降级路径 | ✅ 无需脚本（停服务再问一句） |

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
1. 起 Ollama（服务 + 模型都已就绪，只差手动拉起）  ← 解锁一切
        ↓
2. 跑四类端到端验收，出评测数字                     ← 唯一能证明"答得对"的证据
   （入口已齐：llm_eval / acceptance_eval / 停服务手测）
        ↓
3. 重测延迟（latency_profile 还是 7/27 的旧值）
        ↓
4. Phase 2A 文档上传                                ← 补"知识库"闭环
        ↓
5. Phase 3 包装（Docker / API / 架构图）
```

**两个判断修正**（都来自 2026-09-13/14 的核查）：

1. 此前写"Ollama 有 BM25 降级兜底就够了，不必处理" —— 若目标是**面试演示**，这个判断不成立：
   现场永远显示"回答生成服务暂不可用"，等于核心卖点（能回答问题）缺席。降级路径该保留，但不能是常态。
2. 此前把 P0 描述成"换轻量模型 / 腾内存" —— 也过期了：`qwen2.5:3b` 已装且实测跑通，
   **真正的缺口只剩"手动起服务 + 在新语料上重跑"**（详见 P0 条目）。
