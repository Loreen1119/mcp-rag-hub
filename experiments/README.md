# experiments/ — 实验产物

> ⚠️ **先记住一句话**：**引用任何数字前，先看下面的"语料归属"列。** 标"旧"的不可引用。
>
> ✅ **2026-09-15 起，本目录产物已基本全部更新到当前语料**（`corpora/fastapi-zh`）——
> 包括 `experiments.py --quick` 产出的消融 / 分类表现 / 参数扫描 / 延迟 / 深度追踪五类，
> 以及生成侧的两份评测（18 条可答 + 拒答 10 + 多证据 3）。
> 完整结论见 [`journal/A组验收结果-2026-09-15.md`](../journal/A组验收结果-2026-09-15.md)。

## 为什么需要这份说明

`experiments/` 现在**纳入版本控制了**（2026-09-14 起，此前被 `.gitignore` 整目录忽略）。
纳入之后立刻暴露一个问题：7 月、8 月、9 月的四批产物躺在同一层目录里，
**旧语料的数字和新语料的数字混在一起，光看文件名分不出谁是谁**。

而每次跑评测都是**原地覆盖**同名文件 —— 也就是说，新结果一落盘，旧结果就没了
（现在有 git 兜底，但要点开历史才看得出来）。

## 产物清单与语料归属

| 文件 | 生成时间 | 生成脚本 | 语料 | 能不能引用 |
|---|---|---|---|---|
| `query_deep_dive.json` | **2026-09-15** | `experiments.py` | **新** | ✅ 可用 |
| `ablation_results.json` | **2026-09-15** | `experiments.py` | **新** | ✅ **可用**（7 配置模块消融） |
| `category_breakdown.json` | **2026-09-15** | `experiments.py` | **新** | ✅ 可用（分类别表现） |
| `latency_profile.json` | **2026-09-15** | `experiments.py` | **新** | ✅ **可用**（全链路 12.8s，CE 占 99.8%） |
| `parameter_sweep.json` | **2026-09-15** | `experiments.py` | **新** | ✅ 可用 |
| `report.md` | **2026-09-15** | `experiments.py` | **新** | ⚠️ 可用，但**内含一句硬编码结论与数据矛盾**（见下 C10） |
| `kg_diag_report.json` | 2026-08-04 | `journal/kg_diag.py` | 旧 | ❌ 仅存档（KG 路诊断） |
| `kg_diag_report_control.json` | 2026-08-04 | `journal/kg_diag_control.py` | 旧 | ❌ 仅存档（对照组） |
| `answerability_score_analysis.json` | 2026-09-11 | `phase0_answerability_analysis_v2.py` | 旧 | ✅ 结论可用（见下） |
| `answerability_score_analysis.md` | 2026-09-11 | 同上 | 旧 | ✅ 结论可用（见下） |
| `retrieval_miss_cases.json` | 2026-09-11 | 同上 | 旧 | ❌ 仅存档 |
| `multi_evidence_analysis.json` | 2026-09-11 | 同上 | 旧 | ❌ 仅存档 |
| `phase0_recovery_log.json` | 2026-09-11 | 同上 | 旧 | ❌ 仅存档 |
| `llm_evaluation_results.json` | **2026-09-15** | `llm_eval.py` | **新** | ✅ 可用（18 条可答，**裁判 7b**；含 1 条被截断的假 0 分） |
| `llm_evaluation_results.recovered.json` | **2026-09-15** | 同上 + 后处理恢复 | **新** | ✅ **推荐引用这版**（截断的 0 分已恢复为真实分） |
| `acceptance_eval_results.json` | **2026-09-15** | `acceptance_eval.py` | **新** | ✅ 可用（拒答率 90% / 误答率 10% / 引用覆盖 16.7%） |
| `latency_profile.reproduced.json` | **2026-09-15** | `scripts/reproduce_latency.py` | **新** | ✅ **可用**（**CE 占 99.8%、34s/条**） |

### 两个例外说明

- **`answerability_score_analysis.*` 的结论仍然有效**：它证明的是"CE Top-1 分数对
  '可答 / 不可答'没有区分能力"（可答均值 7.42 vs 不可答均值 7.09，分布严重重叠）。
  这个结论**与具体语料无关** —— 是这个原因导致「用分数卡阈值自动拒答」的方案被废弃，
  相关常量已从 `config.py` 删除。**但表里的具体分数是旧语料的，别引用。**
- **`llm_evaluation_results.json` 已于 2026-09-15 被新语料版本覆盖**（18 条、`corpora/fastapi-zh`、**裁判 7b**）。
  此前那份是 2026-09-12 的**旧语料 3 条**（生成/裁判同为 3b），它曾是"本机能跑通 LLM 生成"的唯一原始证据 ——
  需要时从 git 历史取回：`git show d3270c2:experiments/llm_evaluation_results.json`
  （另有一份 `%TEMP%` 副本：`experiments_backup_20260915/`）。
- **⚠️ 引用三维分数前务必确认"谁当裁判"**：3b 自评无区分度（18 条里 13 条 Faithfulness 都是整 0.70）；
  7b 才有梯度。**两者评的是同一批答案**，分数差异来自裁判能力，不是答案质量变化。
- **⚠️ `llm_evaluation_results.json` 里有 1 条假 0 分**：`llm_eval.py:103` 的 `num_predict=512` 对 7b 太短，
  JSON 被截断 → `_parse_score` 静默记 0（待修项 **C8**）。**要引用请用 `.recovered.json`**，
  它把被截断的真实分（S02 = 0.9）恢复了回来。

## 当前系统的数字在哪（别在这里找）

| 想找什么 | 去哪看 |
|---|---|
| 检索四级指标（BM25 / 向量 / RRF / CE） | 仓库根 `README.md` 的指标表、`journal/ROADMAP.md` |
| 消融分析与"被数据推翻的设计" | `docs_knowledge/chapters/ch09-消融实验与数据分析.md` |
| 拒答率 / 误答率 / 引用覆盖率 / 生成率 | `acceptance_eval_results.json`、`llm_evaluation_results.json`；结论汇总见 `journal/A组验收结果-2026-09-15.md` |

## 怎么重新生成（对应未完成清单的 A 组）

```powershell
$env:HF_HUB_OFFLINE="1"; $env:TRANSFORMERS_OFFLINE="1"; $env:PYTHONIOENCODING="utf-8"
# ⚠️ 必须连 ALL_PROXY 一起清 —— 否则 agent 走 ollama SDK(httpx) 会全线 connection_error，
#    而 llm_eval 走 requests 却正常，表现为"自检通过、主循环全挂"（2026-09-15 实测踩坑）
$env:HTTP_PROXY=""; $env:HTTPS_PROXY=""; $env:ALL_PROXY=""; $env:NO_PROXY="127.0.0.1,localhost"

# 生成层：18 条可答的三维打分（需 Ollama）
.\.venv\Scripts\python.exe -m src.evaluation.llm_eval

# 拒答 + 多证据：拒答率 / 误答率 / 引用覆盖率（需 Ollama）
# → experiments/acceptance_eval_results.json
.\.venv\Scripts\python.exe -m src.evaluation.acceptance_eval

# 消融 / 参数扫描 / 延迟（不需要 Ollama，纯 CPU，较慢）
# → ablation / category / sweep / latency / query_deep_dive / report.md
.\.venv\Scripts\python.exe -m src.evaluation.experiments --quick

# 检索层四级指标（不需要 Ollama，约 8~10 分钟）
.\.venv\Scripts\python.exe -m src.evaluation.retrieval_eval
```

> 跑之前建议先把当前文件拷到 `$TEMP` 存一份：它们是**原地覆盖**的。
> 虽然已纳入 git，但从历史里捞一个文件不如事先拷一下省事。

## 待议：要不要按语料分目录

现在的结构是"一个目录混所有语料"，靠这份文档区分。更彻底的做法是产物带语料标识
（如 `experiments/fastapi-zh/…`、`experiments/vue-zh/…`），但那样要改各脚本的输出路径，
属于代码改动。**暂不改**，先靠这份文档 + git 历史兜着 —— 等真的要在两份语料上并行
做对比实验时再动。
