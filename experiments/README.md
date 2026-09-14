# experiments/ — 实验产物

> ⚠️ **先记住一句话**：**这个目录里目前所有产物都出自"旧语料"**（项目自身的 `docs/`，
> 已于 2026-09-12 换成 `corpora/fastapi-zh`）。**没有任何一条能代表系统当前状态。**
> 引用本目录任何数字前，先看下面的"语料归属"列。

## 为什么需要这份说明

`experiments/` 现在**纳入版本控制了**（2026-09-14 起，此前被 `.gitignore` 整目录忽略）。
纳入之后立刻暴露一个问题：7 月、8 月、9 月的四批产物躺在同一层目录里，
**旧语料的数字和新语料的数字混在一起，光看文件名分不出谁是谁**。

而每次跑评测都是**原地覆盖**同名文件 —— 也就是说，新结果一落盘，旧结果就没了
（现在有 git 兜底，但要点开历史才看得出来）。

## 产物清单与语料归属

| 文件 | 生成时间 | 生成脚本 | 语料 | 能不能引用 |
|---|---|---|---|---|
| `query_deep_dive.json` | 2026-07-01 | `experiments.py` | 旧 | ❌ 仅存档 |
| `ablation_results.json` | 2026-07-27 | `experiments.py` | 旧 | ❌ 仅存档 |
| `category_breakdown.json` | 2026-07-27 | `experiments.py` | 旧 | ❌ 仅存档 |
| `latency_profile.json` | 2026-07-27 | `experiments.py` | 旧 | ⚠️ 已被 README 标"待重测" |
| `parameter_sweep.json` | 2026-07-27 | `experiments.py` | 旧 | ❌ 仅存档 |
| `report.md` | 2026-07-27 | `experiments.py` | 旧 | ❌ 仅存档 |
| `kg_diag_report.json` | 2026-08-04 | `journal/kg_diag.py` | 旧 | ❌ 仅存档（KG 路诊断） |
| `kg_diag_report_control.json` | 2026-08-04 | `journal/kg_diag_control.py` | 旧 | ❌ 仅存档（对照组） |
| `answerability_score_analysis.json` | 2026-09-11 | `phase0_answerability_analysis_v2.py` | 旧 | ✅ 结论可用（见下） |
| `answerability_score_analysis.md` | 2026-09-11 | 同上 | 旧 | ✅ 结论可用（见下） |
| `retrieval_miss_cases.json` | 2026-09-11 | 同上 | 旧 | ❌ 仅存档 |
| `multi_evidence_analysis.json` | 2026-09-11 | 同上 | 旧 | ❌ 仅存档 |
| `phase0_recovery_log.json` | 2026-09-11 | 同上 | 旧 | ❌ 仅存档 |
| `llm_evaluation_results.json` | 2026-09-12 | `llm_eval.py` | 旧 | ⚠️ 见下（唯一"跑通过生成"的证据） |

### 两个例外说明

- **`answerability_score_analysis.*` 的结论仍然有效**：它证明的是"CE Top-1 分数对
  '可答 / 不可答'没有区分能力"（可答均值 7.42 vs 不可答均值 7.09，分布严重重叠）。
  这个结论**与具体语料无关** —— 是这个原因导致「用分数卡阈值自动拒答」的方案被废弃，
  相关常量已从 `config.py` 删除。**但表里的具体分数是旧语料的，别引用。**
- **`llm_evaluation_results.json` 是仓库里唯一保留了"真实生成结果"的文件**：3 条抽样、
  `qwen2.5:3b` 生成，单条耗时 14.5~43.8 秒。语料是旧的，**但"本机能跑通 LLM 生成"这件事是真的**
  （`journal/ROADMAP.md` 的 P0 条目引用的就是它）。注意：这次跑的时候**生成和裁判是同一个 3b 模型**，
  裁判分数不可信 —— 生成/裁判解耦（`JUDGE_MODEL`）是之后才做的。

## 当前系统的数字在哪（别在这里找）

| 想找什么 | 去哪看 |
|---|---|
| 检索四级指标（BM25 / 向量 / RRF / CE） | 仓库根 `README.md` 的指标表、`journal/ROADMAP.md` |
| 消融分析与"被数据推翻的设计" | `docs_knowledge/chapters/ch09-消融实验与数据分析.md` |
| 拒答率 / 误答率 / 引用覆盖率 | **尚未产出** —— 跑 `acceptance_eval.py` 后写入本目录 |

## 怎么重新生成（对应未完成清单的 A 组）

```powershell
$env:HF_HUB_OFFLINE="1"; $env:TRANSFORMERS_OFFLINE="1"; $env:PYTHONIOENCODING="utf-8"

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
