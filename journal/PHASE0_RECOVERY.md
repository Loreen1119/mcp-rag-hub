# Phase 0 环境恢复与重跑指南（下次会话）

## 📋 当前状态

**工作区已有**：
- `src/evaluation/phase0_answerability_analysis_v2.py` — 修正版脚本（已完整）
- `data/multi_evidence_queries_v2.json` — 改名后的数据集（已完整）
- EXECUTION_PLAN.md — 总体计划（已暂存）

**需要做**：
1. 环境恢复（venv 问题）
2. 重跑 v2 脚本
3. 手动改名数据文件
4. 验收产物

---

## 🔧 Step 1：环境恢复（选一个）

### 方案 A：重建 venv（推荐）

```bash
# 1. 进入项目目录
cd d:/1base/computer/Agent/DevRoot/mcp-rag-hub

# 2. 备份旧 venv
mv .venv .venv.backup 2>/dev/null || true

# 3. 创建新 venv
python3 -m venv .venv

# 4. 激活
source .venv/bin/activate  # macOS/Linux
# 或 .venv\Scripts\activate（Windows）

# 5. 装包
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 6. 测试
python -c "import torch; import chromadb; print('✓ 环境 OK')"
```

### 方案 B：等待系统恢复

如果环境由系统管理，等系统自动恢复。

---

## 🚀 Step 2：运行修正版脚本

```bash
# 确保已激活 venv
source .venv/bin/activate

# 运行 v2 脚本
python src/evaluation/phase0_answerability_analysis_v2.py
```

**预期输出**：
```
============================================================
Phase 0 无答案拒答阈值分析（v2）
============================================================
加载数据集...
  answerable: 36 条
  unanswerable: 10 条
  multi_evidence: 3 条
...
✓ Phase 0 v2 分析完成
  推荐阈值：7.8465
  产物位置：experiments/
============================================================
```

**产物检查**：
```bash
ls -la experiments/
# 应该看到：
# ✓ answerability_score_analysis.json （含 candidates）
# ✓ multi_evidence_analysis.json      （新增）
# ✓ retrieval_miss_cases.json
# ✓ answerability_score_analysis.md
# ✓ phase0_recovery_log.json
```

---

## 📝 Step 3：数据文件改名

v2 脚本已经在产物中用 ME01-ME03，但源数据文件还需手动改名：

```bash
# 备份原文件
cp data/multi_evidence_queries.json data/multi_evidence_queries.backup.json

# 改用 v2 版本（ID 已改为 ME01-ME03）
cp data/multi_evidence_queries_v2.json data/multi_evidence_queries.json
```

或者手动编辑 `data/multi_evidence_queries.json`，把 `"query_id": "M0X"` 改为 `"query_id": "ME0X"`。

---

## ✅ Step 4：验收产物

检查以下三件产物是否满足要求：

### ✓ experiments/answerability_score_analysis.json

```bash
python -c "
import json
data = json.load(open('experiments/answerability_score_analysis.json'))
print('Keys:', list(data.keys()))
print('Candidates count:', len(data.get('candidates', [])))
print('Recommended threshold:', data['recommendation_justification']['selected_threshold'])
"
```

**验收标准**：
- [ ] 包含 `candidates` 字段（完整混淆矩阵）
- [ ] 包含 `recommendation_justification`（含理由）
- [ ] recommended_threshold = 7.8465

### ✓ experiments/multi_evidence_analysis.json

```bash
python -c "
import json
data = json.load(open('experiments/multi_evidence_analysis.json'))
print('Queries:')
for q in data['queries']:
    print(f'  {q[\"query_id\"]}: covered={q[\"coverage_in_top3\"]}')"
```

**验收标准**：
- [ ] 包含 3 个问题，ID 为 ME01/ME02/ME03
- [ ] ME01/ME02 coverage=true
- [ ] ME03 coverage=false，原因明确

### ✓ experiments/answerability_score_analysis.md

检查内容包含：
- [ ] 分布重叠现象说明
- [ ] 阈值保守性警告
- [ ] 多证据覆盖率表
- [ ] 已知检索边界说明（ME03）
- [ ] ID 冲突历史记录

---

## 🎯 验收通过后

```bash
# 暂存所有修改
git add EXECUTION_PLAN.md \
        src/evaluation/phase0_answerability_analysis_v2.py \
        data/multi_evidence_queries_v2.json \
        experiments/

# 查看状态
git status

# 不要 commit（让下一个任务决定如何 commit）
```

验收通过后，**直接进 Phase 1A**（Day 2）：实现答案校验 + LLM 生成。

---

## 📞 如果卡住了

| 问题 | 解决方案 |
|------|--------|
| venv 激活失败 | 用 `python3 -m venv .venv` 重建 |
| 包缺失 | 运行 `pip install -r requirements.txt` |
| 脚本挂在 CE 加载 | 重启环境后重试；如仍失败，检查内存 |
| 产物丢失 | 检查 experiments/ 目录是否存在；若不存在用 `mkdir -p experiments` |
