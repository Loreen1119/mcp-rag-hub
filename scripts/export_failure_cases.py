"""失败案例逐条分析（单一来源）—— 从已入库产物**确定性复算**，生成 `data/failure_cases.md`。

为什么单独出一个脚本：
- `data/failure_cases.md` 要同时给 README / EVAL_REPORT / 技术博客 1 引用，必须是**单一来源**，
  而单一来源的前提是「谁都能重算出同一份」——所以生成过程脚本化，复算即校验。
- 复算结果要与 (a) `failure_mode_spec.md` 记的冻结分布、(b) `data/eval_set.jsonl` 里逐条记录的
  `failure_mode.primary` 完全对齐；对不上就非零退出并打印差异 → 这份文档不可能被手改漂移。

口径（严格按 `data/failure_mode_spec.md` 第三、四节）：
- 检索层 F1/F2/F6：基准配置 = **RRF 档（生产 BM25 20 + 向量 20）的 top-5**
- 拒答层 F4/F5：基准 = **CE 分数门控 [B] = 0.98**（⚠️ 这是**模拟口径**，与 W3 真实 agent 链路的
  10.5% / 5% **不是同一回事**，本文件把两者分区摆放、禁止混报）
- 术语鸿沟 F3：只作 `root_cause`（抽检 50 条，seed=42），**不作 primary**
- 独立探针集 31 条：[Z] 危害口径（旧版页进 top-5 且压过正确版页）

用法：
    python -m scripts.export_failure_cases          # 复算校验 + 生成 data/failure_cases.md
    python -m scripts.export_failure_cases --check   # 只复算校验，不写文件（可挂 CI）
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA = PROJECT_ROOT / "data"
EXP = PROJECT_ROOT / "experiments"

EVAL_SET = DATA / "eval_set.jsonl"
EVAL_VERSION = DATA / "eval_set_version.jsonl"
RETRIEVAL = EXP / "baseline_ladder" / "retrieval_375.jsonl"
MISS_HINTS = EXP / "baseline_ladder" / "miss_hints.jsonl"
CE_SCORES = EXP / "baseline_ladder" / "ce_scores.jsonl"          # 生产批（10-14），[B] 门控的分数来源
VERSION_CONF = EXP / "version_confusion" / "version_confusion_31.jsonl"
TERM_GAP = DATA / "failure_mode_term_gap_sample50.json"
JUDGE = EXP / "generation_eval" / "judge_100.jsonl"
REFUSAL = EXP / "generation_eval" / "refusal_60.jsonl"           # W3 真实链路（另一口径，只作诊断）
OUT_MD = DATA / "failure_cases.md"

REJECT_B = 0.98          # 拒答层基准阈值（spec 第三节）
TOP_K = 5
QUERY_CUT = 36

EXPECTED_MAIN = {"召回缺失": 114, "排序压制": 112, "none": 118, "应拒答未拒答": 30, "拒答误杀": 1}
EXPECTED_DENOM = 335
EXPECTED_FALSE_KILL = (1, 20)
EXPECTED_CORRECT_REFUSAL = (10, 40)
EXPECTED_Z_RRF = (6, 31)
EXPECTED_Z_CE = (3, 31)


# ------------------------------------------------------------------
# 工具
# ------------------------------------------------------------------
def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_norm(path: Path) -> str:
    """CRLF 归一到 LF 后再算（与 tests/regression_baselines.json 同一约定）。"""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def top5_hit(row: dict) -> bool:
    top = (row.get("stages", {}).get("rrf", {}) or {}).get("top", [])
    return any(x.get("hit") and x.get("rank", 99) <= TOP_K for x in top)


def cut(text: str, n: int = QUERY_CUT) -> str:
    text = (text or "").replace("\n", " ").strip()
    return text if len(text) <= n else text[: n - 1] + "…"


# ------------------------------------------------------------------
# 复算
# ------------------------------------------------------------------
def recompute() -> dict:
    ev = load_jsonl(EVAL_SET)
    ret = {r["id"]: r for r in load_jsonl(RETRIEVAL)}
    hints = {r["id"]: r for r in load_jsonl(MISS_HINTS)}   # 完整记录（含 golden_rank_in_single_path）
    ce = {r["id"]: r for r in load_jsonl(CE_SCORES)}

    primary: dict[str, str] = {}
    for tc in ev:
        qid = tc["id"]
        score = (ce.get(qid) or {}).get("ce_top1_score") or 0.0
        rejected = score < REJECT_B          # [B]=0.98 门控：低于阈值即拒答
        if tc.get("unanswerable"):
            primary[qid] = "none" if rejected else "应拒答未拒答"
        elif tc["source"] == "miss_kill" and rejected:
            primary[qid] = "拒答误杀"         # 误杀探针被拒 → 覆盖检索层判定
        elif top5_hit(ret.get(qid, {})):
            primary[qid] = "none"
        else:
            primary[qid] = hints[qid]["auto_hint"] if qid in hints else "unclassified"

    # 独立探针集（[Z]，基准配置 = RRF 档）
    ver = load_jsonl(VERSION_CONF)
    ver_primary = {
        r["id"]: ("版本混淆" if (r["stages"].get("rrf") or {}).get("twin_outranks_correct") else "none")
        for r in ver
    }

    unans_ids = [r["id"] for r in ev if r.get("unanswerable")]
    mk_ids = [r["id"] for r in ev if r["source"] == "miss_kill"]
    refused = [r["id"] for r in ev if primary[r["id"]] in ("none",) and r.get("unanswerable")]
    return {
        "eval": ev,
        "retrieval": ret,
        "hints": hints,
        "ce": ce,
        "primary": primary,
        "ver": ver,
        "ver_primary": ver_primary,
        "unans_ids": unans_ids,
        "mk_ids": mk_ids,
        "refused_unans": refused,
    }


def validate(st: dict) -> list[str]:
    """复算 vs 冻结/记录的一致性。返回差异列表（空 = 通过）。"""
    errs: list[str] = []
    ev, primary = st["eval"], st["primary"]

    dist = {k: sum(1 for v in primary.values() if v == k) for k in EXPECTED_MAIN}
    if dist != EXPECTED_MAIN:
        errs.append(f"375 条 primary 分布不符：复算 {dist} vs 冻结 {EXPECTED_MAIN}")

    recorded = {(r.get("failure_mode") or {}).get("primary") for r in ev}
    mismatch = [
        r["id"] for r in ev
        if primary.get(r["id"]) != (r.get("failure_mode") or {}).get("primary")
    ]
    if mismatch:
        errs.append(f"逐条与 eval_set 记录不一致 {len(mismatch)} 条：{mismatch[:10]}")
    if "unclassified" in recorded or None in recorded:
        errs.append("eval_set 里存在 unclassified / 空的 primary")

    denom = sum(1 for r in ev if r.get("golden_slugs"))
    if denom != EXPECTED_DENOM:
        errs.append(f"Hit@5 分母应为 {EXPECTED_DENOM}，实际 {denom}")

    fk = sum(1 for r in ev if primary.get(r["id"]) == "拒答误杀")
    cr = sum(1 for r in ev if r.get("unanswerable") and primary.get(r["id"]) == "none")
    if (fk, len(st["mk_ids"])) != EXPECTED_FALSE_KILL:
        errs.append(f"[B]=0.98 误杀应为 {EXPECTED_FALSE_KILL}，实际 {fk}/{len(st['mk_ids'])}")
    if (cr, len(st["unans_ids"])) != EXPECTED_CORRECT_REFUSAL:
        errs.append(f"[B]=0.98 正确拒答应为 {EXPECTED_CORRECT_REFUSAL}，实际 {cr}/{len(st['unans_ids'])}")

    z_rrf = sum(1 for v in st["ver_primary"].values() if v == "版本混淆")
    if (z_rrf, len(st["ver"])) != EXPECTED_Z_RRF:
        errs.append(f"[Z] RRF 档危害应为 {EXPECTED_Z_RRF}，实际 {z_rrf}/{len(st['ver'])}")
    z_ce = sum(1 for r in st["ver"] if (r["stages"].get("ce") or {}).get("twin_outranks_correct"))
    if (z_ce, len(st["ver"])) != EXPECTED_Z_CE:
        errs.append(f"[Z] CE 档危害应为 {EXPECTED_Z_CE}，实际 {z_ce}/{len(st['ver'])}")
    return errs


# ------------------------------------------------------------------
# 渲染
# ------------------------------------------------------------------
SOURCES = [EVAL_SET, EVAL_VERSION, RETRIEVAL, MISS_HINTS, CE_SCORES, VERSION_CONF, TERM_GAP, JUDGE, REFUSAL]


def render(st: dict) -> str:
    ev, primary, hints, ret = st["eval"], st["primary"], st["hints"], st["retrieval"]
    by_id = {r["id"]: r for r in ev}
    out: list[str] = []
    w = out.append

    miss = [r for r in ev if r["id"] in hints]
    n_none = sum(1 for v in primary.values() if v == "none")
    n_unans_none = sum(1 for r in ev if r.get("unanswerable") and primary[r["id"]] == "none")

    w("# 失败案例逐条分析（单一来源）")
    w("")
    w("> 生成方式：`python -m scripts.export_failure_cases`（**确定性复算**，脚本会对齐冻结分布与逐条记录，")
    w("> 对不上直接非零退出）。README / `EVAL_REPORT.md` / 技术博客 1 引用失败案例时，**引用本文件**，不要各写一份。")
    w(">")
    w("> **口径边界（先读这段）**")
    w("> - 检索层 F1/F2 基准配置 = **RRF 档（生产 BM25 20 + 向量 20）的 top-5**，与 [Y] 口径一致。")
    w(f"> - 拒答层 F4/F5 基准 = **CE 分数门控 [B] = {REJECT_B}**（⚠️ **模拟口径**）。")
    w(">   W3 真实 agent 链路的实测（正确拒答 10.5% / 误杀 5%）**是另一套口径**，只在第 5 节作诊断并列，**不得与本节混报**。")
    w("> - 术语鸿沟 F3 只作 `root_cause`（抽检 50 条 seed=42），**不参与 primary 自动判定**。")
    w("> - 版本混淆 F6 属**独立探针集**，**不在主集 375 里**；主集里新旧版同名页命中仍按 slug 级算命中。")
    w("> - 本文件只放 query 与 slug，**不放语料原文**（`corpora/tidb/` 不入库，原文不随仓库分发）。")
    w("")
    w("## 0. 数据来源（本文件由这些文件复算而来）")
    w("")
    for p in SOURCES:
        rel = p.relative_to(PROJECT_ROOT).as_posix()
        w(f"- `{rel}` — `sha256(LF) = {sha256_norm(p)[:16]}…`")
    w("")
    w("## 1. 主集 375 条 primary 分布（复算 == 冻结）")
    w("")
    w("| primary | 条数 |")
    w("|---|---|")
    for k, v in EXPECTED_MAIN.items():
        w(f"| {k} | {v} |")
    w(f"| **合计** | **{sum(EXPECTED_MAIN.values())}** |")
    w("")
    w(f"- `none` {n_none} = top-5 命中 + `[B]={REJECT_B}` 下正确拒答 {n_unans_none}")
    w(f"- 未命中（`召回缺失` + `排序压制`）= {len(miss)} 条，逐条见第 3 节")
    w("- `unclassified` 0 条；`术语鸿沟` 0 条（该标签只进 `root_cause`，见第 7 节）")
    w("")
    w("## 2. 每类典型 case（详析）")
    w("")
    for label in ("召回缺失", "排序压制"):
        w(f"### {label}")
        w("")
        for r in [x for x in miss if hints[x["id"]]["auto_hint"] == label][:3]:
            qid = r["id"]
            top = (ret.get(qid, {}).get("stages", {}).get("rrf", {}) or {}).get("top", [])[:TOP_K]
            ranks = hints[qid].get("golden_rank_in_single_path")
            w(f"- **{qid}**（{r['source']}）{cut(r['query'], 80)}")
            w(f"  - golden：`{', '.join(r.get('golden_slugs') or [])}`")
            if label == "排序压制":
                w(f"  - 单路已召回（golden 位次）：`{ranks}` → 但 RRF top-5 未排进来")
            else:
                w("  - 单路 top-50 均无此页 → 语料侧就没被召回（术语鸿沟是常见根因，见第 7 节）")
            shown = "、".join(f"{x['rank']}.{cut(x['slug'], 22)}" for x in top)
            w(f"  - RRF top-5：{shown}")
        w("")

    w("### 应拒答未拒答（[B] 门控口径）")
    w("")
    wrong = [r for r in ev if r.get("unanswerable") and primary[r["id"]] == "应拒答未拒答"]
    for r in wrong[:3]:
        w(f"- **{r['id']}** {cut(r['query'], 80)} —— CE top1 = {(st['ce'].get(r['id']) or {}).get('ce_top1_score')} ≥ {REJECT_B}，被当作可答")
    w(f"- 全量 {len(wrong)} 条 id：{', '.join(r['id'] for r in wrong)}")
    w("")
    w("### 拒答误杀（[B] 门控口径）")
    w("")
    fk = [r for r in ev if primary[r["id"]] == "拒答误杀"]
    for r in fk:
        w(f"- **{r['id']}** {cut(r['query'], 80)} —— 该条 top-5 其实命中，但 CE top1 = "
          f"{(st['ce'].get(r['id']) or {}).get('ce_top1_score')} < {REJECT_B} 被误拒")
    w("")
    w("### 版本混淆（独立探针集，[Z] 基准配置 = RRF 档）")
    w("")
    for vid, p in st["ver_primary"].items():
        if p == "版本混淆":
            row = next(x for x in st["ver"] if x["id"] == vid)
            w(f"- **{vid}** {cut(row['query'], 70)} —— 旧版页（`docs-cn-v8.4/`）压过正确版页（`docs-cn/`）")
    w("")
    w("## 3. 未命中 {n} 条逐条".format(n=len(miss)))
    w("")
    w("`auto_hint` = spec 第四节的确定性初判：golden 在单路 top-50 内 → `排序压制`，否则 `召回缺失`。")
    w("")
    w("| id | source | primary | golden | 单路 golden 位次 | query |")
    w("|---|---|---|---|---|---|")
    for r in miss:
        qid = r["id"]
        ranks = hints[qid].get("golden_rank_in_single_path") or {}
        ranks_txt = "、".join(f"{k}={v}" for k, v in ranks.items()) or "无（两路 top-50 都没召回）"
        w(f"| {qid} | {r['source']} | {hints[qid]['auto_hint']} | "
          f"{', '.join(r.get('golden_slugs') or [])} | {ranks_txt} | {cut(r['query'])} |")
    w("")
    w(f"## 4. 拒答两侧清单（[B]={REJECT_B} 模拟口径，共 60 条）")
    w("")
    w(f"- 不可答 40 条：**正确拒答 {len(st['refused_unans'])} 条**（{', '.join(st['refused_unans'])}）；"
      f"**该拒未拒 {len(wrong)} 条**（见 2 节 id 列表）")
    w(f"- 误杀探针 20 条：**误杀 {len(fk)} 条**（{', '.join(r['id'] for r in fk)}）；其余 19 条正常作答")
    w("")
    w("## 5. W3 真实 agent 链路实测（⚠️ 与第 4 节不是同一口径，禁止混报）")
    w("")
    ref = load_json(REFUSAL)
    det = ref["refusal"]["details"] if isinstance(ref.get("refusal"), dict) else ref.get("details", [])
    st_counter = Counter(x.get("status") for x in det)
    un_det = [x for x in det if x.get("category") == "unanswerable"]
    mk_det = [x for x in det if x.get("category") == "miss_kill"]
    un_ok = [x for x in un_det if x.get("status") == "insufficient_evidence"]
    mk_bad = [x for x in mk_det if x.get("status") == "insufficient_evidence"]
    invalid = [x["query_id"] for x in det if x.get("status") == "invalid_output"]
    w(f"- 真实链路（qwen2.5:3b，走 `agent.run_query`）：正确拒答 **{len(un_ok)}/{len(un_det) - len(invalid)}**、"
      f"漏拒 **{len(un_det) - len(un_ok) - len([x for x in invalid])}/{len(un_det) - len(invalid)}**、漏杀/误杀 **{len(mk_bad)}/{len(mk_det)}**")
    w(f"- `status` 分布：{dict(st_counter)}；其中 `invalid_output` {invalid}（归因**模型稳定性**，不计分母）")
    w("- 结论：**拒答能力弱是真实短板、失准单向（该拒却答为主）**；与第 4 节的模拟门控同向，但绝对值不可互推。")
    w("")
    w("## 6. 生成层低分样本（judge_100，W3）")
    w("")
    w("> 🔴 **本节分数来自 7b 裁判自评，未经人工校准**（冻结 30 条盲判：faithfulness κ **+0.0794** / "
      "relevancy κ **+0.0196**，两版 prompt 均远低于 κ≥0.6，见 `EVAL_REPORT.md` 第九节）→ "
      "**只用于定位低分样本，不得当已校准指标引用**。")
    w("")
    judges = load_jsonl(JUDGE)
    def avg(r): return (r["faithfulness"]["score"] + r["answer_relevancy"]["score"] + r["context_recall"]["score"]) / 3
    for r in sorted(judges, key=avg)[:5]:
        w(f"- **{r['id']}**（均分 {avg(r):.2f}）{cut(r['query'], 60)}")
        w(f"  - faithfulness {r['faithfulness']['score']}｜relevancy {r['answer_relevancy']['score']}｜context_recall {r['context_recall']['score']}")
        w(f"  - 裁判理由（context_recall）：{cut(r['context_recall']['reason'], 60)}")
    w("")
    w("## 7. 术语鸿沟（root_cause，抽检 50 条 seed=42）")
    w("")
    tg = load_json(TERM_GAP)
    yes = [x for x in tg["records"] if x.get("term_gap")]
    w(f"- 抽检 {tg['sample_n']} 条 → **术语鸿沟 {tg['term_gap_yes']}/{tg['sample_n']} = {tg['rate']:.0%}**"
      f"（by_source：{tg['by_source']}）")
    w(f"- 判定为是的 {len(yes)} 条 id：{', '.join(x['id'] for x in yes)}")
    w("- 这些 id 的 `failure_mode.root_cause` 已回填为 `术语鸿沟`；**它是抽检比例，不是全量自动判**。")
    w("")
    w("## 8. 引用纪律（红线）")
    w("")
    w("- 引用分布数字 → 第 1 节（375 条，五类合计 375，`unclassified` 0）。")
    w("- 引用拒答 → **必须写明口径**：第 4 节是 `[B]=0.98` 的模拟门控；第 5 节才是真实链路实测。两者**不得混成一句**。")
    w("- 引用版本混淆 → 第 2 节末与独立探针集口径，**不得与主集 Hit@5 混报**。")
    w("- 修改本文件的唯一正路：改 `failure_mode_spec.md` 的规则 → 重跑本脚本；手改会被下一次 `--check` 打回。")
    return "\n".join(out) + "\n"


# ------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="失败案例逐条分析（单一来源）")
    ap.add_argument("--check", action="store_true", help="只复算校验，不写 data/failure_cases.md")
    args = ap.parse_args()

    st = recompute()
    errs = validate(st)
    if errs:
        print("❌ 复算与冻结/记录不一致：")
        for e in errs:
            print("   -", e)
        return 1

    dist = {k: sum(1 for v in st["primary"].values() if v == k) for k in EXPECTED_MAIN}
    print("✅ 复算通过：", dist, "| 分母", EXPECTED_DENOM,
          "| 误杀", EXPECTED_FALSE_KILL, "| 正确拒答", EXPECTED_CORRECT_REFUSAL)

    if args.check:
        print("（--check：未写文件）")
        return 0

    md = render(st)
    OUT_MD.write_text(md, encoding="utf-8", newline="\n")
    print(f"→ 已写入 {OUT_MD.relative_to(PROJECT_ROOT)}（{len(md)} 字符）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
