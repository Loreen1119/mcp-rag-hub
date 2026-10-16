"""Judge 校准（②）第一步：冻结分层抽样 + 出人工判题表 + 长度偏差相关。

设计依据（2026-10-16 勘察）：
- `data/judge_gold_100.jsonl` 的 100 条金标**全部来自 AskTUG** → 没有 source 维度可分层，
  改按 7b 的 faithfulness 二分类分层；
- 7b 分数塌缩：faithfulness 只有 {0.7,0.8,0.9,1.0} 且 58% 挤在 0.7 → 三档退化成二分类。
  二分类规则：`score >= 0.8` 记为「忠实」（正），否则「不忠实」（负）。
- 分层平衡抽 **15 正 / 15 负**（简单随机 30 条会抽到约 17 条 0.7，Kappa 被众数压住）。
- 分层规则与 seed 在**看到人工标注之前冻结**；seed、阈值、抽样 id 全部落到 frozen_sample.json。

确定性：不用时间戳、输出按 id 排序 → 两次运行字节一致（`--check` 校验这一点）。

输出：
- experiments/judge_calibration/frozen_sample.json 分层规则 + seed + 抽样 id + 7b 标签（无原文，可入库）
- experiments/judge_calibration/length_bias.json   长度偏差（Spearman + 置换检验，可入库）
- .workbuddy/tmp/judge_calibration/判题表.md        含语料原文 → **不入库**（同 gen_100 红线）

用法：
    python -m scripts.build_judge_calibration            # 生成
    python -m scripts.build_judge_calibration --check     # 只校验，不写文件（应无差异）
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

JUDGE_GOLD = ROOT / "data" / "judge_gold_100.jsonl"
JUDGE_100 = ROOT / "experiments" / "generation_eval" / "judge_100.jsonl"
GEN_100 = ROOT / "experiments" / "generation_eval" / "gen_100.jsonl"

OUT_DIR = ROOT / "experiments" / "judge_calibration"
OUT_FROZEN = OUT_DIR / "frozen_sample.json"
OUT_LENGTH = OUT_DIR / "length_bias.json"
SHEET_DIR = ROOT / ".workbuddy" / "tmp" / "judge_calibration"
OUT_SHEET = SHEET_DIR / "判题表.md"

SEED = 42
THRESHOLD = 0.8
N_EACH = 15
DIMS = ("faithfulness", "answer_relevancy", "context_recall")
CREATED = "2026-10-16"  # 静态字面量，保证文件可复现（不用运行时钟）


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


# ---------------------------------------------------------------- 统计工具

def _ranks(vals: list[float]) -> list[float]:
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    out = [0.0] * len(vals)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # 1-based，并列取平均秩
        for k in range(i, j + 1):
            out[order[k]] = avg
        i = j + 1
    return out


def spearman(x: list[float], y: list[float]) -> float:
    rx, ry = _ranks(x), _ranks(y)
    n = len(x)
    if n < 3:
        return float("nan")
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    if not dx or not dy:
        return float("nan")
    return num / (dx * dy)


def perm_pvalue(x: list[float], y: list[float], observed: float, n_perm: int = 10000, seed: int = SEED) -> float:
    """双侧置换检验 p 值（避开 scipy 依赖）。"""
    rng = random.Random(seed)
    base = list(y)
    hits = 0
    for _ in range(n_perm):
        rng.shuffle(base)
        r = spearman(x, base)
        if abs(r) >= abs(observed):
            hits += 1
    return (hits + 1) / (n_perm + 1)


# ---------------------------------------------------------------- 主流程

def build() -> dict:
    gold = load_jsonl(JUDGE_GOLD)
    judges = load_jsonl(JUDGE_100)
    gens = load_jsonl(GEN_100)

    gold_by_id = {r["id"]: r for r in gold}
    judge_by_id = {r["id"]: r for r in judges}
    gen_by_id = {r["id"]: r for r in gens}

    ids = sorted(judge_by_id)
    missing = [i for i in ids if i not in gen_by_id or i not in gold_by_id]
    if missing:
        raise SystemExit(f"素材缺 id：{missing[:5]}（共 {len(missing)}）")

    # ---- 分层：7b faithfulness 二分类
    label: dict[str, int] = {}
    for i in ids:
        score = float(judge_by_id[i]["faithfulness"]["score"])
        label[i] = 1 if score >= THRESHOLD else 0

    pos = [i for i in ids if label[i] == 1]
    neg = [i for i in ids if label[i] == 0]
    if len(pos) < N_EACH or len(neg) < N_EACH:
        raise SystemExit(f"分层样本不足：正 {len(pos)} / 负 {len(neg)}，各需 {N_EACH}")

    rng = random.Random(SEED)
    sampled = sorted(rng.sample(pos, N_EACH) + rng.sample(neg, N_EACH))

    # ---- 长度偏差（全 100 条，纯离线）
    length_bias = {}
    ans_len = {i: len(gen_by_id[i]["answer"] or "") for i in ids}
    ctx_len = {i: len(gen_by_id[i]["context"] or "") for i in ids}
    for dim in DIMS:
        scores = [float(judge_by_id[i][dim]["score"]) for i in ids]
        l_ans = [float(ans_len[i]) for i in ids]
        l_ctx = [float(ctx_len[i]) for i in ids]
        r_ans = spearman(l_ans, scores)
        r_ctx = spearman(l_ctx, scores)
        length_bias[dim] = {
            "spearman_answer_len_vs_score": round(r_ans, 4),
            "p_perm_answer_len_vs_score": round(perm_pvalue(l_ans, scores, r_ans), 4),
            "spearman_context_len_vs_score": round(r_ctx, 4),
            "p_perm_context_len_vs_score": round(perm_pvalue(l_ctx, scores, r_ctx), 4),
        }

    frozen = {
        "_doc": (
            "Judge 校准（②）的冻结抽样。分层规则与 seed 在人工标注之前确定，不得事后调整；"
            "判定维度只用 faithfulness（≥阈值记正）+ 同批加判 answer_relevancy；"
            "context_recall 因 7b 分布塌缩（≥0.8 仅 13/100）不作 Kappa。"
        ),
        "created": CREATED,
        "seed": SEED,
        "stratify_by": "judge_7b.faithfulness",
        "binarize_rule": f"score >= {THRESHOLD} -> 忠实（正）; else 不忠实（负）",
        "n_each_stratum": N_EACH,
        "population": {"positive": len(pos), "negative": len(neg), "total": len(ids)},
        "human_scale": {
            "faithfulness": ["忠实", "不忠实", "拿不准"],
            "answer_relevancy": ["切题", "不切题", "拿不准"],
            "拿不准处理": "不计入 Kappa 分母，单独报条数",
        },
        "selected": [
            {
                "order": n + 1,
                "id": i,
                "stratum": "正" if label[i] == 1 else "负",
                **{dim: float(judge_by_id[i][dim]["score"]) for dim in DIMS},
                "answer_len": ans_len[i],
                "context_len": ctx_len[i],
            }
            for n, i in enumerate(sampled)
        ],
    }

    return {
        "frozen": frozen,
        "length_bias": length_bias,
        "sheet": render_sheet(frozen, gold_by_id, judge_by_id, gen_by_id),
    }


def _max_backtick_run(text: str) -> int:
    """文本里最长的连续反引号串长度（决定外围代码围栏要用多长）。"""
    best = 0
    run = 0
    for ch in text:
        if ch == "`":
            run += 1
            best = max(best, run)
        else:
            run = 0
    return best


def render_sheet(frozen, gold_by_id, judge_by_id, gen_by_id) -> str:
    L: list[str] = []
    w = L.append
    w("# Judge 校准 · 人工判题表（30 条）")
    w("")
    w("抽取规则与 seed 已**先冻结**（`experiments/judge_calibration/frozen_sample.json`，seed="
      f"{frozen['seed']}、分层={frozen['binarize_rule']}、正负各 {frozen['n_each_stratum']} 条）。")
    w("本表**隐去了 7b 裁判的分数与理由** —— 若先看到分数再判，等于给它打分背书，Kappa 就没有意义了。")
    w("")
    w("## 怎么判")
    w("")
    w("每条**只填最后两行各一个词**，30 条一共 60 个词。")
    w("")
    w("**忠实**：答案里所有实质信息都能在「检索到的上下文」里找到依据。同义改写、压缩、省略都算忠实。")
    w("**不忠实**：答案出现了上下文里没有的实质信息 —— 具体参数 / 版本号 / 命令 / 数字 / 结论，"
      "哪怕它碰巧是对的也算（判的是「有没有依据」，不是「对不对」）。")
    w("**切题**：答案正面回答了问题。")
    w("**不切题**：答非所问、绕开问题、或只是复述问题。")
    w("**拿不准**：确实两可。可以填，我会把它排除在 Kappa 分母之外单独报 —— 不要为了凑答案硬判。")
    w("")
    w("参考答案是人工写的，只作对照，**不要求系统答案与它一致**（一致与否不影响「忠实 / 切题」）。")
    w("")
    w("---")
    w("")
    for item in frozen["selected"]:
        i = item["id"]
        gen = gen_by_id[i]
        gold = gold_by_id[i]
        w(f"## {item['order']:02d} / `{i}`")
        w("")
        w("**问题**")
        w("")
        w("> " + (gen["query"] or "").replace("\n", " ").strip())
        w("")
        ctx = (gen["context"] or "").strip()
        # 语料里自带 ``` 代码块（5/30 条最长 3 连）→ 外围围栏必须比它长，否则会提前闭合
        fence = "`" * max(4, _max_backtick_run(ctx) + 1)
        w("**检索到的上下文（模型当时看到的全部材料）**")
        w("")
        w(fence + "text")
        w(ctx)
        w(fence)
        w("")
        w("**系统答案（待判对象）**")
        w("")
        w("> " + (gen["answer"] or "").replace("\n", " ").strip())
        w("")
        w("**参考答案**")
        w("")
        w("> " + (gold.get("reference_answer") or gen.get("reference_answer") or "")
          .replace("\n", " ").strip())
        w("")
        w("判定-忠实（忠实 / 不忠实 / 拿不准）：")
        w("")
        w("判定-切题（切题 / 不切题 / 拿不准）：")
        w("")
        w("---")
        w("")
    w(f"*表内共 {len(frozen['selected'])} 条。填完保存即可，我按 `order` 读回。*")
    w("")
    return "\n".join(L)


def dump(obj, path: Path) -> str:
    text = json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    return text


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="只校验，不写文件")
    args = ap.parse_args()

    built = build()
    frozen_text = json.dumps(built["frozen"], ensure_ascii=False, indent=2) + "\n"
    length_text = json.dumps(built["length_bias"], ensure_ascii=False, indent=2) + "\n"

    if args.check:
        bad = False
        for path, text in ((OUT_FROZEN, frozen_text), (OUT_LENGTH, length_text)):
            cur = path.read_text(encoding="utf-8") if path.exists() else None
            if cur != text:
                print(f"❌ {path.relative_to(ROOT)} 与复算不一致")
                bad = True
            else:
                print(f"✅ {path.relative_to(ROOT)} 一致")
        return 1 if bad else 0

    dump(built["frozen"], OUT_FROZEN)
    dump(built["length_bias"], OUT_LENGTH)
    SHEET_DIR.mkdir(parents=True, exist_ok=True)
    OUT_SHEET.write_text(built["sheet"], encoding="utf-8", newline="\n")

    f = built["frozen"]
    print(f"分层：统计总体 正 {f['population']['positive']} / 负 {f['population']['negative']}"
          f"（阈值 {THRESHOLD}），抽 {f['n_each_stratum']} 正 + {f['n_each_stratum']} 负，seed={SEED}")
    print("产物：")
    for p in (OUT_FROZEN, OUT_LENGTH, OUT_SHEET):
        size = p.stat().st_size
        print(f"  - {p.relative_to(ROOT)}  ({size} B)")
    print("长度偏差（Spearman，全 100 条）：")
    for dim, d in built["length_bias"].items():
        print(f"  {dim:<18} 答案长 {d['spearman_answer_len_vs_score']:+.4f} "
              f"(p={d['p_perm_answer_len_vs_score']:.4f})  "
              f"上下文长 {d['spearman_context_len_vs_score']:+.4f} "
              f"(p={d['p_perm_context_len_vs_score']:.4f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
