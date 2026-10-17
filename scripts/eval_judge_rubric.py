"""裁判 v2（rubric 模式）：把 7b 裁判从「吐 0~1 浮点分」改成「按明文准则判 3 档 + 先逐条列依据」。

动机（2026-10-16 实测）：现行裁判（`src/evaluation/llm_eval.py` 的 0~1 打分）在 100 条金标上
**分数塌缩**（faithfulness 只有 {0.7,0.8,0.9,1.0}，58% 挤在 0.7），且人工校准 **κ = −0.067、
AUC = 0.507**（= 无区分力）。假设：让模型「先逐条列主张并抄出处、再落一个离散档位」，
可以逼它真的去比对，而不是给个中庸分。

判定口径（与 v1 / 人工判题表对齐，**在看到 v2 结果之前冻结**）：
- 有依据 / 切题           → 正
- 部分有依据 / 无依据      → 负
- 部分切题 / 不切题        → 负
（v1 把「绝大部分有依据、一处轻微推论」记 0.7 并在二分类里算负 → 这里「部分有依据」同样算负，
口径一致，不因结果调整。）

**只跑 faithfulness 与 answer_relevancy**：这两维有人工标注可比；context_recall 没有人工标注，
重新打分也无法验证，故不做（省约 35 min）。

输出：
- .workbuddy/tmp/judge_calibration/judge_rubric_100.jsonl  原始输出（**含模型抄回的语料片段 → 不入库**）
- experiments/judge_calibration/kappa_report_v2.json        只含标签与统计量（**可入库**）

用法：
    python -m scripts.eval_judge_rubric --stage rubric            # 跑 100 条（可中断续跑）
    python -m scripts.eval_judge_rubric --stage rubric --limit 2  # 冒烟
    python -m scripts.eval_judge_rubric --stage report            # 算 Kappa/AUC（不调模型）
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.llm_eval import JUDGE_MODEL, _call_ollama  # noqa: E402

GEN_100 = ROOT / "experiments" / "generation_eval" / "gen_100.jsonl"
OUT_DIR = ROOT / "experiments" / "judge_calibration"
RAW_FILE = ROOT / ".workbuddy" / "tmp" / "judge_calibration" / "judge_rubric_100.jsonl"
REPORT_FILE = OUT_DIR / "kappa_report_v2.json"
FROZEN = OUT_DIR / "frozen_sample.json"
HUMAN = OUT_DIR / "human_labels.json"

DIMS = {
    "faithfulness": {"positive": "有依据", "labels": ["有依据", "部分有依据", "无依据"]},
    "answer_relevancy": {"positive": "切题", "labels": ["切题", "部分切题", "不切题"]},
}

# prompt 版本号：改了任何 system prompt 就 +1 —— 续跑时只复用同版本的记录，避免同一文件里混两套提问方式
PROMPT_VERSION = 1

FAITH_RUBRIC_SYSTEM = """你是评审专家。任务：判断「生成答案」里的**实质信息**能否在「检索到的上下文」里找到出处。

判定口径（判的是**有没有出处**，不是内容对不对；答案碰巧说对了但上下文里没有，也算没有出处）：
- 有依据：答案的每条实质信息都能在上下文里找到出处。同义改写、合并、压缩、省略都算有依据。
- 部分有依据：答案主体有出处，但有 1~2 处实质信息（具体参数 / 版本号 / 命令 / 数字 / 结论）找不到出处。
- 无依据：多处实质信息找不到出处，或答案与上下文矛盾。

输出必须**按顺序**完成，不得跳步：
1. `claims`：把答案的实质信息逐条列出（最多 8 条），每条给：
   - `claim`：该条信息的简述
   - `supported`：true / false
   - `quote`：从上下文里**原样抄一段**能支撑它的片段（不超过 30 字）；supported 为 false 时写 "无"
2. `label`：综合上面 `claims` 里 false 的条数给出档位，只能是 "有依据" / "部分有依据" / "无依据" 之一
3. `reason`：一句话说明判定理由

只输出 JSON，不要任何解释性文字：
{"claims": [{"claim": "...", "supported": true, "quote": "..."}], "label": "...", "reason": "..."}"""

RELEVANCY_RUBRIC_SYSTEM = """你是评审专家。任务：判断「生成答案」是否正面回答了「用户问题」。

判定口径：
- 切题：答案正面回应了问题的核心诉求，用户读完能拿到答案。
- 部分切题：回应了核心诉求的一部分，但漏掉了问题明确问到的某一点，或夹带明显无关内容。
- 不切题：答非所问、绕开问题，或只是把问题复述一遍。

输出必须**按顺序**完成，不得跳步：
1. `asked`：把问题实际在问的要点拆出来（最多 4 条）
2. `covered`：对每个要点标 true / false，说明答案有没有回应
3. `label`：综合给出档位，只能是 "切题" / "部分切题" / "不切题" 之一
4. `reason`：一句话说明判定理由

只输出 JSON，不要任何解释性文字：
{"asked": [{"point": "...", "covered": true}], "label": "...", "reason": "..."}"""


def log(msg: str) -> None:
    print(msg, flush=True)


def read_jsonl(p: Path) -> list[dict]:
    with open(p, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _parse_label(raw: str, allowed: list[str]) -> tuple[str, str]:
    """从返回里取 label。返回 (label, 备注)；label 为空串表示解析失败。"""
    if not raw:
        return "", "空返回"
    text = raw
    if "```" in text:  # 去掉代码围栏
        parts = text.split("```")
        for seg in parts:
            seg = seg.strip()
            if seg.startswith("json"):
                seg = seg[4:].strip()
            if "{" in seg:
                text = seg
                break
    try:
        data = json.loads(text.strip())
        label = str(data.get("label", "")).strip()
        if label in allowed:
            return label, ""
    except (json.JSONDecodeError, ValueError, AttributeError):
        pass
    # 兜底：直接在原文里找档位词（按长度降序，避免「有依据」命中「部分有依据」）
    for lab in sorted(allowed, key=len, reverse=True):
        if lab in raw:
            return lab, "正则兜底"
    return "", f"解析失败: {raw[:120]}"


def _judge(prompt: str, system: str) -> str:
    return _call_ollama(prompt, system=system, model=JUDGE_MODEL)


def _frozen_ids() -> list[str]:
    return [x["id"] for x in json.loads(FROZEN.read_text(encoding="utf-8"))["selected"]]


def stage_rubric(dims: list[str], limit: int | None = None, subset: str = "all") -> None:
    items = read_jsonl(GEN_100)
    if subset == "frozen":
        keep = set(_frozen_ids())
        items = [it for it in items if it["id"] in keep]
        log(f"subset=frozen：只跑冻结抽样的 {len(items)} 条（够做 Kappa，不必等 100 条）")
    done: dict[tuple[str, str], dict] = {}
    if RAW_FILE.exists():
        for r in read_jsonl(RAW_FILE):
            if r.get("label") and r.get("prompt_version") == PROMPT_VERSION:
                done[(r["id"], r["dim"])] = r
    if done:
        log(f"续跑：已完成 {len(done)} 条（id × dim，prompt_version={PROMPT_VERSION}）")

    RAW_FILE.parent.mkdir(parents=True, exist_ok=True)
    todo = [(it, dim) for it in items for dim in dims]
    if limit:
        todo = todo[: limit * len(dims)]

    n_done, n_new = 0, 0
    for it, dim in todo:
        key = (it["id"], dim)
        if key in done:
            n_done += 1
            continue
        cfg = DIMS[dim]
        if dim == "faithfulness":
            prompt = (
                f"## 检索到的上下文\n\n{it['context']}\n\n"
                f"## 生成答案\n\n{it['answer']}\n\n只输出 JSON。"
            )
            system = FAITH_RUBRIC_SYSTEM
        else:
            prompt = f"## 用户问题\n\n{it['query']}\n\n## 生成答案\n\n{it['answer']}\n\n只输出 JSON。"
            system = RELEVANCY_RUBRIC_SYSTEM

        t0 = time.perf_counter()
        raw = _judge(prompt, system)
        elapsed = time.perf_counter() - t0

        label, note = _parse_label(raw, cfg["labels"])
        rec = {
            "id": it["id"],
            "dim": dim,
            "prompt_version": PROMPT_VERSION,
            "label": label,
            "note": note,
            "is_positive": (label == cfg["positive"]) if label else None,
            "elapsed": round(elapsed, 2),
            "out_chars": len(raw),
            "raw": raw,
        }
        with open(RAW_FILE, "a", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        n_new += 1
        log(f"  [{n_done + n_new}/{len(todo)}] {it['id']}/{dim} -> {label or '解析失败'}"
            f"  {elapsed:.1f}s / {len(raw)}字符")
    log(f"rubric 阶段结束：新增 {n_new}，复用 {n_done}，合计 {n_done + n_new}")


# ---------------------------------------------------------------- 报告

def _ranks(vals: list[float]) -> list[float]:
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    out = [0.0] * len(vals)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            out[order[k]] = avg
        i = j + 1
    return out


def auc(pairs: list[tuple[float, int]]) -> float:
    pos = [s for s, h in pairs if h == 1]
    neg = [s for s, h in pairs if h == 0]
    if not pos or not neg:
        return float("nan")
    w = 0.0
    for a in pos:
        for b in neg:
            w += 1.0 if a > b else (0.5 if a == b else 0.0)
    return w / (len(pos) * len(neg))


def kappa(pairs: list[tuple[int, int]]) -> dict:
    n = len(pairs)
    a = sum(1 for j, h in pairs if j == h)
    po = a / n
    pj = sum(1 for j, _ in pairs if j) / n
    ph = sum(1 for _, h in pairs if h) / n
    pe = pj * ph + (1 - pj) * (1 - ph)
    return dict(n=n, agree=a, po=po, pe=pe, kappa=(po - pe) / (1 - pe) if pe != 1 else float("nan"),
                pj=pj, ph=ph)


def boot_ci(pairs: list[tuple[int, int]], reps: int = 10000, seed: int = 42) -> list[float]:
    rng = random.Random(seed)
    ks = []
    for _ in range(reps):
        s = [pairs[rng.randrange(len(pairs))] for _ in range(len(pairs))]
        v = kappa(s)["kappa"]
        if v == v:
            ks.append(v)
    ks.sort()
    return [ks[int(0.025 * len(ks))], ks[int(0.975 * len(ks))]]


def stage_report() -> int:
    raw = read_jsonl(RAW_FILE)
    got = {(r["id"], r["dim"]): r for r in raw if r.get("prompt_version") == PROMPT_VERSION}
    stale = [r for r in raw if r.get("prompt_version") != PROMPT_VERSION]
    if stale:
        print(f"⚠️ 忽略 {len(stale)} 条 prompt_version 不匹配的旧记录（当前 v{PROMPT_VERSION}）")
    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
    human = json.loads(HUMAN.read_text(encoding="utf-8"))["by_order"]
    sel = frozen["selected"]

    # 3 档 -> 有序数值（仅用于 AUC，不参与二分类）
    ordinal = {"有依据": 2.0, "切题": 2.0, "部分有依据": 1.0, "部分切题": 1.0, "无依据": 0.0, "不切题": 0.0}

    out: dict = {"_doc": "裁判 v2（rubric 3 档 + 逐条列依据）与人工标注的一致性。"
                         "冻结二分类：正 = 有依据 / 切题；部分与无/不切题 记负。",
                 "created": "2026-10-16", "judge_model": JUDGE_MODEL, "dims": {}}
    for dim, cfg in DIMS.items():
        # —— 冻结的 30 条上做校准（与 v1 同一批、同一口径）
        pairs, ords, miss = [], [], []
        for it in sel:
            hid = it["id"]
            r = got.get((hid, dim))
            if r is None or not r.get("label"):
                miss.append(hid)
                continue
            h = 1 if human[str(it["order"])]["faith" if dim == "faithfulness" else "rel"] == (
                "忠实" if dim == "faithfulness" else "切题") else 0
            pairs.append((1 if r["is_positive"] else 0, h))
            ords.append((ordinal[r["label"]], h))
        k = kappa(pairs) if pairs else {}
        ci = boot_ci(pairs) if len(pairs) >= 3 else [float("nan")] * 2
        cm = dict(TP=sum(1 for j, h in pairs if j and h), FP=sum(1 for j, h in pairs if j and not h),
                  FN=sum(1 for j, h in pairs if not j and h), TN=sum(1 for j, h in pairs if not j and not h))
        # —— 全 100 条的档位分布（重新估总体正率，与 v1 的 ≥0.8 口径同义）
        alllab = [got[(it["id"], dim)]["label"] for it in read_jsonl(GEN_100)
                  if got.get((it["id"], dim), {}).get("label")]
        dist = {lab: alllab.count(lab) for lab in cfg["labels"]}
        n_all = len(alllab)
        pos_all = dist[cfg["positive"]] if n_all else 0
        out["dims"][dim] = {
            "n_calibration": len(pairs),
            "missing_in_calibration": miss,
            "kappa": round(k.get("kappa", float("nan")), 4) if k else None,
            "kappa_ci95": [round(x, 4) for x in ci],
            "po": round(k["po"], 4) if k else None,
            "pe": round(k["pe"], 4) if k else None,
            "judge_positive_rate": round(k["pj"], 4) if k else None,
            "human_positive_rate": round(k["ph"], 4) if k else None,
            "confusion": cm,
            "auc_ordinal": round(auc(ords), 4) if ords else None,
            "label_distribution_scored": dist,
            "n_scored": n_all,
            "positive_rate_scored": round(pos_all / n_all, 4) if n_all else None,
            "passes_kappa_0_6": (k.get("kappa", -1) >= 0.6) if k else None,
        }
        print(f"== {dim} ==")
        print(f"  校准 n={len(pairs)}（缺 {len(miss)}）题")
        if k:
            print(f"  κ={k['kappa']:+.4f}  CI[{ci[0]:+.3f},{ci[1]:+.3f}]  po={k['po']:.3f} pe={k['pe']:.3f}")
            print(f"  裁判正率={k['pj']:.3f} 人工正率={k['ph']:.3f} 混淆={cm}")
            print(f"  AUC(3档序数)={auc(ords):.4f}")
            print(f"  κ≥0.6 ? {'✅ 可写' if k['kappa'] >= 0.6 else '❌ 仍不够'}")
        else:
            print("  无可比对样本")
        print(f"  已判 {n_all} 条的档位分布={dist} → 正率 {pos_all}/{n_all} = {pos_all/n_all:.4f}" if n_all else "  还没跑完")

    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"\n报告已写：{REPORT_FILE.relative_to(ROOT)}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["rubric", "report"], required=True)
    ap.add_argument("--limit", type=int, default=None, help="只跑前 N 条 × 维数（冒烟用）")
    ap.add_argument("--dims", default="faithfulness,answer_relevancy",
                    help="要跑的维度，逗号分隔（默认两维；只跑一维可省一半时间）")
    ap.add_argument("--subset", choices=["all", "frozen"], default="all",
                    help="all=100 条；frozen=只跑冻结抽样的 30 条（做 Kappa 够用、省 2/3 时间）")
    args = ap.parse_args()
    if args.stage == "rubric":
        dims = [d.strip() for d in args.dims.split(",") if d.strip()]
        bad = [d for d in dims if d not in DIMS]
        if bad:
            raise SystemExit(f"未知维度：{bad}（可选 {list(DIMS)}）")
        stage_rubric(dims, args.limit, args.subset)
        return 0
    return stage_report()


if __name__ == "__main__":
    sys.exit(main())
