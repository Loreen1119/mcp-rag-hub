#!/usr/bin/env python
"""生成层三维指标 + 端到端拒答链路 —— TiDB 中文评测集（W3）。

## 为什么不是主集 375 条

生成层三维里的 **Context Recall 需要「参考答案」文本**（判断检索到的上下文有没有覆盖
参考答的要点）。主集 `data/eval_set.jsonl` 只给了 `golden_slugs`（指向官方文档页），
**没有参考答文本** → 算不出 Context Recall。

所以两件事各有各的样本集，且**互不重叠**：

| 阶段 | 样本来源 | 条数 |
|---|---|---|
| 生成层三维 | `data/judge_gold_100.jsonl`（带 `reference_answer`） | 100 |
| 端到端拒答 | `data/eval_set.jsonl` 里 `unanswerable=true`(40) + `source=="miss_kill"`(20) | 60 |

合计 160 条。**不跑 375 条全量生成**——那 275 条没有参考标准，评不了，白烧算力。

## 复用而非重写（同一逻辑禁止两处实现）

- 三维裁判的 prompt 与打分解析：`src/evaluation/llm_eval.py`（`_call_ollama` / `_parse_score` /
  三个 `_build_*_prompt` / 三个 `*_SYSTEM`）
- 拒答三态判定：`src/evaluation/acceptance_eval.py::evaluate_refusal`（跑真实 agent 链路，
  含 validator 的 ANSWERED / INSUFFICIENT_EVIDENCE / GENERATION_UNAVAILABLE 三态）

## 阶段（各自可中断续跑，按 id 跳过已完成）

    --stage gen      100 条金标：检索 + 生成 → gen_100.jsonl
    --stage judge    对已生成的 100 条打三维分 → judge_100.jsonl + summary 片段
    --stage refusal  60 条拒答判定（真实 agent 链路）→ refusal_60.jsonl
    --stage report   汇总所有阶段
    --stage all      依次全部

前置：**Ollama 必须在跑**（生成用 3b、裁判用 7b）。拒答阶段自带连通性自检，不通直接退出
—— 否则会跑出一堆 `generation_unavailable`，看着像"系统很爱拒答"，其实是服务没起。

用法：

    cd D:/1base/computer/Agent/DevRoot/mcp-rag-hub
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MCP_RAG_CORPUS=tidb NO_PROXY=127.0.0.1,localhost
    ./.venv/Scripts/python.exe -u -m scripts.eval_generation_tidb --stage all

产物：`experiments/generation_eval/`
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# 离线 / 代理环境（必须在任何 torch / transformers / config 导入之前生效）
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("MCP_RAG_CORPUS", "tidb")
for _k in list(os.environ):
    if _k.endswith("_PROXY") or _k.lower() == "all_proxy":
        os.environ.pop(_k, None)
os.environ["NO_PROXY"] = "127.0.0.1,localhost"

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import JUDGE_MODEL, LLM_MODEL  # noqa: E402
from src.evaluation.llm_eval import (  # noqa: E402
    ANSWER_RELEVANCY_SYSTEM,
    CONTEXT_RECALL_SYSTEM,
    FAITHFULNESS_SYSTEM,
    _build_answer_relevancy_prompt,
    _build_context_recall_prompt,
    _build_faithfulness_prompt,
    _call_ollama,
    _parse_score,
    _retrieve_and_generate,
)

JUDGE_SET = ROOT / "data" / "judge_gold_100.jsonl"
EVAL_SET = ROOT / "data" / "eval_set.jsonl"
OUT_DIR = ROOT / "experiments" / "generation_eval"
GEN_FILE = OUT_DIR / "gen_100.jsonl"
JUDGE_FILE = OUT_DIR / "judge_100.jsonl"
REFUSAL_FILE = OUT_DIR / "refusal_60.jsonl"
SUMMARY_FILE = OUT_DIR / "summary.json"


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def read_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


class Appender:
    """逐条追加写 + flush，中断时已完成条目全部落盘。"""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = open(path, "a", encoding="utf-8")

    def write(self, obj: dict) -> None:
        self.fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
        self.fh.flush()

    def close(self) -> None:
        self.fh.close()


def load_judge_items() -> list[dict]:
    return read_jsonl(JUDGE_SET)


def load_refusal_cases() -> list[dict]:
    """主集里「不可答 40 + 误杀探针 20」→ 转成 evaluate_refusal 需要的 case 形状。"""
    rows = read_jsonl(EVAL_SET)
    cases = []
    for r in rows:
        if r.get("unanswerable"):
            cases.append(
                {
                    "query_id": r["id"],
                    "query": r["query"],
                    "category": "unanswerable",
                    "expected_status": "insufficient_evidence",
                }
            )
        elif r.get("source") == "miss_kill":
            cases.append(
                {
                    "query_id": r["id"],
                    "query": r["query"],
                    "category": "miss_kill",
                    "expected_status": "answered",  # 可答，正确答案是「不要拒答」
                }
            )
    return cases


# ------------------------------------------------------------------
# 阶段 1：生成（检索 → 生成，不做裁判）
# ------------------------------------------------------------------
def stage_gen(limit: int | None = None) -> None:
    items = load_judge_items()
    if limit:
        items = items[:limit]
    done = {r["id"] for r in read_jsonl(GEN_FILE)}
    todo = [x for x in items if x["id"] not in done]
    log(f"生成阶段：共 {len(items)} 条，已完成 {len(done)}，本次待跑 {len(todo)}（生成模型 {LLM_MODEL}）")
    if not todo:
        log("生成阶段已完成，跳过")
        return

    app = Appender(GEN_FILE)
    t0 = time.time()
    try:
        for i, it in enumerate(todo, start=1):
            q = it["query"]
            tq = time.time()
            try:
                answer, context = _retrieve_and_generate(q)
                err = None
            except Exception as e:  # 单条失败不拖垮整批
                answer, context, err = "", "", f"{type(e).__name__}: {e}"
            app.write(
                {
                    "id": it["id"],
                    "topic_id": it.get("topic_id"),
                    "query": q,
                    "reference_answer": it.get("reference_answer", ""),
                    "doc_slugs": it.get("doc_slugs", []),
                    "answer": answer,
                    "context": context,
                    "error": err,
                    "elapsed": round(time.time() - tq, 2),
                    "run_ts": datetime.now().isoformat(timespec="seconds"),
                }
            )
            if i % 5 == 0 or i == len(todo):
                rate = (time.time() - t0) / i
                log(f"  生成 {i}/{len(todo)}  均 {rate:.1f}s/条  ETA {rate*(len(todo)-i)/60:.1f}min")
    finally:
        app.close()
    log(f"生成阶段完成，耗时 {(time.time()-t0)/60:.1f} min")


# ------------------------------------------------------------------
# 阶段 2：三维裁判（Faithfulness / Answer Relevancy / Context Recall）
# ------------------------------------------------------------------
def _judge(prompt: str, system: str) -> dict:
    raw = _call_ollama(prompt, system=system, model=JUDGE_MODEL)
    score, reason = _parse_score(raw)
    return {"score": score, "reason": reason, "judge_model": JUDGE_MODEL}


def stage_judge(limit: int | None = None) -> None:
    gens = read_jsonl(GEN_FILE)
    if limit:
        gens = gens[:limit]
    done = {r["id"] for r in read_jsonl(JUDGE_FILE)}
    todo = [g for g in gens if g["id"] not in done]
    log(f"裁判阶段：共 {len(gens)} 条，已完成 {len(done)}，本次待跑 {len(todo)}（裁判模型 {JUDGE_MODEL}）")
    if not todo:
        log("裁判阶段已完成，跳过")
        return

    app = Appender(JUDGE_FILE)
    t0 = time.time()
    try:
        for i, g in enumerate(todo, start=1):
            tq = time.time()
            answer = g.get("answer") or ""
            context = g.get("context") or ""
            ref = g.get("reference_answer") or ""
            if context:
                f = _judge(_build_faithfulness_prompt(answer, context), FAITHFULNESS_SYSTEM)
            else:
                f = {"score": 0.0, "reason": "无检索上下文", "judge_model": JUDGE_MODEL}
            ar = _judge(_build_answer_relevancy_prompt(answer, g["query"]), ANSWER_RELEVANCY_SYSTEM)
            cr = (
                _judge(_build_context_recall_prompt(context, ref), CONTEXT_RECALL_SYSTEM)
                if ref
                else {"score": 0.0, "reason": "无参考答案", "judge_model": JUDGE_MODEL}
            )
            app.write(
                {
                    "id": g["id"],
                    "query": g["query"],
                    "faithfulness": f,
                    "answer_relevancy": ar,
                    "context_recall": cr,
                    "elapsed": round(time.time() - tq, 2),
                    "run_ts": datetime.now().isoformat(timespec="seconds"),
                }
            )
            if i % 5 == 0 or i == len(todo):
                rate = (time.time() - t0) / i
                log(f"  裁判 {i}/{len(todo)}  均 {rate:.1f}s/条  ETA {rate*(len(todo)-i)/60:.1f}min")
    finally:
        app.close()
    log(f"裁判阶段完成，耗时 {(time.time()-t0)/60:.1f} min")


# ------------------------------------------------------------------
# 阶段 3：端到端拒答（复用 acceptance_eval.evaluate_refusal）
# ------------------------------------------------------------------
def stage_refusal(limit: int | None = None) -> dict | None:
    from src.evaluation import acceptance_eval as AE

    cases = load_refusal_cases()
    if limit:
        cases = cases[:limit]
    # 把它的落盘路径指到本目录，避免覆盖 09 月那批 fastapi-zh 的 acceptance_eval_results.json
    AE.OUT_FILE = REFUSAL_FILE
    if not AE._preflight():
        log("Ollama 未就绪，拒答阶段中止（不产出假数字）")
        return None
    res = AE.evaluate_refusal(cases, use_judge=False)
    log(f"拒答阶段完成：{json.dumps(res['summary'], ensure_ascii=False)[:200]}")
    return res


# ------------------------------------------------------------------
# 阶段 4：汇总
# ------------------------------------------------------------------
def stage_report() -> dict:
    gens = {r["id"]: r for r in read_jsonl(GEN_FILE)}
    judges = read_jsonl(JUDGE_FILE)
    summary: dict = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "corpus": os.environ.get("MCP_RAG_CORPUS", "tidb"),
        "models": {"generation": LLM_MODEL, "judge": JUDGE_MODEL},
        "generation": {
            "n": len(gens),
            "n_error": sum(1 for r in gens.values() if r.get("error")),
            "n_empty_answer": sum(1 for r in gens.values() if not (r.get("answer") or "").strip()),
            "mean_elapsed_s": round(
                sum(r.get("elapsed", 0) for r in gens.values()) / max(1, len(gens)), 2
            ),
        },
    }

    def avg(recs, key):
        v = [r[key]["score"] for r in recs if r.get(key, {}).get("score") is not None]
        return round(sum(v) / len(v), 4) if v else None

    if judges:
        summary["three_metrics"] = {
            "n": len(judges),
            "faithfulness": avg(judges, "faithfulness"),
            "answer_relevancy": avg(judges, "answer_relevancy"),
            "context_recall": avg(judges, "context_recall"),
            "faithfulness_ge_0.8": round(
                sum(1 for r in judges if r["faithfulness"]["score"] >= 0.8) / len(judges), 4),
            "low_faithfulness_ids": [
                r["id"] for r in judges if r["faithfulness"]["score"] < 0.5
            ],
        }

    if REFUSAL_FILE.exists():
        data = json.loads(REFUSAL_FILE.read_text(encoding="utf-8"))
        refusal = data.get("refusal") or data
        if isinstance(refusal, dict) and refusal.get("details"):
            det = refusal["details"]
            valid = [d for d in det if d.get("verdict") in ("correct_refusal", "false_acceptance")]
            un = [d for d in valid if d.get("category") == "unanswerable"]
            mk = [d for d in valid if d.get("category") == "miss_kill"]
            # ⚠️ 判据必须用 status，不能用 verdict。
            # verdict 由 acceptance_eval.evaluate_refusal 生成，其语义是给「期望拒答」的
            # unanswerable 集设计的（作答 → false_acceptance，拒答 → correct_refusal）。
            # miss_kill 是「期望作答」的探针，套同一语义必然写反（曾误报误杀 19/0.95）。
            # 两个口径各自只算一次，避免同一逻辑两处实现再次分叉。
            un_refused = sum(1 for d in un if d["status"] == "insufficient_evidence")
            mk_false_refused = sum(1 for d in mk if d["status"] == "insufficient_evidence")
            summary["refusal_end_to_end"] = {
                "total": len(det),
                "valid": len(valid),
                "skipped": len(det) - len(valid),
                "unanswerable_n": len(un),
                "unanswerable_refused": un_refused,
                "unanswerable_refusal_rate": round(un_refused / len(un), 4) if un else None,
                "miss_kill_n": len(mk),
                "miss_kill_false_refused": mk_false_refused,
                "miss_kill_false_refusal_rate": round(mk_false_refused / len(mk), 4) if mk else None,
                "status_distribution": refusal.get("status_distribution"),
                "details_file": str(REFUSAL_FILE.relative_to(ROOT)),
            }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_FILE.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log("===== 汇总 =====")
    log(f"  生成：{summary['generation']['n']} 条，失败 {summary['generation']['n_error']}，"
        f"空答 {summary['generation']['n_empty_answer']}")
    if "three_metrics" in summary:
        t = summary["three_metrics"]
        log(f"  三维（{t['n']} 条）：Faithfulness {t['faithfulness']} / "
            f"Relevancy {t['answer_relevancy']} / ContextRecall {t['context_recall']}")
    if "refusal_end_to_end" in summary:
        r = summary["refusal_end_to_end"]
        log(f"  拒答：不可答 {r['unanswerable_refused']}/{r['unanswerable_n']}="
            f"{r['unanswerable_refusal_rate']}  误杀 {r['miss_kill_false_refused']}/"
            f"{r['miss_kill_n']}={r['miss_kill_false_refusal_rate']}")
    log(f"汇总已写入 {SUMMARY_FILE.relative_to(ROOT)}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="生成层三维指标 + 端到端拒答（TiDB 评测集）")
    parser.add_argument("--stage", choices=["all", "gen", "judge", "refusal", "report"], default="all")
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 条（冒烟用）")
    parser.add_argument("--dry-run", action="store_true",
                        help="只打印将跑什么、样本量、产物路径；不加载模型、不调 LLM")
    args = parser.parse_args()

    judges = load_judge_items()
    cases = load_refusal_cases()
    log(f"样本：生成层三维 {len(judges)} 条（judge_gold_100）｜ 拒答链路 {len(cases)} 条"
        f"（不可答 {sum(1 for c in cases if c['category']=='unanswerable')} + "
        f"误杀 {sum(1 for c in cases if c['category']=='miss_kill')}）")
    if args.dry_run:
        log(f"[dry-run] 产物目录 {OUT_DIR.relative_to(ROOT)}")
        log(f"[dry-run] 生成={GEN_FILE.name}  裁判={JUDGE_FILE.name}  拒答={REFUSAL_FILE.name}  汇总={SUMMARY_FILE.name}")
        log("[dry-run] 本阶段不加载模型、不调 LLM、不写文件")
        return

    if args.stage in ("all", "gen"):
        stage_gen(args.limit)
    if args.stage in ("all", "judge"):
        stage_judge(args.limit)
    if args.stage in ("all", "refusal"):
        stage_refusal(args.limit)
    if args.stage in ("all", "report", "gen", "judge", "refusal"):
        stage_report()


if __name__ == "__main__":
    main()
