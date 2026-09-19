"""端到端验收评测：拒答集 + 多证据集。

补上四类端到端验收里**原本没有自动化入口**的两类。其它脚本的分工：

| 脚本 | 测什么 | 调 LLM？ |
|---|---|---|
| `retrieval_eval.py` | 检索层（MRR / Hit@K / Prec@K / Recall@K） | 否 |
| `experiments.py` | 消融 / 参数扫描 / 延迟 | 否 |
| `llm_eval.py` | 生成层（18 条**可答**的三维打分） | 是 |
| `agent_eval.py` | Agent 编排（查询改写收益、保真度） | 是 |
| **`acceptance_eval.py`（本文件）** | **该拒的拒没拒（10 条拒答）+ 该跨文档的有没有跨（3 条多证据）** | 是 |

跑的是什么：**真实的 Agent 链路**（`agent.run_query` → 检索 → 生成 → validator），
不是只算检索分数 —— 拒答与跨文档引用都必须由 LLM 参与才测得出来。

用法：

    set MCP_RAG_CORPUS=fastapi-zh
    python -m src.evaluation.acceptance_eval              # 两套都跑
    python -m src.evaluation.acceptance_eval --refusal    # 只跑拒答集（10 条）
    python -m src.evaluation.acceptance_eval --multi      # 只跑多证据集（3 条）
    python -m src.evaluation.acceptance_eval --judge      # 额外给"该拒却答了"的用例打忠实度分
    python -m src.evaluation.acceptance_eval --dry-run    # 只加载验收集自检，不碰模型

前置：**Ollama 必须在跑**（见 `journal/端到端验收演练手册-2026-09-13.md`）。脚本会先做连通性自检，
不通就直接退出 —— 否则会跑出一堆 `generation_unavailable`，看起来像"系统很爱拒答"，
其实是服务没起，属于会误导人的假数字。

环境变量（不设会联网查模型更新，在墙内挂住）：

    $env:HF_HUB_OFFLINE="1"; $env:TRANSFORMERS_OFFLINE="1"

产物：`experiments/acceptance_eval_results.json`（逐条明细 + 汇总）。逐条落盘，
中途 Ctrl+C 也不会丢掉已完成的部分。
"""

from __future__ import annotations

import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from agent import run_query  # noqa: E402
from config import ANSWER_TOP_K, EXPERIMENTS_DIR, JUDGE_MODEL, LLM_MODEL, PROJECT_ROOT  # noqa: E402
from src.answer_validator import AnswerStatus  # noqa: E402

# 复用 llm_eval 的裁判实现（同一套 prompt 与打分解析，避免两处标准漂移）
from src.evaluation.llm_eval import (  # noqa: E402
    FAITHFULNESS_SYSTEM,
    _build_faithfulness_prompt,
    _call_ollama,
    _parse_score,
)

UNANS_FILE = PROJECT_ROOT / "data" / "unanswerable_queries.json"
MULTI_FILE = PROJECT_ROOT / "data" / "multi_evidence_queries.json"
OUT_FILE = EXPERIMENTS_DIR / "acceptance_eval_results.json"

S_ANSWERED = AnswerStatus.ANSWERED.value
S_REFUSED = AnswerStatus.INSUFFICIENT_EVIDENCE.value
S_UNAVAILABLE = AnswerStatus.GENERATION_UNAVAILABLE.value


# ============================================================
# 通用工具
# ============================================================


def _load_cases(path: Path) -> list[dict]:
    """两个验收集都是顶层 JSON 数组（和 test_queries.json 的 test_cases 包裹不同）。"""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path.name} 期望顶层是数组，实际是 {type(data).__name__}")
    return data


def _preflight() -> bool:
    """跑之前先确认 Ollama 真的能出字，避免产出全是降级的假数字。"""
    print(f"  连通性自检（{LLM_MODEL}）...", end=" ", flush=True)
    resp = _call_ollama("回复 OK", system="只回复 OK 两个字。", model=LLM_MODEL)
    if not resp:
        print("失败\n")
        print("  [ERROR] Ollama 未连接或模型未加载。请依次检查：")
        print("    1) 服务是否在跑：  11434 端口应有监听（ollama serve）")
        print('    2) 模型是否存在：  & "D:\\1software\\ollama\\ollama-windows-amd64\\ollama.exe" list')
        print("    3) OLLAMA_MODELS： 不要覆盖，应继承用户级变量 D:\\1software\\ollama_models")
        print("    4) 内存是否够：    3b 需 ≳2.5GB、7b 需 ≳6GB 物理可用")
        print("\n  排查手册：journal/端到端验收演练手册-2026-09-13.md")
        return False
    print(f"OK（{resp.strip()[:20]}）")
    return True


def _run_agent_case(query: str) -> dict:
    """跑一次真实 Agent 链路，返回答案体、用到的候选块、耗时。"""
    t0 = time.perf_counter()
    state = run_query(query)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    chunks = state.get("last_retrieved_chunks") or []
    return {
        "answer": state.get("answer") or {},
        "chunks": chunks[:ANSWER_TOP_K],  # 与 generate_answer 的编号口径一致
        "elapsed_ms": round(elapsed_ms, 1),
        "attempt": state.get("attempt", 0),
        "rewritten_queries": state.get("rewritten_queries", []),
    }


def _cited_sources(chunks: list[dict], used_ids: list) -> list[str]:
    """把 used_source_ids（1-based，指向 Top-K 候选）映射回真实来源文件。"""
    out = []
    for i in used_ids or []:
        try:
            idx = int(i) - 1
        except (TypeError, ValueError):
            continue
        if 0 <= idx < len(chunks):
            src = chunks[idx].get("source_doc", "")
            if src and src not in out:
                out.append(src)
    return out


def _keywords_hit(text: str, keywords: list[str]) -> list[str]:
    """大小写不敏感的子串命中；返回命中的关键词（用于区分"检索没带来"与"带来了没用"）。"""
    low = (text or "").lower()
    return [k for k in keywords if k and k.lower() in low]


def _join_context(chunks: list[dict]) -> str:
    return "\n\n".join(
        f"[来源{i + 1}: {c.get('source_doc', '?')}] {c.get('content', '')}"
        for i, c in enumerate(chunks)
    )


def _save(result: dict) -> None:
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


# ============================================================
# 一、拒答集（10 条）
# ============================================================


def _judge_faithfulness(answer: str, chunks: list[dict]) -> dict:
    """给"本该拒答却答了"的用例打忠实度分，判断它是'跑到题外'还是'编造事实'。"""
    raw = _call_ollama(
        _build_faithfulness_prompt(answer, _join_context(chunks)),
        system=FAITHFULNESS_SYSTEM,
        model=JUDGE_MODEL,
    )
    score, reason = _parse_score(raw)
    return {"score": score, "reason": reason, "judge_model": JUDGE_MODEL}


def evaluate_refusal(cases: list[dict], use_judge: bool = False, verbose: bool = True) -> dict:
    print("\n" + "=" * 62)
    print(f"  拒答验收 —— data/unanswerable_queries.json（{len(cases)} 条）")
    print(f"  期望状态：{S_REFUSED}；误答即 false acceptance（幻觉风险）")
    print("=" * 62)

    details = []
    for i, case in enumerate(cases, 1):
        qid = case.get("query_id") or case.get("id")
        query = case["query"]
        print(f"\n[{i}/{len(cases)}] {qid} | {case.get('category', '-')} | {query}")
        print(f"        ({case.get('expected_status', S_REFUSED)})")

        run = _run_agent_case(query)
        ans = run["answer"]
        status = ans.get("status", "")
        answer_text = ans.get("answer", "") or ""
        cited = _cited_sources(run["chunks"], ans.get("used_source_ids"))

        rec = {
            "query_id": qid,
            "query": query,
            "category": case.get("category"),
            "expected_status": case.get("expected_status", S_REFUSED),
            "status": status,
            "reason": ans.get("reason"),
            "used_source_ids": ans.get("used_source_ids", []),
            "cited_sources": cited,
            "retrieved_sources": [c.get("source_doc") for c in run["chunks"]],
            "answer": answer_text,
            "attempt": run["attempt"],
            "rewritten_queries": run["rewritten_queries"],
            "elapsed_ms": run["elapsed_ms"],
        }

        if status == S_REFUSED:
            rec["verdict"] = "correct_refusal"
            print("        → 拒答 ✅（符合预期）")
        elif status == S_ANSWERED:
            rec["verdict"] = "false_acceptance"
            print("        → 作答 ❌（该拒却答，需人工复核是否幻觉）")
            print(f"          答案前 80 字：{answer_text[:80]}")
            if use_judge:
                rec["faithfulness"] = _judge_faithfulness(answer_text, run["chunks"])
                print(f"          忠实度 {rec['faithfulness']['score']:.2f}（{JUDGE_MODEL}）")
        elif status == S_UNAVAILABLE:
            rec["verdict"] = "skipped_generation_unavailable"
            print(f"        → 服务不可用（{ans.get('reason')}），本条不计入分母")
        else:
            rec["verdict"] = "invalid_output"
            print(f"        → 输出非法（{ans.get('reason')}），本条不计入分母")

        details.append(rec)
        _save({"partial": True, "refusal": {"details": details}})

    valid = [d for d in details if d["verdict"] in ("correct_refusal", "false_acceptance")]
    refused = [d for d in valid if d["verdict"] == "correct_refusal"]
    accepted = [d for d in valid if d["verdict"] == "false_acceptance"]

    by_cat: dict[str, dict] = {}
    for d in valid:
        c = by_cat.setdefault(d["category"] or "-", {"total": 0, "correct_refusal": 0, "false_acceptance": 0})
        c["total"] += 1
        c[d["verdict"]] += 1
    for c in by_cat.values():
        c["refusal_rate"] = round(c["correct_refusal"] / c["total"], 4) if c["total"] else None

    summary = {
        "total": len(details),
        "valid": len(valid),
        "skipped": len(details) - len(valid),
        "correct_refusal": len(refused),
        "false_acceptance": len(accepted),
        "refusal_rate": round(len(refused) / len(valid), 4) if valid else None,
        "false_acceptance_rate": round(len(accepted) / len(valid), 4) if valid else None,
        "status_distribution": dict(Counter(d["status"] for d in details)),
        "by_category": by_cat,
        "false_acceptance_cases": [d["query_id"] for d in accepted],
        "judged": False,
    }

    judged = [d for d in accepted if d.get("faithfulness")]
    if judged:
        scores = [d["faithfulness"]["score"] for d in judged if d["faithfulness"].get("score") is not None]
        if scores:
            summary["judged"] = True
            summary["judge_model"] = JUDGE_MODEL
            summary["mean_faithfulness_on_false_acceptance"] = round(sum(scores) / len(scores), 4)
            summary["hallucination_rate"] = round(sum(1 for s in scores if s < 0.5) / len(scores), 4)

    return {"summary": summary, "details": details}


# ============================================================
# 二、多证据集（3 条）
# ============================================================


def evaluate_multi(cases: list[dict], verbose: bool = True) -> dict:
    print("\n" + "=" * 62)
    print(f"  多证据验收 —— data/multi_evidence_queries.json（{len(cases)} 条）")
    print(f"  期望状态：{S_ANSWERED}；关注'引用是否真的覆盖了两个来源'")
    print("=" * 62)

    details = []
    for i, case in enumerate(cases, 1):
        qid = case.get("query_id") or case.get("id")
        query = case["query"]
        golden = case.get("golden_chunk_sources", [])
        reqs = case.get("required_sources", [])
        print(f"\n[{i}/{len(cases)}] {qid} | {query}")

        run = _run_agent_case(query)
        ans = run["answer"]
        answer_text = ans.get("answer", "") or ""
        cited = _cited_sources(run["chunks"], ans.get("used_source_ids"))
        context = _join_context(run["chunks"])
        retrieved = [c.get("source_doc") for c in run["chunks"]]

        # 引用覆盖：答案实际引用的来源里，命中了几个 golden 来源
        cited_hit = sorted(set(cited) & set(golden))
        citation_coverage = len(cited_hit) / len(golden) if golden else None
        retrieved_hit = sorted(set(s for s in retrieved if s in golden))
        retrieval_coverage = len(retrieved_hit) / len(golden) if golden else None

        per_req = []
        for r in reqs:
            kws = r.get("keywords", [])
            hit_ans = _keywords_hit(answer_text, kws)
            hit_ctx = _keywords_hit(context, kws)
            per_req.append({
                "description": r.get("description"),
                "keywords": kws,
                "hit_in_answer": bool(hit_ans),
                "hit_in_context": bool(hit_ctx),
                "matched_in_answer": hit_ans,
                "matched_in_context": hit_ctx,
            })
        kw_ans = sum(1 for r in per_req if r["hit_in_answer"]) / len(per_req) if per_req else None
        kw_ctx = sum(1 for r in per_req if r["hit_in_context"]) / len(per_req) if per_req else None

        rec = {
            "query_id": qid,
            "query": query,
            "expected_status": case.get("expected_status", S_ANSWERED),
            "status": ans.get("status"),
            "reason": ans.get("reason"),
            "used_source_ids": ans.get("used_source_ids", []),
            "cited_sources": cited,
            "golden_chunk_sources": golden,
            "retrieved_sources": retrieved,
            "answer": answer_text,
            "citation_coverage": citation_coverage,
            "retrieval_coverage": retrieval_coverage,
            "cross_doc_cited": len(set(cited)) >= 2,
            "keyword_coverage_answer": kw_ans,
            "keyword_coverage_context": kw_ctx,
            "per_requirement": per_req,
            "attempt": run["attempt"],
            "elapsed_ms": run["elapsed_ms"],
        }
        details.append(rec)

        print(f"        status={rec['status']}  引用来源={cited}")
        print(f"        golden={golden}")
        print(f"        引用覆盖={_pct(citation_coverage)}  检索覆盖={_pct(retrieval_coverage)}"
              f"  跨文档引用={'是' if rec['cross_doc_cited'] else '否'}")
        print(f"        关键词覆盖（答案/检索证据）= {_pct(kw_ans)} / {_pct(kw_ctx)}")
        _save({"partial": True, "multi": {"details": details}})

    asked = [d for d in details if d["status"] == S_ANSWERED]
    citation_vals = [d["citation_coverage"] for d in details if d["citation_coverage"] is not None]
    kw_ans_vals = [d["keyword_coverage_answer"] for d in details if d["keyword_coverage_answer"] is not None]
    kw_ctx_vals = [d["keyword_coverage_context"] for d in details if d["keyword_coverage_context"] is not None]

    summary = {
        "total": len(details),
        "answered": len(asked),
        "answer_rate": round(len(asked) / len(details), 4) if details else None,
        "cross_doc_cited_rate": round(sum(1 for d in details if d["cross_doc_cited"]) / len(details), 4) if details else None,
        "mean_citation_coverage": round(sum(citation_vals) / len(citation_vals), 4) if citation_vals else None,
        "mean_keyword_coverage_answer": round(sum(kw_ans_vals) / len(kw_ans_vals), 4) if kw_ans_vals else None,
        "mean_keyword_coverage_context": round(sum(kw_ctx_vals) / len(kw_ctx_vals), 4) if kw_ctx_vals else None,
        "status_distribution": dict(Counter(d["status"] for d in details)),
    }
    return {"summary": summary, "details": details}


def _pct(v) -> str:
    return "—" if v is None else f"{v:.1%}"


# ============================================================
# 汇总打印
# ============================================================


def _print_refusal_summary(s: dict) -> None:
    print("\n" + "-" * 62)
    print("  拒答集汇总")
    print("-" * 62)
    print(f"  计入分母 = {s['valid']} 条（跳过 {s['skipped']} 条：服务不可用/输出非法）")
    print(f"  正确拒答 = {s['correct_refusal']}    误答 = {s['false_acceptance']}")
    print(f"  拒答率   = {_pct(s['refusal_rate'])}    误答率 = {_pct(s['false_acceptance_rate'])}")
    if s["false_acceptance_cases"]:
        print(f"  误答用例 = {', '.join(s['false_acceptance_cases'])}")
    if s["judged"]:
        print(f"  误答案忠实度（{s.get('judge_model')}）均值 = {s['mean_faithfulness_on_false_acceptance']:.2f}"
              f"    判定为幻觉的比例 = {_pct(s['hallucination_rate'])}")
    print("  分类别：")
    for cat, c in sorted(s["by_category"].items()):
        print(f"    {cat:<16} 共 {c['total']:>2}  拒答 {c['correct_refusal']:>2}  误答 {c['false_acceptance']:>2}"
              f"  拒答率 {_pct(c['refusal_rate'])}")


def _print_multi_summary(s: dict) -> None:
    print("\n" + "-" * 62)
    print("  多证据集汇总")
    print("-" * 62)
    print(f"  作答率 = {_pct(s['answer_rate'])}（{s['answered']}/{s['total']}）")
    print(f"  跨文档引用率（引用≥2 个不同来源）= {_pct(s['cross_doc_cited_rate'])}")
    print(f"  平均引用覆盖率（命中的 golden 来源占比）= {_pct(s['mean_citation_coverage'])}")
    print(f"  平均关键词覆盖：答案 {_pct(s['mean_keyword_coverage_answer'])}"
          f" / 检索证据 {_pct(s['mean_keyword_coverage_context'])}")
    print("  ↑ 答案 << 检索证据 = 检索没问题，是生成没把两处整合起来")


# ============================================================
# CLI
# ============================================================


def main() -> None:
    logging.basicConfig(level=logging.WARNING)

    only_refusal = "--refusal" in sys.argv
    only_multi = "--multi" in sys.argv
    use_judge = "--judge" in sys.argv
    dry_run = "--dry-run" in sys.argv

    import os

    if not dry_run and not os.environ.get("HF_HUB_OFFLINE"):
        print("[WARN] 未设置 HF_HUB_OFFLINE=1 —— 若本地模型缓存命中失败，会卡在联网检查上（实测可达 10 分钟）。")

    refusal_cases = _load_cases(UNANS_FILE)
    multi_cases = _load_cases(MULTI_FILE)

    if dry_run:
        print(f"拒答集  {UNANS_FILE.name}   {len(refusal_cases)} 条")
        print(f"多证据集 {MULTI_FILE.name}  {len(multi_cases)} 条")
        print(f"输出     {OUT_FILE}")
        print("（--dry-run：只自检，不加载模型、不调 LLM）")
        return

    print("=" * 62)
    print("  端到端验收：拒答集 + 多证据集")
    print(f"  语料 = {_corpus_name()}   生成模型 = {LLM_MODEL}")
    print("=" * 62)

    if not _preflight():
        raise SystemExit(1)

    result: dict = {
        "meta": {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "corpus": _corpus_name(),
            "llm_model": LLM_MODEL,
            "judge_model": JUDGE_MODEL if use_judge else None,
            "answer_top_k": ANSWER_TOP_K,
        }
    }
    t0 = time.perf_counter()
    try:
        if not only_multi:
            result["refusal"] = evaluate_refusal(refusal_cases, use_judge=use_judge)
        if not only_refusal:
            result["multi"] = evaluate_multi(multi_cases)
    except KeyboardInterrupt:
        print("\n[中断] 已完成的部分已落盘（逐条保存），未完成的不计入汇总。")
        _save(result)
        raise SystemExit(130)

    result["meta"]["elapsed_seconds"] = round(time.perf_counter() - t0, 1)
    _save(result)

    if "refusal" in result:
        _print_refusal_summary(result["refusal"]["summary"])
    if "multi" in result:
        _print_multi_summary(result["multi"]["summary"])

    print(f"\n  → 已保存: {OUT_FILE}")


def _corpus_name() -> str:
    from config import CORPUS

    return CORPUS


if __name__ == "__main__":
    main()
