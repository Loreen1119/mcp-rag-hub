"""
Agent 改写评测 — 评测查询改写的有效性和语义保真度。

三个评测任务:
1. 改写效果对比: 原始 query vs 改写 query 的检索指标 delta
2. 语义保真度: LLM 判断改写是否保留了原始信息需求
3. CE 阈值校准: 扫 CE 阈值，找到最能预测「改写是否有效」的最优值

运行:
    python src/evaluation/agent_eval.py              # 跑改写评测 + 保真度
    python src/evaluation/agent_eval.py --calibrate   # 追加 CE 阈值校准

输出:
    experiments/agent_evaluation_results.json
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import List

_PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import (
    TEST_QUERIES_FILE,
    CE_TOP_K,
    CE_THRESHOLD,
    BM25_TOP_K,
    EXPERIMENTS_DIR,
)
from src.evaluation.metrics import load_test_cases, mrr, hit_at_k, precision_at_k, recall_at_k
from src.pipeline import get_pipeline
from src.fusion import reciprocal_rank_fusion

logger = logging.getLogger(__name__)

EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# LLM 调用
# ============================================================


def _call_ollama(prompt: str, system: str = "") -> str:
    try:
        import requests

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        r = requests.post(
            "http://127.0.0.1:11434/api/chat",
            json={
                "model": "qwen2.5:7b",
                "messages": messages,
                "stream": False,
                "options": {"temperature": 0.0, "num_predict": 256},
            },
            timeout=120,
        )
        if r.status_code == 200:
            return r.json()["message"]["content"]
        return ""
    except Exception:
        return ""


# ============================================================
# 语义保真度 Prompt
# ============================================================

FIDELITY_SYSTEM = """你是一个查询改写评审专家。你的任务是判断"改写后的查询"是否保留了"原查询"的信息需求。

注意：评的是"信息需求"是否保留，不是"字面"是否相似。
- "怎么报销" → "差旅费报销审批流程"：信息需求保留 ✅（都是找报销方法）
- "怎么报销" → "财务软件操作指南"：信息需求偏离 ❌（偏到软件操作了）

输出格式（严格 JSON）：
{"score": <0.0~1.0>, "reason": "<一句话说明保留程度或偏离原因>"}"""


def _build_fidelity_prompt(original: str, rewritten: str) -> str:
    return f"""原查询: {original}
改写查询: {rewritten}

改写后的查询是否保留了原查询的信息需求？只输出 JSON。"""


# ============================================================
# 查询改写（独立函数，不依赖 Agent 状态机）
# ============================================================


def _rewrite_query_standalone(original: str) -> str:
    """独立的查询改写函数，拷贝自 agent.py 的 rewrite_query 逻辑。"""
    prompt = f"""你是一个查询改写助手。用户的原始查询检索效果不佳，请改写查询以获得更好的检索结果。

改写规则：
1. 保留原始意图，不要引入新概念
2. 将口语化表达转成书面化技术术语
3. 扩展缩写和专业简称
4. 如果原始查询是中文，改写后也必须是中文
5. 只输出改写后的查询文本，不要加任何解释

原始查询: {original}

改写查询:"""

    rewritten = _call_ollama(prompt)

    # fallback: LLM 不可用时，规则式追加关键词
    if not rewritten:
        parts = [original]
        if "优化" in original:
            parts.append("性能提升 优化方向")
        if "检索" in original:
            parts.append("信息检索 search retrieval")
        rewritten = " ".join(parts)

    return rewritten


# ============================================================
# AgentEvaluator
# ============================================================


class AgentEvaluator:
    """Agent 评测器 — 评测改写策略的有效性。

    三个评测任务共享中间结果（一次 LLM 改写 + 一次检索），
    避免重复调用。
    """

    def __init__(self, test_cases: list[dict]):
        self.test_cases = test_cases
        self.results: list[dict] = []

        # 初始化管线（共享单例）
        logger.info("初始化 RAG 管线...")
        ctx = get_pipeline()
        self.bm25 = ctx.bm25
        self.vector = ctx.vector
        self.pipeline = ctx.pipeline
        logger.info("管线就绪 — %d 个 Chunk", len(ctx.chunks))

    # ----------------------------------------------------------
    # 主入口
    # ----------------------------------------------------------

    def evaluate(self, verbose: bool = True) -> dict:
        """一口气跑完改写效果对比 + 语义保真度，共享中间结果。"""
        if verbose:
            print("=" * 60)
            print("  Agent 改写评测")
            print(f"  Test Cases: {len(self.test_cases)}")
            print("=" * 60)

        per_case: list[dict] = []

        for tc in self.test_cases:
            query = tc["query"]
            goldens = tc["golden_chunk_sources"]

            # Step 1: 改写 query（LLM 调用一次）
            rewritten = _rewrite_query_standalone(query)

            # Step 2: 原始检索 vs 改写检索（共享管线）
            bm25_orig = self.bm25.search(query, top_k=BM25_TOP_K)
            vec_orig = self.vector.search(query, top_k=BM25_TOP_K)
            rrf_orig = reciprocal_rank_fusion([bm25_orig, vec_orig])
            ce_orig = self.pipeline.reranker.rerank(query, rrf_orig, top_k=CE_TOP_K)

            bm25_rw = self.bm25.search(rewritten, top_k=BM25_TOP_K)
            vec_rw = self.vector.search(rewritten, top_k=BM25_TOP_K)
            rrf_rw = reciprocal_rank_fusion([bm25_rw, vec_rw])
            ce_rw = self.pipeline.reranker.rerank(rewritten, rrf_rw, top_k=CE_TOP_K)

            # Step 3: 计算指标（CE 阶段）
            mrr_orig = mrr(ce_orig, goldens)
            hit_orig = hit_at_k(ce_orig, goldens)
            prec_orig = precision_at_k(ce_orig, goldens)
            recall_orig = recall_at_k(ce_orig, goldens, k=CE_TOP_K)
            ce_score_orig = ce_orig[0].score if ce_orig else 0.0

            mrr_rw = mrr(ce_rw, goldens)
            hit_rw = hit_at_k(ce_rw, goldens)
            prec_rw = precision_at_k(ce_rw, goldens)
            recall_rw = recall_at_k(ce_rw, goldens, k=CE_TOP_K)

            delta_mrr = round(mrr_rw - mrr_orig, 4)
            delta_hit = hit_rw - hit_orig
            delta_prec = round(prec_rw - prec_orig, 4)

            # Step 4: 语义保真度（同一对 query，不重复调 LLM）
            fidelity = self._check_fidelity(query, rewritten)

            case_result = {
                "id": tc["id"],
                "category": tc["category"],
                "original_query": query,
                "rewritten_query": rewritten,
                "ce_score_original": round(ce_score_orig, 4),
                "mrr_original": round(mrr_orig, 4),
                "mrr_rewritten": round(mrr_rw, 4),
                "delta_mrr": delta_mrr,
                "hit_original": int(hit_orig),
                "hit_rewritten": int(hit_rw),
                "delta_hit": int(delta_hit),
                "precision_original": round(prec_orig, 4),
                "precision_rewritten": round(prec_rw, 4),
                "delta_precision": delta_prec,
                "recall_original": round(recall_orig, 4),
                "recall_rewritten": round(recall_rw, 4),
                "fidelity_score": fidelity["score"],
                "fidelity_reason": fidelity["reason"],
                "rewrite_was_helpful": delta_mrr > 0,
            }
            per_case.append(case_result)

            if verbose:
                icon = "✅" if delta_mrr > 0 else ("➖" if delta_mrr == 0 else "❌")
                print(
                    f"\n  [{tc['id']}] {query[:50]}"
                    f"\n    → 改写: {rewritten[:60]}"
                    f"\n    MRR: {mrr_orig:.4f} → {mrr_rw:.4f}  Δ={delta_mrr:+.4f}  {icon}"
                    f"\n    保真度: {fidelity['score']:.2f}  {fidelity['reason'][:60]}"
                )

        self.results = per_case

        summary = self._summarize(per_case)

        if verbose:
            self._print_summary(summary)

        output = {"summary": summary, "details": per_case}
        output_path = EXPERIMENTS_DIR / "agent_evaluation_results.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        if verbose:
            print(f"\n  → 已保存: {output_path}")

        return output

    # ----------------------------------------------------------
    # 语义保真度
    # ----------------------------------------------------------

    def _check_fidelity(self, original: str, rewritten: str) -> dict:
        raw = _call_ollama(
            _build_fidelity_prompt(original, rewritten),
            system=FIDELITY_SYSTEM,
        )
        score, reason = self._parse_score(raw)
        return {"score": score, "reason": reason}

    @staticmethod
    def _parse_score(raw: str) -> tuple[float, str]:
        try:
            if "```json" in raw:
                start = raw.index("```json") + 7
                end = raw.index("```", start)
                raw = raw[start:end]
            elif "```" in raw:
                start = raw.index("```") + 3
                end = raw.index("```", start)
                raw = raw[start:end]
            data = json.loads(raw.strip())
            score = float(data.get("score", 0))
            reason = data.get("reason", "")
            return max(0.0, min(1.0, score)), reason
        except (json.JSONDecodeError, ValueError, KeyError):
            return 0.0, f"解析失败: {raw[:80]}"

    # ----------------------------------------------------------
    # CE 阈值校准
    # ----------------------------------------------------------

    def calibrate_threshold(self, verbose: bool = True) -> dict:
        """扫 CE 阈值，找能最好预测「改写是否有效」的最优值。

        Ground truth: rewrite_mrr > original_mrr → 改写有效。
        用 F1 衡量每个阈值在二分类任务上的表现。
        """
        if not self.results:
            logger.warning("请先运行 evaluate() 再校准阈值")
            return {}

        if verbose:
            print("\n" + "=" * 60)
            print("  CE 阈值校准")
            print("=" * 60)

        best_threshold = CE_THRESHOLD
        best_f1 = 0.0
        sweep_results: list[dict] = []

        for threshold in self._sweep_range(1.0, 6.0, 0.5):
            tp = fp = tn = fn = 0

            for case in self.results:
                ce_score = case["ce_score_original"]
                actually_helpful = case["rewrite_was_helpful"]
                agent_would_rewrite = ce_score < threshold

                if actually_helpful and agent_would_rewrite:
                    tp += 1
                elif not actually_helpful and agent_would_rewrite:
                    fp += 1
                elif not actually_helpful and not agent_would_rewrite:
                    tn += 1
                else:
                    fn += 1

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )

            sweep_results.append(
                {
                    "threshold": threshold,
                    "tp": tp,
                    "fp": fp,
                    "tn": tn,
                    "fn": fn,
                    "precision": round(precision, 4),
                    "recall": round(recall, 4),
                    "f1": round(f1, 4),
                }
            )

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        total = len(self.results)
        helpful_count = sum(1 for c in self.results if c["rewrite_was_helpful"])
        not_helpful_count = total - helpful_count

        result = {
            "current_threshold": CE_THRESHOLD,
            "optimal_threshold": best_threshold,
            "best_f1": round(best_f1, 4),
            "recommendation": (
                f"当前阈值 {CE_THRESHOLD}，建议{'保持' if best_threshold == CE_THRESHOLD else f'调至 {best_threshold}'}"
            ),
            "confidence": "low" if total < 30 else "medium",
            "note": f"基于 {total} 组测试集（{helpful_count} 改写有效 / {not_helpful_count} 改写无效），建议积累更多数据后确认",
            "sweep": sweep_results,
        }

        if verbose:
            print(f"\n  当前阈值: {CE_THRESHOLD}")
            print(f"  最优阈值: {best_threshold}  (F1={best_f1:.4f})")
            print(f"  建议: {result['recommendation']}")
            print(f"  置信度: {result['confidence']}")
            print(f"\n  阈值扫描明细:")
            print(f"  {'阈值':>6s}  {'TP':>4s}  {'FP':>4s}  {'TN':>4s}  {'FN':>4s}  {'P':>6s}  {'R':>6s}  {'F1':>6s}")
            for s in sweep_results:
                print(
                    f"  {s['threshold']:>6.1f}"
                    f"  {s['tp']:>4d}  {s['fp']:>4d}  {s['tn']:>4d}  {s['fn']:>4d}"
                    f"  {s['precision']:>6.4f}  {s['recall']:>6.4f}  {s['f1']:>6.4f}"
                )

        # 追加保存到已有输出文件
        output_path = EXPERIMENTS_DIR / "agent_evaluation_results.json"
        if output_path.exists():
            with open(output_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
            existing["threshold_calibration"] = result
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(existing, f, ensure_ascii=False, indent=2)
            if verbose:
                print(f"\n  → 阈值校准已追加: {output_path}")

        return result

    @staticmethod
    def _sweep_range(start: float, end: float, step: float):
        vals = []
        v = start
        while v <= end + 0.001:
            vals.append(round(v, 2))
            v += step
        return vals

    # ----------------------------------------------------------
    # 汇总统计
    # ----------------------------------------------------------

    def _summarize(self, per_case: list[dict]) -> dict:
        n = len(per_case)
        helpful = sum(1 for c in per_case if c["rewrite_was_helpful"])
        degraded = sum(1 for c in per_case if c["delta_mrr"] < 0)
        unchanged = n - helpful - degraded

        deltas = [c["delta_mrr"] for c in per_case if c["delta_mrr"] != 0]
        avg_gain = sum(c["delta_mrr"] for c in per_case) / n if n > 0 else 0.0
        avg_fidelity = sum(c["fidelity_score"] for c in per_case) / n if n > 0 else 0.0

        by_category: dict[str, dict] = {}
        for cat in ["exact_match", "semantic", "mixed"]:
            cat_cases = [c for c in per_case if c["category"] == cat]
            cn = len(cat_cases)
            if cn == 0:
                continue
            cat_helpful = sum(1 for c in cat_cases if c["rewrite_was_helpful"])
            cat_gain = sum(c["delta_mrr"] for c in cat_cases) / cn
            cat_fidelity = sum(c["fidelity_score"] for c in cat_cases) / cn
            by_category[cat] = {
                "cases": cn,
                "helpful_count": cat_helpful,
                "helpful_rate": round(cat_helpful / cn, 4),
                "avg_delta_mrr": round(cat_gain, 4),
                "avg_fidelity": round(cat_fidelity, 4),
            }

        return {
            "total_cases": n,
            "helpful_count": helpful,
            "degraded_count": degraded,
            "unchanged_count": unchanged,
            "helpful_rate": round(helpful / n, 4) if n > 0 else 0.0,
            "avg_delta_mrr": round(avg_gain, 4),
            "avg_fidelity": round(avg_fidelity, 4),
            "max_delta_mrr": round(max(deltas), 4) if deltas else 0.0,
            "min_delta_mrr": round(min(deltas), 4) if deltas else 0.0,
            "by_category": by_category,
        }

    @staticmethod
    def _print_summary(summary: dict) -> None:
        print(f"\n{'='*60}")
        print("  汇总")
        print(f"{'='*60}")
        print(f"  Test Cases: {summary['total_cases']}")
        print(
            f"  改写有效: {summary['helpful_count']}  |  "
            f"改写无效: {summary['degraded_count']}  |  "
            f"无变化: {summary['unchanged_count']}"
        )
        print(f"  改写有效率: {summary['helpful_rate']:.2%}")
        print(f"  平均 Δ MRR: {summary['avg_delta_mrr']:+.4f}")
        print(f"  平均语义保真度: {summary['avg_fidelity']:.4f}")

        if summary.get("by_category"):
            print(f"\n  分类别:")
            print(
                f"  {'类别':<18s} {'数量':>5s}  {'有效率':>7s}  {'ΔMRR':>7s}  {'保真度':>7s}"
            )
            for cat, data in summary["by_category"].items():
                print(
                    f"  {cat:<18s} {data['cases']:>5d}  {data['helpful_rate']:>7.2%}"
                    f"  {data['avg_delta_mrr']:>+7.4f}  {data['avg_fidelity']:>7.4f}"
                )


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)

    test_cases = load_test_cases(TEST_QUERIES_FILE)
    evaluator = AgentEvaluator(test_cases)

    output = evaluator.evaluate(verbose=True)

    if "--calibrate" in sys.argv:
        evaluator.calibrate_threshold(verbose=True)
