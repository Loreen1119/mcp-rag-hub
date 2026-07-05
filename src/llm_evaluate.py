"""
LLM-as-Judge 生成评测 — 基于 Ollama 的三维 LLM 裁判打分。

三个 Ragas 标准指标：
- Faithfulness:  生成答案是否忠实于检索到的上下文？（逐句比对，检测幻觉）
- Answer Relevancy: 生成答案是否紧扣用户问题？（检测跑题）
- Context Recall: 检索上下文是否覆盖了参考答案的关键信息？（检测检索遗漏）

与 evaluate.py 的关系：
- evaluate.py 测检索阶段（MRR / Hit@K / Precision@K）
- llm_evaluate.py 测生成阶段（Faithfulness / Answer Relevancy / Context Recall）
- 两者共用 test_queries.json 的 golden_answer 字段

前置条件：
    1. 安装 Ollama: winget install Ollama.Ollama
    2. 设置模型目录到 D 盘: set OLLAMA_MODELS=D:\ollama_models
    3. 拉取模型: ollama pull qwen2.5:7b
    4. 启动服务: ollama serve  (通常安装后自动启动)

运行：
    python src/llm_evaluate.py              # 跑全部 15 组评测
    python src/llm_evaluate.py --sample 3    # 只跑前 3 组（快速验证）
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import List

# 确保项目根目录在 sys.path 中
_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import TEST_QUERIES_FILE, CE_TOP_K, EXPERIMENTS_DIR
from src.data_pipeline import process_directory
from src.retrievers import BM25Retriever, VectorRetriever
from src.fusion import FusionPipeline

logger = logging.getLogger(__name__)

EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

# 可通过 --model 命令行参数覆盖
LLM_MODEL = "qwen2.5:7b"


# ============================================================
# 管线单例
# ============================================================

_chunks: list = []
_bm25: BM25Retriever | None = None
_vector: VectorRetriever | None = None
_pipeline: FusionPipeline | None = None
_initialized: bool = False


def _ensure_pipeline():
    global _chunks, _bm25, _vector, _pipeline, _initialized
    if _initialized:
        return
    logger.info("初始化 RAG 管线...")
    _chunks = process_directory()
    _bm25 = BM25Retriever(_chunks)
    _vector = VectorRetriever(_chunks, rebuild=True)
    _pipeline = FusionPipeline()
    _initialized = True
    logger.info("管线就绪 — %d 个 Chunk", len(_chunks))


# ============================================================
# Ollama LLM 调用
# ============================================================


def _call_ollama(prompt: str, system: str = "") -> str:
    """通过 HTTP 直接调用 Ollama API（绕过 SDK 版本兼容问题）。"""
    try:
        import requests

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        r = requests.post(
            "http://127.0.0.1:11434/api/chat",
            json={
                "model": LLM_MODEL,
                "messages": messages,
                "stream": False,
                "options": {"temperature": 0.0, "num_predict": 512},
            },
            timeout=300,
        )
        if r.status_code == 200:
            return r.json()["message"]["content"]
        else:
            logger.warning("Ollama API 返回 %d: %s", r.status_code, r.text[:200])
            return ""
    except Exception as exc:
        logger.warning("Ollama 调用失败: %s", exc)
        return ""


# ============================================================
# 评分 Prompt 模板（三个指标各一个独立 prompt）
# ============================================================

FAITHFULNESS_SYSTEM = """你是一个严格的评审专家。你的任务是对比"生成答案"和"检索到的上下文"，判断生成答案中的每一条陈述是否可以在上下文中找到依据。

评分标准（0~1 之间）：
- 1.0: 答案中的所有陈述都能在上下文中直接找到原文支持，没有任何编造
- 0.7~0.9: 绝大部分有依据，有一处轻微的推论性表述（上下文暗示了但没有明确说）
- 0.4~0.6: 有 1~2 处明显的事实错误或编造
- 0.1~0.3: 大量编造，只有少量信息来自上下文
- 0.0: 答案完全脱离上下文，或与上下文矛盾

输出格式（严格 JSON）：
{"score": <0~1 的浮点数>, "reason": "<一句话说明扣分原因，满分则说'全部有据可查'>"}"""

ANSWER_RELEVANCY_SYSTEM = """你是一个严格的评审专家。你的任务是判断"生成答案"是否紧扣"用户问题"，有没有答非所问或跑题。

评分标准（0~1 之间）：
- 1.0: 答案完全围绕问题展开，每句话都在回应问题的核心诉求，没有废话
- 0.7~0.9: 基本扣题，有少量扩展性内容但不影响核心回答
- 0.4~0.6: 部分内容与问题相关，但有明显的跑题或答非所问
- 0.1~0.3: 大部分内容与问题无关，只有一两句沾边
- 0.0: 完全答非所问

输出格式（严格 JSON）：
{"score": <0~1 的浮点数>, "reason": "<一句话说明得分理由>"}"""

CONTEXT_RECALL_SYSTEM = """你是一个严格的评审专家。你的任务是判断"检索到的上下文"是否覆盖了"参考答案"中的所有关键信息点。

评分标准（0~1 之间）：
- 1.0: 参考答案里的每一个关键信息点都能在上下文中找到对应原文
- 0.7~0.9: 绝大部分关键信息被覆盖，有 1 个次要信息点缺失
- 0.4~0.6: 覆盖了主要信息但缺失了 2 个以上关键点
- 0.1~0.3: 只有少量信息被覆盖，大部分关键信息缺失
- 0.0: 上下文完全不包含参考答案中的任何信息

输出格式（严格 JSON）：
{"score": <0~1 的浮点数>, "reason": "<一句话说明缺失了哪些关键信息，满分则说'全部覆盖'>"}"""


def _build_faithfulness_prompt(answer: str, context: str) -> str:
    return f"""## 检索到的上下文

{context}

## 生成答案

{answer}

请逐句比对生成答案与上下文，给 Faithfulness 打分。只输出 JSON。"""


def _build_answer_relevancy_prompt(answer: str, query: str) -> str:
    return f"""## 用户问题

{query}

## 生成答案

{answer}

请判断答案是否紧扣问题，给 Answer Relevancy 打分。只输出 JSON。"""


def _build_context_recall_prompt(context: str, golden_answer: str) -> str:
    return f"""## 检索到的上下文

{context}

## 参考答案（理想情况应该覆盖的信息）

{golden_answer}

请判断上下文是否覆盖了参考答案中的关键信息，给 Context Recall 打分。只输出 JSON。"""


# ============================================================
# 单条评测
# ============================================================


def _parse_score(raw: str) -> tuple[float, str]:
    """从 LLM 返回的 JSON 中提取 score 和 reason。解析失败返回 0 分。"""
    try:
        # 尝试提取 JSON 块
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
        return 0.0, f"解析失败: {raw[:100]}"


def _retrieve_and_generate(query: str) -> tuple[str, str]:
    """检索 + 生成：返回 (generated_answer, context_text)。"""
    _ensure_pipeline()

    bm25_results = _bm25.search(query)
    vector_results = _vector.search(query)
    output = _pipeline.run(bm25_results, vector_results, query, ce_top_k=CE_TOP_K)

    ce_results = output["cross_encoder"]
    if not ce_results:
        return "[无检索结果]", ""

    # 构建上下文（Top-3 Chunk 拼接）
    context_parts = []
    for i, r in enumerate(ce_results[:3]):
        src = r.chunk.metadata.get("source", "unknown")
        context_parts.append(f"[来源{i+1}: {src}] {r.chunk.content}")
    context = "\n\n".join(context_parts)

    # 用 RAG prompt 生成答案
    system = "你是一个知识检索助手。请严格基于提供的文档内容回答用户问题。如果文档内容不足以回答，请如实说明。不要编造文档中没有的信息。控制在 300 字以内。"

    prompt = f"""## 检索到的文档内容

{context}

## 用户问题

{query}

请基于上述文档内容回答用户问题。"""

    answer = _call_ollama(prompt, system=system)

    if not answer:
        # fallback: 返回检索结果拼接
        answer = "\n\n".join(
            f"[来源: {r.chunk.metadata.get('source', '?')}] {r.chunk.content[:300]}"
            for r in ce_results[:2]
        )

    return answer, context


def _save_checkpoint(details: list[dict], output_path: Path):
    """逐条增量写入，以防中途挂掉丢失已有结果。"""
    checkpoint = {"summary": {"model": LLM_MODEL, "total_cases": len(details), "note": "进行中…"}, "details": details}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, ensure_ascii=False, indent=2)


def evaluate_one(tc: dict, verbose: bool = True) -> dict:
    """对单条 test case 跑完整评测链路：检索 → 生成 → 三维 LLM 打分。

    Returns:
        {"id": ..., "query": ..., "answer": ..., "faithfulness": {...},
         "answer_relevancy": {...}, "context_recall": {...}}
    """
    query = tc["query"]
    golden_answer = tc.get("golden_answer", "")

    # Step 1: 检索 + 生成
    t0 = time.perf_counter()
    answer, context = _retrieve_and_generate(query)
    gen_time_ms = (time.perf_counter() - t0) * 1000

    result = {
        "id": tc["id"],
        "query": query,
        "category": tc["category"],
        "answer": answer[:800],
        "context": context[:1000],
        "golden_answer": golden_answer,
        "gen_time_ms": round(gen_time_ms, 1),
    }

    # Step 2: Faithfulness (answer vs context)
    if context:
        raw = _call_ollama(
            _build_faithfulness_prompt(answer, context),
            system=FAITHFULNESS_SYSTEM,
        )
        score, reason = _parse_score(raw)
    else:
        score, reason = 0.0, "无检索上下文"
    result["faithfulness"] = {"score": score, "reason": reason}

    # Step 3: Answer Relevancy (answer vs query)
    raw = _call_ollama(
        _build_answer_relevancy_prompt(answer, query),
        system=ANSWER_RELEVANCY_SYSTEM,
    )
    score, reason = _parse_score(raw)
    result["answer_relevancy"] = {"score": score, "reason": reason}

    # Step 4: Context Recall (context vs golden_answer)
    if context and golden_answer:
        raw = _call_ollama(
            _build_context_recall_prompt(context, golden_answer),
            system=CONTEXT_RECALL_SYSTEM,
        )
        score, reason = _parse_score(raw)
    else:
        score, reason = 0.0, "缺少上下文或参考答案"
    result["context_recall"] = {"score": score, "reason": reason}

    if verbose:
        print(f"\n  [{tc['id']}] {query[:50]}")
        print(f"    Faithfulness:     {result['faithfulness']['score']:.2f}  {result['faithfulness']['reason'][:60]}")
        print(f"    Answer Relevancy: {result['answer_relevancy']['score']:.2f}  {result['answer_relevancy']['reason'][:60]}")
        print(f"    Context Recall:   {result['context_recall']['score']:.2f}  {result['context_recall']['reason'][:60]}")

    return result


# ============================================================
# 批量评测 + 汇总
# ============================================================


def run_llm_evaluation(test_cases: list[dict] | None = None, verbose: bool = True) -> dict:
    """对全部 test cases 跑 LLM 生成评测。

    Returns:
        {"summary": {per_metric_avg, per_category}, "details": [...per test case]}
    """
    if test_cases is None:
        with open(TEST_QUERIES_FILE, "r", encoding="utf-8") as f:
            test_cases = json.load(f)["test_cases"]

    print("=" * 60)
    print("  LLM-as-Judge 生成评测")
    print(f"  模型: {LLM_MODEL}  |  Test Cases: {len(test_cases)}")
    print("=" * 60)

    details: list[dict] = []
    metrics_accum: dict[str, list[float]] = {
        "faithfulness": [],
        "answer_relevancy": [],
        "context_recall": [],
    }
    category_accum: dict[str, dict[str, list[float]]] = {}
    output_path = EXPERIMENTS_DIR / "llm_evaluation_results.json"

    for tc in test_cases:
        result = evaluate_one(tc, verbose=verbose)
        details.append(result)

        cat = result["category"]
        if cat not in category_accum:
            category_accum[cat] = {"faithfulness": [], "answer_relevancy": [], "context_recall": []}

        for metric in metrics_accum:
            s = result[metric]["score"]
            metrics_accum[metric].append(s)
            category_accum[cat][metric].append(s)

        # 逐条即时写入，防止中途挂掉丢数据
        _save_checkpoint(details, output_path)

    # 汇总
    n = len(test_cases)
    summary: dict = {
        "model": LLM_MODEL,
        "total_cases": n,
        "overall": {},
        "by_category": {},
    }

    for metric in metrics_accum:
        vals = metrics_accum[metric]
        summary["overall"][metric] = {
            "mean": round(sum(vals) / len(vals), 4) if vals else 0.0,
            "min": round(min(vals), 4) if vals else 0.0,
            "max": round(max(vals), 4) if vals else 0.0,
        }

    for cat in sorted(category_accum):
        summary["by_category"][cat] = {}
        for metric in metrics_accum:
            vals = category_accum[cat][metric]
            summary["by_category"][cat][metric] = (
                round(sum(vals) / len(vals), 4) if vals else 0.0
            )

    # 打印汇总表
    print(f"\n{'='*60}")
    print("  汇总")
    print(f"{'='*60}")
    print(f"  {'指标':<22s} {'Mean':>8s}  {'Min':>8s}  {'Max':>8s}")
    print(f"  {'-'*50}")
    metric_labels = {
        "faithfulness": "Faithfulness",
        "answer_relevancy": "Answer Relevancy",
        "context_recall": "Context Recall",
    }
    for metric in metrics_accum:
        s = summary["overall"][metric]
        print(f"  {metric_labels[metric]:<22s} {s['mean']:>8.4f}  {s['min']:>8.4f}  {s['max']:>8.4f}")

    print(f"\n  分类别:")
    print(f"  {'类别':<18s} {'Faith':>7s}  {'Relev':>7s}  {'Recall':>7s}")
    print(f"  {'-'*46}")
    for cat in sorted(category_accum):
        c = summary["by_category"][cat]
        print(f"  {cat:<18s} {c['faithfulness']:>7.4f}  {c['answer_relevancy']:>7.4f}  {c['context_recall']:>7.4f}")

    # 保存（最终版含 summary）
    output = {"summary": summary, "details": details}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n  → 已保存: {output_path}")

    return output


# ============================================================
# 快速验证
# ============================================================


def run_quick_check(n: int = 3) -> None:
    """快速验证 Ollama 连接 + prompt 是否正常。只跑前 n 条。"""
    with open(TEST_QUERIES_FILE, "r", encoding="utf-8") as f:
        test_cases = json.load(f)["test_cases"][:n]

    print("=" * 60)
    print(f"  快速验证模式（前 {n} 条）")
    print("=" * 60)

    # 先检查 Ollama 是否可用
    test_resp = _call_ollama("回复'OK'", system="只回复OK两个字。")
    if not test_resp:
        print("\n  [ERROR] Ollama 未连接！请检查:")
        print("    1. ollama serve 是否在运行")
        print("    2. ollama pull qwen2.5:7b 是否完成")
        print("    3. set OLLAMA_MODELS=D:\\ollama_models")
        return

    print(f"  Ollama 连接正常 → 响应: {test_resp.strip()[:50]}")
    run_llm_evaluation(test_cases, verbose=True)


# ============================================================
# CLI
# ============================================================


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)

    if "--model" in sys.argv:
        idx = sys.argv.index("--model")
        if idx + 1 < len(sys.argv):
            LLM_MODEL = sys.argv[idx + 1]
            print(f"[INFO] 使用模型: {LLM_MODEL}")

    if "--sample" in sys.argv:
        idx = sys.argv.index("--sample")
        n = int(sys.argv[idx + 1]) if idx + 1 < len(sys.argv) else 3
        run_quick_check(n)
    elif "--quick" in sys.argv:
        run_quick_check(3)
    else:
        run_llm_evaluation()
