"""
LangGraph Agent — RAG 智能检索问答编排。

状态图流转:
    analyze_query → retrieve → check_results ──[结果不足]──→ rewrite_query → retrieve
                              │                                       │
                              └──[结果充足]──→ generate_answer ──────→ END

核心演示点:
1. TypedDict 定义 Agent 状态（强类型、可观测）
2. 条件边：根据检索质量决定是否需要改写查询
3. 最大重试次数防止无限循环
4. Ollama 本地 LLM 生成答案（不可用时用检索结果拼接做 fallback）

运行方式:
    python agent.py                          # 交互式命令行
    python agent.py --query "RAG 优化方向"    # 单次查询
    python agent.py --diagram                # 输出状态图 mermaid 源码
"""

from __future__ import annotations

import json
import logging
import sys
import urllib.error
import urllib.request
from typing import Annotated, TypedDict

import operator

from langgraph.graph import StateGraph, END

from config import (
    ANSWER_TOP_K,
    CE_TOP_K,
    CE_RELATIVE_THRESHOLD,
    CE_THRESHOLD,
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
    KG_RRF_WEIGHT,
    LLM_BACKEND,
    LLM_MAX_RETRIES,
    LLM_MODEL,
    LLM_TIMEOUT_SECONDS,
    MIN_EVIDENCE_COUNT,
)
from src.answer_validator import AnswerStatus, AnswerValidator, RefusalReason
from src.pipeline import get_pipeline

logger = logging.getLogger(__name__)

# ============================================================
# Agent 状态定义
# ============================================================


class AgentState(TypedDict):
    """LangGraph Agent 的全局状态。

    每个节点接收 state dict，返回部分更新的 dict（add 操作会累积）。
    """

    query: str
    """当前生效的查询（原始查询或改写后的查询）"""

    original_query: str
    """用户原始查询"""

    last_retrieved_chunks: list[dict]
    """最近一轮检索的结果（普通赋值，每轮覆盖）"""

    retrieval_history: Annotated[list[dict], operator.add]
    """按轮次保存的检索观测（可观测性，不参与答案生成）"""

    rewritten_queries: Annotated[list[str], operator.add]
    """已尝试的改写查询列表"""

    attempt: int
    """当前检索尝试次数"""

    answer: dict
    """结构化最终答案（状态、正文、引用和原因）。"""

    search_log: Annotated[list[str], operator.add]
    """检索过程日志（可观测性）"""


# ============================================================
# 管线初始化：复用 src.pipeline.get_pipeline 的全局共享单例
# ============================================================


# ============================================================
# LLM 调用
# ============================================================


def _classify_llm_error(exc: BaseException) -> str:
    """把 LLM 调用异常归类到 RefusalReason 的枚举值。

    用"类名 + 消息"做子串匹配，可同时覆盖 httpx 原始异常和被 ollama
    包装后的异常（ollama 只暴露 RequestError / ResponseError 两类）。
    """
    text = f"{type(exc).__name__} {exc}".lower()
    # 先判 connect：ConnectTimeout 说明连不上（服务没起），比"读超时"更该报连接错误
    if "connect" in text or "refused" in text:
        return RefusalReason.CONNECTION_ERROR.value
    if "timeout" in text or "timed out" in text:
        return RefusalReason.TIMEOUT.value
    return RefusalReason.SERVICE_ERROR.value


def _call_llm(prompt: str, system: str = "", json_mode: bool = False) -> tuple[str, str]:
    """调用本地 Ollama 模型生成文本，返回 (content, error_reason)。

    成功时 error_reason 为空串；失败时 content 为空串，error_reason 取
    RefusalReason 的 timeout / connection_error / service_error 之一，
    供 generate_answer 写进降级结果的 reason 字段（旧实现一律写
    connection_error，把"模型加载慢导致超时"误报成"服务没起"）。

    超时取自 config 的 LLM_TIMEOUT_SECONDS。注意旧代码根本没传 timeout，
    而这个值原本是 20 秒，对纯 CPU 推理（3b 约 15~25 秒）必然超时。

    仅对连接失败和读超时重试 LLM_MAX_RETRIES 次；服务端明确报错
    （如模型不存在、参数非法）不重试，避免无意义等待。

    json_mode=True 时用 Ollama 的 format="json" 约束输出必须是合法 JSON，
    小模型尤其需要，可显著降低 malformed_json 率。

    调用方需自行兜底：generate_answer 会回落到 _bm25_fallback，
    rewrite_query 会规则式改写。
    """
    # 后端开关：优先走 DeepSeek 云端推理（2G 服务器零内存压力）；
    # 本地已装 Ollama 且 LLM_BACKEND="ollama" 时走原本地路径。
    if LLM_BACKEND == "deepseek":
        return _call_deepseek(prompt, system, json_mode)

    try:
        import ollama
    except ImportError:
        logger.warning("[llm] 未安装 ollama 包，跳过生成")
        return "", RefusalReason.SERVICE_ERROR.value

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    kwargs: dict = {"model": LLM_MODEL, "messages": messages}
    if json_mode:
        kwargs["format"] = "json"

    client = ollama.Client(timeout=LLM_TIMEOUT_SECONDS)
    last_error = ""
    for attempt_no in range(1, LLM_MAX_RETRIES + 2):
        try:
            response = client.chat(**kwargs)
            content = (response.get("message") or {}).get("content") or ""
            return content, ""
        except Exception as exc:  # noqa: BLE001 - 需把任意异常归一成枚举原因
            last_error = _classify_llm_error(exc)
            logger.warning(
                "[llm] 第 %d/%d 次调用失败 (%s): %s",
                attempt_no, LLM_MAX_RETRIES + 1, last_error, exc,
            )
            if last_error not in (
                RefusalReason.CONNECTION_ERROR.value,
                RefusalReason.TIMEOUT.value,
            ):
                break  # 服务端明确报错，重试无意义

    return "", last_error


def _call_deepseek(prompt: str, system: str = "", json_mode: bool = False) -> tuple[str, str]:
    """调用 DeepSeek 云端 Chat API（OpenAI 兼容协议）生成文本，返回 (content, error_reason)。

    用于 2G 轻量服务器：避免本地加载 qwen2.5:3b（~2.5GB）挤爆内存。
    使用标准库 urllib 发请求，不依赖额外 SDK。
    """
    if not DEEPSEEK_API_KEY:
        logger.warning("[llm] 未配置 DEEPSEEK_API_KEY，跳过生成")
        return "", RefusalReason.SERVICE_ERROR.value

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload: dict = {"model": DEEPSEEK_MODEL, "messages": messages, "temperature": 0.0}
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    data = json.dumps(payload).encode("utf-8")
    url = DEEPSEEK_BASE_URL.rstrip("/") + "/chat/completions"
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=LLM_TIMEOUT_SECONDS) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        content = (body.get("choices") or [{}])[0].get("message", {}).get("content") or ""
        return content, ""
    except urllib.error.HTTPError as exc:
        logger.warning("[llm] DeepSeek HTTP 错误: %s", exc)
        return "", _classify_llm_error(exc)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[llm] DeepSeek 调用失败 (%s): %s", type(exc).__name__, exc)
        return "", _classify_llm_error(exc)


# ============================================================
# 节点 1: 分析查询
# ============================================================


def analyze_query(state: AgentState) -> dict:
    """分析用户查询，记录初始状态。"""
    query = state["query"].strip()
    log_msg = f"[analyze] 原始查询: '{query}'"
    logger.info(log_msg)

    return {
        "query": query,
        "original_query": query,
        "attempt": 0,
        "rewritten_queries": [],
        "last_retrieved_chunks": [],
        "retrieval_history": [],
        "search_log": [log_msg],
        "answer": {"status": "", "answer": "", "used_source_ids": [], "reason": None},
    }


# ============================================================
# 节点 2: 检索
# ============================================================


def retrieve(state: AgentState) -> dict:
    """执行 RAG 全管线检索。"""
    ctx = get_pipeline()

    query = state["query"]
    attempt = state["attempt"] + 1
    log_msg = f"[retrieve #{attempt}] 查询: '{query[:60]}'"
    logger.info(log_msg)

    bm25_results = ctx.bm25.search(query)
    vector_results = ctx.vector.search(query)
    graph_results = ctx.graph.search(query) if ctx.graph else None
    rrf_weights = [1.0, 1.0, KG_RRF_WEIGHT] if ctx.graph else None
    output = ctx.pipeline.run(
        bm25_results, vector_results, query,
        ce_top_k=CE_TOP_K,
        graph_results=graph_results,
        rrf_weights=rrf_weights,
    )

    # 相对阈值过滤：CE 分数低于 top1 一定比例的视为噪声丢弃。
    # 动机：CE_TOP_K 硬取前 N 条会把无关 chunk 一起塞进 context（实测 bge-reranker 下
    # 正解 0.699 vs 噪声 0.0007，差 1000 倍，但 top5 仍保留噪声）。保底保留 top1。
    ce_results = output["cross_encoder"]
    if ce_results:
        top1 = ce_results[0].score
        if top1 > 0:
            kept = [r for r in ce_results if r.score >= CE_RELATIVE_THRESHOLD * top1]
        else:
            kept = ce_results[:1]
        ce_results = kept or ce_results[:1]

    chunks = [
        {
            "rank": i + 1,
            "content": r.chunk.content[:500],
            "score": round(r.score, 4),
            "source_doc": r.chunk.metadata.get("source", "unknown"),
            "headings": r.chunk.metadata.get("heading_breadcrumb", ""),
            "chunk_id": r.chunk.chunk_id,
        }
        for i, r in enumerate(ce_results)
    ]

    best_score = chunks[0]["score"] if chunks else 0.0
    log_msg += f" | Top-K={len(chunks)} best_score={best_score:.4f}"

    return {
        "query": query,
        "attempt": attempt,
        # 普通赋值：覆盖当前轮结果，不累积
        "last_retrieved_chunks": chunks,
        # 按轮次追加观测记录（可观测性）
        "retrieval_history": [
            {
                "attempt": attempt,
                "query": query,
                "top_k": len(chunks),
                "best_score": round(best_score, 4),
                "source_docs": sorted({c["source_doc"] for c in chunks}),
            }
        ],
        "search_log": [log_msg],
    }


# ============================================================
# 节点 3: 检查检索质量
# ============================================================


def check_results(state: AgentState) -> dict:
    """判断检索结果是否足够回答问题。

    评判标准：Top-1 的 Cross-Encoder 分数 > 阈值 or 达到最大重试次数。
    CE 分数阈值设为 3.0（ms-marco-MiniLM 的经验值，3 以下通常不相关）。
    """
    chunks = state.get("last_retrieved_chunks", [])
    attempt = state["attempt"]
    max_attempts = 2
    ce_threshold = CE_THRESHOLD

    best_score = chunks[0]["score"] if chunks else 0.0
    quality = "good" if best_score >= ce_threshold else "insufficient"

    log_msg = (
        f"[check] attempt={attempt}/{max_attempts} "
        f"best_ce={best_score:.4f} quality={quality}"
    )
    logger.info(log_msg)

    return {
        "search_log": [log_msg],
    }


def _decide_next(state: AgentState) -> str:
    """条件路由：结果不足且未超最大次数 → 改写查询；否则 → 生成答案。"""
    chunks = state.get("last_retrieved_chunks", [])
    attempt = state["attempt"]
    max_attempts = 2
    ce_threshold = CE_THRESHOLD

    best_score = chunks[0]["score"] if chunks else 0.0

    if best_score < ce_threshold and attempt < max_attempts:
        return "rewrite_query"
    return "generate_answer"


# ============================================================
# 节点 4: 改写查询
# ============================================================


def rewrite_query(state: AgentState) -> dict:
    """用 LLM 改写用户查询，尝试更好的检索效果。

    改写策略：扩展缩写、补全专业术语、从口语化转书面化。
    LLM 不可用时使用规则式改写（追加同义表达）。
    """
    original = state["query"]
    attempt = state["attempt"]

    prompt = f"""你是一个查询改写助手。用户的原始查询检索效果不佳，请改写查询以获得更好的检索结果。

改写规则：
1. 保留原始意图，不要引入新概念
2. 将口语化表达转成书面化技术术语
3. 扩展缩写和专业简称
4. 如果原始查询是中文，改写后也必须是中文
5. 只输出改写后的查询文本，不要加任何解释

原始查询: {original}

改写查询:"""

    rewritten, _llm_error = _call_llm(prompt)

    # fallback: LLM 不可用时，规则式追加关键词
    if not rewritten:
        parts = [original]
        if "优化" in original:
            parts.append("性能提升 优化方向")
        if "检索" in original:
            parts.append("信息检索 search retrieval")
        rewritten = " ".join(parts)

    log_msg = f"[rewrite #{attempt}] '{original[:40]}' → '{rewritten[:60]}'"
    logger.info(log_msg)

    return {
        "query": rewritten,
        "rewritten_queries": [rewritten],
        "search_log": [log_msg],
    }


# ============================================================
# 节点 5: 生成答案
# ============================================================


def generate_answer(state: AgentState) -> dict:
    """基于检索结果生成最终答案。

    用 Ollama 做 RAG 生成（retrieval-augmented generation）。

    流程：
    1. 无检索结果 → insufficient_evidence（不是 LLM 问题，保留前置拦截）。
    2. 有结果 → 调用 LLM，由 LLM/validator 判断证据是否足够回答。
       拒答完全由 LLM/validator 决定，不用 CE 分数做任何门控（CE 分数仅用于
       retrieve 阶段的排序与前端展示）。
    3. LLM 返回空（服务未起/超时）→ 回落到 BM25 原文拼接，状态标
       generation_unavailable 但 answer 字段塞拼接文本，保证用户始终看到内容。
    """
    query = state["query"]
    chunks = state.get("last_retrieved_chunks", [])
    attempt = state["attempt"]

    if not chunks:
        return {
            "answer": {"status": AnswerStatus.INSUFFICIENT_EVIDENCE.value, "answer": "", "used_source_ids": [], "reason": RefusalReason.INSUFFICIENT_CONTEXT.value},
            "search_log": ["[generate] 无检索结果，终止"],
        }

    # 构建上下文
    context_parts = []
    for i, ch in enumerate(chunks[:ANSWER_TOP_K]):
        context_parts.append(
            f"[{i+1}] 来源: {ch['source_doc']}\n"
            f"标题: {ch['headings']}\n"
            f"内容: {ch['content']}"
        )
    context = "\n\n".join(context_parts)
    # 编号上限必须跟随实际证据条数：相对阈值过滤后 chunks 可能不足 ANSWER_TOP_K 条，
    # 若 prompt 仍写死 1..ANSWER_TOP_K，模型会引用不存在的编号 → 越界被 validator 拒。
    n_evidence = len(chunks[:ANSWER_TOP_K])

    system = "你是一个知识检索助手。只基于证据回答，不要编造。必须只输出 JSON，不要 Markdown。"

    prompt = f"""## 检索到的文档内容

{context}

## 用户问题

{query}

## 要求

请输出 JSON：{{"status":"answered|insufficient_evidence", "answer":"...", "used_source_ids":[1,2], "reason":"..."}}。
仅当证据足够时使用 answered，并在 used_source_ids 中填写实际引用的编号；否则使用 insufficient_evidence。控制在 300 字以内。
注意：used_source_ids 必须是数字数组（如 [1,2]），不要写成字符串（如 ["1","2"]）；本轮共有 {n_evidence} 条证据，编号只能用 1 到 {n_evidence}。"""

    raw_answer, llm_error = _call_llm(prompt, system=system, json_mode=True)
    if raw_answer:
        validated = AnswerValidator(len(chunks[:ANSWER_TOP_K]), MIN_EVIDENCE_COUNT).validate(raw_answer)
    else:
        # LLM 不可用：回落到 BM25 原文拼接，让用户至少看到检索到的证据内容。
        # reason 用真实错误类型（timeout/connection_error/service_error），
        # 便于区分"服务没起"和"模型加载超时"。
        validated = {
            "status": AnswerStatus.GENERATION_UNAVAILABLE.value,
            "answer": _bm25_fallback(chunks[:ANSWER_TOP_K]),
            "used_source_ids": [],
            "reason": llm_error or RefusalReason.SERVICE_ERROR.value,
        }

    log_msg = f"[generate] attempt={attempt} chunks_used={min(ANSWER_TOP_K, len(chunks))} status={validated['status']}"
    logger.info(log_msg)

    return {
        "answer": validated,
        "search_log": [log_msg],
    }


def _bm25_fallback(chunks: list[dict]) -> str:
    """LLM 不可用时，把检索 Top-K 的原文拼接成一段可读文本。

    与历史 docstring 承诺的"LLM 不可用时返回 Top-3 原文拼接"对齐：
    之前代码只有 generation_unavailable 没有拼接，这里补齐兜底。
    """
    if not chunks:
        return ""

    parts = []
    for i, ch in enumerate(chunks, start=1):
        source = ch.get("source_doc", "未知来源")
        headings = ch.get("headings", "")
        content = ch.get("content", "").strip()
        if not content:
            continue
        head = f"[{i}] {source}"
        if headings:
            head += f" · {headings}"
        parts.append(f"{head}\n{content}")

    body = "\n\n".join(parts)
    return (
        "（回答生成服务暂不可用，以下为检索到的原始证据，供参考）\n\n" + body
    )


# ============================================================
# 构建状态图
# ============================================================


def build_graph() -> StateGraph:
    """构建 LangGraph 状态图并返回编译后的 app。

    图结构:
        START → analyze_query → retrieve → check_results
                                                    ├── [good] → generate_answer → END
                                                    └── [insufficient] → rewrite_query → retrieve
    """
    graph = StateGraph(AgentState)

    # 注册节点
    graph.add_node("analyze_query", analyze_query)
    graph.add_node("retrieve", retrieve)
    graph.add_node("check_results", check_results)
    graph.add_node("rewrite_query", rewrite_query)
    graph.add_node("generate_answer", generate_answer)

    # 设置入口
    graph.set_entry_point("analyze_query")

    # 普通边
    graph.add_edge("analyze_query", "retrieve")
    graph.add_edge("retrieve", "check_results")
    graph.add_edge("rewrite_query", "retrieve")
    graph.add_edge("generate_answer", END)

    # 条件边：根据检索质量决定下一步
    graph.add_conditional_edges(
        "check_results",
        _decide_next,
        {
            "rewrite_query": "rewrite_query",
            "generate_answer": "generate_answer",
        },
    )

    return graph.compile()


# ============================================================
# Mermaid 图输出
# ============================================================


def print_mermaid() -> None:
    """打印状态图的 Mermaid 源码，可粘贴到 mermaid.live 查看。"""
    diagram = """```mermaid
stateDiagram-v2
    [*] --> analyze_query
    analyze_query --> retrieve
    retrieve --> check_results
    check_results --> generate_answer: 结果充足
    check_results --> rewrite_query: 结果不足 & attempt < 2
    rewrite_query --> retrieve
    generate_answer --> [*]
```"""
    print(diagram)
    print()
    print("粘贴到 https://mermaid.live 查看状态图")


# ============================================================
# CLI 入口
# ============================================================


def run_query(query: str, verbose: bool = False) -> dict:
    """执行单次查询，返回完整 Agent 状态。"""
    app = build_graph()

    initial_state: AgentState = {
        "query": query,
        "original_query": query,
        "last_retrieved_chunks": [],
        "retrieval_history": [],
        "rewritten_queries": [],
        "attempt": 0,
        "answer": {"status": "", "answer": "", "used_source_ids": [], "reason": None},
        "search_log": [],
    }

    result = app.invoke(initial_state)

    if verbose:
        print("\n".join(result.get("search_log", [])))

    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)

    if "--diagram" in sys.argv:
        print_mermaid()
        raise SystemExit(0)

    # 单次查询模式
    if "--query" in sys.argv:
        idx = sys.argv.index("--query")
        query = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else ""
        if not query:
            print("用法: python agent.py --query '你的问题'")
            raise SystemExit(1)

        result = run_query(query, verbose=True)

        print(f"\n{'='*60}")
        print(f"  查询: {result['query'][:80]}")
        print(f"  检索次数: {result['attempt']}")
        print(f"  检索到 Chunk 数: {len(result.get('last_retrieved_chunks', []))}")
        print(f"  改写历史: {result.get('rewritten_queries', [])}")
        print(f"{'='*60}")
        print(f"\n{result['answer']}")
        raise SystemExit(0)

    # 交互模式
    print("=" * 60)
    print("  RAG Agent — LangGraph 编排")
    print("  输入查询开始对话，输入 /quit 退出，输入 /diagram 查看状态图")
    print("=" * 60)

    while True:
        try:
            query = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n退出。")
            break

        if not query:
            continue
        if query == "/quit":
            break
        if query == "/diagram":
            print_mermaid()
            continue

        result = run_query(query, verbose=True)
        print(f"\n{'='*60}")
        print(f"  检索次数: {result['attempt']}")
        print(f"  改写历史: {result.get('rewritten_queries', [])}")
        print(f"{'='*60}")
        print(f"\n{result['answer']}")
