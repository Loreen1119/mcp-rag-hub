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

import logging
import sys
from typing import Annotated, TypedDict

import operator

from langgraph.graph import StateGraph, END

from config import CE_TOP_K, CE_THRESHOLD
from src.data_pipeline import process_directory
from src.retrievers import BM25Retriever, VectorRetriever
from src.fusion import FusionPipeline

logger = logging.getLogger(__name__)

# ============================================================
# Agent 状态定义
# ============================================================


class AgentState(TypedDict):
    """LangGraph Agent 的全局状态。

    每个节点接收 state dict，返回部分更新的 dict（add 操作会累积）。
    """

    query: str
    """用户原始查询"""

    retrieved_chunks: Annotated[list[dict], operator.add]
    """累积的检索结果（add 保证多次检索结果合并而非覆盖）"""

    rewritten_queries: Annotated[list[str], operator.add]
    """已尝试的改写查询列表"""

    attempt: int
    """当前检索尝试次数"""

    answer: str
    """最终生成的答案"""

    search_log: Annotated[list[str], operator.add]
    """检索过程日志（可观测性）"""


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
    logger.info("RAG 管线就绪 — %d 个 Chunk 已索引", len(_chunks))


# ============================================================
# LLM 调用
# ============================================================


def _call_llm(prompt: str, system: str = "") -> str:
    """调用本地 Ollama 模型生成文本。

    Ollama 不可用时返回空字符串，由调用方做 fallback 处理。
    """
    try:
        import ollama

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = ollama.chat(model="qwen2.5:7b", messages=messages)
        return response["message"]["content"]
    except Exception:
        return ""


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
        "attempt": 0,
        "rewritten_queries": [],
        "retrieved_chunks": [],
        "search_log": [log_msg],
        "answer": "",
    }


# ============================================================
# 节点 2: 检索
# ============================================================


def retrieve(state: AgentState) -> dict:
    """执行 RAG 全管线检索。"""
    _ensure_pipeline()

    query = state["query"]
    attempt = state["attempt"] + 1
    log_msg = f"[retrieve #{attempt}] 查询: '{query[:60]}'"
    logger.info(log_msg)

    bm25_results = _bm25.search(query)
    vector_results = _vector.search(query)
    output = _pipeline.run(bm25_results, vector_results, query, ce_top_k=CE_TOP_K)

    chunks = [
        {
            "rank": i + 1,
            "content": r.chunk.content[:500],
            "score": round(r.score, 4),
            "source_doc": r.chunk.metadata.get("source", "unknown"),
            "headings": r.chunk.metadata.get("heading_breadcrumb", ""),
            "chunk_id": r.chunk.chunk_id,
        }
        for i, r in enumerate(output["cross_encoder"])
    ]

    best_score = chunks[0]["score"] if chunks else 0.0
    log_msg += f" | Top-K={len(chunks)} best_score={best_score:.4f}"

    return {
        "query": query,
        "attempt": attempt,
        "retrieved_chunks": chunks,
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
    chunks = state.get("retrieved_chunks", [])
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
    chunks = state.get("retrieved_chunks", [])
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

    rewritten = _call_llm(prompt)

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
    LLM 不可用时返回检索 Top-3 的原文拼接。
    """
    query = state["query"]
    chunks = state.get("retrieved_chunks", [])
    attempt = state["attempt"]

    if not chunks:
        return {
            "answer": "未找到相关文档，请尝试更换查询表述。",
            "search_log": ["[generate] 无检索结果，终止"],
        }

    # 构建上下文
    context_parts = []
    for i, ch in enumerate(chunks[:3]):
        context_parts.append(
            f"[文档 {i+1}] 来源: {ch['source_doc']}\n"
            f"标题: {ch['headings']}\n"
            f"内容: {ch['content']}"
        )
    context = "\n\n".join(context_parts)

    system = "你是一个知识检索助手。请基于提供的文档内容回答用户的问题。如果文档内容不足以回答问题，请如实说明。不要编造文档中没有的信息。"

    prompt = f"""## 检索到的文档内容

{context}

## 用户问题

{query}

## 要求

请基于上述文档内容回答用户问题。引用文档中的具体段落支持你的回答。控制在 300 字以内。"""

    answer = _call_llm(prompt, system=system)

    # fallback: 返回检索片段拼接
    if not answer:
        answer = f"[本地 LLM 未连接] 基于检索结果 (共 {attempt} 次检索, Top-{len(chunks)} 结果):\n\n"
        for i, ch in enumerate(chunks[:3]):
            answer += (
                f"--- 来源 {i+1}: {ch['source_doc']} | "
                f"CE Score: {ch['score']:.4f} ---\n"
                f"{ch['content'][:300]}\n\n"
            )

    log_msg = f"[generate] attempt={attempt} chunks_used={min(3, len(chunks))} answer_len={len(answer)}"
    logger.info(log_msg)

    return {
        "answer": answer,
        "search_log": [log_msg],
    }


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
        "retrieved_chunks": [],
        "rewritten_queries": [],
        "attempt": 0,
        "answer": "",
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
        print(f"  检索到 Chunk 数: {len(result.get('retrieved_chunks', []))}")
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
