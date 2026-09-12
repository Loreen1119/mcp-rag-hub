"""冒烟测试：验证 _call_llm 在纯 CPU + qwen2.5:3b 下能否正常工作。

检查三件事：
1. 模型能否在本机内存下加载并生成（此前 7b 必 OOM）
2. timeout 参数是否被真正应用（旧代码根本没传）
3. format="json" 能否让 3b 稳定输出合法 JSON
"""
from __future__ import annotations

import json
import sys
import time

sys.path.insert(0, r"D:\1base\computer\Agent\DevRoot\mcp-rag-hub")

import agent  # noqa: E402
from config import LLM_MODEL, LLM_TIMEOUT_SECONDS  # noqa: E402

print(f"model={LLM_MODEL} timeout={LLM_TIMEOUT_SECONDS}s")

# --- 用例 1：普通文本生成（rewrite_query 走的路径）---
t0 = time.time()
content, err = agent._call_llm("用一句话说明什么是 RRF 融合。")
dt = time.time() - t0
print(f"\n[1] plain  err={err!r}  {dt:.1f}s")
print(f"    content={content[:200]!r}")

# --- 用例 2：JSON 模式（generate_answer 走的路径）---
context = (
    "[1] 来源: rag.md\n标题: 混合检索\n内容: RRF（Reciprocal Rank Fusion）把多路召回的排名取倒数后加权求和，"
    "公式为 score = Σ 1/(k + rank_i)，k 通常取 60，可避免单路召回的偏置。\n\n"
    "[2] 来源: search.md\n标题: 重排序\n内容: Cross-Encoder 对候选做精排，但成本高于双塔模型，一般只用于 Top-K 重排。"
)
prompt = f"""## 检索到的文档内容

{context}

## 用户问题

RRF 如何工作？

## 要求

请输出 JSON：{{"status":"answered|insufficient_evidence", "answer":"...", "used_source_ids":[1,2], "reason":"..."}}。
仅当证据足够时使用 answered，并在 used_source_ids 中填写实际引用的编号；否则使用 insufficient_evidence。控制在 300 字以内。"""

t0 = time.time()
raw, err = agent._call_llm(
    prompt,
    system="你是一个知识检索助手。只基于证据回答，不要编造。必须只输出 JSON，不要 Markdown。",
    json_mode=True,
)
dt = time.time() - t0
print(f"\n[2] json   err={err!r}  {dt:.1f}s")
print(f"    raw={raw[:400]!r}")

if raw:
    try:
        parsed = json.loads(raw)
        print(f"    PARSED OK: status={parsed.get('status')} ids={parsed.get('used_source_ids')}")
        print(f"    answer={str(parsed.get('answer'))[:200]}")
    except json.JSONDecodeError as e:
        print(f"    JSON PARSE FAILED: {e}")
