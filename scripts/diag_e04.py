"""诊断脚本：拆解单条 query 的检索链路，定位噪声来自哪一路。

用法: python -m scripts.diag_e04   （或 python scripts/diag_e04.py，需设 PYTHONPATH）
"""
import logging
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(level=logging.WARNING)

from src.data_pipeline import process_directory
from src.retrievers import BM25Retriever, VectorRetriever
from src.fusion import FusionPipeline
from config import (
    BM25_TOP_K,
    VECTOR_TOP_K,
    EMBEDDING_MODEL,
    CROSS_ENCODER_MODEL,
    CE_RELATIVE_THRESHOLD,
    ANSWER_TOP_K,
)

QUERY = "Faithfulness 忠实度指标"

print("=" * 72)
print(f"QUERY: {QUERY}")
print(f"EMBEDDING_MODEL={EMBEDDING_MODEL}  CROSS_ENCODER={CROSS_ENCODER_MODEL}")
print("=" * 72)

chunks = process_directory()
print(f"chunks total = {len(chunks)}")

bm25 = BM25Retriever(chunks)
vector = VectorRetriever(chunks)

b = bm25.search(QUERY, top_k=BM25_TOP_K)
v = vector.search(QUERY, top_k=VECTOR_TOP_K)

def show(title, results, key="score"):
    print(f"\n[{title}]  n={len(results)}")
    for i, r in enumerate(results[:12], 1):
        src = r.chunk.metadata.get("source", "?")
        head = r.chunk.metadata.get("heading_breadcrumb", "")
        cid = r.chunk.chunk_id
        txt = r.chunk.content[:55].replace("\n", " ")
        print(f"  {i:>2}. {r.score:>8.4f} | {src:<26} | {cid[:18]:<18} | {txt}")
        if head:
            print(f"      heading: {head[:70]}")

show("BM25 召回", b)
show("VECTOR 召回", v)

pipe = FusionPipeline()
out = pipe.run(b, v, QUERY)

print(f"\n[RRF 融合]  n={len(out['rrf'])}")
for i, r in enumerate(out["rrf"][:12], 1):
    src = r.chunk.metadata.get("source", "?")
    orig = r.chunk.metadata.get("original_source", "?")
    txt = r.chunk.content[:55].replace("\n", " ")
    print(f"  {i:>2}. rrf={r.score:.5f} orig={orig:<7} | {src:<26} | {txt}")

print(f"\n[CE 精排 top5]")
for i, r in enumerate(out["cross_encoder"][:5], 1):
    src = r.chunk.metadata.get("source", "?")
    txt = r.chunk.content[:55].replace("\n", " ")
    print(f"  {i:>2}. CE={r.score:>8.4f} | {src:<26} | {txt}")

# 模拟 agent.py retrieve() 里的相对阈值过滤
ce = out["cross_encoder"]
if ce:
    top1 = ce[0].score
    if top1 > 0:
        kept = [r for r in ce if r.score >= CE_RELATIVE_THRESHOLD * top1]
    else:
        kept = ce[:1]
    kept = kept or ce[:1]
    print(f"\n[相对阈值过滤后 → 实际进 LLM 的 context]  "
          f"thr={CE_RELATIVE_THRESHOLD}×top1={CE_RELATIVE_THRESHOLD * top1:.4f}  "
          f"保留 {len(kept)}/{len(ce)} 条，再取 top{ANSWER_TOP_K}")
    for i, r in enumerate(kept[:ANSWER_TOP_K], 1):
        src = r.chunk.metadata.get("source", "?")
        txt = r.chunk.content[:55].replace("\n", " ")
        print(f"  {i:>2}. CE={r.score:>8.4f} | {src:<26} | {txt}")

print("\n" + "=" * 72)
