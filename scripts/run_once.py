"""跑单条 query 端到端，结果写 UTF-8 JSON（避开 PowerShell stdout 中文乱码）。

用法: python scripts/run_once.py "问题" [输出路径]
"""
import json
import logging
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(level=logging.WARNING)

from agent import run_query

query = sys.argv[1] if len(sys.argv) > 1 else "Faithfulness 忠实度指标"
out_path = sys.argv[2] if len(sys.argv) > 2 else str(Path(_PROJECT_ROOT) / "run_once.json")

result = run_query(query, verbose=False)

payload = {
    "query": result.get("query"),
    "attempt": result.get("attempt"),
    "rewritten_queries": result.get("rewritten_queries", []),
    "n_chunks": len(result.get("last_retrieved_chunks", [])),
    "chunks": [
        {"rank": c.get("rank"), "src": c.get("source_doc"), "score": c.get("score"), "cid": c.get("chunk_id")}
        for c in result.get("last_retrieved_chunks", [])
    ],
    "answer": result.get("answer"),
    "search_log": result.get("search_log", []),
}

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(payload, f, ensure_ascii=False, indent=2)

print("done ->", out_path)
