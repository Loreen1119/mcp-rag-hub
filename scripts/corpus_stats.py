"""语料切片体检：chunk 数 / source 分布 / chunk_id 唯一性。

用法：
    python scripts/corpus_stats.py
    set MCP_RAG_CORPUS=vue-zh && python scripts/corpus_stats.py
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import CORPUS, DOCS_DIR  # noqa: E402
from src.data_pipeline import corpus_hash, iter_supported_docs, process_directory  # noqa: E402


def main() -> None:
    print(f"corpus      = {CORPUS}")
    print(f"docs_dir    = {DOCS_DIR}")
    files = iter_supported_docs(DOCS_DIR)
    print(f"files       = {len(files)}")

    chunks = process_directory(DOCS_DIR)
    print(f"chunks      = {len(chunks)}")
    print(f"corpus_hash = {corpus_hash(DOCS_DIR, 256, 38)[:16]}")

    ids = [c.chunk_id for c in chunks]
    dup_ids = [k for k, v in Counter(ids).items() if v > 1]
    print(f"unique ids  = {len(set(ids))}   duplicated = {len(dup_ids)}")
    if dup_ids:
        print("  !! 重复 chunk_id 示例:", dup_ids[:5])

    sources = Counter(c.metadata["source"] for c in chunks)
    print(f"unique sources = {len(sources)}")
    top_dir = Counter(s.split("/")[0] if "/" in s else "(root)" for s in sources)
    print("by top dir (doc count / chunk count):")
    for name, _ in top_dir.most_common():
        dc = sum(1 for s in sources if (s.split("/")[0] if "/" in s else "(root)") == name)
        cc = sum(v for s, v in sources.items() if (s.split("/")[0] if "/" in s else "(root)") == name)
        print(f"  {name:<12} docs={dc:<4} chunks={cc}")

    print("\nsample chunks:")
    for c in chunks[:3]:
        m = c.metadata
        print(
            f"  id={c.chunk_id}  source={m['source']}  idx={m['chunk_index']}  "
            f"tokens={m.get('token_count')}  headings={m.get('heading_breadcrumb', '')[:50]}"
        )
        print(f"    {c.content[:80].replace(chr(10), ' ')}")


if __name__ == "__main__":
    main()
