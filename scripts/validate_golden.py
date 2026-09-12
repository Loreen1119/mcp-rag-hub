"""校验验收测试集：JSON 合法 + golden 可考证。

覆盖三套数据：
  1. data/test_queries.json          —— 可答 golden set
     检查 golden_chunk_sources 源文件存在 + golden_answer 里反引号术语能在源文件找到
  2. data/multi_evidence_queries.json —— 多证据集
     检查 golden_chunk_sources 存在 + 每个 required_source 的关键词至少命中一个
  3. data/unanswerable_queries.json   —— 拒答集
     检查字段齐全 + expected_status 必须是 insufficient_evidence
     并列出查询中「字面出现在语料里」的术语（near_miss 设计上允许命中，仅作提示）

用法：python scripts/validate_golden.py [输出文件]
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import CORPUS, DOCS_DIR, PROJECT_ROOT, TEST_QUERIES_FILE  # noqa: E402

REQUIRED_GOLDEN = ("id", "query", "category", "golden_chunk_sources", "golden_answer")
MULTI_FILE = PROJECT_ROOT / "data" / "multi_evidence_queries.json"
UNANS_FILE = PROJECT_ROOT / "data" / "unanswerable_queries.json"


def main() -> None:
    buf: list[str] = []

    def emit(line: str = "") -> None:
        buf.append(line)

    cache: dict[str, str] = {}
    problems = 0

    def read_src(src: str) -> str | None:
        p = DOCS_DIR / src
        if not p.exists():
            return None
        if src not in cache:
            cache[src] = p.read_text(encoding="utf-8")
        return cache[src]

    emit(f"corpus = {CORPUS}")
    emit(f"docs   = {DOCS_DIR}")

    # ---------------- 1. golden ----------------
    emit()
    emit("=" * 60)
    emit("[1/3] golden set —— data/test_queries.json")
    emit("=" * 60)
    data = json.loads(TEST_QUERIES_FILE.read_text(encoding="utf-8"))
    cases = data["test_cases"]
    emit(f"cases = {len(cases)}")
    ids = [c.get("id") for c in cases]
    if len(set(ids)) != len(ids):
        emit("!! id 有重复")
        problems += 1
    cats: dict[str, int] = {}
    for c in cases:
        cid = c.get("id", "?")
        missing = [k for k in REQUIRED_GOLDEN if not c.get(k)]
        if missing:
            emit(f"[{cid}] !! 缺字段: {missing}")
            problems += 1
        cats[c["category"]] = cats.get(c["category"], 0) + 1
        texts = []
        for src in c["golden_chunk_sources"]:
            t = read_src(src)
            if t is None:
                emit(f"[{cid}] !! 源文件不存在: {src}")
                problems += 1
            else:
                texts.append(t)
        blob = "\n".join(texts)
        terms = re.findall(r"`([^`]+)`", c["golden_answer"])
        absent = [t for t in terms if t not in blob]
        if absent:
            emit(f"[{cid}] !! 这些术语在源文件里找不到: {absent}")
            problems += 1
        else:
            emit(f"[{cid}] OK  sources={c['golden_chunk_sources']}  terms={len(terms)}")
    emit(f"category 分布 = {cats}")

    # ---------------- 2. multi evidence ----------------
    emit()
    emit("=" * 60)
    emit("[2/3] multi-evidence set —— data/multi_evidence_queries.json")
    emit("=" * 60)
    me = json.loads(MULTI_FILE.read_text(encoding="utf-8"))
    emit(f"cases = {len(me)}")
    for c in me:
        qid = c.get("query_id", "?")
        miss = [k for k in ("query_id", "query", "expected_status", "required_sources") if not c.get(k)]
        if miss:
            emit(f"[{qid}] !! 缺字段: {miss}")
            problems += 1
            continue
        srcs = c.get("golden_chunk_sources", [])
        texts = []
        for s in srcs:
            t = read_src(s)
            if t is None:
                emit(f"[{qid}] !! 源文件不存在: {s}")
                problems += 1
            else:
                texts.append(t)
        blob = "\n".join(texts)
        for i, rs in enumerate(c["required_sources"], 1):
            kws = rs.get("keywords", [])
            hit = [k for k in kws if k in blob]
            tag = "OK " if hit else "!! "
            if not hit:
                problems += 1
            emit(
                f"[{qid}] {tag}req#{i} {rs.get('description', '')[:34]} "
                f"hit={len(hit)}/{len(kws)}{'' if hit else '  kws=' + str(kws)}"
            )
        emit(f"[{qid}]     sources={srcs}")

    # ---------------- 3. unanswerable ----------------
    emit()
    emit("=" * 60)
    emit("[3/3] unanswerable set —— data/unanswerable_queries.json")
    emit("=" * 60)
    un = json.loads(UNANS_FILE.read_text(encoding="utf-8"))
    emit(f"cases = {len(un)}")
    ucats: dict[str, int] = {}
    corpus_blob = "\n".join(
        p.read_text(encoding="utf-8") for p in DOCS_DIR.rglob("*.md")
    )
    for c in un:
        qid = c.get("query_id", "?")
        miss = [k for k in ("query_id", "query", "category", "expected_status") if not c.get(k)]
        if miss:
            emit(f"[{qid}] !! 缺字段: {miss}")
            problems += 1
            continue
        ucats[c["category"]] = ucats.get(c["category"], 0) + 1
        if c["expected_status"] != "insufficient_evidence":
            emit(f"[{qid}] !! expected_status 应为 insufficient_evidence，实际 {c['expected_status']}")
            problems += 1
            continue
        # 提示：查询里出现、且确实在语料中存在的技术术语（near_miss 允许，仅提示）
        toks = set(re.findall(r"[A-Za-z][A-Za-z0-9_.\-]{2,}", c["query"]))
        overlap = sorted(t for t in toks if t in corpus_blob)
        note = f"  语料含这些字面词: {overlap}" if overlap else "  语料无字面词命中"
        emit(f"[{qid}] OK  {c['category']:<15}{note}")
    emit(f"category 分布 = {ucats}")

    emit()
    emit("=" * 60)
    emit(f"问题数 = {problems}")
    text = "\n".join(buf)
    if len(sys.argv) > 1:
        Path(sys.argv[1]).write_text(text, encoding="utf-8")
        print("written:", sys.argv[1])
    else:
        print(text)


if __name__ == "__main__":
    main()
