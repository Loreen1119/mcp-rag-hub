#!/usr/bin/env python
"""docs-cn 双版本差异页识别（release-8.4 → release-8.5）。

做什么：
    比对两个分支同名 .md 文件，先滤掉「整行纯版本号」与「整行纯 URL」这两类
    噪声行，再按行级 diff 统计改动量；改动 ≥10 行 或 ≥5% 的页面判为「版本差异页」。

输出：
    _meta/docs-cn-version-diff.jsonl   逐文件指标（含未达阈值的，便于复核）
    _meta/docs-cn-version-diff.source  汇总口径与计数
    docs-cn-v8.4/                      差异页的 8.4 版正文（可选，--copy-changed-old）

说明：
    阈值分母取 两侧行数的较大值（max(len(old), len(new))），避免「新版大幅删减」
    被算成小比例。changed = 新增行 + 删除行（git --numstat 口径）。

用法：
    python scripts/tidb_version_diff.py \
        --old "C:/Users/REDMI/AppData/Local/Temp/mcp_rag_tidb_versions/docs-8.4" \
        --new "C:/Users/REDMI/AppData/Local/Temp/mcp_rag_tidb_versions/docs-8.5"
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORPUS_DIR = PROJECT_ROOT / "corpora" / "tidb"
META_DIR = CORPUS_DIR / "_meta"

# 与 fetch_tidb_corpus.py 保持一致的正文过滤口径
EXCLUDE_DIRS = (".agents/", ".github/", ".vaunt/")
EXCLUDE_NAMES = {"CONTRIBUTING.md", "README.md"}

# 整行就是版本号（8.5 / v8.5.1 / 8.5.0-beta）
RE_VERSION_ONLY = re.compile(r"^\s*[vV]?\d+(\.\d+){1,3}([-_.][0-9A-Za-z]+)*\s*$")
# 整行就是 URL（可带 <> () [] 包裹）
RE_URL_ONLY = re.compile(r"^\s*[<(\[]?\s*(https?://|ftp://)\S+\s*[>)\]]?\s*$")
# 行内 URL（用于「URL 归一化」的参考指标）
RE_URL_INLINE = re.compile(r"(https?://|ftp://)\S+")


def included(rel: str) -> bool:
    if not rel.lower().endswith(".md"):
        return False
    if any(rel.startswith(d) for d in EXCLUDE_DIRS):
        return False
    base = rel.rsplit("/", 1)[-1]
    if base in EXCLUDE_NAMES or base.startswith("_") or base.startswith("TOC"):
        return False
    if any(part.startswith(".") for part in Path(rel).parts):
        return False
    return True


def md_map(root: Path) -> dict[str, Path]:
    return {
        p.relative_to(root).as_posix(): p
        for p in sorted(root.rglob("*"))
        if p.is_file() and included(p.relative_to(root).as_posix())
    }


def filtered_lines(text: str, *, normalize_url: bool = False) -> list[str]:
    out: list[str] = []
    for ln in text.replace("\r\n", "\n").split("\n"):
        if RE_VERSION_ONLY.match(ln) or RE_URL_ONLY.match(ln):
            continue
        if normalize_url:
            ln = RE_URL_INLINE.sub("«URL»", ln)
        out.append(ln.rstrip())
    while out and not out[-1].strip():
        out.pop()
    return out


def line_delta(old: list[str], new: list[str]) -> tuple[int, int]:
    """返回 (新增行, 删除行)。"""
    sm = difflib.SequenceMatcher(a=old, b=new, autojunk=False)
    added = deleted = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        if tag == "replace":
            deleted += i2 - i1
            added += j2 - j1
        elif tag == "delete":
            deleted += i2 - i1
        elif tag == "insert":
            added += j2 - j1
    return added, deleted


def read_text(p: Path) -> str:
    for enc in ("utf-8", "utf-8-sig", "gb18030", "latin-1"):
        try:
            return p.read_text(encoding=enc)
        except (UnicodeDecodeError, UnicodeError):
            continue
    return ""


def main() -> int:
    ap = argparse.ArgumentParser(description="docs-cn 双版本差异页识别")
    ap.add_argument("--old", required=True, help="旧版本分支目录（release-8.4）")
    ap.add_argument("--new", required=True, help="新版本分支目录（release-8.5）")
    ap.add_argument("--min-lines", type=int, default=10)
    ap.add_argument("--min-pct", type=float, default=5.0)
    ap.add_argument("--copy-changed-old", default="", help="把差异页的旧版正文拷到这里")
    args = ap.parse_args()

    old_root = Path(args.old)
    new_root = Path(args.new)
    for r in (old_root, new_root):
        if not r.is_dir():
            raise SystemExit(f"目录不存在：{r}")

    META_DIR.mkdir(parents=True, exist_ok=True)

    old_files = md_map(old_root)
    new_files = md_map(new_root)
    common = sorted(set(old_files) & set(new_files))
    only_old = sorted(set(old_files) - set(new_files))
    only_new = sorted(set(new_files) - set(old_files))

    rows: list[dict] = []
    kept: list[str] = []
    for rel in common:
        o = filtered_lines(read_text(old_files[rel]))
        n = filtered_lines(read_text(new_files[rel]))
        added, deleted = line_delta(o, n)
        changed = added + deleted
        denom = max(len(o), len(n), 1)
        pct = changed / denom * 100

        o2 = filtered_lines(read_text(old_files[rel]), normalize_url=True)
        n2 = filtered_lines(read_text(new_files[rel]), normalize_url=True)
        a2, d2 = line_delta(o2, n2)
        changed_urlnorm = a2 + d2

        hit = changed >= args.min_lines or pct >= args.min_pct
        if hit:
            kept.append(rel)
        rows.append(
            {
                "path": rel,
                "old_lines": len(o),
                "new_lines": len(n),
                "added": added,
                "deleted": deleted,
                "changed": changed,
                "pct": round(pct, 2),
                "changed_url_normalized": changed_urlnorm,
                "included": hit,
            }
        )

    (META_DIR / "docs-cn-version-diff.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8"
    )

    rows_kept = [r for r in rows if r["included"]]
    rows_ident = [r for r in rows if r["changed"] == 0]
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "old_branch": "release-8.4",
        "new_branch": "release-8.5",
        "threshold": {"min_changed_lines": args.min_lines, "min_pct": args.min_pct},
        "denominator": "max(old_lines, new_lines)",
        "changed_definition": "added + deleted（git --numstat 口径）",
        "noise_filter": "整行纯版本号 / 整行纯 URL 行已剔除",
        "md_files": {"old": len(old_files), "new": len(new_files)},
        "same_name": len(common),
        "only_in_old": len(only_old),
        "only_in_new": len(only_new),
        "identical": len(rows_ident),
        "changed_pages": len(rows) - len(rows_ident),
        "version_diff_pages": len(rows_kept),
        "version_diff_pages_url_normalized": sum(
            1
            for r in rows
            if r["changed_url_normalized"] >= args.min_lines
            or (r["changed_url_normalized"] / max(r["old_lines"], r["new_lines"], 1) * 100) >= args.min_pct
        ),
        "top_changed": [
            f"{r['path']}  changed={r['changed']} ({r['pct']}%)"
            for r in sorted(rows, key=lambda r: -r["changed"])[:15]
        ],
    }

    if args.copy_changed_old:
        dest_root = Path(args.copy_changed_old)
        for rel in kept:
            dest = dest_root / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(old_files[rel].read_bytes())
        summary["copied_old_versions_to"] = str(dest_root)
        summary["copied_old_versions_count"] = len(kept)

    (META_DIR / "docs-cn-version-diff.source").write_text(
        "\n".join(f"{k:<34}: {v}" for k, v in summary.items() if k != "top_changed")
        + "\n改动量最大的页面（前 15）:\n"
        + "\n".join(f"  - {x}" for x in summary["top_changed"])
        + "\n",
        encoding="utf-8",
    )
    (META_DIR / "docs-cn-version-diff-pages.list").write_text(
        "\n".join(kept) + "\n", encoding="utf-8"
    )

    print(json.dumps({k: v for k, v in summary.items() if k != "top_changed"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
