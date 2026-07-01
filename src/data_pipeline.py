"""
文档加载与切片管线。

支持 .txt / .md / .pdf 格式。
使用 tiktoken 做 token 级滑窗切片（非字符级），Markdown 文件自动提取标题面包屑。
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import List, Optional

import tiktoken
import pdfplumber

from config import CHUNK_SIZE, CHUNK_OVERLAP, DOCS_DIR
from src.models import Chunk

logger = logging.getLogger(__name__)

# ============================================================
# Token 计数器
# ============================================================

_tokenizer = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """返回 cl100k_base 编码下的 token 数。"""
    return len(_tokenizer.encode(text))


# ============================================================
# 编码自动检测
# ============================================================

def _detect_encoding(file_path: Path, fallback: str = "utf-8") -> str:
    """按优先级降级尝试编码，兼容中文文档。"""
    encodings = [fallback, "gbk", "gb2312", "gb18030", "latin-1"]
    for enc in encodings:
        try:
            file_path.read_text(encoding=enc)
            return enc
        except (UnicodeDecodeError, UnicodeError):
            continue
    return fallback


# ============================================================
# Markdown 标题解析
# ============================================================

_MD_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)


def _parse_md_headings(text: str) -> list[tuple[int, int, str]]:
    """提取全文所有 Markdown 标题。

    Returns:
        [(char_offset, level, title), ...]  按出现位置升序。
    """
    headings: list[tuple[int, int, str]] = []
    for m in _MD_HEADING_RE.finditer(text):
        level = len(m.group(1))
        title = m.group(2).strip()
        headings.append((m.start(), level, title))
    return headings


def _heading_breadcrumb(
    pos: int, headings: list[tuple[int, int, str]]
) -> list[str]:
    """返回给定字符位置 `pos` 处的标题面包屑。

    用栈维护层级：遇到更深或同级标题时弹栈再入栈，
    栈始终是当前位置的作用域链。
    """
    stack: list[tuple[int, str]] = []
    for h_pos, level, title in headings:
        if h_pos > pos:
            break
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
    return [title for _, title in stack]


# ============================================================
# 句子切分
# ============================================================

_SENTENCE_BOUNDARY = re.compile(r"(?<=[。！？.!?])\s+")


def _split_sentences(text: str) -> list[str]:
    """按中英文标点断句，相邻无结束标点的片段自动合并。"""
    raw = _SENTENCE_BOUNDARY.split(text)
    merged: list[str] = []
    for seg in raw:
        stripped = seg.strip()
        if not stripped:
            continue
        if merged and not re.search(r"[。！？.!?]$", merged[-1]):
            merged[-1] += stripped
        else:
            merged.append(stripped)
    return merged


# ============================================================
# 文档加载
# ============================================================

def load_document(file_path: str | Path) -> str:
    """加载单个文档，根据后缀名分发到对应的加载器。"""
    path = Path(file_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"文件不存在: {path}")

    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _load_pdf(path)
    elif suffix in (".md", ".markdown", ".txt"):
        return _load_text(path)
    else:
        raise ValueError(f"暂不支持的文档格式: {suffix}  (支持: .pdf / .md / .txt)")


def _load_pdf(path: Path) -> str:
    """pdfplumber 提取 PDF 文本，不处理表格和图片。"""
    try:
        with pdfplumber.open(str(path)) as pdf:
            pages = [page.extract_text() or "" for page in pdf.pages]
        return "\n\n".join(pages)
    except Exception as exc:
        logger.error("PDF 读取失败 [%s]: %s", path.name, exc)
        return ""


def _load_text(path: Path) -> str:
    """加载纯文本 / Markdown，自动检测编码。"""
    encoding = _detect_encoding(path)
    try:
        return path.read_text(encoding=encoding)
    except Exception as exc:
        logger.error("文本读取失败 [%s]: %s", path.name, exc)
        return ""


# ============================================================
# Token 级滑窗切片
# ============================================================

def sliding_window_chunk(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
    headings: list[tuple[int, int, str]] | None = None,
) -> list[dict]:
    """Token 级滑窗切片 — 先断句，再按 token 数贪心合并。

    Args:
        text: 输入全文。
        chunk_size: 目标 Chunk token 数上限。
        overlap: 相邻 Chunk 间的最小重叠 token 数。
        headings: Markdown 标题位置列表（非 Markdown 传 None）。

    Returns:
        [{"text": str, "headings": [str, ...]}, ...]
    """
    if not text or not text.strip():
        return []

    sentences = _split_sentences(text)
    if not sentences:
        return []

    # 为每个句子建立结构化表示，避免重复 tokenize
    sent_map: list[dict] = []
    for sent in sentences:
        pos = text.find(sent)
        if pos == -1:
            pos = 0
        sent_map.append({
            "text": sent,
            "tokens": count_tokens(sent),
            "char_pos": pos,
        })

    hl = headings or []
    chunks: list[dict] = []
    i = 0

    while i < len(sent_map):
        parts: list[str] = []
        token_count = 0
        j = i

        # 贪心合并 — 按 token 维度
        while j < len(sent_map) and token_count + sent_map[j]["tokens"] <= chunk_size:
            parts.append(sent_map[j]["text"])
            token_count += sent_map[j]["tokens"]
            j += 1

        # 单句 token 数超标 → 按比例硬截断
        if not parts and j < len(sent_map):
            oversized = sent_map[j]["text"]
            ratio = chunk_size / max(sent_map[j]["tokens"], 1)
            cut_chars = max(int(len(oversized) * ratio), 1)
            parts.append(oversized[:cut_chars])
            j += 1

        chunk_text = "".join(parts)
        chunk_headings = _heading_breadcrumb(sent_map[i]["char_pos"], hl)
        chunks.append({"text": chunk_text, "headings": chunk_headings})

        if j >= len(sent_map):
            break

        # 重叠回退 — token 维度
        overlap_acc = 0
        next_i = j
        while next_i > i and overlap_acc < overlap:
            next_i -= 1
            overlap_acc += sent_map[next_i]["tokens"]
        if next_i == i:
            next_i += 1
        i = next_i

    return chunks


# ============================================================
# 主入口
# ============================================================

def process_document(
    file_path: str | Path,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[Chunk]:
    """加载文档 → 切片 → 附加元数据 → 返回 List[Chunk]。"""
    path = Path(file_path)
    full_text = load_document(path)

    if not full_text:
        logger.warning("文档内容为空，无切片产出: %s", path.name)
        return []

    # 仅对 Markdown 文件提取标题结构
    is_md = path.suffix.lower() in (".md", ".markdown")
    md_headings = _parse_md_headings(full_text) if is_md else None

    raw_chunks = sliding_window_chunk(
        full_text,
        chunk_size=chunk_size,
        overlap=overlap,
        headings=md_headings,
    )

    results: list[Chunk] = []
    for idx, rc in enumerate(raw_chunks):
        meta: dict = {
            "source": path.name,
            "chunk_index": idx,
            "token_count": count_tokens(rc["text"]),
        }
        if rc["headings"]:
            meta["headings"] = rc["headings"]
            meta["heading_breadcrumb"] = " > ".join(rc["headings"])

        chunk = Chunk(content=rc["text"], metadata=meta)
        results.append(chunk)

    logger.info(
        "处理完成 [%s] → %d 个 Chunk  (Markdown 标题追踪: %s)",
        path.name,
        len(results),
        "ON" if is_md else "OFF",
    )
    return results


def process_directory(
    dir_path: str | Path = DOCS_DIR,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[Chunk]:
    """批量处理目录下所有支持的文档。"""
    dir_path = Path(dir_path)
    if not dir_path.exists():
        raise FileNotFoundError(f"目录不存在: {dir_path}")

    all_chunks: list[Chunk] = []
    supported = {".pdf", ".md", ".markdown", ".txt"}

    for file_path in sorted(dir_path.iterdir()):
        if file_path.suffix.lower() in supported:
            chunks = process_document(file_path, chunk_size, overlap)
            all_chunks.extend(chunks)

    logger.info(
        "目录处理完成 [%s] → 总计 %d 个 Chunk", dir_path.name, len(all_chunks)
    )
    return all_chunks


# ============================================================
# 演示入口
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    samples = list(Path("docs").glob("sample_*"))
    if not samples:
        print("docs/ 目录下无测试文档，请先放置 .txt / .md / .pdf 文件")
        raise SystemExit(1)

    for sample in sorted(samples):
        print(f"\n{'='*60}")
        print(f"  源文件: {sample.name}")
        print(f"  配置  : chunk_size={CHUNK_SIZE} tokens  overlap={CHUNK_OVERLAP} tokens")
        print(f"{'='*60}\n")

        chunks = process_document(sample)

        for c in chunks:
            print(
                f"--- Chunk #{c.metadata['chunk_index']:02d}  "
                f"| tokens: {c.metadata['token_count']}  "
                f"| 上下文: {c.metadata.get('heading_breadcrumb', '(root)')} ---"
            )
            preview = c.content[:200].replace("\n", "\\n")
            print(f"    {preview}...\n")
