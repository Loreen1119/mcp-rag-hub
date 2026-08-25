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
from src.models import Chunk, make_chunk_id

logger = logging.getLogger(__name__)

# ============================================================
# 支持的文档后缀（全项目唯一来源）
# ============================================================

SUPPORTED_SUFFIXES = {".pdf", ".md", ".markdown", ".txt", ".py"}


def iter_supported_docs(dir_path: str | Path) -> list[Path]:
    """返回目录下所有受支持文档（与 process_directory 的 iterdir 平扫一致）。"""
    dir_path = Path(dir_path)
    return sorted(
        p
        for p in dir_path.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
    )


def corpus_hash(dir_path: str | Path, chunk_size: int, chunk_overlap: int) -> str:
    """文档集内容 hash：文件相对路径 + 内容 + chunk 参数决定。

    任一变化都会改变 hash，从而触发向量索引 / KG 缓存重建。
    必须与 process_directory 使用同一份扫描逻辑，保证 hash 与实际切片一致。
    """
    import hashlib

    dir_path = Path(dir_path)
    h = hashlib.sha256()
    h.update(f"chunk_size={chunk_size}\nchunk_overlap={chunk_overlap}\n".encode())
    for p in iter_supported_docs(dir_path):
        h.update(p.relative_to(dir_path).as_posix().encode())
        h.update(b"\x00")
        try:
            h.update(p.read_bytes())
        except OSError:
            continue
        h.update(b"\x00")
    return h.hexdigest()

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
    elif suffix in (".md", ".markdown", ".txt", ".py"):
        return _load_text(path)
    else:
        raise ValueError(
            f"暂不支持的文档格式: {suffix}  (支持: .pdf / .md / .txt / .py)"
        )


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
# AST 分块
# ============================================================

import ast


def chunk_by_ast(
    source_code: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[dict]:
    """基于 Python AST 按函数/类边界切分源码。

    - 顶层函数/异步函数 → 各自一个 chunk，headings = ["def: 函数名"]
    - 顶层类：
        - token 数 <= chunk_size → 整个类一个 chunk
        - token 数 > chunk_size → 类签名+docstring 作为 overview，其余方法各自 chunk
    - 模块级代码（不在任何函数/类中）→ 一个 chunk，headings = ["module-level"]
    - 超大 chunk（> chunk_size）→ 按顶层语句数硬切，不再复用 sentence 滑窗
    - 解析失败 → 回退到 sliding_window_chunk，source_type = "fallback"

    Args:
        source_code: Python 源码文本。
        chunk_size: 目标 chunk token 上限。
        overlap: 相邻 chunk 重叠 token 数（AST 场景下不实际使用，仅保留签名兼容）。

    Returns:
        [{"text": str, "headings": [str, ...], "start_line": int, "end_line": int}, ...]
    """
    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        # 解析失败，回退到 sentence 滑窗
        raw = sliding_window_chunk(source_code, chunk_size=chunk_size, overlap=overlap)
        for r in raw:
            r["start_line"] = 1
            r["end_line"] = len(source_code.splitlines())
        return raw

    lines = source_code.splitlines()
    n_lines = len(lines)

    def get_source(start: int, end: int) -> str:
        return "\n".join(lines[start - 1 : end])

    def estimate_tokens(text: str) -> int:
        return count_tokens(text)

    chunks: list[dict] = []

    # ----------------------------------------------------------
    # 遍历顶层节点，分发到对应的 chunk 生成逻辑
    # ----------------------------------------------------------
    current_class: str | None = None

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            start = node.lineno
            end = node.end_lineno
            text = get_source(start, end)
            func_name = node.name

            headings = (
                [current_class, f"def: {func_name}"]
                if current_class
                else [f"def: {func_name}"]
            )

            if estimate_tokens(text) <= chunk_size:
                chunks.append(
                    {
                        "text": text,
                        "headings": headings,
                        "start_line": start,
                        "end_line": end,
                    }
                )
            else:
                # 超大函数：按顶层语句数硬切，不复用 sentence 滑窗
                sub_chunks = _split_by_statement(text, start, headings, chunk_size)
                chunks.extend(sub_chunks)

        elif isinstance(node, ast.ClassDef):
            class_name = node.name
            class_start = node.lineno
            class_end = node.end_lineno
            class_text = get_source(class_start, class_end)

            # 统计方法数量（同步 + 异步），用于决定是否拆分
            method_count = sum(
                1
                for item in node.body
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
            )

            # 拆分条件：方法数 >= 2 或 token 超标
            # 单方法小类整体打包，避免过度碎片
            if method_count < 2 and estimate_tokens(class_text) <= chunk_size:
                # 整体打包
                chunks.append(
                    {
                        "text": class_text,
                        "headings": [f"class: {class_name}"],
                        "start_line": class_start,
                        "end_line": class_end,
                    }
                )
            else:
                # 拆解：类签名+docstring 作为 overview，其余方法各自 chunk
                overview_text, method_nodes = _extract_class_parts(
                    node, source_code, lines
                )
                if overview_text:
                    chunks.append(
                        {
                            "text": overview_text,
                            "headings": [f"class: {class_name}"],
                            "start_line": class_start,
                            "end_line": overview_text.count("\n") + class_start,
                        }
                    )
                current_class = f"class: {class_name}"
                for meth in method_nodes:
                    m_start = meth.lineno
                    m_end = meth.end_lineno
                    m_text = get_source(m_start, m_end)
                    headings = [f"class: {class_name}", f"def: {meth.name}"]
                    if estimate_tokens(m_text) <= chunk_size:
                        chunks.append(
                            {
                                "text": m_text,
                                "headings": headings,
                                "start_line": m_start,
                                "end_line": m_end,
                            }
                        )
                    else:
                        sub_chunks = _split_by_statement(
                            m_text, m_start, headings, chunk_size
                        )
                        chunks.extend(sub_chunks)
                current_class = None

        else:
            # 模块级代码（import / 全局变量 / 执行语句等）
            start = node.lineno
            end = node.end_lineno or start
            text = get_source(start, end)
            if estimate_tokens(text) <= chunk_size:
                chunks.append(
                    {
                        "text": text,
                        "headings": ["module-level"],
                        "start_line": start,
                        "end_line": end,
                    }
                )
            else:
                sub_chunks = _split_by_statement(text, start, ["module-level"], chunk_size)
                chunks.extend(sub_chunks)

    # 按 start_line 排序
    chunks.sort(key=lambda c: c["start_line"])
    return chunks


def _extract_class_parts(
    node: ast.ClassDef, source_code: str, lines: list[str]
) -> tuple[str, list[ast.FunctionDef | ast.AsyncFunctionDef]]:
    """从 ClassDef 节点提取类签名+docstring，以及所有方法节点列表。"""
    overview_lines = node.lineno
    method_nodes: list[ast.FunctionDef | ast.AsyncFunctionDef] = []

    for item in node.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            method_nodes.append(item)
        elif overview_lines == node.lineno:
            # 第一个非方法节点之前的行都算类签名/docstring 范围
            if isinstance(item, ast.Expr) and isinstance(item.value, ast.Constant):
                # docstring
                overview_lines = item.end_lineno or node.lineno
            else:
                overview_lines = item.end_lineno or node.lineno

    end_line = overview_lines
    overview_text = "\n".join(lines[node.lineno - 1 : end_line])
    return overview_text, method_nodes


def _split_by_statement(
    text: str, base_start: int, headings: list[str], chunk_size: int
) -> list[dict]:
    """按顶层语句数硬切超大 chunk，不使用 sentence 滑窗。"""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        # 无法解析，直接截断并 warn
        import logging

        logging.getLogger(__name__).warning(
            "AST 解析失败，无法切分超大 chunk，截断处理 (start_line=%d)", base_start
        )
        return [
            {
                "text": text[:500],
                "headings": headings,
                "start_line": base_start,
                "end_line": base_start,
            }
        ]

    lines = text.splitlines()
    chunks: list[dict] = []
    current_lines: list[str] = []
    current_tokens = 0
    chunk_start = base_start

    for node in ast.iter_child_nodes(tree):
        node_text = "\n".join(lines[node.lineno - 1 : (node.end_lineno or node.lineno)])
        node_tokens = count_tokens(node_text)

        if current_tokens + node_tokens > chunk_size and current_lines:
            chunks.append(
                {
                    "text": "\n".join(current_lines),
                    "headings": headings,
                    "start_line": chunk_start,
                    "end_line": chunk_start + len(current_lines) - 1,
                }
            )
            current_lines = []
            current_tokens = 0
            chunk_start = node.lineno + base_start - 1

        current_lines.append(node_text)
        current_tokens += node_tokens

    if current_lines:
        chunks.append(
            {
                "text": "\n".join(current_lines),
                "headings": headings,
                "start_line": chunk_start,
                "end_line": chunk_start + len(current_lines) - 1,
            }
        )

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

    suffix = path.suffix.lower()

    # 文件类型路由
    if suffix in (".md", ".markdown"):
        md_headings = _parse_md_headings(full_text)
        raw_chunks = sliding_window_chunk(
            full_text,
            chunk_size=chunk_size,
            overlap=overlap,
            headings=md_headings,
        )
        source_type = "markdown"
    elif suffix == ".py":
        raw_chunks = chunk_by_ast(full_text, chunk_size=chunk_size, overlap=overlap)
        source_type = "ast"
    else:
        # .pdf / .txt 等兜底
        raw_chunks = sliding_window_chunk(
            full_text,
            chunk_size=chunk_size,
            overlap=overlap,
        )
        source_type = "text"

    results: list[Chunk] = []
    for idx, rc in enumerate(raw_chunks):
        meta: dict = {
            "source": path.name,
            "chunk_index": idx,
            "token_count": count_tokens(rc["text"]),
            "source_type": source_type,
            "start_line": rc.get("start_line", 1),
            "end_line": rc.get("end_line", 1),
        }
        if rc["headings"]:
            meta["headings"] = rc["headings"]
            meta["heading_breadcrumb"] = " > ".join(rc["headings"])

        chunk = Chunk(
            content=rc["text"],
            metadata=meta,
            chunk_id=make_chunk_id(path.name, idx),
        )
        results.append(chunk)

    is_md = suffix in (".md", ".markdown")
    logger.info(
        "处理完成 [%s] → %d 个 Chunk  (Markdown 标题追踪: %s, source_type: %s)",
        path.name,
        len(results),
        "ON" if is_md else "OFF",
        source_type,
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
    for file_path in iter_supported_docs(dir_path):
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
