"""data_pipeline 单元测试。"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from src.data_pipeline import (
    chunk_by_ast,
    load_document,
    process_document,
    process_directory,
)


# ============================================================
# chunk_by_ast — 基础功能
# ============================================================

def test_chunk_by_ast_basic():
    """顶层函数、类（带方法）、模块级代码，各自产出独立 chunk。"""
    code = '''
import os
import sys

GLOBAL_VAR = 42

def helper(x):
    return x + 1

async def async_helper(x):
    return x * 2

class MyClass:
    """MyClass docstring."""

    def method_a(self):
        return "a"

    def method_b(self):
        return "b"

    async def async_method(self):
        return "async"
'''
    chunks = chunk_by_ast(code)

    headings_list = [c["headings"] for c in chunks]

    # 顶层函数
    assert ["def: helper"] in headings_list
    assert ["def: async_helper"] in headings_list
    # 模块级
    assert ["module-level"] in headings_list
    # 顶层类（MyClass token 数 <= chunk_size，整体打包；否则拆成 overview + 方法）
    assert ["class: MyClass"] in headings_list

    # 每个 chunk 不为空
    for c in chunks:
        assert len(c["text"]) > 0
        assert c["start_line"] >= 1
        assert c["end_line"] >= c["start_line"]


def test_chunk_by_ast_small_class_packed():
    """小类（token <= chunk_size）整个打包为一个 chunk。"""
    code = '''
class Small:
    def foo(self):
        return 1
'''
    chunks = chunk_by_ast(code)
    headings_list = [c["headings"] for c in chunks]

    assert ["class: Small"] in headings_list
    # 不应有独立的 method chunk
    assert not any(h == ["class: Small", "def: foo"] for h in headings_list)


def test_chunk_by_ast_large_class_split():
    """大类（token > chunk_size）拆成 class overview + 方法独立 chunk。"""
    code = '''
class Large:
    """Docstring line 1.
    Docstring line 2.
    Docstring line 3."""

    def method_one(self):
        # A method with substantial body to increase token count
        result = []
        for i in range(10):
            result.append(i * 2)
        return result

    def method_two(self):
        # Another substantial method body
        data = {"a": 1, "b": 2, "c": 3}
        return sum(data.values())

    def method_three(self):
        # Yet another method
        return [x for x in range(20) if x % 2 == 0]
'''
    chunks = chunk_by_ast(code)
    headings_list = [c["headings"] for c in chunks]

    # 大类拆成 overview + 方法
    assert ["class: Large"] in headings_list
    assert ["class: Large", "def: method_one"] in headings_list
    assert ["class: Large", "def: method_two"] in headings_list
    assert ["class: Large", "def: method_three"] in headings_list


def test_chunk_by_ast_syntax_error_fallback():
    """语法错误的源码回退到 sliding_window_chunk。"""
    code = "def broken(    # missing closing paren\n    return 42\n"
    chunks = chunk_by_ast(code)

    assert len(chunks) > 0
    # fallback chunk 没有 AST 提供的行号，用默认值
    for c in chunks:
        assert "text" in c
        assert "headings" in c


def test_chunk_by_ast_start_end_lines():
    """每个 chunk 的 start_line / end_line 连续递增。"""
    code = '''
def func_a():
    pass

def func_b():
    pass
'''
    chunks = chunk_by_ast(code)
    starts = [c["start_line"] for c in chunks]
    ends = [c["end_line"] for c in chunks]

    assert starts == sorted(starts)
    assert ends == sorted(ends)
    # 相邻 chunk 首尾相接或接近
    for i in range(len(chunks) - 1):
        assert chunks[i + 1]["start_line"] >= chunks[i]["end_line"]


# ============================================================
# load_document — .py 文件加载
# ============================================================

def test_load_document_python():
    """load_document 能加载 .py 文件。"""
    with tempfile.NamedTemporaryFile(
        suffix=".py", mode="w", delete=False, encoding="utf-8"
    ) as f:
        f.write("def foo(): return 1\n")
        path = f.name

    try:
        text = load_document(path)
        assert "def foo()" in text
    finally:
        Path(path).unlink()


# ============================================================
# process_document — .py 文件处理
# ============================================================

def test_process_python_file():
    """process_document 处理 .py 产出多个 chunk，无碎片。"""
    with tempfile.NamedTemporaryFile(
        suffix=".py", mode="w", delete=False, encoding="utf-8"
    ) as f:
        f.write(
            "import os\n"
            "\n"
            "def add(a, b):\n"
            "    return a + b\n"
            "\n"
            "def multiply(a, b):\n"
            "    return a * b\n"
        )
        path = f.name

    try:
        chunks = process_document(path)
        assert len(chunks) > 0

        for c in chunks:
            # 不应出现单行碎片
            assert len(c.content.strip()) > 5, f"碎片 chunk: {c.content!r}"
            assert c.metadata["source_type"] == "ast"
            assert "start_line" in c.metadata
            assert "end_line" in c.metadata
            assert "source" in c.metadata

        headings_list = [c.metadata.get("headings", []) for c in chunks]
        assert ["def: add"] in headings_list
        assert ["def: multiply"] in headings_list
    finally:
        Path(path).unlink()


# ============================================================
# process_document — Markdown 不被破坏
# ============================================================

def test_process_markdown_file(tmp_path):
    """Markdown 文件走原有滑窗逻辑，heading_breadcrumb 正常产出。"""
    md_file = tmp_path / "sample.md"
    md_file.write_text(
        "# Title\n\nPara one.\n\n## Section\n\nPara two.\n", encoding="utf-8"
    )

    chunks = process_document(md_file)
    assert len(chunks) > 0
    assert any("heading_breadcrumb" in c.metadata for c in chunks)
    assert all(c.metadata["source_type"] == "markdown" for c in chunks)


def test_process_markdown_no_ast(tmp_path):
    """Markdown 不走 AST 分块，source_type = markdown。"""
    md_file = tmp_path / "test.md"
    md_file.write_text("# Hello\n\nWorld.\n", encoding="utf-8")

    chunks = process_document(md_file)
    assert len(chunks) > 0
    for c in chunks:
        assert c.metadata["source_type"] == "markdown"


# ============================================================
# process_directory — .py 支持
# ============================================================

def test_process_directory_python(tmp_path):
    """process_directory 能处理目录下的 .py 文件。"""
    (tmp_path / "a.py").write_text("def func_a(): pass\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("def func_b(): pass\n", encoding="utf-8")

    chunks = process_directory(tmp_path)
    assert len(chunks) >= 2
    sources = {c.metadata["source"] for c in chunks}
    assert "a.py" in sources
    assert "b.py" in sources


# ============================================================
# 端到端：.py 文件不再出现代码碎片
# ============================================================

def test_no_code_fragment_in_python_chunks(tmp_path):
    """每个 chunk 的 content 不是单行语法碎片。"""
    code = """
class Processor:
    def __init__(self):
        self.data = []

    def process(self, item):
        self.data.append(item)
        return item * 2

    def reset(self):
        self.data.clear()
"""
    py_file = tmp_path / "processor.py"
    py_file.write_text(code, encoding="utf-8")

    chunks = process_document(py_file)
    for c in chunks:
        lines = [l.strip() for l in c.content.splitlines() if l.strip()]
        # 不应有只有语法符号的单行碎片
        for line in lines:
            assert len(line) > 3, f"过短 chunk line: {line!r}"
