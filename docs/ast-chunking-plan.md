# mcp-rag-hub 代码文件 AST 分块改造方案

> 目标：让 `src/data_pipeline.py` 支持按文件类型路由分块策略，解决 Python 代码文件被通用滑窗切散的问题。

---

## 背景

当前 `data_pipeline.py` 对所有文件类型都使用同一种 `sliding_window_chunk`（按句子切分 + token 级滑窗）。对 Markdown 文档效果好，但对 Python 代码文件会把一个完整函数切成 `def run(`、`continue`、`}` 等几字符碎片，导致 RAG 检索质量差。

本方案在不破坏现有 Markdown / PDF / TXT 处理逻辑的前提下，为 `.py` 文件新增 **AST 分块器**。

---

## 改动范围

1. `src/data_pipeline.py`：核心改造文件
2. `config.py`：可选，增加 AST 分块默认参数（如 `AST_CHUNK_SIZE`）
3. `tests/test_data_pipeline.py`：新增/扩展测试用例
4. `README.md`：可选，更新文档说明支持 `.py` 文件

---

## 设计

### 1. 新增 `chunk_by_ast()` 函数

函数签名：

```python
def chunk_by_ast(
    source_code: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[dict]:
    """基于 Python AST 按函数/类边界切分源码。

    Returns:
        [{"text": str, "headings": [str, ...]}, ...]
    """
```

行为要求：

1. **解析源码**：使用 `ast.parse(source_code)`，异常时 fallback 到 `sliding_window_chunk`。
2. **识别顶层定义**：遍历 `ast.Module.body`，只处理以下节点类型：
   - `ast.FunctionDef`
   - `ast.AsyncFunctionDef`
   - `ast.ClassDef`
3. **提取源码片段**：通过 `node.lineno` 和 `node.end_lineno` 从源码行切片。
4. **函数 chunk**：
   - 每个函数作为一个 chunk
   - `headings = ["def: 函数名"]`
   - 如果函数位于类内部，额外加上 `f"class: 类名"`，即 `headings = ["class: 类名", "def: 函数名"]`
5. **类 chunk**：
   - 如果类整体 token 数 <= `chunk_size`：整个类作为一个 chunk，`headings = ["class: 类名"]`
   - 如果类整体 token 数 > `chunk_size`：拆成多个子 chunk
     - 类签名 + docstring 作为类 overview chunk，`headings = ["class: 类名"]`
     - 每个方法单独作为 chunk，`headings = ["class: 类名", "def: 方法名"]`
6. **模块级代码**：不在任何函数/类中的顶层代码（import、全局变量、执行语句）作为一个 chunk，`headings = ["module-level"]`。
7. **超大 chunk 截断**：任何 chunk 超过 `chunk_size` 时，调用 `sliding_window_chunk` 进一步切分，但保留原始 `headings`。
8. **重叠处理**：相邻 chunk 之间按 token 数重叠 `overlap`，类似现有 `sliding_window_chunk` 的回退逻辑。

---

### 2. 修改 `process_document()` 增加文件类型路由

在 `process_document()` 中，根据文件后缀选择分块策略：

```python
def process_document(file_path, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    path = Path(file_path)
    full_text = load_document(path)

    if not full_text:
        logger.warning("文档内容为空，无切片产出: %s", path.name)
        return []

    suffix = path.suffix.lower()

    if suffix in (".md", ".markdown"):
        md_headings = _parse_md_headings(full_text)
        raw_chunks = sliding_window_chunk(
            full_text, chunk_size=chunk_size, overlap=overlap, headings=md_headings
        )
    elif suffix == ".py":
        raw_chunks = chunk_by_ast(full_text, chunk_size=chunk_size, overlap=overlap)
    else:
        # .pdf / .txt 等兜底
        raw_chunks = sliding_window_chunk(
            full_text, chunk_size=chunk_size, overlap=overlap
        )

    # 后续 metadata 包装逻辑保持不变
    ...
```

---

### 3. 扩展 `process_directory()` 支持 `.py`

```python
supported = {".pdf", ".md", ".markdown", ".txt", ".py"}
```

保持不递归处理子目录，与现有行为一致。

---

### 4. 可选：`config.py` 增加 AST 参数

```python
# AST 分块默认参数
AST_CHUNK_SIZE = int(os.getenv("MCP_RAG_AST_CHUNK_SIZE", CHUNK_SIZE))
AST_CHUNK_OVERLAP = int(os.getenv("MCP_RAG_AST_CHUNK_OVERLAP", CHUNK_OVERLAP))
```

如果短期内不想增加配置项，也可以直接复用 `CHUNK_SIZE` 和 `CHUNK_OVERLAP`。

---

## 测试用例

在 `tests/test_data_pipeline.py` 中新增以下测试：

### 测试 1：AST 分块基本功能

```python
def test_chunk_by_ast_basic():
    code = '''
import os

GLOBAL_VAR = 42

def helper(x):
    return x + 1

class MyClass:
    def method_a(self):
        return "a"

    def method_b(self):
        return "b"
'''
    chunks = chunk_by_ast(code)
    assert len(chunks) >= 4  # module-level, helper, MyClass(method_a), method_b
    headings_list = [c["headings"] for c in chunks]
    assert ["def: helper"] in headings_list
    assert ["class: MyClass", "def: method_a"] in headings_list
```

### 测试 2：`process_document` 能处理 `.py`

```python
def test_process_python_file():
    chunks = process_document("src/retrievers.py")
    assert len(chunks) > 0
    # 每个 chunk 应该包含完整的函数或类，不应出现几字符碎片
    for chunk in chunks:
        assert len(chunk.content) > 20
        assert "source" in chunk.metadata
```

### 测试 3：`.md` 处理逻辑不被破坏

```python
def test_process_markdown_file():
    chunks = process_document("README.md")
    assert len(chunks) > 0
    # 至少有一个 chunk 包含 heading_breadcrumb 元数据
    assert any("heading_breadcrumb" in c.metadata for c in chunks)
```

---

## 验收标准

1. `python -m src.data_pipeline src/retrievers.py` 能正常跑通，产出多个 chunk。
2. 每个 chunk 都包含完整函数/类/方法源码，不再出现 `def run(`、`}`、`continue` 等碎片。
3. `chunk.metadata["headings"]` 中至少包含函数名或类名。
4. 现有 `.md` / `.pdf` / `.txt` 处理逻辑完全不受影响。
5. `process_directory("docs")` 可以同时处理目录下的 `.md` 和 `.py` 文件。

---

## 面试讲述要点

如果被问到，可以这样说：

> "我发现通用滑窗分块会把 Python 代码切散，于是给 `mcp-rag-hub` 加了文件类型路由：代码文件走 AST 分块，Markdown 保持标题面包屑 + 滑窗。AST 分块按函数/类边界切，每个 chunk 的 headings 里带上函数名或类名，检索时能直接命中。这是一个低投入高回报的优化，不需要改动 embedding 或检索模型，只改数据管线。"

---

## 下一步（可选）

- 实现 **父子分块（Parent-Child）**：类作为父块，方法作为子块，检索时命中子块、生成时引用父块上下文。
- 递归扫描 `docs/` 子目录中的 `.py` 文件。
- 给 `.py` chunk 的 `metadata` 增加 `start_line` / `end_line`，方便定位源码位置。
