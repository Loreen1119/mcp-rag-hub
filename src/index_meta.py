"""索引命名的最小公共实现（不依赖任何模型 / 向量库）。

**为什么单独一个模块**：`corpus_collection_name` 是 collection 命名的**唯一来源**
（`scripts/audit_chroma_index.py` 的注释也这么写）。但它原来的宿主 `src/pipeline.py`
在模块顶层就 `import` 了 `src.data_pipeline` / `src.fusion` / `src.retrievers`，
进而拉起 `sentence-transformers` / `chromadb` / `torch`。

`GET /health` 这类**绝不能加载模型**的路径需要用到命名规则，如果从
`src.pipeline` 导入，健康检查就会背上几秒启动 + 数百 MB 常驻 —— 探针看到的是
「服务慢/挂」。所以把这段纯函数搬到这里：只依赖 `config`（os + pathlib），
可安全轻量导入。

`src/pipeline.py` 从本模块再导出同名函数，因此
`from src.pipeline import corpus_collection_name` 的既有调用点全部不变。
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from config import INDEX_SCHEMA_VERSION


def corpus_collection_name(corpus: str, model_name: str) -> str:
    """每份语料一个稳定的 collection 名，使多语料索引共存、切换时零重建。

    前缀刻意不用 knowledge_base__：VectorRetriever._gc_stale_collections 只清理
    该前缀下「hash 与当前不同」的集合，若沿用默认命名，切换语料会把另一份语料的
    索引直接 GC 掉，每次切回来都要重新 embedding。
    """
    base = Path(model_name).name
    name = f"kb__{corpus}__{base}__v{INDEX_SCHEMA_VERSION}"
    if len(name) > 63:  # Chroma 上限 63 字符，超了就用模型名 hash 压缩
        base = hashlib.sha256(model_name.encode()).hexdigest()[:8]
        name = f"kb__{corpus}__{base}__v{INDEX_SCHEMA_VERSION}"
    return name
