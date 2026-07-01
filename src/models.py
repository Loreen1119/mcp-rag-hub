"""
核心数据结构。

整个项目流通两种对象：
- Chunk: 文档切片，从 data_pipeline 产出，被 retrievers 消费
- RetrievalResult: 检索结果，从 retrievers/fusion 产出，被 app/evaluate 消费

下游永远消费上游输出，结构固定，字段只增不减。
"""

from dataclasses import dataclass, field
import uuid


@dataclass
class Chunk:
    """文档切片。metadata 用 dict 做弹性扩展，不同文档类型附加不同元信息。"""
    content: str
    metadata: dict = field(default_factory=dict)
    chunk_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    def __repr__(self) -> str:
        source = self.metadata.get("source", "unknown")
        idx = self.metadata.get("chunk_index", "?")
        preview = self.content[:50].replace("\n", " ")
        return f"Chunk({source}#{idx}, '{preview}...')"


@dataclass
class RetrievalResult:
    """检索结果。source 标记来自哪一路：bm25 / vector / rrf / cross_encoder。"""
    chunk: Chunk
    score: float
    source: str

    def __repr__(self) -> str:
        return f"RetrievalResult({self.source}, score={self.score:.4f}, chunk={self.chunk})"
