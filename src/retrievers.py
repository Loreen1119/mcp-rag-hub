"""
BM25 + ChromaDB 双路检索引擎。

- BM25Retriever: 稀疏关键词召回，jieba 分词 + rank_bm25
- VectorRetriever: 稠密语义召回，手动 sentence-transformers embedding + ChromaDB 持久化

两路各自返回原始分数（不做归一化），量纲差异由 fusion.py 的 RRF 统一处理。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

# 先导入 src（触发 src/__init__.py 的 OMP 修复），再导入任何 torch 相关库
from src.models import Chunk, RetrievalResult
from config import (
    BM25_TOP_K,
    VECTOR_TOP_K,
    EMBEDDING_MODEL,
    CHROMA_PERSIST_DIR,
)

import chromadb
import jieba
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer  # 此时 OMP 修复已生效

logger = logging.getLogger(__name__)

# ============================================================
# BM25 关键词检索器
# ============================================================


class BM25Retriever:
    """基于 jieba 分词 + BM25Okapi 的关键词检索器。

    BM25 对专有名词、数字、编号等字面匹配敏感，
    对同义词和语义相近的表达较弱——恰好与向量检索互补。
    """

    def __init__(self, chunks: List[Chunk]):
        if not chunks:
            raise ValueError("chunks 不能为空")

        self.chunks = chunks

        # jieba 分词 → 构建 BM25 稀疏索引
        self._tokenized_corpus: list[list[str]] = []
        self._bm25_map: list[int] = []  # bm25_local_idx → chunks_idx

        for idx, chunk in enumerate(chunks):
            tokens = list(jieba.cut(chunk.content))
            tokens = [t.strip() for t in tokens if t.strip()]
            if tokens:
                self._tokenized_corpus.append(tokens)
                self._bm25_map.append(idx)

        if not self._tokenized_corpus:
            logger.warning("所有 Chunk 分词后均为空，BM25 索引为空")
            self._bm25_index = None
        else:
            self._bm25_index = BM25Okapi(self._tokenized_corpus)

        logger.info(
            "BM25Retriever 初始化完成 — 索引 %d/%d 个 Chunk",
            len(self._tokenized_corpus),
            len(chunks),
        )

    def search(self, query: str, top_k: int = BM25_TOP_K) -> List[RetrievalResult]:
        """BM25 关键词检索。"""
        if self._bm25_index is None:
            return []

        query_tokens = [t.strip() for t in jieba.cut(query) if t.strip()]
        if not query_tokens:
            return []

        scores = self._bm25_index.get_scores(query_tokens)

        # 取 Top-K
        top_indices = np.argsort(-scores)[: min(top_k, len(scores))]

        results: List[RetrievalResult] = []
        for local_idx in top_indices:
            bm25_score = float(scores[local_idx])
            if bm25_score <= 0:
                continue
            chunk_idx = self._bm25_map[local_idx]
            results.append(RetrievalResult(
                chunk=self.chunks[chunk_idx],
                score=bm25_score,
                source="bm25",
            ))
        return results


# ============================================================
# ChromaDB 向量检索器
# ============================================================


class VectorRetriever:
    """手动 embedding + ChromaDB 持久化存储的语义检索器。

    不使用 ChromaDB 内置 embedding 函数，而是用 sentence-transformers
    手动生成向量后存入 ChromaDB。面试时能讲清向量生成全流程。
    """

    def __init__(
        self,
        chunks: List[Chunk],
        model_name: str = EMBEDDING_MODEL,
        persist_dir: str | Path = CHROMA_PERSIST_DIR,
        rebuild: bool = True,
    ):
        if not chunks:
            raise ValueError("chunks 不能为空")

        self.chunks = chunks
        persist_dir = Path(persist_dir)

        # 初始化模型
        logger.info("加载 Embedding 模型: %s", model_name)
        self.model = SentenceTransformer(model_name)
        logger.info(
            "模型加载完成 — 维度: %d, 最大长度: %d",
            self.model.get_sentence_embedding_dimension(),
            self.model.max_seq_length,
        )

        # 初始化 ChromaDB
        self.client = chromadb.PersistentClient(path=str(persist_dir))

        collection_name = "knowledge_base"
        if rebuild:
            try:
                self.client.delete_collection(collection_name)
            except Exception:
                pass
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            self._index_chunks(chunks)
        else:
            self.collection = self.client.get_collection(collection_name)
            logger.info("复用已有 ChromaDB 集合: %s", collection_name)

        logger.info(
            "VectorRetriever 初始化完成 — 索引 %d 个 Chunk, 持久化路径: %s",
            len(chunks),
            persist_dir,
        )

    def _index_chunks(self, chunks: List[Chunk]) -> None:
        """手动对所有 Chunk 做 embedding 并存入 ChromaDB。"""
        texts = [c.content for c in chunks]
        ids = [c.chunk_id for c in chunks]
        metadatas = [c.metadata for c in chunks]

        logger.info("开始批量 Embedding (%d 条文本)...", len(texts))
        embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            batch_size=32,
        )

        self.collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
        )
        logger.info("Embedding + 写入完成")

    def search(self, query: str, top_k: int = VECTOR_TOP_K) -> List[RetrievalResult]:
        """向量语义检索。

        手动 encode query → 用 query_embeddings 查询 ChromaDB
        （不使用 query_texts，确保全程可见 embedding 过程）
        """
        query_embedding = self.model.encode(query)

        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        output: List[RetrievalResult] = []
        if results["ids"] and results["ids"][0]:
            for i, chunk_id in enumerate(results["ids"][0]):
                distance = results["distances"][0][i]
                # ChromaDB cosine distance → similarity: sim = 1 - distance
                similarity = 1.0 - distance
                document = results["documents"][0][i]
                metadata = results["metadatas"][0][i]

                chunk = Chunk(
                    content=document,
                    metadata=metadata,
                    chunk_id=chunk_id,
                )
                output.append(RetrievalResult(
                    chunk=chunk,
                    score=float(similarity),
                    source="vector",
                ))
        return output


# ============================================================
# 演示入口
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    from src.data_pipeline import process_directory

    # 1. 加载文档
    chunks = process_directory()
    if not chunks:
        print("docs/ 目录下无文档，请先放置测试文件")
        raise SystemExit(1)

    # 2. 初始化两路检索器
    bm25 = BM25Retriever(chunks)
    vector = VectorRetriever(chunks)

    # 3. 测试查询
    queries = [
        "RAG 的核心优化方向有哪些？",
        "混合检索策略",
        "Transformer 自注意力",
    ]

    for q in queries:
        print(f"\n{'='*60}")
        print(f"  Query: {q}")
        print(f"{'='*60}")

        print("\n  [BM25 关键词召回]")
        bm25_results = bm25.search(q, top_k=5)
        for r in bm25_results:
            print(f"    score={r.score:.4f}  |  {r.chunk}")

        print("\n  [向量语义召回]")
        vector_results = vector.search(q, top_k=5)
        for r in vector_results:
            print(f"    score={r.score:.4f}  |  {r.chunk}")

        # 对比两路差异
        bm25_ids = {r.chunk.chunk_id for r in bm25_results}
        vector_ids = {r.chunk.chunk_id for r in vector_results}
        overlap = bm25_ids & vector_ids
        print(f"\n  两路重叠: {len(overlap)}/{len(bm25_ids) | len(vector_ids)}")
