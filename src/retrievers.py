"""
BM25 + ChromaDB 双路检索引擎。

- BM25Retriever: 稀疏关键词召回，jieba 分词 + rank_bm25
- VectorRetriever: 稠密语义召回，手动 sentence-transformers embedding + ChromaDB 持久化

两路各自返回原始分数（不做归一化），量纲差异由 fusion.py 的 RRF 统一处理。
"""

from __future__ import annotations

import hashlib
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
    INDEX_SCHEMA_VERSION,
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

    索引生命周期:
        collection 名 = knowledge_base__{model}__v{INDEX_SCHEMA_VERSION}__{corpus_hash[:8]}，
        内容 / 切片变化 → corpus_hash 变化 → collection 名变化 → 新旧天然隔离，
        构建失败不影响旧集合。同名集合每次初始化校验两个摘要:
        - 文档集内容 hash（CHUNK_CORPUS_HASH）
        - schema version + chunk_id 数
        摘要都匹配时直接复用，避免重复 embedding；不匹配则删除同名集合重建。
        构建成功后才 GC 旧版本集合。

        两种重建语义需区分:
        - 默认（rebuild=False）：hash 变化 → 新 collection 名 → 安全新建，旧索引不受影响。
        - rebuild=True：破坏性强制重建，先 delete_collection 当前名集合再重建，
          构建失败会让当前索引不可用，仅供实验/显式维护使用（如 experiments.py、force_rebuild）。
    """

    def __init__(
        self,
        chunks: List[Chunk],
        model_name: str = EMBEDDING_MODEL,
        persist_dir: str | Path = CHROMA_PERSIST_DIR,
        rebuild: bool = False,
        corpus_hash: str | None = None,
        collection_name: str | None = None,
    ):
        if not chunks:
            raise ValueError("chunks 不能为空")

        self.chunks = chunks
        self.persist_dir = Path(persist_dir)
        self.corpus_hash = corpus_hash or ""

        # 初始化模型
        logger.info("加载 Embedding 模型: %s", model_name)
        self.model = SentenceTransformer(model_name)
        logger.info(
            "模型加载完成 — 维度: %d, 最大长度: %d",
            self.model.get_sentence_embedding_dimension(),
            self.model.max_seq_length,
        )

        # 初始化 ChromaDB
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))

        collection_name = self._resolve_collection_name(
            model_name,
            INDEX_SCHEMA_VERSION,
            self.corpus_hash,
            collection_name,
        )
        schema_version = INDEX_SCHEMA_VERSION

        self._doc_to_id: dict[int, str] = {id(c): c.chunk_id for c in chunks}
        self._by_id: dict[str, Chunk] = {c.chunk_id: c for c in chunks}

        if rebuild:
            self._rebuild_collection(collection_name, chunks, schema_version)
        else:
            self.collection = self._get_or_build_collection(
                collection_name, chunks, schema_version
            )

        logger.info(
            "VectorRetriever 初始化完成 — 索引 %d 个 Chunk, collection=%s, 持久化路径: %s",
            len(chunks),
            self.collection.name,
            self.persist_dir,
        )

    # ----------------------------------------------------------
    # Collection 名解析
    # ----------------------------------------------------------

    @staticmethod
    def _resolve_collection_name(
        model_name: str,
        schema_version: int,
        corpus_hash: str,
        collection_name_override: str | None,
    ) -> str:
        """决定本次使用的 Chroma collection 名。

        - collection_name_override 非空 → 直接用 override（hash 只进 metadata）。
        - 否则 knowledge_base__{basename}__v{schema}__{hash8}；hash 为空 → nohash 段。
        - 超过 Chroma 63 字符限制 → basename 段替换为 sha256(model)[:8]，不截断模型名。
        """
        if collection_name_override:
            name = collection_name_override
        else:
            basename = Path(model_name).name
            hash_seg = corpus_hash[:8] if corpus_hash else "nohash"
            name = f"knowledge_base__{basename}__v{schema_version}__{hash_seg}"
            if len(name) > 63:
                basename = hashlib.sha256(model_name.encode()).hexdigest()[:8]
                name = f"knowledge_base__{basename}__v{schema_version}__{hash_seg}"
        if len(name) > 63:
            raise ValueError(f"collection 名超限(63): {name}")
        return name

    def _gc_stale_collections(self, current_name: str) -> None:
        """清理 persist_dir 下旧版本的 collection。

        仅在当前为默认命名（knowledge_base__*）且带真实 hash 时触发；
        只删前缀匹配 knowledge_base__、段数 >= 4、末段 hash != 当前且 != nohash 的集合。
        实验用的 exp__ 前缀集合不在模式内，永不误删。删除失败仅告警。
        """
        if not current_name.startswith("knowledge_base__"):
            return
        segments = current_name.split("__")
        if len(segments) < 4:
            return
        current_hash = segments[-1]
        if current_hash == "nohash":
            return

        try:
            collections = self.client.list_collections()
        except Exception as exc:
            logger.warning("GC: 无法列出 collections: %s", exc)
            return

        for coll in collections:
            # 兼容不同 Chroma 版本：list_collections() 可能返回字符串名或 Collection 对象
            name = coll if isinstance(coll, str) else getattr(coll, "name", None)
            if not isinstance(name, str) or not name.startswith("knowledge_base__"):
                continue
            parts = name.split("__")
            if len(parts) < 4:
                continue
            if parts[-1] in (current_hash, "nohash"):
                continue
            try:
                self.client.delete_collection(name)
                logger.info("GC: 删除过期 collection: %s", name)
            except Exception as exc:
                logger.warning("GC: 删除 collection %s 失败: %s", name, exc)

    # ----------------------------------------------------------
    # Collection 生命周期
    # ----------------------------------------------------------

    def _collection_signature(self, schema_version: int) -> dict:
        """返回可写入 collection metadata 的摘要信息。

        hnsw:space=cosine 既让 HNSW 使用 cosine 距离（与 search() 的
        similarity = 1 - distance 语义一致），也写入 metadata 供复用校验，
        避免已有 collection 配置不一致时被误复用。
        """
        return {
            "hnsw:space": "cosine",
            "corpus_hash": self.corpus_hash,
            "chunk_count": len(self.chunks),
            "chunk_id_count": len(self._by_id),
            "schema_version": str(schema_version),
        }

    def _get_or_build_collection(
        self, collection_name: str, chunks: List[Chunk], schema_version: int
    ) -> "chromadb.Collection":
        """校验已有 collection：摘要匹配则复用，否则删除并重建。

        同名 collection 定义上就是"这份语料的半成品"：内容变化 → hash 变化
        → 名字变化 → 重建发生在另一个 collection 名上，旧集合不受影响。
        只有内容没变但校验失败（数据损坏）时，才删除同名集合重建。
        """
        try:
            existing = self.client.get_collection(collection_name)
        except Exception:
            existing = None

        if existing is not None:
            meta = existing.metadata or {}
            if (
                meta.get("hnsw:space") == "cosine"
                and meta.get("schema_version") == str(schema_version)
                and meta.get("corpus_hash") == self.corpus_hash
                and meta.get("chunk_count") == len(chunks)
                and meta.get("chunk_id_count") == len(self._by_id)
            ):
                logger.info(
                    "复用已有 ChromaDB 集合（摘要匹配，无重建）: %s", collection_name
                )
                return existing
            logger.info(
                "ChromaDB 集合摘要不匹配（schema/corpus hash/chunk 数变化），重建: %s",
                collection_name,
            )
            try:
                self.client.delete_collection(collection_name)
            except Exception:
                pass

        return self._build_collection(collection_name, chunks, schema_version)

    def _rebuild_collection(
        self, collection_name: str, chunks: List[Chunk], schema_version: int
    ) -> "chromadb.Collection":
        logger.info("强制重建 ChromaDB 集合: %s", collection_name)
        try:
            self.client.delete_collection(collection_name)
        except Exception:
            pass
        return self._build_collection(collection_name, chunks, schema_version)

    def _build_collection(
        self, collection_name: str, chunks: List[Chunk], schema_version: int
    ) -> "chromadb.Collection":
        collection = self.client.create_collection(
            name=collection_name,
            metadata=self._collection_signature(schema_version),
        )
        self._index_chunks(collection, chunks)
        # 构建成功后才 GC 旧版本集合，避免构建失败时回滚空间消失
        self._gc_stale_collections(collection_name)
        return collection

    # ----------------------------------------------------------
    # Embedding
    # ----------------------------------------------------------

    def _index_chunks(self, collection: "chromadb.Collection", chunks: List[Chunk]) -> None:
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

        collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
        )
        logger.info("Embedding + 写入完成")

    def search(self, query: str, top_k: int = VECTOR_TOP_K) -> List[RetrievalResult]:
        """向量语义检索。

        手动 encode query → 用 query_embeddings 查询 ChromaDB
        （不使用 query_texts，确保全程可见 embedding 过程）。
        Chroma 返回的命中优先关联回内存 chunk（保持与 get_chunk / list_documents
        同一数据源），未命中的 ID 用文档内容重建 Chunk 兜底。
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

                chunk = self._by_id.get(chunk_id)
                if chunk is None:
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
