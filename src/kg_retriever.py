"""
基于知识图谱三元组的检索器 KGRetriever。

基于 kg_builder.py 抽取的三元组构建有向图，支持多跳路径搜索与 Chunk 评分。
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import jieba.posseg as pseg
import networkx as nx

from config import KG_MAX_HOP, KG_TOP_K, KG_TRIPLES_FILE
from src.graph_retriever import extract_entities_from_text
from src.models import Chunk, RetrievalResult

logger = logging.getLogger(__name__)

# ============================================================
# 停用词（用于实体过滤）
# ============================================================

_STOP_WORDS: Set[str] = {
    "的", "了", "是", "在", "和", "与", "或", "不", "也", "都", "就",
    "要", "会", "可以", "能够", "这个", "那个", "一个", "一种", "其中",
    "使用", "进行", "通过", "以及", "用于", "对于", "基于", "利用",
    "此外", "同时", "因此", "所以", "但是", "然而", "虽然", "如果",
    "基本", "说", "来", "去", "做", "让", "把", "被", "从", "到",
    "等", "该", "并", "而", "且", "所", "上", "下", "中", "内",
}


# ============================================================
# 实体标准化
# ============================================================

def _standardize_entity(text: str) -> str:
    """标准化实体名称：去首尾空格、英文小写、去除标点残渣、移除版本号噪音。"""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"^[，。！？、；：""''（）()【】]+", "", text)
    text = re.sub(r"[，。！？、；：""''（）()【】]+$", "", text)
    text = re.sub(r"\b(v?\d+(?:\.\d+)*)\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"-+", "-", text).strip(" -_")
    if text:
        text = text[0].upper() + text[1:] if text[0].isascii() and text[0].islower() else text
    return text


def _is_valid_entity(text: str) -> bool:
    """过滤无效实体：单字、纯数字、停用词、纯标点。"""
    if not text or len(text.strip()) < 2:
        return False
    if text in _STOP_WORDS:
        return False
    if re.match(r"^[\d\.\+\-\s]+$", text):
        return False
    if re.match("^[，。！？、；：""''（）()【】\\s]+$", text):
        return False
    return True


# ============================================================
# 三元组加载与清洗
# ============================================================

def _load_and_clean_triples(
    triples_file: Path,
) -> List[dict]:
    """加载并清洗三元组。

    清洗规则：
    - 过滤 subject/relation/object 任一为空或纯空格
    - 过滤 relation 纯数字或过长（>20 字）
    - subject/object 标准化
    - relation 截断超长部分并去首尾空格
    """
    if not triples_file.exists():
        logger.warning("三元组文件不存在: %s", triples_file)
        return []

    triples: List[dict] = []

    with triples_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            source_doc = record.get("source_doc", "")

            for triple in record.get("triples", []):
                raw_subj = triple.get("subject", "")
                raw_rel = triple.get("relation", "")
                raw_obj = triple.get("object", "")

                if not raw_subj or not raw_rel or not raw_obj:
                    continue
                if not raw_subj.strip() or not raw_rel.strip() or not raw_obj.strip():
                    continue

                rel = raw_rel.strip()
                if re.match(r"^[\d\.\+\-\s]+$", rel):
                    continue
                if len(rel) > 20:
                    rel = rel[:20]

                subj = _standardize_entity(raw_subj)
                obj = _standardize_entity(raw_obj)

                if not subj or not obj:
                    continue
                if not _is_valid_entity(subj) or not _is_valid_entity(obj):
                    continue

                triples.append({
                    "subject": subj,
                    "relation": rel,
                    "object": obj,
                    "source_doc": source_doc,
                    "chunk_id": record.get("chunk_id"),
                })

    logger.info("三元组清洗完成 — %d 条有效三元组", len(triples))
    return triples


# ============================================================
# 图谱构建
# ============================================================

def _build_graph(
    triples: List[dict],
) -> nx.DiGraph:
    """从三元组构建有向图。

    - 节点 = 标准化后的实体
    - 有向边 subject → object，属性 relation
    - 同一 (subject, object) 对有多个 relation 时合并为列表
    """
    graph = nx.DiGraph()

    for triple in triples:
        s, rel, o = triple["subject"], triple["relation"], triple["object"]

        graph.add_node(s)
        graph.add_node(o)

        if graph.has_edge(s, o):
            existing = graph[s][o]["relations"]
            if rel not in existing:
                existing.append(rel)
        else:
            graph.add_edge(s, o, relations=[rel])

    logger.info(
        "图谱构建完成 — %d 节点, %d 有向边",
        graph.number_of_nodes(),
        graph.number_of_edges(),
    )
    return graph


# ============================================================
# 倒排索引构建（基于 source_doc 关联）
# ============================================================

def _build_entity_chunk_index(
    triples: List[dict],
    chunks: List[Chunk],
) -> Dict[str, Set[int]]:
    """将三元组实体关联到 Chunk 索引。

    优先使用 triple 自带的 chunk_id 精确关联到对应 Chunk；
    若 chunk_id 缺失（兼容旧缓存），再回退到 source_doc 级关联。
    """
    # chunk_id -> 索引（精确关联）
    chunk_id_to_index: Dict[str, int] = {}
    for idx, chunk in enumerate(chunks):
        if chunk.chunk_id:
            chunk_id_to_index[chunk.chunk_id] = idx

    # source_doc -> chunk indices（兼容旧数据）
    doc_to_indices: Dict[str, Set[int]] = defaultdict(set)
    for idx, chunk in enumerate(chunks):
        source = chunk.metadata.get("source", "")
        if source:
            doc_to_indices[source].add(idx)

    entity_to_chunks: Dict[str, Set[int]] = defaultdict(set)

    for triple in triples:
        subj, obj = triple["subject"], triple["object"]
        chunk_id = triple.get("chunk_id")

        if chunk_id and chunk_id in chunk_id_to_index:
            ci = chunk_id_to_index[chunk_id]
            entity_to_chunks[subj].add(ci)
            entity_to_chunks[obj].add(ci)
        else:
            # 兼容旧缓存：按 source_doc 关联全文所有 chunks
            doc = triple.get("source_doc", "")
            if doc in doc_to_indices:
                for ci in doc_to_indices[doc]:
                    entity_to_chunks[subj].add(ci)
                    entity_to_chunks[obj].add(ci)

    return dict(entity_to_chunks)


# ============================================================
# Query 实体抽取
# ============================================================

def _extract_query_entities(query: str) -> List[str]:
    """从 Query 中抽取名词性实体，与图节点保持一致的标准化处理。"""
    raw_entities = extract_entities_from_text(query, top_n=20)
    entities = []
    for entity, _ in raw_entities:
        std = _standardize_entity(entity)
        if _is_valid_entity(std):
            entities.append(std)
    return entities


# ============================================================
# 节点匹配
# ============================================================

def _find_matched_nodes(
    entities: List[str],
    all_nodes: Set[str],
) -> List[str]:
    """将 Query 实体匹配到图节点：精确匹配 + 子串包含匹配。"""
    matched: Set[str] = set()

    for e in entities:
        e_lower = e.lower()
        # L1: 精确匹配
        if e in all_nodes:
            matched.add(e)
            continue
        # L2: 子串包含
        for node in all_nodes:
            if e_lower in node.lower() or node.lower() in e_lower:
                matched.add(node)

    return list(matched)


# ============================================================
# 路径搜索
# ============================================================

def _search_paths(
    nodes: List[str],
    graph: nx.DiGraph,
    max_hop: int = KG_MAX_HOP,
) -> List[Tuple[List[str], int]]:
    """对节点两两找最短路径。

    Returns:
        [(path_nodes, path_length), ...]
        path_length = 路径边数（hop 数）
    """
    paths: List[Tuple[List[str], int]] = []
    for i in range(len(nodes)):
        for j in range(len(nodes)):
            if i == j:
                continue
            try:
                path = nx.shortest_path(graph, nodes[i], nodes[j])
                length = len(path) - 1
                if length <= max_hop:
                    paths.append((path, length))
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                pass
    return paths


# ============================================================
# Chunk 评分
# ============================================================

def _score_chunks(
    matched_nodes: List[str],
    paths: List[Tuple[List[str], int]],
    entity_to_chunks: Dict[str, Set[int]],
) -> List[Tuple[int, float]]:
    """对 Chunk 评分并归一化到 [0, 1]。"""
    chunk_scores: Dict[int, float] = defaultdict(float)

    # 收集路径上所有中间节点
    path_entities: Set[str] = set()
    for path, _ in paths:
        for node in path:
            path_entities.add(node)

    all_relevant = set(matched_nodes) | path_entities

    for entity in all_relevant:
        if entity not in entity_to_chunks:
            continue
        is_direct = entity in matched_nodes
        base_score = 1.0 if is_direct else 0.5

        for ci in entity_to_chunks[entity]:
            chunk_scores[ci] += base_score

    if not chunk_scores:
        return []

    max_score = max(chunk_scores.values())
    normalized = {ci: s / max_score for ci, s in chunk_scores.items()}
    return sorted(normalized.items(), key=lambda x: x[1], reverse=True)


# ============================================================
# KGRetriever
# ============================================================

class KGRetriever:
    """基于知识图谱三元组的检索器。"""

    def __init__(
        self,
        chunks: List[Chunk],
        triples_file: Optional[Path] = None,
    ):
        if not chunks:
            raise ValueError("chunks 不能为空")

        self.chunks = chunks
        self.triples_file = triples_file or KG_TRIPLES_FILE

        # 加载三元组并构建图
        self.triples = _load_and_clean_triples(self.triples_file)
        self.graph = _build_graph(self.triples)

        # 基于 source_doc 构建实体→Chunk 倒排索引
        self.entity_to_chunks = _build_entity_chunk_index(self.triples, chunks)

        self._all_nodes: Set[str] = set(self.graph.nodes())

        logger.info(
            "KGRetriever 初始化完成 — %d Chunk, %d 三元组, %d 图节点, %d 实体索引",
            len(chunks),
            len(self.triples),
            len(self._all_nodes),
            len(self.entity_to_chunks),
        )

    def search(
        self,
        query: str,
        top_k: int = KG_TOP_K,
    ) -> List[RetrievalResult]:
        """知识图谱检索入口。"""
        # Step 1: Query 实体抽取
        query_entities = _extract_query_entities(query)
        if not query_entities:
            logger.debug("Query 未抽取出有效实体")
            return []

        # Step 2: 节点匹配
        matched = _find_matched_nodes(query_entities, self._all_nodes)
        if not matched:
            logger.debug("Query 实体未匹配到图节点: %s", query_entities[:5])
            return []

        # Step 3: 路径搜索（>=2 节点时）
        if len(matched) >= 2:
            paths = _search_paths(matched, self.graph, KG_MAX_HOP)
        else:
            paths = []

        # Step 4: Chunk 评分
        scored = _score_chunks(
            matched_nodes=matched,
            paths=paths,
            entity_to_chunks=self.entity_to_chunks,
        )

        # Step 5: 包装结果
        results: List[RetrievalResult] = []
        for chunk_idx, score in scored[:top_k]:
            chunk = self.chunks[chunk_idx]
            meta = dict(chunk.metadata)
            meta["kg_score"] = round(score, 4)
            meta["kg_matched_entities"] = matched[:10]
            results.append(RetrievalResult(
                chunk=Chunk(
                    content=chunk.content,
                    metadata=meta,
                    chunk_id=chunk.chunk_id,
                ),
                score=float(score),
                source="kg",
            ))

        logger.info(
            "KG 检索完成 — Query: '%s' | 匹配节点: %s | 路径: %d | 返回: %d Chunk",
            query[:50],
            matched[:5],
            len(paths),
            len(results),
        )
        return results


# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
    )

    from src.data_pipeline import process_directory

    chunks = process_directory()
    if not chunks:
        print("docs/ 目录下无文档，请先放置测试文件")
        raise SystemExit(1)

    kg = KGRetriever(chunks)

    queries = [
        "BM25 和 Cross-Encoder 在 RAG 中怎么配合？",
        "混合检索和重排序是什么关系？",
        "Embedding 模型的选择",
    ]

    for q in queries:
        print(f"\n{'='*60}")
        print(f"  Query: {q}")
        print(f"{'='*60}")

        results = kg.search(q, top_k=5)
        for i, r in enumerate(results):
            print(
                f"  [{i+1}] score={r.score:.4f}  "
                f"entities={r.chunk.metadata.get('kg_matched_entities', [])}  "
                f"| {r.chunk}"
            )
            print(f"       {r.chunk.content[:80].replace(chr(10), ' ')}")
