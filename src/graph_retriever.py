"""
知识图谱增强检索器（GraphRAG）。

基于文档内实体共现关系构建无向图索引，实现子图遍历与邻居扩展检索，
弥补向量/关键词检索在多跳推理与实体关联类查询上的盲区。

核心流程:
    1. 实体抽取 — jieba TF-IDF 关键词 + 词性过滤（名词优先）
    2. 图构建 — NetworkX 无向图, 节点=实体, 边=Chunk 内共现, 权重=共现次数
    3. 图检索 — Query 实体匹配(精确→模糊降级) → 1-hop 邻居扩展 → 评分排序

检索公式:
    Score(Chunk) = Σ Weight(e) / (1 + Hop(e))
    - e: 命中的实体
    - Hop(e): 0=直接命中, 1=1-hop 邻居, 权重随距离衰减
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from typing import List, Dict, Set, Tuple

import jieba
import jieba.posseg as pseg
import networkx as nx

from config import GRAPH_TOP_K, GRAPH_ENTITY_TOP_N, GRAPH_HOP
from src.models import Chunk, RetrievalResult

logger = logging.getLogger(__name__)

# ============================================================
# 实体标准化
# ============================================================

# 停用词 — 高频无信息词，过滤掉避免噪声节点
_STOP_WORDS: Set[str] = {
    "的", "了", "是", "在", "和", "与", "或", "不", "也", "都", "就",
    "要", "会", "可以", "能够", "这个", "那个", "一个", "一种", "其中",
    "使用", "进行", "通过", "以及", "用于", "对于", "基于", "利用",
    "此外", "同时", "因此", "所以", "但是", "然而", "虽然", "如果",
    "基本", "说", "来", "去", "做", "让", "把", "被", "从", "到",
    "等", "该", "并", "而", "且", "所", "上", "下", "中", "内",
}


def _normalize_entity(text: str) -> str:
    """实体标准化：去空格、去标点残留、统一小写(英文部分)。"""
    text = text.strip()
    text = re.sub(r"[，。！？、；：""'']+$", "", text)
    text = re.sub(r"^[，。！？、；：""'']+", "", text)
    if text:
        text = text[0].upper() + text[1:] if text[0].isascii() and text[0].islower() else text
    return text


def _is_valid_entity(text: str) -> bool:
    """过滤无效实体：单字、纯数字、纯标点、停用词、Markdown 标记、纯空格。"""
    if not text or len(text) < 2:
        return False
    if text in _STOP_WORDS:
        return False
    if re.match(r"^[\d\.\+\-\s]+$", text):
        return False
    if re.match(r"^[，。！？、；：""''（）\(\)\[\]【】\s]+$", text):
        return False
    # 过滤 Markdown 标题标记残留
    if re.match(r"^#+$", text):
        return False
    return True


# ============================================================
# 实体抽取
# ============================================================


def extract_entities_from_text(
    text: str,
    top_n: int = GRAPH_ENTITY_TOP_N,
) -> List[Tuple[str, float]]:
    """从单段文本中抽取实体及其重要性权重。

    两路并行抽取后合并去重:
    - TF-IDF 关键词: 基于词频-逆文档频率提取 Top-N 关键词作为实体候选
    - 词性标注: 提取名词/专有名词词组 (nr/ns/nt/nz/n/eng)，补充 TF-IDF 遗漏的领域术语

    Returns:
        [(entity_text, weight), ...]  按权重降序
    """
    # 路 1: TF-IDF 关键词
    tfidf_keywords: Dict[str, float] = {}
    try:
        import jieba.analyse
        raw_keywords = jieba.analyse.extract_tags(
            text, topK=top_n, withWeight=True
        )
        for word, weight in raw_keywords:
            normalized = _normalize_entity(word)
            if _is_valid_entity(normalized):
                tfidf_keywords[normalized] = max(tfidf_keywords.get(normalized, 0), weight)
    except Exception:
        pass

    # 路 2: 词性标注 — 提取名词词组
    pos_entities: Dict[str, float] = {}
    try:
        for word, flag in pseg.cut(text):
            if flag in ("nr", "ns", "nt", "nz", "n", "eng"):
                normalized = _normalize_entity(word)
                if _is_valid_entity(normalized):
                    pos_entities[normalized] = pos_entities.get(normalized, 0) + 1.0
    except Exception:
        pass

    # 合并: TF-IDF 权重为主，POS 词频归一化后做补充
    merged: Dict[str, float] = dict(tfidf_keywords)
    if pos_entities:
        max_pos = max(pos_entities.values())
        for entity, count in pos_entities.items():
            normalized_weight = count / max_pos * 0.5  # POS 权重系数 0.5 (低于 TF-IDF)
            if entity not in merged:
                merged[entity] = normalized_weight
            else:
                merged[entity] = max(merged[entity], normalized_weight)

    sorted_entities = sorted(merged.items(), key=lambda x: x[1], reverse=True)
    return sorted_entities[:top_n]


# ============================================================
# 知识图谱构建
# ============================================================


def build_entity_graph(
    chunks: List[Chunk],
    top_n: int = GRAPH_ENTITY_TOP_N,
) -> Tuple[nx.Graph, Dict[str, List[int]], Dict[str, float]]:
    """基于 Chunk 集合构建实体共现无向图。

    构建逻辑:
    - 节点 = 标准化后的实体文本
    - 节点属性 `global_weight` = 实体在所有 Chunk 中的重要度累加
    - 边 = 同一 Chunk 内共现的实体对, 权重 = 共现次数
    - 同时构建 entity → [chunk_idx] 倒排索引供检索时快速反查

    Returns:
        (graph, entity_to_chunks, entity_weights)
    """
    graph = nx.Graph()
    entity_to_chunks: Dict[str, List[int]] = defaultdict(list)
    entity_weights: Dict[str, float] = defaultdict(float)
    cooccurrence: Dict[Tuple[str, str], int] = defaultdict(int)

    for chunk_idx, chunk in enumerate(chunks):
        entities = extract_entities_from_text(chunk.content, top_n=top_n)
        if not entities:
            continue

        entity_names = [e for e, _ in entities]

        for e_name, e_weight in entities:
            entity_to_chunks[e_name].append(chunk_idx)
            entity_weights[e_name] += e_weight

        # 同 Chunk 内实体两两之间有共现关系
        for i in range(len(entity_names)):
            for j in range(i + 1, len(entity_names)):
                a, b = entity_names[i], entity_names[j]
                if a < b:
                    cooccurrence[(a, b)] += 1
                else:
                    cooccurrence[(b, a)] += 1

    # 建图节点
    for entity_name, weight in entity_weights.items():
        graph.add_node(entity_name, global_weight=round(weight, 4))

    # 建图边
    for (a, b), count in cooccurrence.items():
        graph.add_edge(a, b, weight=count)

    logger.info(
        "实体图构建完成 — %d 节点, %d 边, %d 条倒排索引",
        graph.number_of_nodes(),
        graph.number_of_edges(),
        len(entity_to_chunks),
    )
    return graph, dict(entity_to_chunks), dict(entity_weights)


# ============================================================
# 图检索
# ============================================================


class GraphRetriever:
    """基于实体共现图的检索器。

    接口与 BM25Retriever / VectorRetriever 一致，
    返回 List[RetrievalResult], source="graph"。
    """

    def __init__(self, chunks: List[Chunk]):
        if not chunks:
            raise ValueError("chunks 不能为空")

        self.chunks = chunks
        self.graph, self.entity_to_chunks, self.entity_weights = build_entity_graph(chunks)
        self._node_set: Set[str] = set(self.graph.nodes())

        logger.info(
            "GraphRetriever 初始化完成 — %d Chunk, %d 实体节点",
            len(chunks),
            len(self._node_set),
        )

    # ----------------------------------------------------------
    # Query 实体映射
    # ----------------------------------------------------------

    def _map_query_to_nodes(
        self,
        query_entities: List[str],
    ) -> Dict[str, List[str]]:
        """将 Query 抽取的实体映射到图节点。

        两级降级匹配策略:
        1. 精确匹配: query_entity 与节点名完全相同
        2. 模糊匹配: 子串包含 (query_entity in node 或 node in query_entity)

        Returns:
            {query_entity: [matched_node_names]}
        """
        mapping: Dict[str, List[str]] = {}

        for qe in query_entities:
            qe_normalized = _normalize_entity(qe)
            if not _is_valid_entity(qe_normalized):
                continue

            # L1: 精确匹配
            if qe_normalized in self._node_set:
                mapping[qe_normalized] = [qe_normalized]
                continue

            # L2: 子串包含匹配
            fuzzy_matches: List[str] = []
            qe_lower = qe_normalized.lower()
            for node_name in self._node_set:
                node_lower = node_name.lower()
                if qe_lower in node_lower or node_lower in qe_lower:
                    fuzzy_matches.append(node_name)

            if fuzzy_matches:
                mapping[qe_normalized] = fuzzy_matches

        return mapping

    # ----------------------------------------------------------
    # 子图扩展
    # ----------------------------------------------------------

    def _expand_subgraph(
        self,
        matched_nodes: List[str],
        hop: int = GRAPH_HOP,
    ) -> Dict[str, int]:
        """从匹配节点出发做 k-hop 邻居扩展。

        返回命中节点的 hop 距离:
        - 0: 直接匹配的 Query 实体节点
        - 1: 1-hop 邻居
        - n: n-hop 邻居

        Returns:
            {entity_name: hop_distance}
        """
        entity_hops: Dict[str, int] = {}

        # 0-hop: 直接匹配节点
        for node in matched_nodes:
            entity_hops[node] = 0

        if hop < 1:
            return entity_hops

        # BFS 逐跳扩展
        frontier = set(matched_nodes)
        visited = set(matched_nodes)

        for h in range(1, hop + 1):
            next_frontier: Set[str] = set()
            for node in frontier:
                if node not in self.graph:
                    continue
                for neighbor in self.graph.neighbors(node):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_frontier.add(neighbor)
                        entity_hops[neighbor] = h
            frontier = next_frontier
            if not frontier:
                break

        return entity_hops

    # ----------------------------------------------------------
    # Chunk 评分
    # ----------------------------------------------------------

    def _score_chunks(
        self,
        entity_hops: Dict[str, int],
    ) -> List[Tuple[int, float]]:
        """按图检索公式对 Chunk 打分排序。

        公式: Score(Chunk) = Σ Weight(e) / (1 + Hop(e))
        - Weight(e): 实体 e 在图中的全局权重 (TF-IDF 累加)
        - Hop(e): 实体距离 Query 实体的图跳数 (0=直接命中, 1=邻居)

        实现: 对每个被命中实体关联的 Chunk，累加衰减后的实体权重，
        最终按得分降序。
        """
        chunk_scores: Dict[int, float] = defaultdict(float)

        for entity, hop in entity_hops.items():
            if entity not in self.entity_to_chunks:
                continue

            e_weight = self.entity_weights.get(entity, 1.0)
            decay = e_weight / (1.0 + hop)

            for chunk_idx in self.entity_to_chunks[entity]:
                chunk_scores[chunk_idx] += decay

        sorted_chunks = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_chunks

    # ----------------------------------------------------------
    # 公开检索入口
    # ----------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = GRAPH_TOP_K,
    ) -> List[RetrievalResult]:
        """图检索入口。

        流程:
        1. 从 Query 抽取实体
        2. 实体映射到图节点 (精确→模糊)
        3. k-hop 邻居扩展
        4. Chunk 评分排序
        5. 包装为 RetrievalResult 返回
        """
        # Step 1: Query 实体抽取
        query_entities_raw = extract_entities_from_text(query, top_n=GRAPH_ENTITY_TOP_N)
        if not query_entities_raw:
            logger.debug("Query 未抽取出有效实体，图检索返回空")
            return []

        query_entity_names = [e for e, _ in query_entities_raw]

        # Step 2: 实体映射 (精确 + 模糊降级)
        node_mapping = self._map_query_to_nodes(query_entity_names)
        matched_nodes: List[str] = []
        for mapped_list in node_mapping.values():
            matched_nodes.extend(mapped_list)
        matched_nodes = list(dict.fromkeys(matched_nodes))  # 保序去重

        if not matched_nodes:
            logger.debug(
                "Query 实体未匹配到图节点: %s",
                query_entity_names[:5],
            )
            return []

        # Step 3: 子图扩展
        entity_hops = self._expand_subgraph(matched_nodes, hop=GRAPH_HOP)

        # Step 4: 评分排序
        scored = self._score_chunks(entity_hops)

        # Step 5: 包装结果
        results: List[RetrievalResult] = []
        for chunk_idx, score in scored[:top_k]:
            meta = dict(self.chunks[chunk_idx].metadata)
            meta["graph_score"] = round(score, 4)
            # 记录匹配的实体信息供调试
            chunk_entities = [
                e for e, h in entity_hops.items()
                if chunk_idx in self.entity_to_chunks.get(e, [])
            ]
            meta["matched_entities"] = chunk_entities[:10]

            results.append(RetrievalResult(
                chunk=Chunk(
                    content=self.chunks[chunk_idx].content,
                    metadata=meta,
                    chunk_id=self.chunks[chunk_idx].chunk_id,
                ),
                score=float(score),
                source="graph",
            ))

        logger.info(
            "图检索完成 — Query: '%s' | 匹配节点: %d | 扩展实体: %d | 返回: %d Chunk",
            query[:50],
            len(matched_nodes),
            len(entity_hops),
            len(results),
        )
        return results


# ============================================================
# 演示入口
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    from src.data_pipeline import process_directory

    chunks = process_directory()
    if not chunks:
        print("docs/ 目录下无文档，请先放置测试文件")
        raise SystemExit(1)

    gr = GraphRetriever(chunks)

    queries = [
        "RAG 系统中怎么样才能提高检索的准确率？",
        "混合检索和 Cross-Encoder 重排序是什么关系？",
        "BM25 算法和向量语义检索的优缺点",
    ]

    for q in queries:
        print(f"\n{'='*60}")
        print(f"  Query: {q}")
        print(f"{'='*60}")

        results = gr.search(q)
        for i, r in enumerate(results):
            entities = r.chunk.metadata.get("matched_entities", [])
            print(
                f"  [{i+1}] score={r.score:.4f}  "
                f"entities={entities[:5]}  "
                f"|  {r.chunk}"
            )
