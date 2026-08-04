---
title: "LightRAG 学习笔记：可借鉴到 mcp-rag-hub 的要点"
status: "概念评估 / 待实验"
last_updated: "2026-08-04"
source: "LightRAG-main"
---

# LightRAG 学习笔记：可借鉴到 mcp-rag-hub 的要点

> 学习方式：Dify 导入 README 做问答 + 核对源码 prompt/operate.py  
> 当前状态：**概念层已理解，尚未在 mcp-rag-hub 中实现**。源码层只看了入口函数，深度还不够。

---

## LightRAG 与传统 RAG 的核心差异

传统 RAG：文本切分 → 向量化 → 向量检索 → 生成。  
LightRAG：在向量检索之外，额外构建 **实体-关系-文本块** 三层索引：

- **实体节点**：带 type + description + 向量
- **关系边**：带 keywords + description + 向量
- **文本块**：保留原始向量索引

查询时不再只搜文本块，而是同时去图里搜实体和关系。

---

## 最有价值的两个设计

### 1. 查询关键词分层（low-level / high-level）

`lightrag/prompt.py:488-490` 明确定义：

- **low-level keywords**：具体实体、术语、名词（如 `BM25`、`HybridRetriever`）
- **high-level keywords**：主题、意图、问题类型（如 `区别`、`实现方式`、`优缺点`）

用途：

- low-level → `_get_node_data()` → 查实体节点 → **local query**
- high-level → `_get_edge_data()` → 查关系边 → **global query**

这和 mcp-rag-hub 当前做法的差异：

| | mcp-rag-hub | LightRAG |
|---|---|---|
| 从 query 抽什么 | 只抽实体 | 同时抽实体 + 主题/关系词 |
| 查图里的什么 | 节点（实体） | 节点 + 边 |
| 查询模式 | 单一 KG 路 | local / global / hybrid / mix |

### 2. local / global 双查询策略

- **local**：从具体实体出发，扩展它的直接邻居和关联文本块。适合 `"BM25 是什么？"` `"HybridRetriever 有哪些方法？"`
- **global**：从关系/主题出发，沿关系链遍历多个实体。适合 `"LightRAG 和传统 RAG 有什么区别？"` `"这个系统的优缺点是什么？"`
- **hybrid**：两路合并。
- **mix**：local + global + naive 三路合并，最全面也最贵。

**mode 是人指定的配置参数**，默认 `"mix"`。想要自动选 mode，需要在外层自己加 Router（规则 / LLM / 分类器）。

---

## mcp-rag-hub 可探索的 3 个改进方向（尚未实现）

### 改进 1：给三元组加描述

当前：`kg_builder.py` 只抽 `(subject, relation, object)`。  
建议：让 LLM 同时输出：

```json
{
  "subject": "BM25",
  "subject_description": "基于词频统计的稀疏检索算法",
  "relation": "用于",
  "object": "关键词召回",
  "object_description": "从文档中检索包含查询词的候选文本块",
  "relationship_keywords": "检索, 召回, 关键词匹配"
}
```

收益：节点和边有语义描述后，向量检索更准；关系关键词可用于 global query。

### 改进 2：查询时抽 high-level / low-level 关键词

当前：`kg_retriever.py` 只从 query 抽实体。  
建议：增加一个 prompt，把 query 拆成两类关键词：

```json
{
  "low_level_keywords": ["BM25", "向量检索", "HybridRetriever"],
  "high_level_keywords": ["并行执行", "实现方式", "工作流程"]
}
```

然后：

- low-level → 实体匹配（现有逻辑）
- high-level → 关系关键词匹配（新增）

### 改进 3：实现 local / global 两种查询模式

当前：KG 路只有一套逻辑。  
建议：把 `kg_retriever.py` 拆成：

- `local_search(query, entities)`：从命中实体出发，扩展 1-hop 邻居，返回相关 chunks
- `global_search(query, keywords)`：从关系关键词出发，找相关关系边，沿边聚合多实体信息
- `hybrid_search`：两者融合

实现成本中等，但能让 KG 路从“补充召回”升级为“结构化推理”。

---

## 当前学习的局限

通过 Dify 读 README 只能得到**概念层**理解：

- ✅ 知道 local/global 是什么
- ✅ 知道关键词分层的设计
- ✅ 知道图构建的基本思路
- ❌ 没看具体 prompt 怎么写（只看了一眼定义）
- ❌ 没看 `_get_node_data` / `_get_edge_data` 的具体实现
- ❌ 没看上下文怎么组装、token 怎么截断
- ❌ 没跑实验对比效果

**要深入，必须读源码**。重点函数（行号来自当时拉取的源码快照，后续版本可能会漂移，建议以函数名搜索）：

- `lightrag/operate.py` `kg_query()` — 查询总入口
- `lightrag/operate.py` `_perform_kg_search()` — local/global 检索逻辑
- `lightrag/operate.py` `_build_query_context()` — 上下文组装
- `lightrag/prompt.py` `keywords_extraction` — 关键词抽取 prompt

---

## 源码层理解（operate.py 关键函数）

### `_perform_kg_search()` — 检索调度中心

`lightrag/operate.py:4551`

根据 `query_param.mode` 决定走哪条路：

```python
if mode == "local" and ll_keywords:
    local_entities, local_relations = await _get_node_data(ll_keywords, ...)
elif mode == "global" and hl_keywords:
    global_relations, global_entities = await _get_edge_data(hl_keywords, ...)
else:  # hybrid / mix
    if ll_keywords:
        local_entities, local_relations = await _get_node_data(ll_keywords, ...)
    if hl_keywords:
        global_relations, global_entities = await _get_edge_data(hl_keywords, ...)
```

关键点：**它会预先把 query / low-level / high-level 三个文本分别 embedding**，避免多次调用 embedding API。

### `_get_node_data()` — local query 的具体实现

`lightrag/operate.py:5396`

1. 用 `entities_vdb.query(ll_keywords)` 做向量检索，召回最相关的实体节点
2. 批量取节点属性（description、type 等）和节点度数（rank）
3. 对每个命中实体，调用 `_find_most_related_edges_from_entities()` 找它的出边/入边
4. 边按 `(rank, weight)` 排序，rank 是边两端节点的度数综合，weight 是共现/语义权重

**本质**：实体 → 邻居 → 关系 → 文本块。

### `_get_edge_data()` — global query 的具体实现

`lightrag/operate.py:5671`

1. 用 `relationships_vdb.query(hl_keywords)` 做向量检索，召回最相关的关系边
2. 批量取边属性（description、keywords、weight、source_id 等）
3. 调用 `_find_most_related_entities_from_relationships()` 从边的 src/tgt 反推实体

**本质**：关系/主题 → 相关边 → 边两端的实体 → 文本块。

### `_build_query_context()` — 上下文组装

`lightrag/operate.py:5273`

四阶段流水线：

1. **Search**：调用 `_perform_kg_search` 拿到 entities/relations/chunks
2. **Truncate**：按 token 预算截断 entities/relations/chunks
3. **Merge chunks**：把实体关联的 chunks 和关系关联的 chunks 合并去重
4. **Build context**：把实体描述、关系描述、文本块拼接成最终 prompt

这比你项目目前的“直接返回 chunk 列表”更工程化：它控制了上下文长度，避免了 prompts 爆掉。

---

## mcp-rag-hub 与 LightRAG 的实现差异（更具体）

| 环节 | mcp-rag-hub | LightRAG |
|---|---|---|
| **图构建** | NetworkX DiGraph，边只有 `relations` 列表 | 抽象 `BaseGraphStorage`，节点/边都有完整属性 + 向量索引 |
| **实体关联 chunks** | 通过 `source_doc` 间接关联 | 实体/关系直接存储 `source_id`（chunk IDs 列表） |
| **查询入口** | jieba 抽实体 → 精确/子串匹配节点 | embedding 检索实体/关系向量 |
| **邻居扩展** | 最短路径 `_search_paths` | 直接取节点相连的所有边 |
| **得分依据** | 直接命中 1.0，路径节点 0.5，简单求和 | 节点度数 rank + 边 weight + 向量相似度 |
| **上下文控制** | 无，直接返回 top chunks | 四阶段组装 + token 截断 |

LightRAG 比你当前实现重的点：

1. **存储层更复杂**：实体/关系都要单独做向量存储
2. **检索层更细**：用向量相似度而不是字符串匹配
3. **上下文管理层更完整**：有明确的 token 预算控制

但也并不是所有都要抄。你当前实现的**轻量**是优势。

---

## 建议的“最小可行升级”

不要一次性把 LightRAG 全搬过来。先做一个最小升级：

### 阶段 1：在现有架构上加 local/global（1-2 天）

1. **给 `kg_builder.py` 增加 `relationship_keywords`**
   - 在 prompt 里要求 LLM 为每个关系输出 1-3 个概括性关键词
   - 存到 `knowledge_triples.jsonl`

2. **给 `kg_retriever.py` 加关键词分层 prompt**
   - 新增 `_extract_keywords(query)` 返回 `{"low_level": [...], "high_level": [...]}`

3. **实现 local/global/hybrid 三种检索模式**
   - `local`：用 low-level 关键词匹配实体节点（保留现有 `_find_matched_nodes`）
   - `global`：用 high-level 关键词匹配 `relationship_keywords`
   - `hybrid`：两路结果 RRF 融合

4. **在 `mcp_server.py` 暴露 mode 参数**
   - 让检索工具支持 `mode=local|global|hybrid`

### 阶段 2：评估效果（1 天）

- 在 test set 里加几类问题：
  - 事实性问题 → 预期 local 更好
  - 对比/总结问题 → 预期 global 更好
  - 混合问题 → 预期 hybrid 更好
- 跑一遍看 MRR 变化

### 阶段 3（可选）：向量化实体/关系

- 如果阶段 1 效果不够，再给实体/关系做 embedding
- 这是大改动，放到后面

---

## 面试可用的一句话

> "我参考 LightRAG 的设计，规划把 KG 检索从单一的实体匹配升级为 local/global 分层检索：low-level 关键词定位实体节点，high-level 关键词定位关系边，分别处理事实性问题和宏观对比性问题。"
