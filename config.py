"""
全局配置中心。

所有可调参数集中在这里。消融实验时只改这一个文件即可。
"""

import os
import sys

# 修复 Windows 下 Anaconda + PyTorch 的 OpenMP 重复加载问题
# 根因：torch 自带 libiomp5md.dll，Anaconda 的 MKL 也带了一份，两者冲突
if sys.platform == "win32":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pathlib import Path

# 注意：HF_HOME 环境变量已由 src/__init__.py 在第一时间设置

# ============================================================
# 路径
# ============================================================
PROJECT_ROOT = Path(__file__).parent
DOCS_DIR = PROJECT_ROOT / "docs"
CHROMA_PERSIST_DIR = PROJECT_ROOT / "chroma_db"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"

# ============================================================
# 切片参数
# ============================================================
# all-MiniLM-L6-v2 的有效编码窗口为 256 token（Sentence-BERT 微调长度），
# chunk_size 对齐模型窗口，确保向量与原文语义一致，避免尾部截断丢失信息。
CHUNK_SIZE = 256           # 每个切片的 token 数上限，对齐 embedding 模型 max_seq_len
CHUNK_OVERLAP = 38         # ≈15% of chunk_size，行业推荐 10%~20% 区间

# ============================================================
# 模型名称
# ============================================================
EMBEDDING_MODEL = "all-MiniLM-L6-v2"                       # 384 维, 轻量本地推理
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # 重排序模型

# ============================================================
# 检索参数
# ============================================================
BM25_TOP_K = 20            # BM25 召回数量
VECTOR_TOP_K = 20          # ChromaDB 向量召回数量
RRF_K = 60                 # RRF 平滑常数
CE_TOP_K = 5               # Cross-Encoder 最终返回数量
CE_THRESHOLD = 3.0           # CE 分数阈值：低于此值触发查询改写（ms-marco-MiniLM 经验值）

# ============================================================
# GraphRAG 参数
# ============================================================
GRAPH_TOP_K = 20             # 图检索召回数量
GRAPH_ENTITY_TOP_N = 12      # 每个 Chunk 提取的关键词/实体数上限
GRAPH_HOP = 1                # 图遍历跳数（1=1-hop 邻居，2=邻居的邻居）
GRAPH_RRF_WEIGHT = 0.6       # 图检索在 RRF 融合中的权重（<1.0 可降低噪声影响）

# ============================================================
# Ragas 评测参数
# ============================================================
RAGAS_LLM = "ollama/qwen2.5:7b"   # 评测用 LLM（本地 Ollama 模型）
TEST_QUERIES_FILE = PROJECT_ROOT / "data" / "test_queries.json"
TRAIN_QUERIES_FILE = PROJECT_ROOT / "data" / "train_queries.json"
KG_TRIPLES_FILE = PROJECT_ROOT / "data" / "knowledge_triples.jsonl"

# DeepSeek API
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL = "deepseek-chat"

# ============================================================
# KG Retriever 参数
# ============================================================
KG_TOP_K = 20             # KG 检索召回数量
KG_MAX_HOP = 2            # 两节点间最大路径跳数
KG_RRF_WEIGHT = 0.5      # KG 路在 RRF 融合中的权重（<1.0 降低噪声影响）
KG_TRIPLES_FILE = PROJECT_ROOT / "data" / "knowledge_triples.jsonl"
