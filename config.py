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
CHUNK_SIZE = 512          # 每个切片的 token 数上限
CHUNK_OVERLAP = 128       # 相邻切片的重叠 token 数

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
# Ragas 评测参数
# ============================================================
RAGAS_LLM = "ollama/qwen2.5:7b"   # 评测用 LLM（本地 Ollama 模型）
TEST_QUERIES_FILE = PROJECT_ROOT / "data" / "test_queries.json"
