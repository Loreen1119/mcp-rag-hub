"""
mcp-rag-hub — RAG 智能知识检索系统
"""

import sys
import os
from pathlib import Path

# 修复 Windows 下 Anaconda + PyTorch 的 OpenMP 重复加载问题
# 必须在任何 torch 相关 import 之前执行
if sys.platform == "win32":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# HuggingFace 模型缓存路径 — 必须在任何 huggingface_hub import 之前设置
_HF_CACHE = Path("D:/huggingface_cache")
if _HF_CACHE.parent.exists():
    _HF_CACHE.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(_HF_CACHE))

# 强制离线模式 — 模型已缓存在本地，不需要每次连 huggingface.co 检查更新
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
