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

# 知识库语料根目录：每份语料一个子目录（见 corpora/README.md）。
# 用环境变量 MCP_RAG_CORPUS 切换当前生效的语料，各自独立的 Chroma collection 互不覆盖。
# 注意：PROJECT_ROOT/"docs" 是项目自身文档（人读的），不再参与索引。
CORPUS_ROOT = PROJECT_ROOT / "corpora"
CORPUS = os.environ.get("MCP_RAG_CORPUS", "fastapi-zh")
DOCS_DIR = CORPUS_ROOT / CORPUS
CHROMA_PERSIST_DIR = PROJECT_ROOT / "chroma_db"

# 索引 schema 版本：改变 chunk 参数 / ID 规则 / collection 元数据含义时递增，
# 旧版本 collection 不会在新版本下被误复用（版本化 collection 命名的一部分）。
INDEX_SCHEMA_VERSION = 2
# chunk_id 生成规则版本：make_chunk_id 算法变化时递增，
# 用于 KG 缓存 sidecar 判定旧缓存是否仍可用（只依赖文件名+序号，与内容无关）。
# v3：chunk_id 的 source 参数由「文件名」改为「相对语料根的路径」——
# 嵌套语料里 index.md 等重名文件会撞出相同 ID（FastAPI 中文档里有 11 个 index.md）。
CHUNK_ID_RULE_VERSION = 3
# KG 缓存 sidecar 格式版本：sidecar 字段含义变化时递增。
KG_META_VERSION = 1
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
# experiments.py 独立索引目录：与生产 chroma_db 隔离，防止实验删掉线上集合。
EXPERIMENTS_CHROMA_DIR = EXPERIMENTS_DIR / "chroma_db"

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
# 中文知识库必须用中文/多语言 embedding。all-MiniLM-L6-v2 是纯英文模型，
# 对中文的余弦相似度全部挤在 0.24~0.37 的窄区间、毫无区分度 —— 实测会把
# 无关文档排到正确 chunk 前面（正确 chunk 只排第 10）。改用中文优化的
# bge-small-zh-v1.5：512 维，C-MTEB 中文检索表现优秀，仅 ~95MB，CPU 推理快。
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"                    # 512 维, 中文优化
# CE 也是英文基座的话，对中文 query-doc 判定不准（实测把无关文档打成第 2 名）。
# 换中文优化的 bge-reranker-base：C-MTEB reranking 强，~1.1GB。
CROSS_ENCODER_MODEL = "BAAI/bge-reranker-base"                # 中文优化 reranker

# 生成答案用的本地 Ollama 模型。
# 选 3b 而非 7b 的原因（本机实测）：GPU 是 Intel UHD 集显、无 CUDA，只能纯 CPU 推理。
#   - qwen2.5:7b 权重 4.36 GB，加上 KV cache 与计算缓冲需 5.0~5.5 GB，
#     开发环境全开时 Commit Free 只剩约 3.9 GB，加载必 OOM；即便腾出内存，
#     CPU 推理仅 2~4 tok/s，300 字答案要 50~100 秒，演示不可用。
#   - qwen2.5:3b 权重 1.80 GB，总占用约 2.4~2.8 GB，CPU 约 8~15 tok/s，
#     15~25 秒出答案。本项目的生成任务是"给 3 条证据做抽取归纳 + 标引用 + 输出 JSON"，
#     属窄任务，3b 足够。
LLM_MODEL = "qwen2.5:3b"

# LLM-as-Judge 评测的裁判模型。裁判比生成更需要模型能力：评测时要「逐句比对答案与
# 上下文」并给出符合评分标准的分数，3b 做不到 —— 实测出现自相矛盾的判词（同一句话里
# 说「大部分信息来自上下文」却按「大量编造」档打 0.10 分）。项目早期已记录过同一结论：
# 3b 当裁判无区分度，换 7b 后 Faithfulness 才从 0.23 拉到 0.95。
# 因此生成与裁判解耦：生成用 3b（窄任务、CPU 可跑），裁判用 7b（跑评测时才加载）。
JUDGE_MODEL = "qwen2.5:7b"

# ============================================================
# 检索参数
# ============================================================
BM25_TOP_K = 20            # BM25 召回数量
VECTOR_TOP_K = 20          # ChromaDB 向量召回数量
RRF_K = 60                 # RRF 平滑常数
CE_TOP_K = 5               # Cross-Encoder 最终返回数量
# bge-reranker-base 经 sigmoid 输出 0~1（0.5 为判定中点），与英文 ms-marco 的 logits
# 量纲完全不同，故阈值从 3.0 调到 0.3。此阈值仅用于「是否触发查询改写」的质量门控。
CE_THRESHOLD = 0.3
# 相对阈值：CE 结果中低于 top1 分数该比例的视为噪声丢弃。动机：CE_TOP_K=5 硬取前 5 会把
# 无关 chunk 塞进 context（实测 bge-reranker 下正解 0.699 vs 噪声 0.0007，差 1000 倍）。
CE_RELATIVE_THRESHOLD = 0.2
ANSWER_TOP_K = 3
RETRIEVAL_CHECK_TOP_K = 5
MIN_EVIDENCE_COUNT = 1
# 超时必须远大于 CPU 推理耗时：3b 出 300 字答案约 15~25 秒，7b 约 50~100 秒。
# 此前这个值没被 _call_llm 读取（死配置），且 20 秒对纯 CPU 推理必然超时。
LLM_TIMEOUT_SECONDS = 120
LLM_MAX_RETRIES = 1          # 仅对连接失败 / 超时重试；JSON 解析失败不重试

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
# 单一 Golden Test Set（18 条，四类分层：exact_match / semantic / mixed / graph）。
# 历史说明：曾存在 train/test 拆分（train_queries.json + test_queries_all.json），
# 那是 Phase 0「用 CE 分数做拒答门控」调参用的。该方案已被证伪并删除
# （见 MEMORY.md「设计决策」与 docs_knowledge/开发过程中遇到的问题.md），
# 拆分随之失去意义 —— 不存在需要留出训练集不看的调参流程，故收敛为单一测试集。
TEST_QUERIES_FILE = PROJECT_ROOT / "data" / "test_queries.json"
KG_TRIPLES_FILE = PROJECT_ROOT / "data" / "knowledge_triples.jsonl"
# KG 缓存 sidecar：记录缓存对应的 chunk_id_rule_version + corpus_hash，
# 用于判断旧缓存是否仍可复用（规则/内容任一变化 → 全量重建）。
KG_TRIPLES_META_FILE = PROJECT_ROOT / "data" / "knowledge_triples.jsonl.meta"

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
# KG_TRIPLES_FILE 定义在上方「Ragas 评测参数」区（与 KG_TRIPLES_META_FILE 相邻），此处不再重复。
ENABLE_KG = os.environ.get("MCP_RAG_ENABLE_KG", "false").lower() in ("1", "true", "yes")  # 默认关闭 KG 路
