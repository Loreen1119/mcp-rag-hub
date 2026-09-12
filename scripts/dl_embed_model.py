"""下载中文 embedding / reranker 模型到本地 HF 缓存。

endpoint 由外部环境变量 HF_ENDPOINT 控制（不设则用 huggingface.co 原站）。
DL_KIND=st（默认，SentenceTransformer）| ce（CrossEncoder reranker）
DL_MODEL=模型名
"""
import os

name = os.environ.get("DL_MODEL", "BAAI/bge-small-zh-v1.5")
kind = os.environ.get("DL_KIND", "st")
print(f"downloading {name}  kind={kind}  endpoint={os.environ.get('HF_ENDPOINT', '(default huggingface.co)')} ...")

if kind == "ce":
    from sentence_transformers import CrossEncoder

    m = CrossEncoder(name)
    print("OK (reranker)")
else:
    from sentence_transformers import SentenceTransformer

    m = SentenceTransformer(name)
    print("OK")
    print("dim =", m.get_sentence_embedding_dimension())
    print("max_seq_length =", m.max_seq_length)
