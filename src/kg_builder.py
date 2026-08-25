"""
知识图谱三元组抽取。

调用 DeepSeek-V3 API 从每个 Chunk 中提取 (subject, relation, object) 三元组，
缓存到 data/knowledge_triples.jsonl（每行一个 Chunk 的结果）。
已处理的 chunk_id 自动跳过，避免重复调用 API。
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import time
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

from config import (
    DOCS_DIR,
    KG_TRIPLES_FILE,
    KG_TRIPLES_META_FILE,
    CHUNK_ID_RULE_VERSION,
    KG_META_VERSION,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
)
from src.data_pipeline import corpus_hash
from src.pipeline import get_document_chunks
from src.models import Chunk

logger = logging.getLogger(__name__)

# ============================================================
# Prompt 模板
# ============================================================

_EXTRACT_PROMPT = """请从以下技术文档片段中提取实体之间的关系三元组。
每个三元组格式为 (subject, relation, object)。
只提取明确在文本中体现的关系，不要编造。
subject 和 object 应为技术术语、组件名称或核心概念。
relation 用简洁的动词或动词短语表示。

输出必须是 JSON 数组，例如：
[
  {{"subject": "BM25", "relation": "用于", "object": "关键词召回"}},
  {{"subject": "Cross-Encoder", "relation": "解决", "object": "精排"}}
]

文本：
{chunk_content}"""

# ============================================================
# 实体标准化
# ============================================================

_VERSION_PATTERN = re.compile(
    r"\b(v?\d+(?:\.\d+)*(?:[-_]?\w+)?)\b", re.IGNORECASE
)


def normalize_entity(text: str) -> str:
    """标准化实体名称：去首尾空格、统一大小写、移除版本号等噪音。"""
    text = text.strip()
    # 统一英文为小写（保留大小写敏感的技术名）
    # 仅对全大写的缩写词做小写归一（如 HNSW → hnsw → HNSW 保持原样）
    # 简单策略：去首尾空格 + 折叠多余空白
    text = re.sub(r"\s+", " ", text)
    # 移除常见版本/版本号噪音
    text = _VERSION_PATTERN.sub("", text)
    # 去掉残留的连续空格和首尾横线
    text = re.sub(r"-+", "-", text).strip(" -_")
    return text


def normalize_triple(triple: dict) -> dict:
    """对三元组的 subject 和 object 做标准化。"""
    return {
        "subject": normalize_entity(triple.get("subject", "")),
        "relation": triple.get("relation", "").strip(),
        "object": normalize_entity(triple.get("object", "")),
    }


# ============================================================
# API 调用
# ============================================================

_MAX_RETRIES = 3


def extract_triples(chunk: Chunk, client: OpenAI) -> list[dict]:
    """调用 DeepSeek API 提取三元组，支持重试。"""
    prompt = _EXTRACT_PROMPT.format(chunk_content=chunk.content)

    last_error: Exception | None = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "你是一个专业的技术知识图谱构建助手，擅长从技术文档中精确提取实体关系三元组。",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=1024,
            )
            raw = response.choices[0].message.content or ""

            # 尝试从 markdown 代码块中提取 JSON
            json_match = re.search(
                r"```(?:json)?\s*\n?(.*?)\n?```",
                raw,
                re.DOTALL,
            )
            if json_match:
                raw = json_match.group(1)

            triples = json.loads(raw)
            if not isinstance(triples, list):
                logger.warning(
                    "API 返回非数组格式 [chunk=%s]: type=%s",
                    chunk.chunk_id,
                    type(triples).__name__,
                )
                return []

            # 过滤空三元组并标准化
            normalized = [
                normalize_triple(t)
                for t in triples
                if t.get("subject") and t.get("relation") and t.get("object")
            ]
            return normalized

        except json.JSONDecodeError as exc:
            last_error = exc
            logger.warning(
                "JSON 解析失败 [attempt %d/%d, chunk=%s]: %s",
                attempt,
                _MAX_RETRIES,
                chunk.chunk_id,
                exc,
            )
        except Exception as exc:
            last_error = exc
            logger.warning(
                "API 调用失败 [attempt %d/%d, chunk=%s]: %s",
                attempt,
                _MAX_RETRIES,
                chunk.chunk_id,
                exc,
            )

        if attempt < _MAX_RETRIES:
            wait = 2 ** attempt
            logger.info("等待 %.1f 秒后重试 ...", wait)
            time.sleep(wait)

    logger.error("API 调用彻底失败 [chunk=%s]，跳过: %s", chunk.chunk_id, last_error)
    return []


# ============================================================
# 缓存读写（sidecar 校验）
# ============================================================


def _load_sidecar() -> dict | None:
    """读取 KG 缓存 sidecar；缺失或解析失败返回 None（视为不可验证）。"""
    if not KG_TRIPLES_META_FILE.exists():
        return None
    try:
        return json.loads(KG_TRIPLES_META_FILE.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("KG sidecar 读取失败，按全量重建处理: %s", exc)
        return None


def _write_sidecar(corpus_hash: str, docs_dir: str) -> None:
    """写入 KG 缓存 sidecar（临时文件 + 原子替换）。

    调用方应保证 JSONL 已成功落盘。sidecar 损坏/缺失会被视为缓存不可验证，
    下次安全地全量重建，因此原子性主要避免读到一个半写的 meta。
    """
    meta = {
        "docs_dir": docs_dir,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "chunk_id_rule_version": CHUNK_ID_RULE_VERSION,
        "corpus_hash": corpus_hash,
        "meta_version": KG_META_VERSION,
    }
    try:
        tmp = KG_TRIPLES_META_FILE.with_suffix(KG_TRIPLES_META_FILE.suffix + ".tmp")
        tmp.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        os.replace(tmp, KG_TRIPLES_META_FILE)
    except Exception as exc:
        logger.warning("KG sidecar 写入失败（下次将全量重建）: %s", exc)


def _cache_is_valid(corpus_hash: str) -> bool:
    """判断缓存是否可复用：JSONL 存在 + sidecar 版本/规则/内容 hash 全部匹配。

    chunk_id 只依赖（文件名, 序号），与内容无关——所以仅 ID 对上不能说明
    内容未变，必须用 corpus_hash 校验。任一条件不满足 → 全量重建。
    """
    if not KG_TRIPLES_FILE.exists():
        return False
    meta = _load_sidecar()
    if not meta:
        return False
    return (
        meta.get("meta_version") == KG_META_VERSION
        and meta.get("chunk_id_rule_version") == CHUNK_ID_RULE_VERSION
        and meta.get("corpus_hash") == corpus_hash
    )


def _load_cache_ids() -> set[str]:
    """读取已缓存的 chunk_id 集合。"""
    cache: set[str] = set()
    if not KG_TRIPLES_FILE.exists():
        return cache

    with KG_TRIPLES_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                if record.get("chunk_id"):
                    cache.add(record["chunk_id"])
            except json.JSONDecodeError:
                continue

    logger.info("缓存读取完成，共 %d 条已处理记录", len(cache))
    return cache


def _tmp_path() -> Path:
    return KG_TRIPLES_FILE.with_suffix(KG_TRIPLES_FILE.suffix + ".tmp")


def _write_line(fh, chunk_id: str, source_doc: str, triples: list[dict]) -> None:
    """向文件句柄追加一条三元组记录。"""
    record = {
        "chunk_id": chunk_id,
        "source_doc": source_doc,
        "triples": triples,
    }
    fh.write(json.dumps(record, ensure_ascii=False) + "\n")


# ============================================================
# 主流程
# ============================================================


def build_knowledge_graph(
    docs_dir: Path | str = DOCS_DIR,
    verbose: bool = True,
) -> dict:
    """构建知识图谱。

    Returns:
        {
            "total": int,       # 总 chunk 数
            "skipped": int,     # 缓存跳过数
            "processed": int,   # 实际调用 API 数
            "total_triples": int,  # 累计抽取三元组数
            "failed": int,      # API 失败数
        }
    """
    # API Key 检查
    if not DEEPSEEK_API_KEY:
        raise RuntimeError(
            "未设置 DEEPSEEK_API_KEY 环境变量。\n"
            "请在终端执行:  $env:DEEPSEEK_API_KEY='your-key-here'"
        )

    client = OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
    )

    # 加载 chunks（只切片，不加载 embedding 模型）
    chunks = get_document_chunks(docs_dir, CHUNK_SIZE, CHUNK_OVERLAP)
    if not chunks:
        logger.warning("未找到任何 Chunk，退出。")
        return {"total": 0, "skipped": 0, "processed": 0, "total_triples": 0, "failed": 0}

    # 校验缓存：规则版本 + 内容 hash 任一不匹配 → 全量重建
    cur_hash = corpus_hash(docs_dir, CHUNK_SIZE, CHUNK_OVERLAP)
    if _cache_is_valid(cur_hash):
        cache = _load_cache_ids()
        # 以现有缓存文件为基底，追加到临时文件
        shutil.copyfile(KG_TRIPLES_FILE, _tmp_path())
        rebuild_reason = None
    else:
        cache = set()
        # 清空临时文件，从零构建
        _tmp_path().write_text("", encoding="utf-8")
        rebuild_reason = "缓存缺失、规则版本或内容 hash 不匹配"

    pending = [c for c in chunks if c.chunk_id not in cache]

    stats = {"total": len(chunks), "skipped": len(cache), "processed": 0, "total_triples": 0, "failed": 0}

    if verbose:
        print(f"\n>>> 总 Chunk 数  : {stats['total']}")
        if rebuild_reason:
            print(f">>> 缓存重建    : {rebuild_reason}")
        print(f">>> 缓存命中    : {stats['skipped']} (跳过)")
        print(f">>> 待处理      : {len(pending)}")
        if stats["skipped"] > 0:
            print(f">>> 缓存文件    : {KG_TRIPLES_FILE}")

    if not pending:
        logger.info("所有 Chunk 均已缓存，无需处理。")
        return stats

    # 抽取结果写入临时文件（不直接追加原文件，保证原子替换）
    with _tmp_path().open("a", encoding="utf-8") as f:
        for chunk in tqdm(pending, desc="抽取三元组", disable=not verbose):
            triples = extract_triples(chunk, client)
            _write_line(
                f,
                chunk_id=chunk.chunk_id,
                source_doc=chunk.metadata.get("source", ""),
                triples=triples,
            )
            if triples:
                stats["processed"] += 1
                stats["total_triples"] += len(triples)
            else:
                stats["failed"] += 1

    # 先原子替换 JSONL，成功后才写 sidecar
    os.replace(_tmp_path(), KG_TRIPLES_FILE)
    _write_sidecar(cur_hash, str(docs_dir))

    if verbose:
        print(f"\n>>> 完成！")
        print(f"    实际处理 : {stats['processed']} 个 Chunk")
        print(f"    抽取失败 : {stats['failed']} 个 Chunk")
        print(f"    三元组总数: {stats['total_triples']}")
        print(f"    输出文件 : {KG_TRIPLES_FILE}")

    return stats


# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="知识图谱三元组抽取")
    parser.add_argument(
        "--docs",
        type=str,
        default=None,
        help="文档目录路径（默认使用 config.DOCS_DIR）",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="静默模式，不显示进度条",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
    )

    docs_dir = Path(args.docs) if args.docs else DOCS_DIR
    build_knowledge_graph(docs_dir=docs_dir, verbose=not args.quiet)
