"""Read-only audit for the persisted Chroma index.

Usage:
    python scripts/audit_chroma_index.py [--corpus fastapi-zh]

This script does not rebuild, delete, or write to the index.  It only reads
Chroma metadata/content and, optionally, loads the existing pipeline to compare
the runtime chunk count.

集合命名已随多语料改造从 knowledge_base__{模型}__v{schema}__{hash8} 改为
kb__{语料}__{模型}__v{schema}，故本脚本按「语料名」定位集合：
默认取 config.CORPUS，可用 --corpus 指定。集合名本身复用
src.pipeline.corpus_collection_name（单一命名来源，避免两处实现漂移）。
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import chromadb

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import CHROMA_PERSIST_DIR, CORPUS, CORPUS_ROOT, EMBEDDING_MODEL

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# 分页读取 payload 的页大小。审计必须遍历全量（否则「声明条数 == 实际条数」的
# 校验失去意义），但一次性 get() 十万级 collection 会把全部 embeddings 拉进内存
# （104,192 × 512 维 ≈ 数百 MB 起步），故改为分页累加计数与维度。
_PAGE_SIZE = 5000


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    return value


def _expected_collection_name(corpus: str, model_name: str = EMBEDDING_MODEL) -> str:
    """按生产规则推导该语料的 collection 名（复用 src.pipeline 的实现）。"""
    from src.pipeline import corpus_collection_name

    return corpus_collection_name(corpus, model_name)


def _collection_candidates(client: Any, corpus: str | None = None) -> tuple[list[Any], str]:
    """返回 (候选集合, 期望集合名)。

    优先精确命中 kb__{corpus}__{model}__v{schema}；未命中时列出所有 kb__* 集合
    供诊断（例如语料名拼错、索引尚未构建）。
    """
    corpus = corpus or CORPUS
    expected_name = _expected_collection_name(corpus)
    names = [item if isinstance(item, str) else getattr(item, "name", "") for item in client.list_collections()]
    if expected_name in names:
        return [client.get_collection(expected_name)], expected_name
    return [client.get_collection(name) for name in names if name.startswith("kb__")], expected_name


def _audit_collection(collection: Any, page_size: int = _PAGE_SIZE) -> dict[str, Any]:
    metadata = collection.metadata or {}
    declared_count = metadata.get("chunk_count")
    count_method = int(collection.count())

    # 分页遍历全量 payload，只累加计数/维度，样本只留前 3 条（内存有界）
    ids_count = documents_count = embeddings_count = 0
    dimensions: set[int] = set()
    sample_documents: list[str] = []
    offset = 0
    while offset < count_method:
        page = collection.get(
            include=["documents", "embeddings", "metadatas"],
            limit=page_size,
            offset=offset,
        )
        page_ids = page.get("ids") or []
        if not page_ids:
            break
        page_documents = page.get("documents") or []
        page_embeddings = page.get("embeddings")
        ids_count += len(page_ids)
        documents_count += len(page_documents)
        if page_embeddings is not None:
            embeddings_count += len(page_embeddings)
            dimensions.update(len(vector) for vector in page_embeddings if vector is not None)
        if len(sample_documents) < 3:
            for item in page_documents:
                if len(sample_documents) >= 3:
                    break
                sample_documents.append(item[:160])
        offset += len(page_ids)

    counts_match = ids_count == documents_count == embeddings_count
    return {
        "name": collection.name,
        "metadata": metadata,
        "declared_chunk_count": declared_count,
        "actual_counts": {
            "collection_count": count_method,
            "ids": ids_count,
            "documents": documents_count,
            "embeddings": embeddings_count,
        },
        "embedding_dimensions": sorted(dimensions),
        "consistency": {
            "count_method_eq_ids": count_method == ids_count,
            "ids_eq_documents": ids_count == documents_count,
            "ids_eq_embeddings": ids_count == embeddings_count,
            "all_payload_counts_match": counts_match,
            "declared_eq_actual": declared_count is None or declared_count == count_method,
        },
        "sample_documents": sample_documents,
    }


def _diagnose(index_meta: dict[str, Any] | None, collection: dict[str, Any] | None, runtime_count: int | None, context: dict[str, Any] | None = None) -> dict[str, Any]:
    context = context or {}
    if collection is None:
        all_collections = context.get("all_collections", [])
        expected_name = context.get("expected_name", "")
        summary = "未找到该语料的 Chroma collection"
        if all_collections:
            summary += f"；找到的 collections：{all_collections}；期望集合名：{expected_name}"
        return {"verdict": "MISSING", "summary": summary, "action": "检查语料名、索引命名或执行显式重建", "risk_level": "HIGH"}
    counts = collection["actual_counts"]
    declared = collection["declared_chunk_count"]
    actual = counts["collection_count"]
    consistency = collection["consistency"]
    if actual == 0 and declared not in (None, 0):
        verdict, summary, action, risk = "CORRUPTED", "空壳集合：metadata 声明有数据但实际为空", "隔离旧索引后强制重建", "HIGH"
    elif not consistency.get("all_payload_counts_match", True):
        verdict, summary, action, risk = "PAYLOAD_MISMATCH", f"载荷数量不一致：ids={counts['ids']}, documents={counts['documents']}, embeddings={counts['embeddings']}", "隔离旧索引后强制重建", "HIGH"
    elif not consistency.get("declared_eq_actual", True):
        verdict, summary, action, risk = "DECLARATION_MISMATCH", f"声明数量不匹配：metadata={declared}, 实际={actual}", "检查索引版本和语料后再决定是否重建", "MEDIUM"
    elif runtime_count is not None and actual != runtime_count:
        verdict, summary, action, risk = "RUNTIME_MISMATCH", f"运行时数量不匹配：process_directory={runtime_count}, 索引={actual}", "检查语料 hash 和索引版本后再决定是否重建", "MEDIUM"
    else:
        verdict, summary, action, risk = "HEALTHY", "集合数据和运行时数量初步一致", "继续执行向量查询和 RRF 人工回归", "LOW"
    result = {"verdict": verdict, "summary": summary, "action": action, "risk_level": risk}
    if index_meta and index_meta.get("corpus_hash"):
        result["index_meta_corpus_hash"] = index_meta["corpus_hash"]
    return result


def audit_chroma_index(
    corpus: str | None = None,
    docs_dir: str | Path | None = None,
    skip_runtime: bool = False,
) -> dict[str, Any]:
    """审计指定语料的持久化索引（只读）。

    corpus: 语料名，决定 collection 名与默认语料目录；默认取 config.CORPUS。
    docs_dir: 覆盖运行时切片目录；默认 CORPUS_ROOT / corpus。
    skip_runtime: 跳过 process_directory 全量切片（十万级语料要 1~2 分钟且吃内存），
        跳过时运行时计数不参与诊断。
    """
    corpus = corpus or CORPUS
    docs_dir = Path(docs_dir) if docs_dir else (CORPUS_ROOT / corpus)

    report: dict[str, Any] = {"timestamp": datetime.now().isoformat(), "corpus": corpus, "stages": {}}
    meta_path = CHROMA_PERSIST_DIR / "index_meta.json"
    index_meta = None
    if meta_path.exists():
        index_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        report["stages"]["metadata_file"] = {"status": "OK", "path": str(meta_path), "content": index_meta, "has_chunk_count": "chunk_count" in index_meta}
    else:
        report["stages"]["metadata_file"] = {"status": "MISSING", "path": str(meta_path)}

    report["stages"]["filesystem"] = {
        "status": "OK" if CHROMA_PERSIST_DIR.exists() else "MISSING",
        "path": str(CHROMA_PERSIST_DIR),
        "files": [{"name": item.name, "size": item.stat().st_size, "is_dir": item.is_dir()} for item in CHROMA_PERSIST_DIR.iterdir()] if CHROMA_PERSIST_DIR.exists() else [],
    }

    client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
    all_collections_raw = client.list_collections()
    all_collection_names = [item if isinstance(item, str) else getattr(item, "name", "") for item in all_collections_raw]
    collections, expected_name = _collection_candidates(client, corpus)
    report["stages"]["chroma_connection"] = {
        "status": "OK",
        "candidate_collections": [item.name for item in collections],
        "all_collections": all_collection_names,
        "expected_name": expected_name,
    }
    collection_data = _audit_collection(collections[0]) if collections else None
    report["stages"]["collection_data"] = {"status": "OK" if collection_data else "MISSING", "data": collection_data}

    runtime_count = None
    if skip_runtime:
        report["stages"]["runtime_chunks"] = {
            "status": "SKIPPED",
            "reason": "--skip-runtime；docs_dir=未切片",
            "docs_dir": str(docs_dir),
        }
    else:
        try:
            from src.data_pipeline import process_directory
            runtime_count = len(process_directory(docs_dir))
            report["stages"]["runtime_chunks"] = {"status": "OK", "count": runtime_count, "source": f"process_directory({docs_dir})"}
        except Exception as exc:
            report["stages"]["runtime_chunks"] = {"status": "PARTIAL", "error": str(exc)}

    query_test = None
    if collection_data and collection_data["actual_counts"]["collection_count"] > 0:
        query_test = {"status": "SKIPPED", "reason": "审计脚本不加载 embedding 模型；collection 内容已验证"}
    else:
        query_test = {"status": "NOT_RUN", "reason": "collection 为空或不存在"}
    report["stages"]["vector_query"] = query_test
    report["diagnosis"] = _diagnose(
        index_meta,
        collection_data,
        runtime_count,
        context={"all_collections": all_collection_names, "expected_name": expected_name},
    )
    print_report(report)
    return report


def print_report(report: dict[str, Any]) -> None:
    print("\n" + "=" * 60)
    print("CHROMA INDEX AUDIT REPORT")
    print("=" * 60)
    print(f"Timestamp: {report['timestamp']}")
    if report.get("corpus"):
        print(f"Corpus: {report['corpus']}")
    for name, stage in report["stages"].items():
        print(f"\n[{name}] {stage.get('status')}")
        if "error" in stage:
            print(f"  error: {stage['error']}")
        if name == "chroma_connection":
            print(f"  matched collections: {stage.get('candidate_collections', [])}")
            print(f"  all collections: {stage.get('all_collections', [])}")
            print(f"  expected name: {stage.get('expected_name', '')}")
        elif name == "collection_data" and stage.get("data"):
            data = stage["data"]
            print(f"  collection: {data['name']}")
            print(f"  metadata.chunk_count: {data['declared_chunk_count']}")
            print(f"  actual counts: {data['actual_counts']}")
            print(f"  embedding dimensions: {data['embedding_dimensions']}")
            print(f"  consistency: {data['consistency']}")
        elif name == "runtime_chunks":
            print(f"  count: {stage.get('count')}")
        elif name == "vector_query":
            print(f"  {stage.get('reason', '')}")
    diagnosis = report["diagnosis"]
    print("\n[diagnosis]")
    print(f"  verdict: {diagnosis['verdict']}")
    print(f"  summary: {diagnosis['summary']}")
    print(f"  action: {diagnosis['action']}")
    print(f"  risk: {diagnosis['risk_level']}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="只读审计某个语料的 Chroma 持久化索引")
    parser.add_argument("--corpus", default=CORPUS, help="语料名（corpora/ 下的子目录名），默认取 config.CORPUS")
    parser.add_argument("--docs-dir", default=None, help="覆盖运行时切片目录，默认 CORPUS_ROOT/<corpus>")
    parser.add_argument("--skip-runtime", action="store_true", help="跳过 process_directory 全量切片（十万级语料要 1~2 分钟）")
    args = parser.parse_args()
    audit_chroma_index(corpus=args.corpus, docs_dir=args.docs_dir, skip_runtime=args.skip_runtime)
