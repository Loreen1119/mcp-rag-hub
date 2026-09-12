"""Read-only audit for the persisted Chroma index.

Usage:
    python scripts/audit_chroma_index.py

This script does not rebuild, delete, or write to the index.  It only reads
Chroma metadata/content and, optionally, loads the existing pipeline to compare
the runtime chunk count.
"""

from __future__ import annotations

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

from config import CHROMA_PERSIST_DIR, DOCS_DIR, EMBEDDING_MODEL, INDEX_SCHEMA_VERSION

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    return value


def _collection_candidates(client: Any) -> list[Any]:
    collections = client.list_collections()
    expected_prefix = f"knowledge_base__{EMBEDDING_MODEL}__v{INDEX_SCHEMA_VERSION}__"
    names = [item if isinstance(item, str) else getattr(item, "name", "") for item in collections]
    return [client.get_collection(name) for name in names if name.startswith(expected_prefix)]


def _audit_collection(collection: Any) -> dict[str, Any]:
    metadata = collection.metadata or {}
    declared_count = metadata.get("chunk_count")
    count_method = int(collection.count())
    data = collection.get(include=["documents", "embeddings", "metadatas"])
    ids = data.get("ids") or []
    documents = data.get("documents") or []
    embeddings = data.get("embeddings")
    ids_count = len(ids)
    documents_count = len(documents)
    embeddings_count = len(embeddings) if embeddings is not None else 0
    dimensions = sorted({len(vector) for vector in embeddings if vector is not None}) if embeddings is not None else []
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
        "embedding_dimensions": dimensions,
        "consistency": {
            "count_method_eq_ids": count_method == ids_count,
            "ids_eq_documents": ids_count == documents_count,
            "ids_eq_embeddings": ids_count == embeddings_count,
            "all_payload_counts_match": counts_match,
            "declared_eq_actual": declared_count is None or declared_count == count_method,
        },
        "sample_documents": [item[:160] for item in documents[:3]],
    }


def _diagnose(index_meta: dict[str, Any] | None, collection: dict[str, Any] | None, runtime_count: int | None, context: dict[str, Any] | None = None) -> dict[str, Any]:
    context = context or {}
    if collection is None:
        all_collections = context.get("all_collections", [])
        expected_prefix = context.get("expected_prefix", "")
        summary = "未找到当前版本的 Chroma collection"
        if all_collections:
            summary += f"；找到的 collections：{all_collections}；期望前缀：{expected_prefix}"
        return {"verdict": "MISSING", "summary": summary, "action": "检查索引命名、路径或执行显式重建", "risk_level": "HIGH"}
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


def audit_chroma_index() -> dict[str, Any]:
    report: dict[str, Any] = {"timestamp": datetime.now().isoformat(), "stages": {}}
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
    expected_prefix = f"knowledge_base__{EMBEDDING_MODEL}__v{INDEX_SCHEMA_VERSION}__"
    collections = _collection_candidates(client)
    report["stages"]["chroma_connection"] = {
        "status": "OK",
        "candidate_collections": [item.name for item in collections],
        "all_collections": all_collection_names,
        "expected_prefix": expected_prefix,
    }
    collection_data = _audit_collection(collections[0]) if collections else None
    report["stages"]["collection_data"] = {"status": "OK" if collection_data else "MISSING", "data": collection_data}

    runtime_count = None
    try:
        from src.data_pipeline import process_directory
        runtime_count = len(process_directory(DOCS_DIR))
        report["stages"]["runtime_chunks"] = {"status": "OK", "count": runtime_count, "source": "process_directory(docs)"}
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
        context={"all_collections": all_collection_names, "expected_prefix": expected_prefix},
    )
    print_report(report)
    return report


def print_report(report: dict[str, Any]) -> None:
    print("\n" + "=" * 60)
    print("CHROMA INDEX AUDIT REPORT")
    print("=" * 60)
    print(f"Timestamp: {report['timestamp']}")
    for name, stage in report["stages"].items():
        print(f"\n[{name}] {stage.get('status')}")
        if "error" in stage:
            print(f"  error: {stage['error']}")
        if name == "chroma_connection":
            print(f"  matched collections: {stage.get('candidate_collections', [])}")
            print(f"  all collections: {stage.get('all_collections', [])}")
            print(f"  expected prefix: {stage.get('expected_prefix', '')}")
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
    audit_chroma_index()
