"""回放层回归测试 —— 不加载模型、不联网、不依赖重型第三方包。

设计（2026-10-16）：
- 这层只做一件事：把「README / 面经 / 简历里引用的检索指标」从**仓库里已入库的产物 JSONL**
  重新算一遍，与 `tests/regression_baselines.json` 里冻结的值对比。
- 每个产物先过 **sha256 闸门**：产物被重跑覆盖后，断言若不同步更新就会静默测到另一份文件、
  还显示绿 —— 那是假绿。所以产物一变就先红，报错里直接给出复算到的新值，逼着显式改基线。
- 「本地全量重跑层」（要装模型 / 建索引 / 跑 LLM）**不进 CI，也不在本文件里**。

依赖纪律：本文件只用标准库 + pytest。CI 的 replay job 因此无需安装
torch / transformers / chromadb / rank-bm25 —— 回放层与模型层是解耦的，这一点本身是设计目标。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BASELINE_FILE = ROOT / "tests" / "regression_baselines.json"

_TOPK = 5


# ============================================================
# 基线装载与复算工具
# ============================================================


def _load_baselines() -> dict:
    if not BASELINE_FILE.exists():
        pytest.fail(f"基线文件不存在：{BASELINE_FILE}")
    return json.loads(BASELINE_FILE.read_text(encoding="utf-8"))


_BASELINES = _load_baselines()
_CASES = _BASELINES["baselines"]
_FROZEN = _BASELINES["frozen_artifacts"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(rel_path: str) -> list[dict]:
    path = ROOT / rel_path
    if not path.exists():
        pytest.fail(f"复算所需的产物不存在：{rel_path}（CI 依赖它已入库）")
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _top5_slugs(candidates: list[dict]) -> set[str]:
    """按 CE 分数取 top-5 的 slug。"""
    return {
        c["slug"]
        for c in sorted(candidates, key=lambda c: -float(c["score"]))[:_TOPK]
    }


def _recount(entry: dict) -> tuple[int, int]:
    """按条目声明的 metric 复算，返回 (命中数, 分母)。"""
    metric = entry["metric"]
    rows = _read_jsonl(entry["artifact"])
    hits = total = 0

    for row in rows:
        golden = set(row.get("golden_slugs") or [])
        if not golden:  # 不可答样本不进检索分母
            continue
        total += 1

        if metric == "stage_hit@5":
            stage_block = row["stages"][entry["stage"]]
            top = [x for x in stage_block["top"] if x.get("rank", 9999) <= _TOPK]
            if any(x.get("slug") in golden for x in top):
                hits += 1

        elif metric == "ce_top_hit@5":
            # 该产物没有 ce_candidates，只能按自带 ce_top（已是 top-5）判
            if any(x.get("slug") in golden for x in row["ce_top"]):
                hits += 1

        elif metric == "ce_candidates_hit@5":
            candidates = row["ce_candidates"]
            cap = entry.get("rrf_rank_max")
            if cap is not None:
                # 候选截断：只保留 RRF 排名在前 cap 的候选，再按 CE 分取 top-5
                candidates = [
                    c for c in candidates if (c.get("rrf_rank") or 0) <= cap
                ]
            if any(slug in golden for slug in _top5_slugs(candidates)):
                hits += 1

        else:
            pytest.fail(f"基线 {entry['name']} 声明了未知 metric：{metric}")

    return hits, total


# ============================================================
# 闸门 1：冻结产物 / 基线产物未被静默替换
# ============================================================


def _artifact_cases() -> list[dict]:
    """所有需要校验 sha256 的文件（冻结产物 + 基线引用的产物），按路径去重。"""
    seen: dict[str, dict] = {}
    for item in _FROZEN:
        seen.setdefault(item["path"], {"path": item["path"], "sha256": item["sha256"], "kind": "冻结产物"})
    for item in _CASES:
        seen.setdefault(
            item["artifact"],
            {"path": item["artifact"], "sha256": item["sha256"], "kind": "基线产物"},
        )
    return sorted(seen.values(), key=lambda x: x["path"])


@pytest.mark.parametrize("item", _artifact_cases(), ids=lambda x: x["path"])
def test_artifact_sha256_locked(item):
    path = ROOT / item["path"]
    if not path.exists():
        pytest.fail(f"{item['kind']}缺失：{item['path']}")
    actual = _sha256(path)
    assert actual == item["sha256"], (
        f"{item['kind']}已被改动：{item['path']}\n"
        f"  基线 sha256 = {item['sha256']}\n"
        f"  当前 sha256 = {actual}\n"
        f"→ 若这是有意的重跑/换批，请**同一次**把 tests/regression_baselines.json 里的 "
        f"sha256 与 expected（含 expect_hits）一起更新，并在 note 写清是哪一批、样本量多少；\n"
        f"→ 若不是有意的，说明产物被覆盖了，先恢复产物再查是谁改的。"
    )


# ============================================================
# 闸门 2：基线值可从同一份产物确定性复算
# ============================================================


@pytest.mark.parametrize("entry", _CASES, ids=lambda x: x["name"])
def test_baseline_recomputes(entry):
    hits, total = _recount(entry)

    assert total == entry["denominator"], (
        f"{entry['name']}：分母变了。基线 {entry['denominator']}，复算 {total}。"
        f"分母 = 有 golden 的非不可答样本；变了说明产物或评测集被换过。"
    )
    assert hits == entry["expect_hits"], (
        f"{entry['name']}：命中数变了。基线 {entry['expect_hits']}/{entry['denominator']}，"
        f"复算 {hits}/{total}。"
    )

    value = round(hits / total, 4)
    assert abs(value - entry["expected"]) <= entry["tolerance"], (
        f"{entry['name']}：复算 {value}，基线 {entry['expected']}"
        f"（容差 {entry['tolerance']}）。"
    )


def test_每个基线都写清了出处与口径():
    """基线条目必须自带 artifact + note，防止后来者抄一个数字却不知从哪来。"""
    for entry in _CASES:
        for field in ("name", "artifact", "sha256", "metric", "expected", "note"):
            assert entry.get(field), f"基线 {entry.get('name')} 缺字段 {field}"
