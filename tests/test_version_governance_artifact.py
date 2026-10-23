"""版本治理对照实验的产物守护测试 —— 不加载模型、不联网、只用标准库 + pytest。

为什么单独一个文件而不是并进 test_regression_replay.py：
- `_recount()` 是「一条基线 = 一份产物 + 一个数字」的极简结构，而本实验是
  「一份输入 → 4 个臂 × 5 个指标 + 配对检验 + 交叉表」，塞进去会把那个函数撑成 if-else 丛林。
- 但守护目标与回放层一致：**ablation.json 里的每个数字，都必须能从
  experiments/baseline_ladder/retrieval_375.jsonl 确定性复算出来。**
  做不到就说明 json 被手改过、或输入产物被换过 —— 那正是要红的场景。

输入产物本身的 sha256 已由 test_regression_replay.py 的 rrf_hit@5 条目锁定，
这里只校验 ablation.json 自己记的那个哈希，防止两处记录漂移。
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GOV_JSON = ROOT / "experiments" / "version_governance" / "ablation.json"
INPUT = ROOT / "experiments" / "baseline_ladder" / "retrieval_375.jsonl"
OLD_PREFIX = "docs-cn-v8.4/"
K = 5


# ============================================================
# 装载与工具
# ============================================================


def _load() -> dict:
    if not GOV_JSON.exists():
        pytest.fail(f"产物不存在：{GOV_JSON}")
    return json.loads(GOV_JSON.read_text(encoding="utf-8"))


def _sha256_lf(path: Path) -> str:
    """CRLF 归一成 LF 后再算 —— 与 test_regression_replay._sha256 同一算法。

    不归一会得到 Windows 工作区的 CRLF 哈希，在 Linux CI 上永远对不上，
    闸门就红在换行符上而不是真实变化上（假阳性会把闸门废掉）。
    """
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _records() -> list[dict]:
    if not INPUT.exists():
        pytest.fail(f"复算所需的输入产物不存在：{INPUT}")
    with open(INPUT, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _denominator(rows: list[dict]) -> list[dict]:
    """335 口径：跳过 unanswerable=True 与无 golden 的条目。"""
    return [r for r in rows if not r.get("unanswerable", False) and r.get("golden_slugs")]


def _cands(row: dict) -> list[dict]:
    top = (row.get("stages", {}).get("rrf") or {}).get("top") or []
    out = [
        {
            "rank": int(c.get("rank", i + 1)),
            "slug": c.get("slug"),
            "src": c.get("src") or "",
            "score": float(c.get("score") or 0.0),
        }
        for i, c in enumerate(top)
    ]
    out.sort(key=lambda c: c["rank"])
    return out


def _is_old(c: dict) -> bool:
    return c["src"].startswith(OLD_PREFIX)


def _arm(name: str, cands: list[dict]) -> list[dict]:
    if name == "baseline":
        return list(cands)
    if name == "filter":
        return [c for c in cands if not _is_old(c)]
    factor = {"decay_0.5": 0.5, "decay_0.9": 0.9}.get(name)
    if factor is None:
        raise AssertionError(f"未知臂：{name}")
    decorated = [{**c, "score": c["score"] * (factor if _is_old(c) else 1.0)} for c in cands]
    decorated.sort(key=lambda c: (-c["score"], c["rank"]))
    return decorated


def _first_hit(ranked: list[dict], golden: set, k: int = K):
    for c in ranked[:k]:
        if c["slug"] in golden:
            return c
    return None


def _hit(ranked: list[dict], golden: set, k: int = K) -> bool:
    return _first_hit(ranked, golden, k) is not None


def _vhit(ranked: list[dict], golden: set, k: int = K) -> bool:
    """VersionCorrect@k：首个命中存在，且不来自旧版目录。"""
    first = _first_hit(ranked, golden, k)
    return first is not None and not _is_old(first)


def _mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    m = min(b, c)
    tail = sum(math.comb(n, i) for i in range(m + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


ARMS = ["baseline", "filter", "decay_0.5", "decay_0.9"]


# ============================================================
# 闸门 1：输入产物的哈希记录与「归一后」算法一致
# ============================================================


def test_input_sha256_is_lf_normalized():
    """ablation.json 记的 input_sha256 必须是 CRLF→LF 归一后的值。

    2026-10-23 实测踩过的坑：初版记的是 CRLF 原始哈希（bfaaaf89...），
    在 Linux CI 上永远对不上 —— 这个断言就是防止再犯。
    """
    meta = _load()["meta"]
    actual = _sha256_lf(INPUT)
    assert meta["input_sha256"] == actual, (
        f"input_sha256 与归一后的实测值不一致：\n"
        f"  记载 = {meta['input_sha256']}\n"
        f"  实测 = {actual}\n"
        f"→ 若记载值形如 CRLF 原始哈希，说明算哈希时没做归一（本机 375 行均为 CRLF）。"
    )


# ============================================================
# 闸门 2：每个臂的指标都能从输入复算
# ============================================================


@pytest.mark.parametrize("arm", ARMS)
def test_hit_at_5_recomputes(arm):
    doc = _load()
    den = _denominator(_records())
    assert len(den) == doc["meta"]["denominator"], (
        f"分母变了：json 记 {doc['meta']['denominator']}，复算 {len(den)}。"
        f"分母 = 有 golden 的非不可答样本。"
    )
    hits = sum(1 for r in den if _hit(_arm(arm, _cands(r)), set(r["golden_slugs"])))
    got = doc["per_arm"][arm]
    assert hits == got["hit@5_num"], (
        f"{arm} 的 Hit@5 分子变了：json 记 {got['hit@5_num']}，复算 {hits}。"
    )
    assert abs(hits / len(den) - got["hit@5"]) < 1e-9, (
        f"{arm} 的 Hit@5 比率与分子不自洽：{got['hit@5']} vs {hits / len(den)}"
    )


@pytest.mark.parametrize("arm", ARMS)
def test_version_correct_at_5_recomputes(arm):
    """VersionCorrect@5 是 2026-10-23 主线补算的口径，同样必须可复算。"""
    doc = _load()
    den = _denominator(_records())
    num = sum(1 for r in den if _vhit(_arm(arm, _cands(r)), set(r["golden_slugs"])))
    got = doc["version_correct"]["per_arm"][arm]
    assert num == got["num"], (
        f"{arm} 的 VersionCorrect@5 分子变了：json 记 {got['num']}，复算 {num}。"
    )
    assert abs(num / len(den) - got["rate"]) < 1e-9


def test_interference_rate_recomputes():
    """干扰率 = top-5 含旧版候选的 query 占比（只看 baseline 臂）。"""
    doc = _load()
    den = _denominator(_records())
    num = sum(1 for r in den if any(_is_old(c) for c in _arm("baseline", _cands(r))[:K]))
    assert num == doc["per_arm"]["baseline"]["interference_num"], (
        f"baseline 干扰数变了：json 记 {doc['per_arm']['baseline']['interference_num']}，复算 {num}。"
    )
    # filter 臂的定义就是剔干净旧版，干扰必须为 0（若不为 0 说明过滤逻辑坏了）
    assert doc["per_arm"]["filter"]["interference_num"] == 0


# ============================================================
# 闸门 3：配对检验与交叉表
# ============================================================


def test_paired_test_on_hit_at_5():
    doc = _load()
    den = _denominator(_records())
    base = {r["id"]: _hit(_arm("baseline", _cands(r)), set(r["golden_slugs"])) for r in den}
    filt = {r["id"]: _hit(_arm("filter", _cands(r)), set(r["golden_slugs"])) for r in den}
    b = sum(1 for i in base if base[i] and not filt[i])
    c = sum(1 for i in base if not base[i] and filt[i])

    vs = doc["baseline_vs_filter"]
    assert (b, c) == (vs["base_only"], vs["filter_only"]), (
        f"Hit@5 不一致对变了：json 记 丢{vs['base_only']}/得{vs['filter_only']}，复算 丢{b}/得{c}。"
    )
    assert abs(_mcnemar_exact(b, c) - vs["mcnemar_exact_p"]) < 1e-12, (
        f"p 值变了：json 记 {vs['mcnemar_exact_p']}，复算 {_mcnemar_exact(b, c)}。"
    )


@pytest.mark.parametrize("arm", ["filter", "decay_0.5", "decay_0.9"])
def test_paired_test_on_version_correct(arm):
    """三个治理臂在 VersionCorrect@5 上的配对检验应完全一致（都是 丢0/得14）。"""
    doc = _load()
    den = _denominator(_records())
    base = {r["id"]: _vhit(_arm("baseline", _cands(r)), set(r["golden_slugs"])) for r in den}
    other = {r["id"]: _vhit(_arm(arm, _cands(r)), set(r["golden_slugs"])) for r in den}
    b = sum(1 for i in base if base[i] and not other[i])
    c = sum(1 for i in base if not base[i] and other[i])

    vs = doc["version_correct"]["vs_baseline"]
    assert (b, c) == (vs["base_only"], vs["arm_only"]), (
        f"{arm}：不一致对变了，json 记 丢{vs['base_only']}/得{vs['arm_only']}，复算 丢{b}/得{c}。"
    )
    assert abs(_mcnemar_exact(b, c) - vs["mcnemar_exact_p"]) < 1e-12


def test_old_version_hit_disposition():
    """baseline 靠旧版命中的那批 query，各治理臂的处置（纠正/仍旧/丢）必须可复算。"""
    doc = _load()
    den = _denominator(_records())

    # 先找出 baseline 上「首个命中来自旧版」的那些 query
    targets = []
    for r in den:
        first = _first_hit(_arm("baseline", _cands(r)), set(r["golden_slugs"]))
        if first is not None and _is_old(first):
            targets.append(r)

    block = doc["version_correct"]["old_version_hits_in_baseline"]
    assert len(targets) == block["count"], (
        f"baseline 靠旧版命中的条数变了：json 记 {block['count']}，复算 {len(targets)}。"
    )

    for arm in ("filter", "decay_0.5", "decay_0.9"):
        new = old = lost = 0
        for r in targets:
            first = _first_hit(_arm(arm, _cands(r)), set(r["golden_slugs"]))
            if first is None:
                lost += 1
            elif _is_old(first):
                old += 1
            else:
                new += 1
        want = block["disposition"][arm]
        assert (new, old, lost) == (
            want["corrected_to_new"],
            want["still_old"],
            want["lost_hit"],
        ), (
            f"{arm} 的处置表变了：json 记 纠正{want['corrected_to_new']}/仍旧{want['still_old']}/丢{want['lost_hit']}，"
            f"复算 纠正{new}/仍旧{old}/丢{lost}。"
        )


def test_preferred_arm_is_really_zero_loss():
    """decay_0.9 被称为「零召回代价」，这个断言就是那句话的证据。"""
    doc = _load()
    pa = doc["version_correct"]["preferred_arm"]["arm"]
    assert pa == "decay_0.9"
    assert doc["per_arm"]["decay_0.9"]["hit@5_num"] == doc["per_arm"]["baseline"]["hit@5_num"], (
        "decay_0.9 的 Hit@5 分子必须与 baseline 相同，否则「零召回代价」不成立。"
    )
    assert doc["version_correct"]["per_arm"]["decay_0.9"]["num"] > doc["version_correct"]["per_arm"]["baseline"]["num"]


def test_verdict_scope_is_explicit():
    """结论必须带 scope_limit —— 防止「slug 口径无收益」被当成「治理无价值」引用。"""
    v = _load()["verdict"]
    assert v["passed"] is False
    assert v.get("scope_limit"), "verdict 缺 scope_limit：这条负面结论必须写明它只覆盖 slug 口径。"
    assert "版本" in v["scope_limit"]
