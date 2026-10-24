"""检索阶梯配对显著性的产物守护测试 —— 不加载模型、不联网、只用标准库 + pytest。

守护目标：`experiments/baseline_ladder/significance.json` 里的每个数字，都必须能从
仓库里已入库的两份产物（`retrieval_375.jsonl` + `ce_scores.jsonl`）确定性复算出来。
做不到就说明 json 被手改过、或输入产物被换过 —— 那正是要红的场景。

与 `test_regression_replay.py` 的分工：
- 回放层管「一条基线 = 一份产物 + 一个数字」（点估计）。
- 本文件管「同一批 query 的**配对**比较」（p 值与 CI），且**不往
  tests/regression_baselines.json 里新增条目** —— 输入产物的 sha256 直接复用
  回放层已冻结的那两个值，两处记录一漂移就红。

依赖纪律（与回放层一致）：只用标准库 + pytest。统计函数**只 import 复用**
`scripts/eval_ladder_significance`，不在这里复制 McNemar / bootstrap 实现。

期望值来源：`EVAL_REPORT.md` 第十一节「配对检验」表（2026-10-18 定稿），
本文件把它变成可执行断言；**数字对不上就改代码或产物，不许改这里的期望值**。
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import eval_ladder_significance as els

ROOT = Path(__file__).resolve().parents[1]
BASELINE_FILE = ROOT / "tests" / "regression_baselines.json"
SIG_JSON = ROOT / "experiments" / "baseline_ladder" / "significance.json"
RETRIEVAL_REL = "experiments/baseline_ladder/retrieval_375.jsonl"
CE_REL = "experiments/baseline_ladder/ce_scores.jsonl"

DENOMINATOR = 335
ALPHA_EXPECTED = 0.05

# 顺序固定为输出顺序，与 significance.json 的 comparisons 一一对应。
# (对比名, base, arm, base_hits, arm_hits, Δpt, McNemar p, 跨 0, 显著)
EXPECTED = [
    ("rrf_vs_vector", "vector", "rrf", 88, 109, 6.27, 0.0125, False, True),
    ("rrf_vs_bm25", "bm25", "rrf", 102, 109, 2.09, 0.2478, True, False),
    ("vector_vs_bm25", "bm25", "vector", 102, 88, -4.18, 0.1702, True, False),
    ("ce_vs_rrf", "rrf", "ce", 109, 109, 0.00, 1.0, True, False),
]

# 四条臂各自的命中数（规格里单列断言，防止「互相抵消的错位」蒙过关）。
EXPECTED_ARM_HITS = {"bm25": 102, "vector": 88, "rrf": 109, "ce": 109}


# ============================================================
# 工具
# ============================================================
def _sha256_lf(path: Path) -> str:
    """CRLF 归一成 LF 后再算 —— 与 test_regression_replay._sha256 同一算法。

    本机 core.autocrlf=true 时工作区是 CRLF、仓库 blob 是 LF；不归一会让本地
    与 Linux CI 对同一份产物算出两个哈希，闸门就红在换行符上（假阳性会废掉闸门）。
    """
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _baseline_sha256(artifact: str) -> str:
    """从回放层基线里取某个产物的 sha256（同一产物可能被多条基线共用，值必须唯一）。"""
    if not BASELINE_FILE.exists():
        pytest.fail(f"基线文件不存在：{BASELINE_FILE}")
    data = json.loads(BASELINE_FILE.read_text(encoding="utf-8"))
    values = {e["sha256"] for e in data["baselines"] if e["artifact"] == artifact}
    if not values:
        pytest.fail(f"基线里没有 {artifact} 的条目 —— 本测试不允许新增基线条目，请复用既有值")
    if len(values) != 1:
        pytest.fail(f"{artifact} 在基线里记了 {len(values)} 个不同 sha256，基线自身已漂移")
    return values.pop()


@pytest.fixture(scope="module")
def payload() -> dict:
    if not SIG_JSON.exists():
        pytest.fail(f"产物不存在：{SIG_JSON}（跑 python -m scripts.eval_ladder_significance 生成）")
    return json.loads(SIG_JSON.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def fresh() -> dict:
    """内存里重算一遍，不落盘 —— 用来证明 json 不是手写的。"""
    return els.compute()


def _by_name(comparisons: list[dict]) -> dict[str, dict]:
    return {c["comparison"]: c for c in comparisons}


# ============================================================
# 1. 输入产物的完整性（sha256 复用回放层已冻结的值）
# ============================================================
@pytest.mark.parametrize("rel", [RETRIEVAL_REL, CE_REL])
def test_input_artifact_sha256_matches_frozen_baseline(rel: str) -> None:
    actual = _sha256_lf(ROOT / rel)
    expected = _baseline_sha256(rel)
    assert actual == expected, (
        f"{rel} 的 sha256 与回放层冻结值不一致（CRLF 归一后）：\n"
        f"  实测 {actual}\n  基线 {expected}\n"
        "产物被重跑覆盖过 —— 先确认本次替换是有意的，再同一次改完基线的 sha256 与 expected。"
    )


def test_significance_records_same_input_hashes(payload: dict) -> None:
    """significance.json 自己记的输入哈希，也必须与基线一致（防两处记录漂移）。"""
    inputs = payload["inputs"]
    assert [i["path"] for i in inputs] == [RETRIEVAL_REL, CE_REL]
    for entry in inputs:
        assert entry["sha256"] == _baseline_sha256(entry["path"]), (
            f"{entry['path']}：significance.json 记的哈希与基线不一致"
        )
        assert entry["denominator"] == DENOMINATOR


def test_denominator_and_bootstrap_settings(payload: dict) -> None:
    assert payload["denominator"] == DENOMINATOR
    assert payload["alpha"] == ALPHA_EXPECTED
    assert payload["bootstrap"]["n_resamples"] == els.BOOTSTRAP_N == 10000
    assert payload["bootstrap"]["seed"] == els.BOOTSTRAP_SEED == 20261018
    assert payload["bootstrap"]["method"] == "percentile, paired resampling of query indices"


# ============================================================
# 2. 四条臂的命中数（精确）
# ============================================================
def test_arm_hits_exact(payload: dict) -> None:
    observed: dict[str, int] = {}
    for c in payload["comparisons"]:
        for role, value in (("base", c["base_hits"]), ("arm", c["arm_hits"])):
            arm = c[role]
            if arm in observed:
                assert observed[arm] == value, f"{arm} 在两处对比里的命中数不一致：{observed[arm]} / {value}"
            else:
                observed[arm] = value
    assert observed == EXPECTED_ARM_HITS, f"臂命中数不符：{observed} ≠ {EXPECTED_ARM_HITS}"


# ============================================================
# 3. 点估计与 p 值（seed 无关，必须一位不差）
# ============================================================
@pytest.mark.parametrize("row", EXPECTED, ids=[r[0] for r in EXPECTED])
def test_point_estimate_and_p_value(payload: dict, row) -> None:
    name, base, arm, base_hits, arm_hits, delta_pt, p_value, _crosses, _sig = row
    c = _by_name(payload["comparisons"])[name]

    assert (c["base"], c["arm"]) == (base, arm)
    assert c["base_hits"] == base_hits
    assert c["arm_hits"] == arm_hits
    # Δ 必须与命中数自洽：Δpt = (arm − base) / 335 × 100
    assert round(c["delta_pt"], 2) == pytest.approx(delta_pt, abs=1e-9), (
        f"{name}：Δ 点估计 {c['delta_pt']} ≠ 期望 {delta_pt}（按命中数应为 "
        f"{(arm_hits - base_hits) / DENOMINATOR * 100:+.4f}pt）"
    )
    assert round(c["mcnemar_p"], 4) == pytest.approx(p_value, abs=1e-9), (
        f"{name}：McNemar p {c['mcnemar_p']} ≠ 期望 {p_value}"
    )
    # 不一致对总数必须等于 lost + gained（McNemar 只用这些对，写出来防口径漂移）
    assert c["mcnemar_discordant"] == c["lost"] + c["gained"]
    # gained − lost 必须等于命中数之差，否则 Δ 与配对表说的不是一回事
    assert c["gained"] - c["lost"] == arm_hits - base_hits


def test_discordant_counts_match_report(fresh: dict) -> None:
    """不一致对数与 EVAL_REPORT 第十一节逐条对齐（22/43、10/17、52/38、27/27）。"""
    expected = {
        "rrf_vs_vector": (22, 43),
        "rrf_vs_bm25": (10, 17),
        "vector_vs_bm25": (52, 38),
        "ce_vs_rrf": (27, 27),
    }
    got = {c["comparison"]: (c["lost"], c["gained"]) for c in fresh["comparisons"]}
    assert got == expected, f"不一致对数不符：{got} ≠ {expected}"


# ============================================================
# 4. 显著性判定
# ============================================================
def test_significant_flags_exact(payload: dict) -> None:
    got = {c["comparison"]: c["significant"] for c in payload["comparisons"]}
    expected = {r[0]: r[8] for r in EXPECTED}
    assert got == expected


def test_significance_requires_both_p_and_ci(payload: dict) -> None:
    """显著 = p < alpha **且** CI 不含 0；两者必须与落盘的布尔值自洽。"""
    for c in payload["comparisons"]:
        p_ok = c["mcnemar_p"] < payload["alpha"]
        expected = bool(p_ok and not c["ci_crosses_zero"])
        assert c["significant"] is expected, (
            f"{c['comparison']}：significant={c['significant']} 与 "
            f"p<alpha({p_ok}) 且 CI 不跨 0({not c['ci_crosses_zero']}) 不自洽"
        )


# ============================================================
# 5. CI 只断言「是否跨 0」（端点受 seed 影响，不逐位断言）
# ============================================================
@pytest.mark.parametrize("row", EXPECTED, ids=[r[0] for r in EXPECTED])
def test_ci_crosses_zero_matches_expected(payload: dict, row) -> None:
    name, _base, _arm, _bh, _ah, _d, _p, crosses, _sig = row
    c = _by_name(payload["comparisons"])[name]
    lo, hi = c["ci95_pt"]
    assert lo <= hi, f"{name}：CI 端点顺序反了 [{lo}, {hi}]"
    assert c["ci_crosses_zero"] is (lo <= 0.0 <= hi)
    assert c["ci_crosses_zero"] is crosses, (
        f"{name}：CI [{lo:+.2f}, {hi:+.2f}] 的跨 0 结论与报告不符（期望 crosses_zero={crosses}）"
    )


def test_ci_width_is_plausible(payload: dict) -> None:
    """CI 只能反映配对差的不确定性：宽度必须远小于「0~100pt」，端点不得越界。"""
    for c in payload["comparisons"]:
        lo, hi = c["ci95_pt"]
        assert -100.0 < lo <= hi < 100.0
        assert (hi - lo) < 40.0, f"{c['comparison']}：CI 宽度 {(hi - lo):.2f}pt 不合理"


# ============================================================
# 6. 配对必须按 id 对齐（行序错配曾经算出 65/65 的假不一致对）
# ============================================================
def test_pairing_is_by_id_not_row_order(fresh: dict) -> None:
    assert fresh["pairing"]["pairing_key"] == "id"
    assert fresh["pairing"]["ce_hit_field"] == "hit@5", "CE 命中字段探测结果变了，先确认产物格式"
    # 两个产物行序本来就不同 —— 这正是不能按行号 zip 的原因。若哪天变成同序，
    # 这条断言会红：那是去掉「按 id 对齐」唯一防线前的最后一次提醒。
    assert fresh["pairing"]["row_order_identical"] is False


def test_ce_vs_rrf_discordant_is_27_not_65(fresh: dict) -> None:
    """CE 与 RRF 在 335 条上各命中 109，不一致对是 27/27，不是 65/65。

    65/65 是按文件行号 zip（两份产物行序不同）算出来的配对假象；按 id 对齐后
    净 0 但两侧各 27 条互换，与 regression_baselines.json 里 ce_hit@5 的
    note「与 RRF 逐位相同（+27/−27 净 0）」一致。
    """
    c = _by_name(fresh["comparisons"])["ce_vs_rrf"]
    assert (c["lost"], c["gained"]) == (27, 27)
    assert c["delta_pt"] == 0.0
    assert c["mcnemar_p"] == 1.0


# ============================================================
# 7. CLI 与 import 链纪律
# ============================================================
def test_cli_check_mode_exits_zero() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "scripts.eval_ladder_significance", "--check"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, f"--check 失败：\n{proc.stdout}\n{proc.stderr}"
    assert "一致" in proc.stdout


def test_import_chain_has_no_model_deps() -> None:
    """CI 的 replay job 只装 pytest —— import 链里不许出现模型栈。"""
    code = (
        "import sys, scripts.eval_ladder_significance as m;"
        "forbidden = [n for n in ('torch','transformers','chromadb','rank_bm25','ollama','sentence_transformers')"
        " if n in sys.modules];"
        "print(','.join(forbidden));"
        "print(m.ALPHA, m.BOOTSTRAP_SEED)"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], cwd=str(ROOT), capture_output=True, text=True, timeout=300
    )
    assert proc.returncode == 0, f"导入失败：\n{proc.stderr}"
    # 第一行是禁用依赖清单（空串=干净），第二行是常量回显，用来确认模块真被导入了
    lines = proc.stdout.splitlines()
    assert lines and lines[0] == "", f"import 链里出现了禁用依赖：{lines[0] if lines else '(无输出)'}"
    assert lines[1] == "0.05 20261018"
