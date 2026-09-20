r"""重截 RAG 界面截图（E4）+ 验证三态文案（A2.4）。

前置：Streamlit 已在跑 ——
    .\.venv\Scripts\python.exe -m streamlit run app.py --server.headless true --server.port 8510

依赖：`pip install playwright`（**不在 requirements.txt 里** —— 它只是截图工具，不进运行时镜像）。
      本脚本用 `executable_path` 指向系统 Chrome，**不需要** `playwright install` 下载浏览器。

三个阶段（因为要切换 LLM 可用性）：

    # 一、LLM 不可用（Ollama 不启动）→ 空态 + 降级态
    $env:MCP_RAG_LLM_BACKEND="ollama"; .\.venv\Scripts\python.exe scripts\capture_ui.py --phase no-llm

    # 二、LLM 可用（先启动 Ollama）→ 正常回答 + 调试模式四阶段
    .\.venv\Scripts\python.exe scripts\capture_ui.py --phase with-llm

    # 三、只补"拒答态"（不动已拍好的图）—— 这一步慢，可单独跑、可重试
    .\.venv\Scripts\python.exe scripts\capture_ui.py --phase refusal

输出固定写到**项目根**的 screenshots/（不随 cwd 变化）：

    ui-initial.png          空态
    ui-degraded.png         LLM 服务不可用 → 「回答生成服务暂不可用」+ 已找出的检索证据
    ui-query-results.png    正常回答（带 [n] 引用编号）
    ui-refusal.png          知识库里确实没有 → 「知识库信息不足，未生成答案」
    ui-tab-{BM25,vector,RRF,Cross-Encoder}.png   调试模式的四个阶段

设计要点（2026-09-20 重写；五条都是当天踩出来的）：

1. **每次截图前先断言三态文案**，断言不过就报错退出。
   否则可能把"降级"截成"正常回答"还浑然不觉 —— 这正是 A2.4 要防的：
   **技术故障被误报成业务判断**（用户会去知识库里翻找，而真正坏的是服务）。

2. **"拒答态"只能在 LLM 可用时拍**：三态逻辑是「无 chunk → `insufficient_evidence`；
   **有 chunk 必调 LLM**；LLM 空 → `generation_unavailable`」，而混合检索对**任何**问题都会
   返回若干 chunk（BM25/向量总有一个"最像"的）。所以 LLM 挂掉时，"知识库里根本没有"这类问题
   **也会走降级**，拍不到拒答；拒答的正常触发路径是 **LLM 判定"未在文档中找到"**。

3. **拒答这一步要等很久（5~8 分钟/条）**：UI 走的是完整 Agent 链路（含查询改写迭代），
   而调试模式每次 rerun 还会重算一遍 CE。⏰ 一开始按 200 秒设超时，结果两条候选"没拒答"——
   实际是**还在跑**（A2.2 实测这两条都是 `insufficient_evidence`）。现设 480 秒。

4. **调试模式的四个阶段渲染慢**：那段会重跑一遍管线（含 CE 精排），所以要**等 6 个 tab
   都出现**再点，否则第一个 Tab 会点空（实测：等 2 秒就点，`BM25 关键词` 超时）。
   另注意实际标签带计数，形如 `BM25 关键词 (15)` —— 按子串匹配即可。

5. **流式输出要用 `-u` 跑**：不带 `-u` 时 python 会缓冲，中途被中断就什么都看不到
   （实测丢过一次全部进度输出，只能靠截图时间戳反推跑到哪）。
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

BASE = "http://localhost:8510"
# 输出固定到项目根：本脚本 2026-09-18 从根目录移入 scripts/，
# 原先的相对路径会写进 scripts/screenshots/
OUT = Path(__file__).resolve().parents[1] / "screenshots"
OUT.mkdir(exist_ok=True)
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"

# 能答的问题（取自 golden 集 M06）——检索能命中，LLM 能给出带 [n] 引用的答案
Q_ANSWERABLE = "如何让某段耗时的处理在响应返回之后再执行？"

# 拒答候选：全部取自 `data/unanswerable_queries.json`。
# 顺序依据 2026-09-15 的 A2.2 实测（experiments/acceptance_eval_results.json）：
# 10 条里 9 条被判 `insufficient_evidence`，只有「FastAPI 内置的图形化管理后台…」被判 `answered`
# —— 故把它排在最后（它已知不会拒答）。
Q_UNANSWERABLE_CANDIDATES = [
    "这个知识库目前一共收录了多少篇文档，上一次更新是什么时候？",
    "本系统在生产环境承诺的 SLA 和可用性指标是多少？",
    "知识库里有哪些维护者的个人简历和联系方式？",
    "Vue 3 的 Composition API 里 ref 和 reactive 有什么区别？",
    "FastAPI 背后的公司去年营收是多少？",
    "Spring Boot 的自动配置（AutoConfiguration）是怎么实现的？",
    "Django ORM 中 ForeignKey 的 on_delete 参数有哪些可选值？",
    "Flask 里 Blueprint 的 url_prefix 参数应该怎么配置？",
    "把一个已有的 Django 项目迁移到 FastAPI，模型层和视图层分别要怎么改写？",
    "FastAPI 内置的图形化管理后台（Admin UI）要怎么开启？",
]

STAGE_TABS = [
    ("BM25 关键词", "ui-tab-BM25.png"),
    ("向量语义", "ui-tab-vector.png"),
    ("RRF 融合", "ui-tab-RRF.png"),
    ("Cross-Encoder 精排", "ui-tab-Cross-Encoder.png"),
]

# 三态文案 —— 与 app.py 里的 st.success / st.warning / st.error 一一对应
TXT_ANSWERED = "引用来源"
TXT_REFUSAL = "知识库信息不足，未生成答案"
TXT_DEGRADED = "回答生成服务暂不可用"

# 单条查询的等待上限。实测这条链路很慢（Agent 迭代 + 每次 rerun 重算 CE），给足。
QUERY_TIMEOUT = 480.0


def wait_for(page, markers: list[str], timeout: float = QUERY_TIMEOUT, poll: float = 3.0) -> str | None:
    """轮询直到页面出现任一 marker，返回命中的那个；超时返回 None。"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        text = page.inner_text("body")
        for m in markers:
            if m in text:
                return m
        time.sleep(poll)
    return None


def wait_for_tab_count(page, minimum: int, timeout: float = 300.0, poll: float = 2.0) -> bool:
    """等 tab 数量达到 minimum（调试模式渲染完会出现 2 顶层 + 4 阶段 = 6 个）。"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if page.get_by_role("tab").count() >= minimum:
            return True
        time.sleep(poll)
    return False


def submit(page, query: str) -> None:
    box = page.locator("input[type=text]").first
    box.fill(query)
    box.press("Enter")


def guard_no_exception(page) -> None:
    exc = page.locator('[data-testid="stException"]')
    if exc.count() > 0:
        raise SystemExit(f"!! 页面出现异常面板，中止：\n{exc.first.inner_text()[:600]}")


def alerts(page) -> str:
    els = page.locator('[data-testid="stAlert"]')
    return " / ".join(els.nth(i).inner_text()[:120] for i in range(els.count()))


def shoot(page, name: str) -> None:
    page.screenshot(path=str(OUT / name))
    print(f"     ✅ {name}", flush=True)


def capture_refusal(page) -> None:
    """遍历候选问题，取第一个真拒答的进截图（见文件头说明 2、3）。"""
    for idx, q in enumerate(Q_UNANSWERABLE_CANDIDATES, 1):
        started = time.time()
        print(f"     [{idx}/{len(Q_UNANSWERABLE_CANDIDATES)}] {q}", flush=True)
        submit(page, q)
        hit = wait_for(page, [TXT_REFUSAL, TXT_ANSWERED, TXT_DEGRADED])
        guard_no_exception(page)
        spent = time.time() - started
        if hit == TXT_REFUSAL:
            print(f"     ✅ 被正确拒答（{spent:.0f}s）", flush=True)
            shoot(page, "ui-refusal.png")
            return
        print(f"     ✗ 没拒答（命中 {hit!r}，{spent:.0f}s），换下一条", flush=True)
    raise SystemExit("!! 所有候选都没触发拒答 —— 与 A2.2 的 90% 拒答率不符，需人工看一眼")


def phase_no_llm(page) -> None:
    print("  [1] 能答的问题 + LLM 不可用 → 期望「降级」而不是「拒答」", flush=True)
    submit(page, Q_ANSWERABLE)
    hit = wait_for(page, [TXT_DEGRADED, TXT_REFUSAL])
    guard_no_exception(page)
    if hit != TXT_DEGRADED:
        raise SystemExit(
            f"!! A2.4 失败：期望「{TXT_DEGRADED}」，实际命中 {hit!r}。\n   告警区：{alerts(page)}"
        )
    print("     ✅ 显示的是「服务暂不可用」—— 技术故障没有被误报成业务拒答", flush=True)
    shoot(page, "ui-degraded.png")


def phase_with_llm(page) -> None:
    print("  [1] 能答的问题 + LLM 可用 → 期望正常回答（带 [n] 引用）", flush=True)
    submit(page, Q_ANSWERABLE)
    hit = wait_for(page, [TXT_ANSWERED, TXT_DEGRADED, TXT_REFUSAL])
    guard_no_exception(page)
    if hit != TXT_ANSWERED:
        raise SystemExit(f"!! 期望正常回答，实际命中 {hit!r}。\n   告警区：{alerts(page)}")
    print("     ✅ 正常回答", flush=True)
    shoot(page, "ui-query-results.png")

    print("  [2] 调试模式：四个阶段（等 6 个 tab 出现再点）", flush=True)
    page.get_by_role("tab", name="调试模式").first.click(timeout=20000)
    if not wait_for_tab_count(page, 6):
        raise SystemExit("!! 调试模式的四个阶段 Tab 未出现（渲染超时）")
    print("     阶段 Tab 已渲染", flush=True)
    for name, filename in STAGE_TABS:
        try:
            page.get_by_role("tab", name=name).first.click(timeout=30000)
            time.sleep(2)
            shoot(page, filename)
        except Exception as e:  # noqa: BLE001
            print(f"     ! 点击 Tab '{name}' 失败: {e}", flush=True)


def run(phase: str) -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=CHROME)
        page = browser.new_context(viewport={"width": 1400, "height": 1000}).new_page()
        page.goto(BASE, wait_until="networkidle", timeout=90000)
        page.wait_for_selector("input[type=text]", timeout=60000)
        time.sleep(3)

        if phase == "refusal":
            print("  [*] 只补拒答态（不动已拍好的图）", flush=True)
            capture_refusal(page)
        else:
            print("  [0] 空态", flush=True)
            shoot(page, "ui-initial.png")
            if phase == "no-llm":
                phase_no_llm(page)
            else:
                phase_with_llm(page)

        browser.close()
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["no-llm", "with-llm", "refusal"], required=True,
                    help="no-llm: 空态+降级；with-llm: 正常回答+四阶段；refusal: 只补拒答态")
    run(ap.parse_args().phase)
