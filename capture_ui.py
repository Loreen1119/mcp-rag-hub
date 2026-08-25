"""用 Playwright 重新截取 RAG 界面全量截图（覆盖旧的报错图）。

前置：streamlit run app.py --server.port 8510 已在运行。
输出：screenshots/ui-initial.png / ui-query-results.png / ui-tab-{BM25,vector,RRF,Cross-Encoder}.png
"""
import time
from pathlib import Path
from playwright.sync_api import sync_playwright

BASE = "http://localhost:8510"
OUT = Path("screenshots")
OUT.mkdir(exist_ok=True)
QUERY = "RAG 混合检索策略"

TABS = [
    ("BM25 关键词", "ui-tab-BM25.png"),
    ("向量语义", "ui-tab-vector.png"),
    ("RRF 融合", "ui-tab-RRF.png"),
    ("Cross-Encoder 精排", "ui-tab-Cross-Encoder.png"),
]


def click_tab(page, name_fragment):
    try:
        page.get_by_role("tab", name=name_fragment).first.click(timeout=8000)
        time.sleep(1.5)
        return True
    except Exception as e:
        print(f"  ! 点击 Tab '{name_fragment}' 失败: {e}")
        return False


with sync_playwright() as p:
    browser = p.chromium.launch(executable_path=r"C:\Program Files\Google\Chrome\Application\chrome.exe")
    ctx = browser.new_context(viewport={"width": 1400, "height": 1000})
    page = ctx.new_page()
    page.goto(BASE, wait_until="networkidle", timeout=90000)
    page.wait_for_selector("input", timeout=60000)
    time.sleep(3)

    # 1) 空态
    page.screenshot(path=str(OUT / "ui-initial.png"))
    print("captured ui-initial.png")

    # 2) 填查询并提交
    box = page.locator("input[type=text]").first
    box.fill(QUERY)
    box.press("Enter")
    time.sleep(6)

    # 出错即停，避免截到报错图
    err = page.locator('[data-testid="stException"]')
    if err.count() > 0:
        print("!! 页面出现异常面板，中止：")
        print(page.locator('[data-testid="stException"]').first.inner_text()[:500])
        browser.close()
        raise SystemExit(1)

    page.screenshot(path=str(OUT / "ui-query-results.png"))
    print("captured ui-query-results.png")

    # 3) 四个阶段 Tab
    for name, filename in TABS:
        if click_tab(page, name):
            page.screenshot(path=str(OUT / filename))
            print(f"captured {filename}")

    browser.close()
print("ALL DONE")
