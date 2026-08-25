"""Capture Streamlit UI screenshots for README showcase."""
import time
from playwright.sync_api import sync_playwright

BASE = "http://localhost:8510"
OUT = "screenshots"
QUERY = "RAG 混合检索策略"  # a query with known good matches in docs/

def click_tab(page, name_fragment):
    """Click a tab by its text fragment."""
    try:
        page.get_by_role("tab", name=name_fragment).first.click(timeout=5000)
        time.sleep(1.2)
        return True
    except Exception:
        return False

with sync_playwright() as p:
    browser = p.chromium.launch(executable_path=r"C:\Program Files\Google\Chrome\Application\chrome.exe")
    ctx = browser.new_context(viewport={"width": 1400, "height": 1000})
    page = ctx.new_page()
    page.goto(BASE, wait_until="networkidle")
    time.sleep(15)  # let pipeline build & render

    # 1) initial empty state
    page.screenshot(path=f"{OUT}/ui-initial.png", full_page=False)

    # 2) submit a real query
    box = page.locator("input[type='text']").first
    box.fill(QUERY)
    box.press("Enter")
    time.sleep(4)  # wait for retrieval + CE rerank

    page.screenshot(path=f"{OUT}/ui-query-results.png", full_page=False)

    # 3) four stage tabs (click each, capture)
    tabs = ["BM25", "向量语义", "RRF", "Cross-Encoder"]
    for t in tabs:
        if click_tab(page, t):
            page.screenshot(path=f"{OUT}/ui-tab-{t}.png", full_page=False)

    browser.close()
print("DONE")
