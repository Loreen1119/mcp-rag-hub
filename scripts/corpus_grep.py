"""按 (文件, 正则) 批量摘录语料关键句，用于编写可考证的 golden_answer。

用法：python scripts/corpus_grep.py  > 输出重定向自己管
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import DOCS_DIR  # noqa: E402

SPECS: list[tuple[str, str]] = [
    ("tutorial/first-steps.md", r"路径操作|装饰器|get\(|uvicorn|运行"),
    ("tutorial/path-params.md", r"路径参数|path parameter|item_id|类型"),
    ("tutorial/query-params.md", r"查询参数|query|skip|limit|默认值"),
    ("tutorial/body.md", r"请求体|BaseModel|Pydantic|body"),
    ("tutorial/dependencies/index.md", r"依赖|Depends|注入"),
    ("tutorial/middleware.md", r"中间件|middleware|请求|响应"),
    ("tutorial/handling-errors.md", r"HTTPException|404|错误"),
    ("advanced/events.md", r"lifespan|生命周期|启动|关闭"),
    ("tutorial/cors.md", r"CORSMiddleware|allow_origins|跨域"),
    ("advanced/settings.md", r"BaseSettings|环境变量|Settings"),
    ("tutorial/background-tasks.md", r"BackgroundTasks|后台任务"),
    ("tutorial/testing.md", r"TestClient|测试"),
    ("advanced/custom-response.md", r"ORJSONResponse|HTMLResponse|Response"),
    ("advanced/websockets.md", r"WebSocket|websocket"),
    ("tutorial/sql-databases.md", r"SQLAlchemy|Session|数据库|get_db"),
    ("tutorial/security/oauth2-jwt.md", r"JWT|token|OAuth2"),
    ("deployment/docker.md", r"Dockerfile|uvicorn|docker|镜像"),
    ("tutorial/response-status-code.md", r"status_code|状态码"),
    ("tutorial/request-files.md", r"UploadFile|File|文件"),
    ("tutorial/security/first-steps.md", r"安全|OAuth2|密码|token"),
]


def main() -> None:
    buf: list[str] = []

    def emit(line: str = "") -> None:
        buf.append(line)

    for rel, pat in SPECS:
        p = DOCS_DIR / rel
        emit("=" * 78)
        if not p.exists():
            emit(f"[缺失] {rel}")
            continue
        lines = p.read_text(encoding="utf-8").splitlines()
        rx = re.compile(pat)
        hits = [(i + 1, ln.strip()) for i, ln in enumerate(lines) if rx.search(ln)]
        emit(f"[{rel}]  行数={len(lines)}  命中={len(hits)}")
        for no, ln in hits[:10]:
            if ln:
                emit(f"  {no:>4}| {ln[:150]}")
        emit()

    text = "\n".join(buf)
    if len(sys.argv) > 1:
        # 直接写 UTF-8 文件：避免 PowerShell 管道按 GBK 解码把中文变成乱码
        Path(sys.argv[1]).write_text(text, encoding="utf-8")
        print(f"written: {sys.argv[1]}")
    else:
        print(text)


if __name__ == "__main__":
    main()
