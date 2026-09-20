r"""C5 联调证据：MCP 服务能被真实客户端按 stdio 拉起来，并列出它承诺的工具。

背景（2026-09-20 补遗）：`src/mcp_server.py` 早就写好了 4 个工具，但 `tests/` 里
没有任何 mcp 测试、`screenshots/` 里也没有 MCP 截图 —— "能被客户端调用"这件事
**只有文档、没有证据**。本文件就是那份证据。

做法不是 import 模块再断言属性，而是**真的把服务当子进程拉起来**，用官方 `mcp` SDK
的 stdio 传输走一遍 `initialize` → `tools/list`。这正是 Claude Desktop / Cursor 等
客户端的行为，只是省掉了 GUI。

⚠️ 三个刻意的设计：

1. **不调用 `search_knowledge` 本体** —— 它会触发 `_ensure_pipeline()`，加载 embedding
   与 1.1GB 的 reranker；测试会从 28 秒变成几分钟，还可能 OOM（2G 目标机上实测过）。
   只断言"工具列表 + 入参 schema"，这已足以证明"可被调用"。

2. **覆盖两种启动方式**：`-m src.mcp_server` 与 `src/mcp_server.py`。后者正是 README
   与模块 docstring 里写给 MCP 客户端的写法，**曾因 sys.path 问题直接崩**
   （`ModuleNotFoundError: No module named 'config'`）—— 参数化这一维就是防它回退。

3. **env 必须显式传**：`mcp` 的 `stdio_client` 默认只继承一组"安全"环境变量，
   `HF_HUB_OFFLINE` / `TRANSFORMERS_OFFLINE` 传不进去 → 子进程会去联网查模型更新，
   本机实测会挂约 10 分钟。这条坑 2026-09-20 踩过一次。

⚠️ 成本：每次拉起服务约 28 秒（导入 torch / transformers），本文件 2 次 ≈ 56 秒。
这是买"真联调"的代价 —— 换成 import 模块的假测试可以 0.1 秒，但它什么也证明不了。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import anyio
import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

PROJECT_ROOT = Path(__file__).resolve().parents[1]

#: 服务对外承诺的四个工具。改这里等于改对外契约 —— 须同步 README 与模块 docstring。
EXPECTED_TOOLS = {"search_knowledge", "list_documents", "get_chunk", "get_chunk_count"}

#: 单次握手上限。正常约 28 秒（导入成本），给足余量让慢机器也能过。
HANDSHAKE_TIMEOUT_SECONDS = 180


def _server_params(args: list[str]) -> StdioServerParameters:
    """构造 stdio 启动参数（cwd 固定为仓库根，与服务实际部署方式一致）。"""
    env = {
        **os.environ,
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "PYTHONIOENCODING": "utf-8",
    }
    return StdioServerParameters(
        command=sys.executable, args=args, cwd=str(PROJECT_ROOT), env=env
    )


def _handshake(args: list[str]):
    """拉起服务 → initialize → tools/list，返回 (初始化结果, 工具列表)。"""

    async def main():
        with anyio.fail_after(HANDSHAKE_TIMEOUT_SECONDS):
            async with stdio_client(_server_params(args)) as (read, write):
                async with ClientSession(read, write) as session:
                    init = await session.initialize()
                    tools = await session.list_tools()
                    return init, tools

    try:
        return anyio.run(main)
    except Exception as exc:  # 把启动失败翻译成人能看懂的一句话（防回退时最有用）
        raise AssertionError(
            f"MCP 服务未能完成 stdio 握手：args={args}\n"
            f"原始异常：{exc!r}\n"
            f"提示：若为 ModuleNotFoundError，多半是启动方式缺少 sys.path 引导 —— "
            f"见 src/mcp_server.py 顶部的 __package__ 分支。"
        ) from exc


# ============================================================
# 1. 握手 + 工具列表（两种启动方式各一遍）
# ============================================================


@pytest.mark.parametrize(
    "args",
    [["-m", "src.mcp_server"], ["src/mcp_server.py"]],
    ids=["module", "script"],
)
def test_stdio_handshake_exposes_promised_tools(args):
    """客户端能连上、服务名正确、工具集合与承诺一致、检索工具的入参 schema 可用。

    `script` 这一路尤其重要：它是文档里给客户端的写法，历史上是坏的。
    """
    init, tool_list = _handshake(args)

    assert init.serverInfo.name == "RAG Knowledge Server"

    names = {t.name for t in tool_list.tools}
    assert names == EXPECTED_TOOLS, (
        f"工具集合与契约不符：多出 {sorted(names - EXPECTED_TOOLS)}，"
        f"缺少 {sorted(EXPECTED_TOOLS - names)}"
    )

    # 每个工具都得有描述 —— 客户端靠它决定调不调，空描述等于接口没写完
    missing_desc = sorted(t.name for t in tool_list.tools if not (t.description or "").strip())
    assert not missing_desc, f"以下工具缺 description：{missing_desc}"

    search = next(t for t in tool_list.tools if t.name == "search_knowledge")
    props = set((search.inputSchema or {}).get("properties", {}))
    assert {"query", "top_k"} <= props, f"search_knowledge 入参变了：{sorted(props)}"


# ============================================================
# 2. 启动方式本身（不依赖子进程，快速）
# ============================================================


def test_module_importable_without_side_effects():
    """import 服务模块不应有重活（管线是懒加载）—— 否则每次客户端连接都要等模型。

    这里只验证"导入不报错、不触发模型加载"，因此是秒级测试；真正的进程级证据在上面。
    """
    import importlib

    mod = importlib.import_module("src.mcp_server")
    assert getattr(mod, "mcp", None) is not None
    # `_ensure_pipeline` 是懒加载入口，模块级不应已经建好管线
    assert getattr(mod, "_ctx", None) is None, "模块导入时就建了管线 —— 应保持懒加载"
