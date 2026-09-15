r"""C7 回归测试：评测链路的 Ollama 地址必须读 OLLAMA_HOST，不能硬编码。

背景（2026-09-15 清单 C7）：`llm_eval.py` / `agent_eval.py` 曾把
`http://127.0.0.1:11434/api/chat` 写死在代码里。容器化后 `127.0.0.1` 指容器自己，
连不到宿主上的 Ollama，而运行时链路（`agent.py` 走 ollama SDK）本就读这个环境变量，
于是**只有评测脚本不通**。

修法：`llm_eval._ollama_chat_url()` 运行时读 `OLLAMA_HOST`；`agent_eval` 的重复实现删掉、
改为复用 `llm_eval` 那一份。

这几条测试是防回退守卫 —— 以后谁再把地址写死，或者又复制出第二份实现，这里会红。

不依赖真实 Ollama：第 4 条自己起一个临时 HTTP 服务，只验证"请求发到哪去了"。
"""

from __future__ import annotations

import pytest

from src.evaluation.llm_eval import _call_ollama, _ollama_chat_url

# ============================================================
# 1. 地址解析
# ============================================================


def test_url_defaults_to_localhost(monkeypatch):
    """没设 OLLAMA_HOST → 退回本机默认地址（保证宿主直接跑评测不受影响）。"""
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    assert _ollama_chat_url() == "http://127.0.0.1:11434/api/chat"


def test_url_accepts_host_without_scheme(monkeypatch):
    """Ollama 官方允许省略 scheme（如 host.docker.internal:11434），要自动补 http://。"""
    monkeypatch.setenv("OLLAMA_HOST", "host.docker.internal:11434")
    assert _ollama_chat_url() == "http://host.docker.internal:11434/api/chat"


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("http://ollama:11434", "http://ollama:11434/api/chat"),
        ("http://ollama:11434/", "http://ollama:11434/api/chat"),
        ("https://rag.example.com/ollama/", "https://rag.example.com/ollama/api/chat"),
        ("  127.0.0.1:11434  ", "http://127.0.0.1:11434/api/chat"),
    ],
)
def test_url_normalization(monkeypatch, raw, expected):
    """补 scheme、去尾部斜杠、去空白 —— 各种写法都要能拼出正确的 /api/chat。"""
    monkeypatch.setenv("OLLAMA_HOST", raw)
    assert _ollama_chat_url() == expected


def test_url_is_read_at_call_time_not_import_time(monkeypatch):
    """地址必须**每次调用**重新读环境变量。

    如果实现改成模块级常量（`_URL = os.environ.get(...)`），
    测试里改环境变量就不会生效，容器/本地切换也会失效 —— 这条守住这个语义。
    """
    monkeypatch.setenv("OLLAMA_HOST", "http://first:11434")
    assert _ollama_chat_url().startswith("http://first")
    monkeypatch.setenv("OLLAMA_HOST", "http://second:11434")
    assert _ollama_chat_url().startswith("http://second")


# ============================================================
# 2. 实际发出去的 URL 就是 OLLAMA_HOST 拼出来的
# ============================================================


def test_call_ollama_posts_to_ollama_host(monkeypatch):
    """最硬的一条：拦下 `requests.post`，断言它收到的 URL 来自 OLLAMA_HOST。

    能同时抓住两种情况：① 地址被硬编码（URL 会是 11434）；
    ② 变量读了但没用在 `post` 上（URL 会是默认值或空）。

    为什么不用"起真实 HTTP 服务"来测：本机环境里代理注入和 Windows 的
    `ConnectionAbortedError(10053)` 会让这种测试时过时挂（实测单跑过、全量挂），
    而它想验证的只是"URL 对不对"——拦 `post` 更准也更稳，且完全不需要网络。
    """
    monkeypatch.setenv("OLLAMA_HOST", "host.docker.internal:11434")  # 故意省略 scheme

    seen: dict = {}

    class _Resp:
        status_code = 200

        @staticmethod
        def json():
            return {"message": {"content": "pong"}}

    def _fake_post(url, **kwargs):
        seen["url"] = url
        seen["payload"] = kwargs.get("json")
        return _Resp()

    import requests

    monkeypatch.setattr(requests, "post", _fake_post)

    assert _call_ollama("ping") == "pong"
    assert seen["url"] == "http://host.docker.internal:11434/api/chat"
    assert seen["payload"]["stream"] is False
    assert seen["payload"]["messages"][-1] == {"role": "user", "content": "ping"}


# ============================================================
# 3. 全链路只有一份实现
# ============================================================


def test_eval_modules_share_one_implementation():
    """agent_eval / acceptance_eval 必须复用 llm_eval 的 `_call_ollama`。

    原先 agent_eval 自己复制了一份，修 C8（num_predict 截断）时只改了 llm_eval，
    复制品仍是旧值 —— 同一逻辑两处维护必漏一处。这条守住"只有一份"。
    """
    from src.evaluation import acceptance_eval, agent_eval, llm_eval

    assert agent_eval._call_ollama is llm_eval._call_ollama
    assert acceptance_eval._call_ollama is llm_eval._call_ollama
