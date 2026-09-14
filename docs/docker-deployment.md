# Docker 部署

> 目标：把这个项目做成「一条命令拉起、不依赖本机 Python 环境」的可交付形态。
> 涉及文件：`Dockerfile`、`.dockerignore`、`docker-compose.yml`（均在项目根）。

---

## 一、两条前置条件（都不是可选的）

### 1. 指向宿主的 HuggingFace 模型缓存

容器**不烧模型**，而是挂载宿主已缓存的模型目录。原因：国内直连 `huggingface.co`
不稳定，构建期下载几乎必然失败；而本机缓存里已经有全部需要的模型
（`BAAI/bge-small-zh-v1.5` 嵌入 + `BAAI/bge-reranker-base` 精排）。

```powershell
$env:HF_CACHE = "$env:USERPROFILE\.cache\huggingface"   # 本机 = C:\Users\REDMI\.cache\huggingface
```

`docker-compose.yml` 里这个变量用了 `${HF_CACHE:?...}` 语法，**没设会直接报错退出**。
这是故意的：如果静默挂一个空目录，容器会以为没有模型、转而联网下载 → 在墙内卡死，
而且报错信息完全不指向真实原因（这个坑在 `开发过程中遇到的问题.md` 里记过同类现象）。

### 2. 宿主上把 Ollama 起起来

Ollama 跑在宿主，不在容器里。另开一个窗口（保持不关）：

```powershell
& "D:\1software\ollama\ollama-windows-amd64\ollama.exe" serve
```

容器通过 `OLLAMA_HOST=http://host.docker.internal:11434` 访问它。

> 没起也行 —— 界面会显示"回答生成服务暂不可用"并回落为检索原文拼接，不会白屏。
> 但要演示问答，就得起。

---

## 二、跑起来

```powershell
# 在项目根目录
$env:HF_CACHE = "$env:USERPROFILE\.cache\huggingface"
docker compose up --build
```

浏览器打开 <http://localhost:8501>。停止：`docker compose down`。

换语料不用改文件：

```powershell
$env:MCP_RAG_CORPUS = "vue-zh"; docker compose up
```

---

## 三、验证清单

```powershell
# ① 配置语法 + 变量校验（不需要守护进程，秒出）
docker compose config

# ② 构建前先确认"能不能拉到基础镜像"（国内 Docker Hub 直连不通，靠 daemon.json 里的镜像源）
docker pull python:3.11-slim

# ③ 构建
docker compose build
#    想用国内 PyPI 加速：
#    docker compose build --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

# ④ 跑起来之后：健康检查是否通过
docker compose ps                    # STATUS 里应出现 (healthy)

# ⑤ 容器里能不能真的读到模型（不该有任何联网行为）
docker compose exec rag-hub python -c "import os; print(os.environ['HF_HUB_OFFLINE']); from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-zh-v1.5'); print('model ok')"

# ⑥ 容器能不能连到宿主 Ollama
docker compose exec rag-hub curl -s http://host.docker.internal:11434/api/tags
```

### 本机网络实况（2026-09-14 预检）

| 目标 | 结果 | 说明 |
|---|---|---|
| `registry-1.docker.io`（Docker Hub） | **直连超时** | 国内常态。靠 `~/.docker/daemon.json` 里的 `registry-mirrors` 兜底 |
| `docker.m.daocloud.io` | 401 | 正常（registry 要求 token，Docker 会自动取） |
| `docker.1panel.live` | **200，能取到 `python:3.11-slim` 的 manifest** | 当前可用 |
| `hub.rat.dev` | 302 | 可达 |
| `download.pytorch.org/whl/cpu` | 200 / 0.9s | 可达，不需要镜像 |
| `pypi.org` / `files.pythonhosted.org` | 200 | 可达，但 wheel 下载偏慢 → 可用清华镜像加速 |
| `pypi.tuna.tsinghua.edu.cn` | 200 / 0.5s | 可用 |

> 镜像源会失效，别当永久事实。构建前先跑上面第 ② 步探一下，比直接 build 失败再看日志快得多。

---

## 四、已知限制（写清楚，不当没看见）

### 1. 本机尚未 `build` 验证过

写这份文档时 Docker 守护进程没有运行，而且启动 Docker Desktop 会占用数 GB 内存 ——
那和"腾出内存跑 7b 裁判"直接冲突，所以当时没有启动它。
**首次 `build` 时请留意两点**：

- **网络**：会拉 `python:3.11-slim` + 从 `download.pytorch.org/whl/cpu` 装 torch + 从 PyPI 装其余依赖。
  若本机代理不稳，可能中途失败，重试即可（Docker 层缓存会保留已完成的部分）。
- **体积**：预计 1.5~2.5 GB（torch CPU 约 200MB，`ragas` + `datasets` + `transformers` 占大头）。

### 2. `llm_eval.py` / `agent_eval.py` 在容器内**跑不了评测**

这两个脚本把 Ollama 地址**硬编码**成 `http://127.0.0.1:11434/api/chat`
（`src/evaluation/llm_eval.py:98`、`agent_eval.py:61`），在容器里 `127.0.0.1` 指的是容器自己，
不是宿主。而 `agent.py` 走的是 ollama SDK，会读 `OLLAMA_HOST`，所以**问答链路在容器里是通的**，
只有评测脚本不通。

→ 现阶段：**评测在宿主直接跑**（各脚本都已就绪）。
→ 要修的话：把这两处改成读 `OLLAMA_HOST` 环境变量（约 4 行），已在未完成清单里记为可选项。

### 3. MCP 服务不适合当常驻 compose service

MCP 用的是 **stdio 传输** —— 由客户端（Claude Desktop / Cursor）把它当子进程拉起来，通过标准输入输出通信，
不是监听端口的服务。所以容器化 MCP 的正确形态是 `docker run -i`：

```jsonc
// claude_desktop_config.json 片段
{
  "mcpServers": {
    "rag-knowledge": {
      "command": "docker",
      "args": [
        "run", "-i", "--rm",
        "--add-host", "host.docker.internal:host-gateway",
        "-e", "OLLAMA_HOST=http://host.docker.internal:11434",
        "-e", "HF_HUB_OFFLINE=1", "-e", "TRANSFORMERS_OFFLINE=1",
        "-v", "C:\\Users\\REDMI\\.cache\\huggingface:/root/.cache/huggingface",
        "-v", "D:\\1base\\computer\\Agent\\DevRoot\\mcp-rag-hub\\chroma_db:/app/chroma_db",
        "mcp-rag-hub:local", "python", "src/mcp_server.py"
      ]
    }
  }
}
```

### 4. 语料是烧进镜像的

`corpora/` 会 `COPY` 进镜像（两份语料合计 1.6MB，让镜像自带可演示的知识库）。
新增语料后需要 `docker compose up --build` 重建。

---

## 五、这次容器化暴露出的两个隐含假设

写 Dockerfile 的价值不只是"能跑在容器里"，更在于它逼你把**隐含假设**一个个写出来。
这次挖出两条，都是在本机跑得好好的、一进容器就挂的：

| 隐含假设 | 容器里的现实 | 处置 |
|---|---|---|
| "Ollama 就在本机 127.0.0.1:11434" | 容器里的 127.0.0.1 是容器自己 | `agent.py` 走 SDK 读 `OLLAMA_HOST`，容器里能用；两个评测脚本硬编码，不通（见限制 2） |
| "Streamlit 起来就能访问" | 默认只绑 127.0.0.1，容器外访问不到 | CMD 里显式 `--server.address=0.0.0.0` |

这两条比 Dockerfile 本身更适合拿去讲 —— 它们说明的是**"能在我机器上跑"和"能交付"之间的距离**。
