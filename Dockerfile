# ============================================================
# MCP-RAG-Hub —— 单容器镜像（Streamlit 界面 + MCP 服务共用一份代码）
#
# 设计要点（每条都是踩过或差点踩到的坑，不是模板抄来的）：
#   1. 只装 CPU 版 torch：直接从 PyPI 装 torch 在 Linux 上会拉 2GB+ 的 CUDA 依赖，
#      本项目无 GPU（Intel 集显），必须走 download.pytorch.org/whl/cpu。
#   2. 模型在构建期从 ModelScope 下载并塞进镜像内的 HuggingFace 缓存：
#      服务器连不上 HuggingFace 原站，改用 ModelScope（已实测可达）；HF_HUB_OFFLINE=1 离线加载。
#   3. HF_HUB_OFFLINE=1 必须开：否则 transformers 启动时会尝试联网查更新，
#      实测在墙内会卡住约 10 分钟（详见 docs_knowledge/开发过程中遇到的问题.md）。
#   4. Streamlit 必须绑 0.0.0.0：默认只监听 127.0.0.1，容器外访问不到（经典坑）。
#   5. OLLAMA_HOST 指向宿主：Ollama 跑在宿主而非容器内，不装进镜像。
#      （旧问题已修：src/evaluation/{llm_eval,agent_eval}.py 曾硬编码 127.0.0.1:11434，
#       容器内跑评测连不上宿主 Ollama；2026-09-15 起统一由 llm_eval._ollama_chat_url()
#       在**运行时**读 OLLAMA_HOST，容器内可直接跑评测，不再需要 network_mode: host 变通。）
#      另注：config.LLM_BACKEND 默认 "deepseek"（云端，零本地内存），
#      只有设 MCP_RAG_LLM_BACKEND=ollama 时才走本地 Ollama 这条路。
# ============================================================

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONIOENCODING=utf-8 \
    LANG=C.UTF-8 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    OLLAMA_HOST=http://host.docker.internal:11434 \
    MCP_RAG_CORPUS=fastapi-zh

WORKDIR /app

# libgomp1：torch CPU 运行时依赖；curl：下面 HEALTHCHECK 用
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 curl \
 && rm -rf /var/lib/apt/lists/*

# 先装依赖，利用 Docker 层缓存（改代码不必重装依赖）
COPY requirements.txt ./

# 国内源：服务器连不上 pytorch.org / 官方 PyPI，统一走清华镜像（已实测可达）。
# 注：清华 PyPI 上的 torch 是含 CUDA 的版本（约 2GB+），比 pytorch.org 的纯 CPU 版大，
# 但 CPU 推理功能完全一致。要更小镜像可改回 pytorch.org/whl/cpu（需服务器能连外网）。
ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

# 先装 torch（清华 PyPI 镜像），后面 -r requirements.txt 就不会再拉它。
RUN pip install --index-url "$PIP_INDEX_URL" torch \
 && pip install --index-url "$PIP_INDEX_URL" -r requirements.txt

# 模型：服务器连不上 HuggingFace 原站，改用 ModelScope 下载并放入镜像内 HF 缓存，
# 容器以 HF_HUB_OFFLINE=1 离线加载（见下方 ENV）。放在 COPY . . 之前以复用缓存层。
RUN pip install --index-url "$PIP_INDEX_URL" modelscope \
 && python -c "from modelscope.hub.snapshot_download import snapshot_download; snapshot_download('BAAI/bge-small-zh-v1.5', local_dir='/root/.cache/huggingface/hub/models--BAAI--bge-small-zh-v1.5/snapshots/local'); snapshot_download('BAAI/bge-reranker-base', local_dir='/root/.cache/huggingface/hub/models--BAAI--bge-reranker-base/snapshots/local')"
RUN printf 'local' > /root/.cache/huggingface/hub/models--BAAI--bge-small-zh-v1.5/refs/main \
 && printf 'local' > /root/.cache/huggingface/hub/models--BAAI--bge-reranker-base/refs/main

# 再拷代码与语料（corpora/ 一并烧进镜像，让镜像自带可演示的知识库）
COPY . .

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8501/_stcore/health || exit 1

CMD ["python", "-m", "streamlit", "run", "app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.headless=true"]
