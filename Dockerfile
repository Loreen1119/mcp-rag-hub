# ============================================================
# MCP-RAG-Hub —— 单容器镜像（Streamlit 界面 + MCP 服务共用一份代码）
#
# 设计要点（每条都是踩过或差点踩到的坑，不是模板抄来的）：
#   1. 只装 CPU 版 torch：直接从 PyPI 装 torch 在 Linux 上会拉 2GB+ 的 CUDA 依赖，
#      本项目无 GPU（Intel 集显），必须走 download.pytorch.org/whl/cpu。
#   2. 模型不烧进镜像，而是挂载宿主已有的 HuggingFace 缓存（~/.cache/huggingface）：
#      国内直连 huggingface.co 不稳定，构建期下载几乎必然失败；宿主已缓存全部模型。
#   3. HF_HUB_OFFLINE=1 必须开：否则 transformers 启动时会尝试联网查更新，
#      实测在墙内会卡住约 10 分钟（详见 docs_knowledge/开发过程中遇到的问题.md）。
#   4. Streamlit 必须绑 0.0.0.0：默认只监听 127.0.0.1，容器外访问不到（经典坑）。
#   5. OLLAMA_HOST 指向宿主：Ollama 跑在宿主而非容器内。
#      注意 ollama SDK 读 OLLAMA_HOST，但 src/evaluation/{llm_eval,agent_eval}.py 里
#      是**硬编码** 127.0.0.1:11434 的 —— 容器内跑评测会连不上宿主 Ollama，
#      评测请在宿主直接跑，或用 docker-compose 的 network_mode: host 变通。
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
# 先单独用 CPU 专用源满足 torch，后面的 sentence-transformers 就不会再拉 CUDA 版
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch \
 && pip install -r requirements.txt

# 再拷代码与语料（corpora/ 一并烧进镜像，让镜像自带可演示的知识库）
COPY . .

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8501/_stcore/health || exit 1

CMD ["python", "-m", "streamlit", "run", "app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.headless=true"]
