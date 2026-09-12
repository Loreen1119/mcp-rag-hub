# corpora/ — RAG 知识库语料库

每个子目录是一份独立的**真实文档语料**。用环境变量 `MCP_RAG_CORPUS` 选择当前生效的语料，
不同语料使用各自独立的 Chroma collection，互不覆盖。

```bash
# 默认 fastapi-zh
python -m streamlit run app.py

# 切到另一份语料
set MCP_RAG_CORPUS=vue-zh && python -m streamlit run app.py
```

## 已有语料

| 目录 | 来源 | 体量 | 许可 |
|---|---|---|---|
| `fastapi-zh/` | [fastapi/fastapi](https://github.com/fastapi/fastapi) `docs/zh/docs/` | 122 文件 / 677KB | MIT |

> 原来的 `docs/` 目录**不再作为知识库**，它是项目自身的文档（设计说明、速查表）。
> 语料换成真实第三方文档后，`docs/` 保留原样仅供人阅读。

---

## fastapi-zh 来源声明

- **上游仓库**：https://github.com/fastapi/fastapi
- **目录**：`docs/zh/docs/`
- **固定 commit**：`50113da16fec53b66b80d75e80a89296de4fa5a5`
- **许可**：MIT（见上游仓库 `LICENSE`，版权归 Sebastián Ramírez 及 FastAPI 贡献者）
- **改动**：仅剔除了 MkDocs 非正文文件（`_` 前缀的 partial 与 `translation-banner.md`），
  正文内容未做任何修改；目录结构原样保留（`tutorial/` `advanced/` `how-to/` `deployment/` 等）。

### 更新方式

```powershell
# 重新拉取（记得同步 README 里的 commit 号）
$zip = "$env:TEMP\fastapi_master.zip"
Invoke-WebRequest "https://codeload.github.com/fastapi/fastapi/zip/refs/heads/master" -OutFile $zip
Expand-Archive $zip -DestinationPath "$env:TEMP\fastapi_extract" -Force
# 再把 ...\docs\zh\docs\**\*.md 拷进 corpora\fastapi-zh\（保留相对路径）
```

语料内容变化会改变 `corpus_hash`，下次启动时向量索引自动重建。
