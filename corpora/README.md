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
| `fastapi-zh/` | [fastapi/fastapi](https://github.com/fastapi/fastapi) `docs/zh/docs/` | 122 文件 / 677KB / 1445 chunk | MIT |
| `vue-zh/` | [vuejs-translations/docs-zh-cn](https://github.com/vuejs-translations/docs-zh-cn) `src/` | 110 文件 / 941KB / 1804 chunk | **CC BY 4.0**（图片除外） |

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

---

## vue-zh 来源声明

- **上游仓库**：https://github.com/vuejs-translations/docs-zh-cn （Vue 3 官方中文文档）
- **目录**：`src/`
- **固定 commit**：`dda601fe33187dfd913641d58b6cefe829bf1a0d`（2026-09-07）
- **许可**：**CC BY 4.0**，版权归 Yuxi (Evan) You 及 Vue 文档贡献者。
  ⚠️ 与 fastapi-zh 的 MIT **不同**——CC BY 4.0 **要求署名**：使用或分发本语料时必须保留本节署名与许可说明。
  上游 `LICENSE` 明确写着「**所有图片文件**不在 CC BY 4.0 范围内」，因此本语料**只取 `.md` 正文、不含任何图片**。
- **改动**：正文一字未改，嵌套目录结构原样保留（`guide/essentials/`、`api/`、`tutorial/` 等）。剔除项：
  - 贡献者元文档：`style-guide/`（内含 `PLEADE_DO_NOT_TRANSLATE.md`）、`translations/`
  - 无正文占位页：`partners/`、`examples/`（合计不足 1KB，只有链接与动态路由占位符）
  - 资源与配置：`src/public/`、`.vitepress/` 及所有非 `.md` 文件

  > 之所以强调"剔除贡献者元文档"：本项目早先把`docs/`（项目自身的施工笔记）当语料用过，
  > 结果检索被"怎么改这份文档"之类的内容污染。同样的错不犯第二次。
- **落地校验**：110 文件逐字节比对一致；切片 **1804 chunk / chunk_id 零重复 / 110 个 source**。
  语料里有 **6 个不同层级的 `index.md`**，全部靠相对路径区分 —— 这正是 `CHUNK_ID_RULE_VERSION=3` 存在的原因。

### 更新方式

```powershell
# 重新拉取（记得同步 README 里的 commit 号）
$sha = "dda601fe33187dfd913641d58b6cefe829bf1a0d"
Invoke-WebRequest "https://codeload.github.com/vuejs-translations/docs-zh-cn/zip/$sha" -OutFile "$env:TEMP\vue_zh.zip"
Expand-Archive "$env:TEMP\vue_zh.zip" -DestinationPath "$env:TEMP\vue_zh_extract" -Force
# 再把 ...\docs-zh-cn-<sha>\src\**\*.md 按上面的剔除规则拷进 corpora\vue-zh\（保留相对路径）
```

> 本机踩到的两个坑：① 沙箱注入的 `HTTP_PROXY` 可能已失效，PowerShell 里 `Invoke-WebRequest` 会
> 直接连不上；在 Git Bash 里 `export HTTP_PROXY= HTTPS_PROXY=` 清掉后用
> `C:\Windows\System32\curl.exe -sL -o <Windows风格路径> <url>` 直连即可。
> ② `curl.exe` 是原生程序，**不认 `/c/...` 这种 MSYS 路径**，`-o` 必须给 `C:/...` 或 `C:\...`。

---

## ⚠️ vue-zh 尚未配齐的两件事

**换语料 = 索引和验收集都得重做**，这不是可选项（语料与 golden 强绑定）。

| 项 | fastapi-zh | vue-zh |
|---|---|---|
| 语料 | ✅ | ✅ 已落地 |
| 向量索引 | ✅ 已建（`chroma_db/index_meta.json` 指向它） | ❌ 首次启动时自动重建 |
| 可答 golden（18 条） | ✅ | ❌ **需重写**（旧 18 条钉死在 FastAPI 文件上） |
| 拒答（10 条）/ 多证据（3 条） | ✅ | ❌ 需重写 |

→ 切到 `vue-zh` 之前：先 `set MCP_RAG_CORPUS=vue-zh` + `python scripts/corpus_stats.py` 体检，
再按 `scripts/corpus_grep.py`（摘句出题）→ `scripts/validate_golden.py`（可考证性校验）的流程配齐。

---

语料内容变化会改变 `corpus_hash`，下次启动时向量索引自动重建。
