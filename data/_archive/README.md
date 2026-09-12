# data/_archive/

这个目录不是代码依赖，是**2026-09-12 数据丢失事件的归档**。

## 背景

当天执行 `git rm data/train_queries.json data/test_queries_all.json`（方案 A：删掉为已废弃的 CE 阈值方案服务的 train/test 拆分）后，
**整个 `data/` 目录被文件系统层静默清空**——10 个文件全不见，git 完全不知情。

排查与恢复全过程见 [`docs_knowledge/开发过程中遇到的问题.md`](../../docs_knowledge/开发过程中遇到的问题.md) 第 29、30 条。

## 这里的文件是什么

| 文件 | 字节 | 来源 | 说明 |
|---|---|---|---|
| `train_queries.json` | 8931 | `git show HEAD:data/train_queries.json` | 旧语料 18 条训练集。丢失前的工作区版本是 9087 字节 = 本文件 **+156 字节 CRLF**（156 行全部 CRLF 化），**内容逐字相同** |
| `test_queries_all.json` | 17895 | `modify_backup/27.d.3b73618a.test_queries_all.json` | 旧语料 36 条全集。字节数与丢失前完全一致（该备份是 `git rm` 触发的"删除前"副本） |
| `multi_evidence_queries.old-v1.json` | 1924 | `modify_backup/25.m.deb32cc7.multi_evidence_queries.json` | 旧版多证据集 v1。它派生出的 v2（2137B）**没有留下任何快照，已不可恢复**，这是该文件线下唯一幸存的旧版 |

## 为什么放在 `_archive/` 而不是 `data/`

方案 A 的结论（现有测试集收敛为**单一 golden set**）依然成立，这三个文件**不参与任何评测**：

- `train_queries.json` / `test_queries_all.json` 的 golden 指向旧 `docs/` 语料，跑起来全 0
- `multi_evidence_queries.old-v1.json` 已被重写的 `data/multi_evidence_queries.json` 取代

放在 `data/` 根目录会让它们被误当成"当前测试集"，所以隔离到这里。**只作史料，不作输入。**

## 本次事故中确认不可恢复的（4 项）

| 文件 | 字节 | 性质 | 影响 |
|---|---|---|---|
| `knowledge_triples.jsonl` | 42470 | `.gitignore` 声明"可通过 kg_builder.py 重新生成" | 无。且它是旧 `docs/` 语料抽的，`CHUNK_ID_RULE_VERSION` 2→3 后本就整体失效 |
| `knowledge_triples.jsonl.bak` | 42470 | 上述文件的手工副本 | 无 |
| `test_queries.json.bak` | 10907 | 被 `*.bak` 规则忽略的备份 | 无。正本在 git HEAD |
| `multi_evidence_queries_v2.json` | 2137 | 旧版多证据集 v2 | 轻微。已被重写版取代 |

部分幸存：`data/knowledge_triples.control.jsonl`（14669B，tracked 已恢复）是同一批三元组的过滤子集。

## 给未来的自己

四条恢复通道，**出事前就该知道**：

1. `git show HEAD:<path>` — 覆盖所有 tracked 文件
2. `~/.workbuddy/file-history/<session-id>/<pathhash>@v<N>` — 本会话**改过**的文件的完整快照（文件名是路径哈希，用旧版字节数+时间戳反推匹配）
3. `~/.workbuddy/workspace/sessions/<session-id>/modify_backup/<seq>.<a|m|d>.<hash>.<basename>` — `d`=删除前、`m`=修改前
4. 同目录 `.modify_backup_meta/<hash>.<basename>` — 内容就是完整原始路径

**untracked + 不可再生 = 裸奔。** 按这个判据筛一遍，就知道哪些文件该提交。
