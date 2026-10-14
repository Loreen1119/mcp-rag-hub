#!/usr/bin/env python
"""TiDB 语料抓取（数据落盘，不建索引）。

来源与落点：
    docs-cn  pingcap/docs-cn 的 release-8.5 分支      -> corpora/tidb-zh/docs-cn/
    asktug   AskTUG 论坛（Discourse）已采纳答案的主题  -> corpora/tidb-zh/asktug/
    tidbnet  TiDB 社区专栏（中文技术长文）            -> corpora/tidb-zh/tidb-net/

设计要点：
    - 可断点续跑：输出文件已存在则跳过，重跑不重复下载。
    - 只落数据：不触碰 chroma_db / data/ 下的评测集，改索引是后续独立步骤。
    - 元数据（话题索引、抓取清单）落在 corpora/tidb-zh/_meta/，后缀非 .md，
      不会被 src/data_pipeline.iter_supported_docs 扫进语料。

用法：
    python scripts/fetch_tidb_corpus.py docs
    python scripts/fetch_tidb_corpus.py asktug --target-solved 5000
    python scripts/fetch_tidb_corpus.py stats
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tarfile
import time
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

import requests

# ---------------------------------------------------------------- 路径与常量

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORPUS_DIR = PROJECT_ROOT / "corpora" / "tidb"
META_DIR = CORPUS_DIR / "_meta"

DOCS_DIR = CORPUS_DIR / "docs-cn"
ASKTUG_DIR = CORPUS_DIR / "asktug"
TIDBNET_DIR = CORPUS_DIR / "tidb-net"

DOCS_REPO = "pingcap/docs-cn"
DOCS_BRANCH = "release-8.5"
DOCS_TARBALL = (
    f"https://codeload.github.com/{DOCS_REPO}/tar.gz/refs/heads/{DOCS_BRANCH}"
)

ASKTUG_BASE = "https://asktug.com"

TIDBNET_BASE = "https://tidb.net"
TIDBNET_PAGE_SIZE = 20          # 上游固定 20 条/页
# 专栏里的非技术体裁（社区动态 / 活动招募），干货率统计要单独剔出来
TIDBNET_NON_TECH_CATEGORY = {"社区与动态"}
TIDBNET_NON_TECH_PATTERN = re.compile(
    r"(活动|直播|预告|回顾|报名|招募|招聘|颁奖|大赛|Happy Hour|Meetup|茶话会|"
    r"问卷|抽奖|获奖|公示|通知|邀请函|用户组|读书会|打卡|征稿)",
    re.I,
)
# 判定「干货」的技术信号词
TIDBNET_TECH_TERMS = (
    "sql", "select", "insert", "update", "delete", "tiup", "tikv", "tiflash", "pd ", "region",
    "集群", "扩容", "缩容", "配置", "参数", "报错", "日志", "慢查询", "索引", "事务", "优化",
    "备份", "恢复", "监控", "性能", "压测", "迁移", "升级", "版本", "排障", "故障", "内存",
    "dashboard", "grafana", "prometheus", "tidb-lightning", "dumpling", "br ", "cdc",
)
TIDBNET_MIN_CHARS = 600         # 正文太短的当水文

UA = {"User-Agent": "mcp-rag-hub-corpus-fetcher/1.0 (+research corpus build)"}

# docs-cn 中不属于正文的内容：贡献者元文档 / 导航页 / 站务配置
DOCS_EXCLUDE_DIRS = (".agents/", ".github/", ".vaunt/")
DOCS_EXCLUDE_NAMES = {"CONTRIBUTING.md", "README.md"}

# AskTUG：单帖正文上限，防止个别超长帖把语料撑爆
ASKTUG_MAX_CHARS = 16000
ASKTUG_MIN_POST_CHARS = 40


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------- 断点状态

def _load_state(path: Path) -> dict:
    """读取断点状态；不存在或损坏都返回空字典（等价于从头跑）。"""
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_state(path: Path, state: dict) -> None:
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------- HTTP

def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update(UA)
    return s


def _get(sess: requests.Session, url: str, *, tries: int = 4, timeout: int = 30):
    """带指数退避的 GET；429/5xx 重试，其余直接抛，返回 response。"""
    delay = 2.0
    for attempt in range(1, tries + 1):
        try:
            r = sess.get(url, timeout=timeout)
        except requests.RequestException as exc:
            if attempt == tries:
                raise
            log(f"    网络异常 {exc.__class__.__name__}，{delay:.0f}s 后重试 ({attempt}/{tries})")
            time.sleep(delay)
            delay *= 2
            continue

        if r.status_code in (429, 500, 502, 503, 504):
            if attempt == tries:
                r.raise_for_status()
            ra = r.headers.get("Retry-After")
            wait = float(ra) if (ra or "").replace(".", "", 1).isdigit() else delay
            log(f"    HTTP {r.status_code}，{wait:.0f}s 后重试 ({attempt}/{tries})")
            time.sleep(wait)
            delay *= 2
            continue

        r.raise_for_status()
        return r

    raise RuntimeError(f"重试耗尽: {url}")


def get_json(sess: requests.Session, url: str, *, tries: int = 4, timeout: int = 30):
    return _get(sess, url, tries=tries, timeout=timeout).json()


def get_text(sess: requests.Session, url: str, *, tries: int = 4, timeout: int = 30) -> str:
    return _get(sess, url, tries=tries, timeout=timeout).text


# ---------------------------------------------------------------- HTML -> 文本

_BLOCK_TAGS = {"p", "div", "li", "tr", "blockquote", "pre", "h1", "h2", "h3", "h4", "h5", "h6"}
_DROP_TAGS = {"script", "style", "head", "noscript"}


class _TextExtractor(HTMLParser):
    """把 Discourse 的 cooked HTML 转成可读纯文本（不引入 bs4 依赖）。"""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.out: list[str] = []
        self._drop_depth = 0
        self._pre_depth = 0
        self._li_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in _DROP_TAGS:
            self._drop_depth += 1
            return
        if self._drop_depth:
            return
        if tag == "br":
            self.out.append("\n")
        elif tag == "pre":
            self._pre_depth += 1
            self.out.append("\n```\n")
        elif tag in ("ul", "ol"):
            self.out.append("\n")
        elif tag == "li":
            self._li_depth += 1
            self.out.append("\n- ")
        elif tag == "img":
            self.out.append("[图片]")
        elif tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            self.out.append("\n" + "#" * int(tag[1]) + " ")
        elif tag in _BLOCK_TAGS:
            self.out.append("\n")

    def handle_endtag(self, tag):
        if tag in _DROP_TAGS:
            self._drop_depth = max(0, self._drop_depth - 1)
            return
        if self._drop_depth:
            return
        if tag == "pre":
            self._pre_depth = max(0, self._pre_depth - 1)
            self.out.append("\n```\n")
        elif tag == "li":
            self._li_depth = max(0, self._li_depth - 1)
        elif tag in _BLOCK_TAGS:
            self.out.append("\n")

    def handle_data(self, data):
        if self._drop_depth:
            return
        self.out.append(data if self._pre_depth else re.sub(r"\s+", " ", data))

    def text(self) -> str:
        raw = "".join(self.out)
        raw = re.sub(r"[ \t]+\n", "\n", raw)
        raw = re.sub(r"\n{3,}", "\n\n", raw)
        return raw.strip()


def html_to_text(html: str) -> str:
    if not html:
        return ""
    p = _TextExtractor()
    p.feed(html)
    p.close()
    return p.text()


def safe_name(name: str, limit: int = 60) -> str:
    cleaned = re.sub(r"[\x00-\x1f/\\:*?\"<>|]+", "_", name).strip("_ .")
    return (cleaned or "untitled")[:limit]


# ---------------------------------------------------------------- docs-cn

def cmd_docs(args) -> int:
    branch = getattr(args, "branch", None) or DOCS_BRANCH
    dest_dir = Path(args.dest) if getattr(args, "dest", None) else DOCS_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)

    tarball = f"https://codeload.github.com/{DOCS_REPO}/tar.gz/refs/heads/{branch}"
    sess = _session()

    log(f"下载 {DOCS_REPO}@{branch} 并流式抽取 .md（图片等非 md 不落盘）…")
    kept = 0
    skipped = 0
    total_md = 0

    with sess.get(tarball, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        resp.raw.decode_content = True
        with tarfile.open(fileobj=resp.raw, mode="r|gz") as tar:
            for member in tar:
                if not member.isfile():
                    continue
                parts = member.name.split("/", 1)
                rel = parts[1] if len(parts) == 2 else parts[0]
                if not rel.lower().endswith(".md"):
                    continue
                total_md += 1

                if any(rel.startswith(d) for d in DOCS_EXCLUDE_DIRS):
                    skipped += 1
                    continue
                base = rel.rsplit("/", 1)[-1]
                if base in DOCS_EXCLUDE_NAMES or base.startswith("_") or base.startswith("TOC"):
                    skipped += 1
                    continue

                fh = tar.extractfile(member)
                if fh is None:
                    continue
                dest = dest_dir / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(fh.read())
                kept += 1

    manifest = META_DIR / f"docs-cn-{branch}.source"
    manifest.write_text(
        "\n".join(
            [
                f"repo        : https://github.com/{DOCS_REPO}",
                f"branch      : {branch}",
                f"license     : CC BY-SA 3.0（见上游 LICENSE；需署名 + 相同方式共享）",
                f"fetched_at  : {datetime.now(timezone.utc).isoformat()}",
                f"md_in_tar   : {total_md}",
                f"md_kept     : {kept}",
                f"md_excluded : {skipped}  ({'+'.join(DOCS_EXCLUDE_DIRS)} / {sorted(DOCS_EXCLUDE_NAMES)} / _前缀 / TOC*)",
                "改动        : 仅剔除贡献者元文档与导航页，正文一字未改；目录结构原样保留。",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    log(f"docs-cn 完成：tar 内 {total_md} 个 md，保留 {kept}，剔除 {skipped} -> {dest_dir}")
    return 0


# ---------------------------------------------------------------- AskTUG

def _enumerate_topics(sess, target_solved: int, max_pages: int, state: dict, state_path: Path) -> list[dict]:
    """翻 /latest.json 收集全站主题，直到攒够 target_solved 个已采纳主题。

    续跑：每页处理完就把新主题**逐行追加**到 asktug_topics.jsonl，并把
    enum_last_page 写进状态文件；重启时从该页的下一页继续，已存在的行不重复写。
    """
    index_path = META_DIR / "asktug_topics.jsonl"
    all_topics: list[dict] = []
    seen: set[int] = set()

    if index_path.exists():
        for line in index_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                t = json.loads(line)
            except json.JSONDecodeError:
                continue
            if t.get("id") is not None and t["id"] not in seen:
                seen.add(t["id"])
                all_topics.append(t)
        log(f"    已有索引 {len(all_topics)} 条，从第 {state.get('enum_last_page', -1) + 1} 页续跑")

    page = int(state.get("enum_last_page", -1)) + 1

    while page < max_pages:
        solved_n = sum(1 for t in all_topics if t["has_accepted_answer"])
        if solved_n >= target_solved:
            log(f"    已攒够 {solved_n} 个已采纳主题，无需继续翻页")
            break

        url = f"{ASKTUG_BASE}/latest.json?no_definitions=true&page={page}"
        try:
            data = get_json(sess, url)
        except Exception as exc:
            log(f"    列表第 {page} 页失败：{exc}（断点已保留在第 {page - 1} 页）")
            break

        tl = data.get("topic_list") or {}
        topics = tl.get("topics") or []
        if not topics:
            log(f"列表翻到第 {page} 页已空，枚举结束")
            state["enum_done"] = True
            _save_state(state_path, state)
            break

        new_rows: list[dict] = []
        for t in topics:
            tid = t.get("id")
            if tid is None or tid in seen:
                continue
            seen.add(tid)
            row = {
                "id": tid,
                "title": t.get("title"),
                "slug": t.get("slug"),
                "posts_count": t.get("posts_count"),
                "reply_count": t.get("reply_count"),
                "views": t.get("views"),
                "like_count": t.get("like_count"),
                "has_accepted_answer": bool(t.get("has_accepted_answer")),
                "created_at": t.get("created_at"),
                "last_posted_at": t.get("last_posted_at"),
                "category_id": t.get("category_id"),
                "tags": t.get("tags") or [],
            }
            all_topics.append(row)
            new_rows.append(row)

        with index_path.open("a", encoding="utf-8") as fh:
            for row in new_rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

        state["enum_last_page"] = page
        state["topics_total"] = len(all_topics)
        state["solved_total"] = sum(1 for t in all_topics if t["has_accepted_answer"])
        _save_state(state_path, state)

        if page % 25 == 0:
            log(f"    列表 page={page} 累计主题 {len(all_topics)} 其中已采纳 {state['solved_total']}")

        if not tl.get("more_topics_url"):
            log("    more_topics_url 为空，枚举到末尾")
            state["enum_done"] = True
            _save_state(state_path, state)
            break

        page += 1
        time.sleep(0.4)

    log(f"主题索引 {index_path}（{len(all_topics)} 条）")
    return all_topics


def _fetch_post_by_number(sess, tid: int, post_number: int) -> dict | None:
    url = f"{ASKTUG_BASE}/t/{tid}/posts.json?post_number={post_number}"
    try:
        data = get_json(sess, url, tries=2)
    except Exception:
        return None
    for p in (data.get("post_stream") or {}).get("posts") or []:
        if p.get("post_number") == post_number:
            return p
    return None


def _render_topic(topic: dict, detail: dict, sess) -> str | None:
    stream = (detail.get("post_stream") or {}).get("posts") or []
    if not stream:
        return None
    by_number = {p.get("post_number"): p for p in stream}

    accepted = detail.get("accepted_answer") or {}
    acc_no = accepted.get("post_number")

    question = by_number.get(1) or min(stream, key=lambda p: p.get("post_number") or 0)

    acc_post = by_number.get(acc_no)
    if acc_post is None and acc_no:
        acc_post = _fetch_post_by_number(sess, topic["id"], acc_no)

    tags = topic.get("tags") or []
    head = [
        f"# {topic.get('title')}",
        "",
        f"> 来源：AskTUG  话题 #{topic.get('id')}  "
        f"发布：{(topic.get('created_at') or '')[:10]}  "
        f"回复：{topic.get('posts_count')}  "
        f"浏览：{topic.get('views')}",
        f"> 链接：{ASKTUG_BASE}/t/{topic.get('slug')}/{topic.get('id')}",
    ]
    if tags:
        head.append(f"> 标签：{', '.join(tags)}")

    body = ["", "## 问题", "", html_to_text(question.get("cooked") or "")]

    acc_text = html_to_text((acc_post or {}).get("cooked") or "")
    if not acc_text:
        acc_text = accepted.get("excerpt") or ""
    if acc_text:
        body += ["", "## 已采纳答案", "", acc_text]

    others = [
        p
        for p in sorted(stream, key=lambda p: p.get("post_number") or 0)
        if p.get("post_number") != 1
        and p.get("post_number") != acc_no
        and not p.get("deleted_at")
    ]
    extra: list[str] = []
    for p in others:
        txt = html_to_text(p.get("cooked") or "")
        if len(txt) >= ASKTUG_MIN_POST_CHARS:
            extra.append(f"**#{p.get('post_number')} @{p.get('username')}**：{txt}")
    if extra:
        body += ["", "## 其他回复", ""] + extra

    text = "\n".join(head + body).strip() + "\n"
    if len(text) > ASKTUG_MAX_CHARS:
        text = text[:ASKTUG_MAX_CHARS].rstrip() + "\n\n（正文超长，已截断）\n"
    return text


def cmd_asktug(args) -> int:
    ASKTUG_DIR.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)
    sess = _session()

    state_path = META_DIR / "asktug.state.json"
    state = _load_state(state_path)
    log(
        f"Phase 1/2 枚举主题（目标已采纳 {args.target_solved} 条）"
        f"{'｜检测到断点，续跑' if state else ''}…"
    )
    topics = _enumerate_topics(sess, args.target_solved, args.max_pages, state, state_path)
    solved = [t for t in topics if t["has_accepted_answer"]]
    log(f"枚举完成：主题 {len(topics)}，已采纳 {len(solved)}")

    log("Phase 2/2 抓取已采纳主题详情（已存在的 .md 直接跳过）…")
    ok = fail = skipped = 0
    for i, t in enumerate(solved, 1):
        dest = ASKTUG_DIR / f"{t['id']}-{safe_name(t.get('slug') or 'topic')}.md"
        if dest.exists():
            skipped += 1
        else:
            try:
                detail = get_json(sess, f"{ASKTUG_BASE}/t/{t['id']}.json")
                text = _render_topic(t, detail, sess)
                if not text:
                    fail += 1
                else:
                    dest.write_text(text, encoding="utf-8")
                    ok += 1
            except Exception as exc:
                fail += 1
                log(f"    #{t['id']} 失败：{exc}")
            time.sleep(args.delay)

        state["fetch_done_count"] = i
        state["fetch_last_topic_id"] = t["id"]
        state["fetch_ok"] = ok
        state["fetch_skipped"] = skipped
        state["fetch_fail"] = fail
        if i % 50 == 0 or i == len(solved):
            _save_state(state_path, state)
        if i % 100 == 0 or i == len(solved):
            log(f"    进度 {i}/{len(solved)}  成功 {ok}  跳过 {skipped}  失败 {fail}")

    state["fetch_done"] = True
    _save_state(state_path, state)
    log(f"AskTUG 完成：新增 {ok}，已存在跳过 {skipped}，失败 {fail} -> {ASKTUG_DIR}")
    return 0


# ---------------------------------------------------------------- TiDB 社区专栏

def _next_data(html: str) -> dict | None:
    m = re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', html, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return None


def _block_text(node) -> str:
    """递归把块内联节点折成纯文本（leaf 节点有 text，container 有 children）。"""
    if node is None:
        return ""
    if isinstance(node, str):
        return node
    if isinstance(node, list):
        return "".join(_block_text(n) for n in node)
    if isinstance(node, dict):
        parts = []
        if isinstance(node.get("text"), str):
            parts.append(node["text"])
        if node.get("children"):
            parts.append(_block_text(node["children"]))
        return "".join(parts)
    return ""


def _list_lines(items, depth: int = 0) -> list[str]:
    lines: list[str] = []
    for it in items or []:
        if isinstance(it, dict) and str(it.get("type") or "").lower() in ("listitem", "list_item"):
            inner = " ".join(_render_blocks(it.get("children") or [], depth + 1))
            if inner.strip():
                lines.append("  " * depth + "- " + inner.strip())
        else:
            txt = _block_text(it).strip()
            if txt:
                lines.append("  " * depth + "- " + txt)
    return lines


def _render_blocks(blocks, depth: int = 0) -> list[str]:
    out: list[str] = []
    for b in blocks or []:
        if not isinstance(b, dict):
            txt = _block_text(b).strip()
            if txt:
                out.append(txt)
            continue
        t = str(b.get("type") or "").lower()
        if t == "heading":
            out.append("\n" + "#" * int(b.get("depth") or 2) + " " + _block_text(b.get("children")).strip())
        elif t == "paragraph":
            txt = _block_text(b.get("children")).strip()
            if txt:
                out.append(txt)
        elif t == "list":
            out.extend(_list_lines(b.get("children"), depth))
        elif t in ("codeblock", "code", "code_block", "codesnippet", "codemirror"):
            lang = str(b.get("language") or "").strip()
            out.append(f"\n```{lang}\n{_block_text(b.get('children')).strip()}\n```\n")
        elif t in ("image", "img", "figure"):
            out.append("[图片]" + _block_text(b.get("children")).strip())
        elif t in ("quote", "blockquote"):
            out.append("> " + _block_text(b.get("children")).strip())
        elif t in ("divider", "hr"):
            out.append("---")
        else:
            out.extend(_render_blocks(b.get("children") or [], depth))
    return [x for x in out if x and x.strip()]


def _article_text(blog: dict) -> str:
    raw = blog.get("content")
    if not raw:
        return ""
    try:
        blocks = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return html_to_text(str(raw))
    text = "\n\n".join(_render_blocks(blocks))
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def classify_article(blog: dict, text: str) -> tuple[bool, str]:
    """判断是否「干货」：非社区动态体裁 + 正文够长 + 技术信号词够多。"""
    cat = (blog.get("category") or {}).get("name") or ""
    title = (blog.get("title") or "").strip()
    if cat in TIDBNET_NON_TECH_CATEGORY:
        return False, f"分类={cat}"
    if TIDBNET_NON_TECH_PATTERN.search(title):
        return False, "标题像活动/通知/招募"
    if len(text) < TIDBNET_MIN_CHARS:
        return False, f"正文过短（{len(text)} 字）"
    low = text.lower()
    hits = sum(1 for kw in TIDBNET_TECH_TERMS if kw in low)
    if hits < 5:
        return False, f"技术信号词仅 {hits} 个"
    return True, f"技术信号词 {hits} 个"


def cmd_tidbnet(args) -> int:
    TIDBNET_DIR.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)
    sess = _session()

    state_path = META_DIR / "tidb-net.state.json"
    state = _load_state(state_path)
    limit = args.limit or 0
    log(f"Phase 1/2 枚举专栏列表（limit={limit or '全部'}）{'｜检测到断点，续跑' if state else ''}…")

    index_path = META_DIR / "tidb-net-topics.jsonl"
    index: list[dict] = []
    seen_slugs: set[str] = set()
    if index_path.exists():
        for line in index_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                t = json.loads(line)
            except json.JSONDecodeError:
                continue
            if t.get("slug") and t["slug"] not in seen_slugs:
                seen_slugs.add(t["slug"])
                index.append(t)
        log(f"    已有索引 {len(index)} 条，从第 {state.get('enum_last_page', 0) + 1} 页续跑")

    page = int(state.get("enum_last_page", 0)) + 1
    if limit and len(index) >= limit:
        page = 10 ** 9  # 已够量，跳过枚举
        log(f"    已有索引 {len(index)} 条 ≥ limit {limit}，无需继续翻页")
    while page < 10 ** 8:
        nd = _next_data(get_text(sess, f"{TIDBNET_BASE}/blog?page={page}"))
        bl = (((nd or {}).get("props") or {}).get("pageProps") or {}).get("blogs") or {}
        items = bl.get("content") or []
        pg = bl.get("page") or {}
        if page == 1:
            log(f"    上游共 {pg.get('totalElements')} 篇 / {pg.get('totalPages')} 页")
        if not items:
            log(f"    第 {page} 页无条目，枚举结束")
            state["enum_done"] = True
            _save_state(state_path, state)
            break

        new_rows: list[dict] = []
        for it in items:
            slug = it.get("slug")
            if not slug or slug in seen_slugs:
                continue
            seen_slugs.add(slug)
            row = {
                "id": it.get("id"),
                "slug": slug,
                "title": (it.get("title") or "").strip(),
                "category": ((it.get("category") or {}) or {}).get("name"),
                "publishedAt": it.get("publishedAt"),
                "visits": it.get("visits"),
                "likes": it.get("likes"),
                "comments": it.get("comments"),
            }
            index.append(row)
            new_rows.append(row)
            if limit and len(index) >= limit:
                break

        with index_path.open("a", encoding="utf-8") as fh:
            for row in new_rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

        state["enum_last_page"] = page
        state["topics_total"] = len(index)
        _save_state(state_path, state)

        if limit and len(index) >= limit:
            break
        if page >= (pg.get("totalPages") or 0):
            state["enum_done"] = True
            _save_state(state_path, state)
            break
        page += 1
        time.sleep(args.delay)

    log(f"列表索引 {index_path}（{len(index)} 条）")

    log("Phase 2/2 抓正文…")
    judged_path = META_DIR / "tidb-net-judged.jsonl"
    judged: set[str] = set()
    if judged_path.exists():
        for line in judged_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                judged.add(json.loads(line)["slug"])
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
        log(f"    已判定过的文章 {len(judged)} 篇（含被拒稿），本次直接跳过")
    judged_fh = judged_path.open("a", encoding="utf-8")

    def mark_judged(slug: str, verdict: str, reason: str) -> None:
        if slug in judged:
            return
        judged.add(slug)
        judged_fh.write(
            json.dumps({"slug": slug, "verdict": verdict, "reason": reason}, ensure_ascii=False) + "\n"
        )
        judged_fh.flush()

    ok = fail = skipped = 0
    kept: list[dict] = []
    rejected: list[dict] = []
    for i, t in enumerate(index, 1):
        slug = t.get("slug")
        if not slug:
            fail += 1
            continue
        if slug in judged:
            skipped += 1
            continue
        dest = TIDBNET_DIR / f"{slug}-{safe_name(t.get('title') or 'untitled')}.md"
        if dest.exists():
            skipped += 1
            mark_judged(slug, "干货", "文件已存在")
            continue
        try:
            nd = _next_data(get_text(sess, f"{TIDBNET_BASE}/blog/{slug}"))
            blog = ((nd or {}).get("props") or {}).get("pageProps", {}).get("blog") if nd else None
            if not blog:
                fail += 1
                mark_judged(slug, "失败", "页面无 __NEXT_DATA__.blog")
                log(f"    {slug} 无 __NEXT_DATA__.blog")
                continue
            text = _article_text(blog)
            good, why = classify_article(blog, text)
            rec = {
                "slug": slug,
                "title": (blog.get("title") or "").strip(),
                "category": ((blog.get("category") or {}) or {}).get("name"),
                "chars": len(text),
                "verdict": "干货" if good else "非干货",
                "reason": why,
            }
            (kept if good else rejected).append(rec)
            mark_judged(slug, rec["verdict"], why)
            if good or args.keep_non_tech:
                author = blog.get("author") or {}
                head = [
                    f"# {rec['title']}",
                    "",
                    f"> 来源：TiDB 社区专栏  |  作者：{(author.get('name') if isinstance(author, dict) else '') or '佚名'}  "
                    f"|  发布：{(blog.get('publishedAt') or '')[:10]}  "
                    f"|  分类：{rec['category'] or '-'}  "
                    f"|  浏览：{blog.get('visits')}",
                    f"> 链接：{TIDBNET_BASE}/blog/{slug}",
                    f"> 标签：{', '.join((tg or {}).get('name') or '' for tg in (blog.get('tags') or []))}",
                    "",
                    "（用户投稿内容，版权归原作者；本语料仅用于非商业研究，引用请回溯原文。）",
                    "",
                ]
                dest.write_text("\n".join(head) + text.strip() + "\n", encoding="utf-8")
                ok += 1
        except Exception as exc:
            fail += 1
            mark_judged(slug, "失败", str(exc)[:160])
            log(f"    {slug} 失败：{exc}")
        if i % 20 == 0 or i == len(index):
            state["fetch_done_count"] = i
            state["fetch_last_slug"] = slug
            state["fetch_ok"] = ok
            state["fetch_skipped"] = skipped
            state["fetch_fail"] = fail
            _save_state(state_path, state)
            log(f"    进度 {i}/{len(index)}  干货 {len(kept)}  非干货 {len(rejected)}  跳过 {skipped}  失败 {fail}")
        time.sleep(args.delay)

    judged_fh.close()
    total_judged = len(kept) + len(rejected)
    rate = (len(kept) / total_judged * 100) if total_judged else 0.0
    report = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "sampled": total_judged,
        "kept": len(kept),
        "rejected": len(rejected),
        "dry_goods_rate": round(rate, 1),
        "top_reject_reasons": _top_reasons(rejected),
        "rejected_samples": [f"{r['title'][:40]}  <-- {r['reason']}" for r in rejected[:20]],
        "kept_samples": [f"{r['title'][:40]}  ({r['chars']} 字)" for r in kept[:20]],
    }
    (META_DIR / "tidb-net-audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    log(f"专栏完成：写入 {ok}，跳过 {skipped}，失败 {fail}")
    log(f"干货率：{len(kept)}/{total_judged} = {rate:.1f}%")
    log("抽样驳斥理由：" + "; ".join(f"{k}×{v}" for k, v in report["top_reject_reasons"].items()))
    state["fetch_done"] = True
    state["fetch_ok"] = ok
    state["fetch_skipped"] = skipped
    state["fetch_fail"] = fail
    state["files_on_disk"] = len(list(TIDBNET_DIR.glob("*.md")))
    _save_state(state_path, state)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def _top_reasons(rows: list[dict]) -> dict:
    from collections import Counter

    c = Counter(re.sub(r"（\d+ 字）| \d+ 个", "", r["reason"]) for r in rows)
    return dict(c.most_common(8))


# ---------------------------------------------------------------- stats

def cmd_stats(args) -> int:
    if not CORPUS_DIR.exists():
        print(f"语料目录不存在：{CORPUS_DIR}")
        return 1
    total = 0
    for sub in sorted(p for p in CORPUS_DIR.iterdir() if p.is_dir() and p.name != "_meta"):
        mds = [p for p in sub.rglob("*") if p.is_file() and p.suffix.lower() in (".md", ".txt")]
        size = sum(p.stat().st_size for p in mds)
        total += len(mds)
        print(f"  {sub.name:<12} {len(mds):>6} 文件  {size/1024/1024:>7.1f} MB")
    others = [p for p in CORPUS_DIR.glob("*") if p.is_file()]
    if others:
        size = sum(p.stat().st_size for p in others)
        total += len(others)
        print(f"  {'(根目录)':<12} {len(others):>6} 文件  {size/1024/1024:>7.1f} MB")
    print(f"  {'合计':<12} {total:>6} 文件")
    return 0


# ---------------------------------------------------------------- CLI

def main() -> int:
    ap = argparse.ArgumentParser(description="TiDB 语料抓取（只落数据，不建索引）")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_docs = sub.add_parser("docs", help="抓 pingcap/docs-cn 的 md")
    p_docs.add_argument("--branch", default=DOCS_BRANCH, help="分支名，默认最新稳定版")
    p_docs.add_argument("--dest", default="", help="落盘目录，默认 corpora/<语料>/docs-cn")
    p_docs.set_defaults(func=cmd_docs)

    p_at = sub.add_parser("asktug", help="抓 AskTUG 已采纳答案主题")
    p_at.add_argument("--target-solved", type=int, default=5000)
    p_at.add_argument("--max-pages", type=int, default=2000)
    p_at.add_argument("--delay", type=float, default=0.5, help="详情请求间隔秒")
    p_at.set_defaults(func=cmd_asktug)

    p_st = sub.add_parser("stats", help="统计已落盘语料")
    p_st.set_defaults(func=cmd_stats)

    p_net = sub.add_parser("tidbnet", help="抓 TiDB 社区专栏中文长文")
    p_net.add_argument("--limit", type=int, default=0, help="只抓前 N 篇（0=全部）")
    p_net.add_argument("--delay", type=float, default=0.6, help="详情请求间隔秒")
    p_net.add_argument(
        "--keep-non-tech", action="store_true", help="连非干货也落盘（默认只留干货）"
    )
    p_net.set_defaults(func=cmd_tidbnet)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
