# PROGRESS — ai-lowcode-assistant

## 项目一句话

用户输入自然语言 → LangGraph 五 Agent 协作 → SSE 流式推送 → 前端实时渲染图表。全栈：Python/FastAPI/LangGraph + React/TypeScript。

## 分层架构

```
网关层: main.py (鉴权→令牌桶限流→SSE推流→SSE/REST双轨异常路由)
  ↓
Agent层: app.py build_graph() → requirement → retrieval → generate ⇄ audit → fix → END
  ↓                                                      (条件路由: success/retry<3/fix≥3)
数据层: db.py (路由表+4级漏斗+LIKE兜底) + sql_guard.py (5层审查) + vector_store.py (纯NumPy 384维哈希) + field_registry.py (全栈唯一数据源)
  ↓
前端层: App.tsx → appReducer.ts (SequenceID竞态防御+惰性擦除+双轨渲染) → DashboardCanvas
```

## 源码速查（backend/）

| 文件 | 做什么 |
|------|--------|
| main.py | FastAPI SSE 网关 + 中间件栈（鉴权/令牌桶/指标/异常双轨） |
| app.py | AgentState 定义 + 5Agent 状态图编译 + build_system_prompt() 硬约束注入 |
| db.py | SQLite(aiosqlite+WAL) + query_fields_with_agent() 双轨查询 + _fuzzy_match_metric() 4级漏斗 |
| sql_guard.py | 5层SQL静态审查：L1 SELECT白名单→L2 21类高危关键字+字面量剥离→L3 表名白名单→L4 系统表黑名单→L5 注入载荷 |
| field_registry.py | FIELD_ENTRIES 唯一数据源 → build_seed_assets/build_field_query_map/build_metric_alias_map |
| vector_store.py | Unigram+Bigram滑窗→SHA-256→struct.unpack 4字节→mod 384→余弦相似度→chroma_db持久化 |
| schemas.py | Pydantic v2: LowCodeSchema (title/chart_type:bar|line|pie/fields) |
| agents/requirement.py | 大白话→structured_task (task_summary+chart_type_hint+title_suggestion+data_fields) |
| agents/retrieval.py | LLM关键词提纯→query_assets()→retrieved_assets |
| agents/generate.py | async astream + 错题本注入_build_user_message() |
| agents/audit.py | L1 json.loads + L2~L4 Pydantic model_validate + classify_validation_error (syntax/missing_field/enum_error/type_error) |
| agents/fix.py | 三级修复: 语法愈合(_heal_json_syntax)→字段补齐(_fill_missing_fields)→骨架重建(_build_fallback_schema) |

## 前端源码速查（frontend/src/）

| 文件 | 做什么 |
|------|--------|
| App.tsx | 三栏布局 + fetch SSE + fetch /api/fetch-data + securityAlert感知 |
| state/appReducer.ts | 中央状态机: SSE_GENERATE(jsonRepair→Zod双模)+SSE_AUDIT_FAILED(awaitingRetry信号旗)+FETCH_DATA_SUCCESS(nullCount+securityAlert) |
| constants/index.ts | ChartSchema + ChartTypeOnlySchema (Zod旁路快检) |
| utils/jsonRepair.ts | 栈状态机流式JSON修复 |

## AgentState 字段速查

user_requirement → structured_task → user_query/refined_query → retrieved_assets → generated_schema → error_message/error_category → retry_count → reflection_history

## 文档导航

- docs/course/README.md → 4章13课索引 + 面试延伸14题
- docs/course/附录-测试覆盖.md → 8个测试文件一览
- docs/course/附录-工程踩坑记录.md → 5个踩坑故事

## 当前状态：全部完成

无待处理。已推送 GitHub: github.com/Loreen1119/ai-lowcode-assistant
