# 失败案例逐条分析（单一来源）

> 生成方式：`python -m scripts.export_failure_cases`（**确定性复算**，脚本会对齐冻结分布与逐条记录，
> 对不上直接非零退出）。README / `EVAL_REPORT.md` / 技术博客 1 引用失败案例时，**引用本文件**，不要各写一份。
>
> **口径边界（先读这段）**
> - 检索层 F1/F2 基准配置 = **RRF 档（生产 BM25 20 + 向量 20）的 top-5**，与 [Y] 口径一致。
> - 拒答层 F4/F5 基准 = **CE 分数门控 [B] = 0.98**（⚠️ **模拟口径**）。
>   W3 真实 agent 链路的实测（正确拒答 10.5% / 误杀 5%）**是另一套口径**，只在第 5 节作诊断并列，**不得与本节混报**。
> - 术语鸿沟 F3 只作 `root_cause`（抽检 50 条 seed=42），**不参与 primary 自动判定**。
> - 版本混淆 F6 属**独立探针集**，**不在主集 375 里**；主集里新旧版同名页命中仍按 slug 级算命中。
> - 本文件只放 query 与 slug，**不放语料原文**（`corpora/tidb/` 不入库，原文不随仓库分发）。

## 0. 数据来源（本文件由这些文件复算而来）

- `data/eval_set.jsonl` — `sha256(LF) = 4b9b3c594efa8834…`
- `data/eval_set_version.jsonl` — `sha256(LF) = d3b29201c87239a6…`
- `experiments/baseline_ladder/retrieval_375.jsonl` — `sha256(LF) = 5c5da08963988811…`
- `experiments/baseline_ladder/miss_hints.jsonl` — `sha256(LF) = c732592308670f93…`
- `experiments/baseline_ladder/ce_scores.jsonl` — `sha256(LF) = df1d6265bbd3c85f…`
- `experiments/version_confusion/version_confusion_31.jsonl` — `sha256(LF) = 2f67ee972314aaf0…`
- `data/failure_mode_term_gap_sample50.json` — `sha256(LF) = b9dadf99fcbbb00a…`
- `experiments/generation_eval/judge_100.jsonl` — `sha256(LF) = 06453e0080033ab8…`
- `experiments/generation_eval/refusal_60.jsonl` — `sha256(LF) = 22abd6275b796b43…`

## 1. 主集 375 条 primary 分布（复算 == 冻结）

| primary | 条数 |
|---|---|
| 召回缺失 | 114 |
| 排序压制 | 112 |
| none | 118 |
| 应拒答未拒答 | 30 |
| 拒答误杀 | 1 |
| **合计** | **375** |

- `none` 118 = top-5 命中 + `[B]=0.98` 下正确拒答 10
- 未命中（`召回缺失` + `排序压制`）= 226 条，逐条见第 3 节
- `unclassified` 0 条；`术语鸿沟` 0 条（该标签只进 `root_cause`，见第 7 节）

## 2. 每类典型 case（详析）

### 召回缺失

- **ask_1**（asktug）TiDB 集群被安全扫描出 Prometheus 和 ETCD 未授权访问漏洞，应该怎么修复？
  - golden：`enable-tls-between-clients-and-servers, enable-tls-between-components`
  - 单路 top-50 均无此页 → 语料侧就没被召回（术语鸿沟是常见根因，见第 7 节）
  - RRF top-5：1.665857-topic、2.665857-topic、3.665857-topic、4.665857-topic、5.1011059-topic
- **ask_3**（asktug）往 TiDB 批量 insert 数据很慢，有什么优化方向？
  - golden：`dumpling-overview, tidb-lightning-overview`
  - 单路 top-50 均无此页 → 语料侧就没被召回（术语鸿沟是常见根因，见第 7 节）
  - RRF top-5：1.1020515-topic、2.1010947-topic、3.1010756-topic、4.ebf64fb2-TiDB 社区智慧合集丨…、5.1010756-topic
- **ask_4**（asktug）用什么工具对 TiDB 做性能压测？
  - golden：`benchmark-tidb-using-ch, benchmark-tidb-using-sysbench, benchmark-tidb-using-tpcc`
  - 单路 top-50 均无此页 → 语料侧就没被召回（术语鸿沟是常见根因，见第 7 节）
  - RRF top-5：1.1020837-topic、2.f8a41154-大量 SET autoc…、3.1014973-topic、4.1024113-topic、5.f8a41154-大量 SET autoc…

### 排序压制

- **ask_2**（asktug）TiKV 触发 TiKV_approximate_region_size 告警，手动 split region 超时怎么办？
  - golden：`tikv-control`
  - 单路已召回（golden 位次）：`{'bm25:tikv-control': 12}` → 但 RRF top-5 未排进来
  - RRF top-5：1.1021535-topic、2.1038796-topic、3.alert-rules、4.alert-rules、5.1021535-topic
- **ask_6**（asktug）TiDB 查询有没有缓存机制？
  - golden：`coprocessor-cache, manage-cluster-faq, sql-non-prepared-plan-cache, sql-prepared-plan-cache`
  - 单路已召回（golden 位次）：`{'vector:coprocessor-cache': 37, 'vector:sql-prepared-plan-cache': 23, 'vector:sql-non-prepared-plan-cache': 45}` → 但 RRF top-5 未排进来
  - RRF top-5：1.1048113-topic、2.1027034-topic、3.1031821-topic、4.1048113-topic、5.22606340-破解“潮汐负载”与海量数…
- **ask_10**（asktug）TiDB 服务器内存耗尽被重启，怎么排查原因？
  - golden：`troubleshoot-tidb-oom, tune-tikv-memory-performance`
  - 单路已召回（golden 位次）：`{'vector:troubleshoot-tidb-oom': 48}` → 但 RRF top-5 未排进来
  - RRF top-5：1.1025198-topic、2.1015152-topic、3.1036908-topic、4.ada59a4c-一次SSD磁盘寿命耗尽导…、5.1034138-topic

### 应拒答未拒答（[B] 门控口径）

- **una_1** TiDB 的 tidb_vector_index 参数怎么配置？ —— CE top1 = 0.9986 ≥ 0.98，被当作可答
- **una_2** TiDB 怎么开启自动分库分表（sharding）功能？ —— CE top1 = 0.9909 ≥ 0.98，被当作可答
- **una_3** TiDB 支持存储过程吗，怎么写一个存储过程？ —— CE top1 = 0.9999 ≥ 0.98，被当作可答
- 全量 30 条 id：una_1, una_2, una_3, una_4, una_5, una_6, una_7, una_8, una_9, una_10, una_11, una_12, una_13, una_14, una_15, una_16, una_17, una_18, una_19, una_21, una_22, una_26, una_27, una_30, una_33, una_36, una_37, una_38, una_39, una_40

### 拒答误杀（[B] 门控口径）

- **mk_14** TiDB 支持 WITH RECURSIVE 递归查询吗，这个好像比较冷门？ —— 该条 top-5 其实命中，但 CE top1 = 0.978 < 0.98 被误拒

### 版本混淆（独立探针集，[Z] 基准配置 = RRF 档）

- **ver_2** TiDB v8.5 的 max_execution_time 变量是控制所有语句还是只控制 SELECT 的执行时长？ —— 旧版页（`docs-cn-v8.4/`）压过正确版页（`docs-cn/`）
- **ver_9** TiDB v8.5 分区表的主键和唯一键必须包含分区键吗，全局索引能突破这个限制吗？ —— 旧版页（`docs-cn-v8.4/`）压过正确版页（`docs-cn/`）
- **ver_11** TiDB v8.5 里 TiCDC Changefeed 的配置参数有什么新增？ —— 旧版页（`docs-cn-v8.4/`）压过正确版页（`docs-cn/`）
- **ver_13** TiDB v8.5 的 TiCDC Debezium Protocol 有哪些更新？ —— 旧版页（`docs-cn-v8.4/`）压过正确版页（`docs-cn/`）
- **ver_23** TiDB v8.5 的 Top SQL 页面有哪些新功能？ —— 旧版页（`docs-cn-v8.4/`）压过正确版页（`docs-cn/`）
- **ver_24** TiDB v8.5 新增支持了哪些第三方工具？ —— 旧版页（`docs-cn-v8.4/`）压过正确版页（`docs-cn/`）

## 3. 未命中 226 条逐条

`auto_hint` = spec 第四节的确定性初判：golden 在单路 top-50 内 → `排序压制`，否则 `召回缺失`。

| id | source | primary | golden | 单路 golden 位次 | query |
|---|---|---|---|---|---|
| ask_1 | asktug | 召回缺失 | enable-tls-between-clients-and-servers, enable-tls-between-components | 无（两路 top-50 都没召回） | TiDB 集群被安全扫描出 Prometheus 和 ETCD 未授权… |
| ask_2 | asktug | 排序压制 | tikv-control | bm25:tikv-control=12 | TiKV 触发 TiKV_approximate_region_siz… |
| ask_3 | asktug | 召回缺失 | dumpling-overview, tidb-lightning-overview | 无（两路 top-50 都没召回） | 往 TiDB 批量 insert 数据很慢，有什么优化方向？ |
| ask_4 | asktug | 召回缺失 | benchmark-tidb-using-ch, benchmark-tidb-using-sysbench, benchmark-tidb-using-tpcc | 无（两路 top-50 都没召回） | 用什么工具对 TiDB 做性能压测？ |
| ask_5 | asktug | 召回缺失 | non-transactional-dml | 无（两路 top-50 都没召回） | TiDB 大表按时间条件删历史数据 delete 很慢，怎么快速删除？ |
| ask_6 | asktug | 排序压制 | coprocessor-cache, manage-cluster-faq, sql-non-prepared-plan-cache, sql-prepared-plan-cache | vector:coprocessor-cache=37、vector:sql-prepared-plan-cache=23、vector:sql-non-prepared-plan-cache=45 | TiDB 查询有没有缓存机制？ |
| ask_10 | asktug | 排序压制 | troubleshoot-tidb-oom, tune-tikv-memory-performance | vector:troubleshoot-tidb-oom=48 | TiDB 服务器内存耗尽被重启，怎么排查原因？ |
| ask_11 | asktug | 召回缺失 | expressions-pushed-down | 无（两路 top-50 都没召回） | 一条 SQL 慢在 not(isnull(字段)) 上，是什么原因？ |
| ask_13 | asktug | 排序压制 | statistics, system-variables | bm25:system-variables=46、bm25:statistics=20、vector:system-variables=6 | tidb_auto_analyze_ratio 参数怎么优化设置？ |
| ask_14 | asktug | 召回缺失 | dumpling-overview | 无（两路 top-50 都没召回） | DM 没有 LOCK 权限能不能做全量备份？ |
| ask_15 | asktug | 排序压制 | tidb-configuration-file, tiup-component-cluster-edit-config | bm25:tidb-configuration-file=5、vector:tidb-configuration-file=8 | TiDB 节点的 tidb.log 和慢日志太大想清理，log.fil… |
| ask_16 | asktug | 召回缺失 | br-pitr-guide, br-snapshot-guide | 无（两路 top-50 都没召回） | 怎么查看 BR 的历史备份信息（做过哪些备份、是全量还是增量）？ |
| ask_17 | asktug | 召回缺失 | generated-columns, sql-statement-create-index | 无（两路 top-50 都没召回） | TiDB 里 JSON 查询比 like 模糊查询慢很多，怎么优化？ |
| ask_18 | asktug | 召回缺失 | explain-overview | 无（两路 top-50 都没召回） | TiDB 多表关联时是怎么把关联下推到 TiKV 执行的？ |
| ask_19 | asktug | 排序压制 | dumpling-overview, system-variables | vector:dumpling-overview=21 | dumpling 导出大表时怎么限速，减少对生产的影响？ |
| ask_21 | asktug | 召回缺失 | tikv-configuration-file | 无（两路 top-50 都没召回） | 删除一个 TiKV 挂载盘导致 IO 跌零 10 秒以上，怎么回事？ |
| ask_22 | asktug | 召回缺失 | use-tidb-to-read-tiflash | 无（两路 top-50 都没召回） | 表设置了 TiFlash 副本后，执行计划却没有走 TiFlash，怎… |
| ask_23 | asktug | 召回缺失 | tso | 无（两路 top-50 都没召回） | TiDB 里的 TSO 是什么？ |
| ask_24 | asktug | 召回缺失 | dm-webui-guide, dm-worker-intro | 无（两路 top-50 都没召回） | 同一个 TiDB 集群里，上游下游都是这个集群，只是不同数据库的表，能… |
| ask_25 | asktug | 排序压制 | pd-scheduling-best-practices | vector:pd-scheduling-best-practices=23 | TiDB MergeRegion 调度不工作，空 region 持续增… |
| ask_26 | asktug | 召回缺失 | sql-statement-load-data, sql-statement-rollback | 无（两路 top-50 都没召回） | 事务里用 Bulk Copy 插入数据后，未提交时其他事务也能读到，R… |
| ask_27 | asktug | 召回缺失 | br-pitr-guide, br-snapshot-guide | 无（两路 top-50 都没召回） | 生产环境的 BR 全量备份和增量备份、恢复脚本，用 TSO 做增量还是… |
| ask_28 | asktug | 排序压制 | tikv-configuration-file | bm25:tikv-configuration-file=15 | WAL 日志和 RocksDB 里的 Raft 日志有什么区别，各自作… |
| ask_29 | asktug | 召回缺失 | dm-best-practices, tidb-lightning-physical-import-mode | 无（两路 top-50 都没召回） | 快速把大量数据导入 TiDB，为什么建议用 DM 而不是直接 inse… |
| ask_30 | asktug | 召回缺失 | titan-configuration | 无（两路 top-50 都没召回） | 怎么查看 TiKV 里 Blob 内容都属于哪些表？ |
| ask_31 | asktug | 排序压制 | sql-prepared-plan-cache | bm25:sql-prepared-plan-cache=31 | 同一个 SQL 指纹为什么会产生不同的执行计划 id，但计划内容又一样？ |
| ask_32 | asktug | 召回缺失 | stale-read, dev-guide-use-stale-read | 无（两路 top-50 都没召回） | 对视图分页查询期间，视图涉及的表数据变化导致查询结果不一致，怎么解决？ |
| ask_34 | asktug | 排序压制 | dm-binlog-event-filter, handle-failed-ddl-statements, mysql-compatibility | bm25:handle-failed-ddl-statements=16 | dm-worker 同步上游 MySQL 的 DDL 时暂停了，但 q… |
| ask_35 | asktug | 召回缺失 | high-concurrency-best-practices | 无（两路 top-50 都没召回） | TiKV 三台机器压测写入时只有一台负载高，另外两台空闲，为什么？ |
| ask_36 | asktug | 排序压制 | dumpling-overview | bm25:dumpling-overview=21、vector:dumpling-overview=15 | dumpling 导出数据到 csv，longtext 字段解析不符合… |
| ask_37 | asktug | 排序压制 | configure-memory-usage | vector:configure-memory-usage=48 | TiDB 内存持续升高，间歇性 OOM，heapinuse 有什么参数… |
| ask_38 | asktug | 召回缺失 | haproxy-best-practices, tiproxy-deployment-topology | 无（两路 top-50 都没召回） | 有两个 TiDB 节点的情况下，怎么做负载均衡？ |
| ask_39 | asktug | 排序压制 | migrate-from-tidb-to-mysql, ticdc-overview | bm25:ticdc-overview=21、vector:ticdc-overview=46 | TiCDC 或 TiDB-to-MySQL 同步，能同步用户和数据库吗… |
| ask_40 | asktug | 排序压制 | glossary, tidb-scheduling | vector:glossary=25 | region 和 peer 怎么理解，3 副本是指 3 个 peer … |
| ask_41 | asktug | 召回缺失 | grafana-tikv-dashboard, tikv-configuration-file | 无（两路 top-50 都没召回） | 并发查询导致其他查询变慢，从 1 秒变 50 秒，怎么排查？ |
| ask_42 | asktug | 召回缺失 | latency-breakdown | 无（两路 top-50 都没召回） | DM 向多副本容灾集群写入速度较慢（5 副本），怎么优化？ |
| ask_43 | asktug | 排序压制 | auto-increment | bm25:auto-increment=7、vector:auto-increment=2 | AUTO_INCREMENT=1000001 且 AUTO_ID_CA… |
| ask_44 | asktug | 排序压制 | time-to-live | bm25:time-to-live=6 | TiDB 表设置 TTL 后导致系统读 IO 飙升、慢查询增多，怎么处… |
| ask_45 | asktug | 召回缺失 | monitor-ticdc, ticdc-architecture | 无（两路 top-50 都没召回） | 用 TiCDC 把增量数据同步到 Kafka 给 BI 用，需要部署几… |
| ask_46 | asktug | 召回缺失 | multi-data-centers-in-one-city-deployment, schedule-replicas-by-topology-labels | 无（两路 top-50 都没召回） | 三台物理机各部署 4 个 TiKV，同一副本的数据会落在同一台服务器上… |
| ask_47 | asktug | 排序压制 | upgrade-tidb-using-tiup | bm25:upgrade-tidb-using-tiup=36、vector:upgrade-tidb-using-tiup=42 | TiDB 从 v7.5.2 平滑升级到 v7.5.5 的最佳实践？ |
| ask_48 | asktug | 排序压制 | dm-best-practices, dm-overview | bm25:dm-overview=14 | 用 DM 从 MySQL 迁移 1TB 数据到 TiDB，大概要多久？… |
| ask_49 | asktug | 召回缺失 | dev-guide-timeouts-in-tidb, tidb-configuration-file | 无（两路 top-50 都没召回） | TiDB 事务里每隔 10 秒 send TxnHeartBeat 更… |
| ask_50 | asktug | 召回缺失 | information-schema-slow-query, statement-summary-tables | 无（两路 top-50 都没召回） | RU 占用高，怎么找到占用 RU 较高的 SQL？ |
| ask_52 | asktug | 召回缺失 | tidb-architecture | 无（两路 top-50 都没召回） | MySQL 有实例的概念，TiDB 有这个概念吗？ |
| ask_53 | asktug | 召回缺失 | dm-overview | 无（两路 top-50 都没召回） | TiDB v3.1.0 的 40T 数据迁移到新集群，有什么成熟的方案… |
| ask_55 | asktug | 召回缺失 | sql-statement-batch | 无（两路 top-50 都没召回） | 普通表改成分区表后，怎么快速把普通表的数据导入分区表？ |
| ask_56 | asktug | 召回缺失 | massive-regions-best-practices | 无（两路 top-50 都没召回） | 一个 TiKV 节点 region 数量近 20 万（官方建议不超 5… |
| ask_57 | asktug | 召回缺失 | dev-guide-connection-parameters | 无（两路 top-50 都没召回） | 单表 2 亿条数据一次 select 返回全表，内存 24G 报溢出，… |
| ask_58 | asktug | 排序压制 | use-tidb-to-read-tiflash | vector:use-tidb-to-read-tiflash=29 | INSERT 语句的 select 里不会走 TiFlash 吗？ |
| ask_59 | asktug | 召回缺失 | enable-tls-between-clients-and-servers | 无（两路 top-50 都没召回） | TiDB 在数据安全方面做了哪些保障？ |
| ask_60 | asktug | 召回缺失 | sql-statement-trace | 无（两路 top-50 都没召回） | 怎么在查询计划里看每个算子基于代价模型预估的代价？ |
| ask_61 | asktug | 召回缺失 | dev-guide-timeouts-in-tidb | 无（两路 top-50 都没召回） | 客户端终止长事务后，TiDB 数据库端不终止任务且最终提交，怎么解决？ |
| ask_62 | asktug | 排序压制 | sql-prepared-plan-cache | bm25:sql-prepared-plan-cache=44 | TiDB 程序执行和手动执行同一 SQL，执行计划不一致，为什么？ |
| ask_64 | asktug | 排序压制 | configure-memory-usage | bm25:configure-memory-usage=9、vector:configure-memory-usage=22 | TiDB 执行 SQL 完成后内存不会立即释放，导致占用越来越高，怎么… |
| ask_65 | asktug | 召回缺失 | identify-expensive-queries | 无（两路 top-50 都没召回） | 生产环境 TiDB 主库集群 OOM 内存溢出，怎么排查？ |
| ask_66 | asktug | 召回缺失 | use-tiflash-mpp-mode | 无（两路 top-50 都没召回） | TiFlash 查 1.5 亿数据的表，执行计划走了 TiFlash … |
| ask_67 | asktug | 召回缺失 | transaction-overview | 无（两路 top-50 都没召回） | 事务内 delete 是作用在 buffer 上吗，commit 前不… |
| ask_68 | asktug | 召回缺失 | high-reliability-faq | 无（两路 top-50 都没召回） | TiDB 升级到 v7.1.1 后安全扫描出漏洞，怎么解决？ |
| ask_69 | asktug | 召回缺失 | massive-regions-best-practices | 无（两路 top-50 都没召回） | region 心跳定时忽高忽低正常吗？ |
| ask_70 | asktug | 召回缺失 | dumpling-overview | 无（两路 top-50 都没召回） | TiDB 能用 mysqldump 工具备份吗，兼容吗？ |
| ask_71 | asktug | 召回缺失 | tidb-configuration-file | 无（两路 top-50 都没召回） | TiDB 的日志可以直接删吗，有没有自动删除策略？ |
| ask_72 | asktug | 召回缺失 | rocksdb-overview | 无（两路 top-50 都没召回） | TiDB 插数据很占内存，为什么？ |
| ask_73 | asktug | 召回缺失 | sql-statement-grant-privileges | 无（两路 top-50 都没召回） | TiDB 可以多表同时授权吗？ |
| ask_76 | asktug | 召回缺失 | hardware-and-software-requirements | 无（两路 top-50 都没召回） | TiKV 节点存储有大小限制吗？ |
| ask_77 | asktug | 召回缺失 | sql-statement-show-backups | 无（两路 top-50 都没召回） | BR 快照备份时，怎么监控备份进度？ |
| ask_79 | asktug | 排序压制 | coprocessor-cache | vector:coprocessor-cache=39 | TiDB 生产环境建议使用查询缓存功能吗？ |
| ask_80 | asktug | 排序压制 | character-set-and-collation | bm25:character-set-and-collation=21、vector:character-set-and-collation=16 | TiDB v6.0 的 Collation 能改成 MySQL 8.0… |
| ask_81 | asktug | 召回缺失 | ticdc-avro-protocol | 无（两路 top-50 都没召回） | TiCDC 同步数据异常（多条数据合并），怎么排查？ |
| ask_82 | asktug | 排序压制 | migration-tidb-faq | bm25:migration-tidb-faq=20、vector:migration-tidb-faq=25 | 删除数据报 transaction is too large, siz… |
| ask_83 | asktug | 召回缺失 | dev-guide-third-party-tools-compatibility | 无（两路 top-50 都没召回） | 怎么用 flyway 连接 TiDB？ |
| ask_84 | asktug | 排序压制 | explain-joins | bm25:explain-joins=38、vector:explain-joins=9 | hash join 前先过滤数据，怎么提高 SQL 查询速度？ |
| ask_85 | asktug | 召回缺失 | tikv-configuration-file | 无（两路 top-50 都没召回） | TiKV 无法写入，磁盘使用限制参数在哪看，怎么修改？ |
| ask_86 | asktug | 召回缺失 | garbage-collection-overview | 无（两路 top-50 都没召回） | truncate table 是对表还是对行做删除标记？drop ta… |
| ask_87 | asktug | 召回缺失 | use-tiflash-mpp-mode | 无（两路 top-50 都没召回） | TiDB 分区表怎么做全表分组查询？ |
| ask_89 | asktug | 召回缺失 | dashboard-ops-deploy | 无（两路 top-50 都没召回） | dashboard 用户登录认证失败 authenticate fai… |
| ask_90 | asktug | 召回缺失 | dev-guide-choose-driver-or-orm | 无（两路 top-50 都没召回） | TiDB 有提供 C++ 的 API 吗？ |
| ask_91 | asktug | 召回缺失 | garbage-collection-configuration, garbage-collection-overview | 无（两路 top-50 都没召回） | TiDB GC 清理异常导致空间不断上涨，怎么处理？ |
| ask_92 | asktug | 排序压制 | tiup-component-cluster-scale-in | bm25:tiup-component-cluster-scale-in=41 | TiKV 一个节点下线很慢，怎么加速？ |
| ask_94 | asktug | 召回缺失 | ticdc-filter | 无（两路 top-50 都没召回） | 归档产生 delete 事件导致 CDC 链路延迟，怎么处理？ |
| ask_95 | asktug | 召回缺失 | subquery-optimization | 无（两路 top-50 都没召回） | 为什么语句的执行计划没有算子？ |
| ask_96 | asktug | 召回缺失 | sql-statement-select | 无（两路 top-50 都没召回） | TiDB 查询结果怎么导出到指定文件？ |
| ask_97 | asktug | 召回缺失 | tidb-computing | 无（两路 top-50 都没召回） | TiDB 索引结构是怎么存储的，会在 RocksDB 新建一个列簇吗？ |
| ask_98 | asktug | 召回缺失 | explain-index-merge | 无（两路 top-50 都没召回） | 主从两个库同一个 SQL 执行计划不一样，为什么？ |
| ask_99 | asktug | 召回缺失 | grafana-overview-dashboard | 无（两路 top-50 都没召回） | 哪里有 TiDB 的 Grafana 比较完整的使用说明？ |
| ask_100 | asktug | 召回缺失 | hardware-and-software-requirements | 无（两路 top-50 都没召回） | /tmp/tidb 目录空间大小怎么计算？ |
| ask_101 | asktug | 召回缺失 | identify-expensive-queries | 无（两路 top-50 都没召回） | 为什么简单的查询会产生慢查询？ |
| ask_103 | asktug | 召回缺失 | explain-overview | 无（两路 top-50 都没召回） | SQL 执行计划的顺序怎么看？ |
| ask_104 | asktug | 召回缺失 | use-tiflash-mpp-mode | 无（两路 top-50 都没召回） | 表亿级数据，查询速度慢，怎么优化？ |
| ask_106 | asktug | 召回缺失 | auto-increment | 无（两路 top-50 都没召回） | 如何高效修改大表 id 列改成递增数字？ |
| ask_107 | asktug | 召回缺失 | statement-summary-tables | 无（两路 top-50 都没召回） | 怎么查询一个月前的 DML SQL 执行语句？ |
| ask_108 | asktug | 排序压制 | data-type-date-and-time | vector:data-type-date-and-time=4 | timestamp(0) 的默认值问题，怎么处理？ |
| ask_109 | asktug | 召回缺失 | sql-statement-create-index | 无（两路 top-50 都没召回） | 基于 TiCDC 不能同步无有效索引表，怎么从源头解决无有效索引表的创… |
| ask_110 | asktug | 排序压制 | partitioned-table | bm25:partitioned-table=6、vector:partitioned-table=35 | TiDB 目前只支持 hash 和 range 分区吗？ |
| ask_111 | asktug | 排序压制 | br-snapshot-architecture | bm25:br-snapshot-architecture=14、vector:br-snapshot-architecture=5 | BR 备份从 TiKV 集群读数据是从 leader 还是所有 pee… |
| ask_112 | asktug | 召回缺失 | optimistic-transaction | 无（两路 top-50 都没召回） | Write conflict 写写冲突怎么处理？ |
| ask_113 | asktug | 召回缺失 | rocksdb-overview | 无（两路 top-50 都没召回） | TiKV 里 SST 文件、Raft 日志和 KV 数据是怎么对应的？ |
| ask_114 | asktug | 召回缺失 | dm-block-allow-table-lists | 无（两路 top-50 都没召回） | DM 不支持多库迁移吗？ |
| ask_115 | asktug | 召回缺失 | check-before-deployment | 无（两路 top-50 都没召回） | TiDB 单机离线部署怎么做？ |
| ask_116 | asktug | 排序压制 | auto-increment | vector:auto-increment=20 | 用 navicat 在 TiDB 建表时没办法添加自增 id，为什么？ |
| ask_117 | asktug | 召回缺失 | maintain-tidb-using-tiup | 无（两路 top-50 都没召回） | 用 tiup 部署后，等保修改了 ssh 端口，还需要配合做什么吗？ |
| ask_118 | asktug | 召回缺失 | pd-control | 无（两路 top-50 都没召回） | tikv-ctl 扫描数据的时候的警告是什么原因？ |
| ask_119 | asktug | 召回缺失 | configure-placement-rules | 无（两路 top-50 都没召回） | 配置冷热隔离分区策略后，新增 label disk=hhd 导致数据写… |
| ask_120 | asktug | 排序压制 | ticdc-faq | bm25:ticdc-faq=13、vector:ticdc-faq=10 | TiCDC 同步任务可以暂停多长时间？ |
| ask_121 | asktug | 召回缺失 | dashboard-key-visualizer | 无（两路 top-50 都没召回） | TiDB 读取热点是什么意思，怎么消除读热点？ |
| ask_122 | asktug | 排序压制 | dev-guide-use-temporary-tables | bm25:dev-guide-use-temporary-tables=28、vector:dev-guide-use-temporary-tables=4 | 临时表插入大量数据报错 the table is full，怎么处理？ |
| ask_123 | asktug | 排序压制 | dumpling-overview | vector:dumpling-overview=24 | dumpling 导数据很慢，卡在 0% 不动，怎么处理？ |
| ask_124 | asktug | 召回缺失 | sql-statement-create-table | 无（两路 top-50 都没召回） | TiDB 支持表级别压缩吗？ |
| ask_125 | asktug | 召回缺失 | multi-data-centers-in-one-city-deployment | 无（两路 top-50 都没召回） | 单机房 3 节点 TiKV 集群怎么在线扩容成 3 机房 9 节点集群？ |
| ask_126 | asktug | 召回缺失 | grafana-tikv-dashboard | 无（两路 top-50 都没召回） | TiKV 之间的网络带宽占用看哪个监控图？ |
| ask_127 | asktug | 召回缺失 | migration-tools | 无（两路 top-50 都没召回） | DM、TiCDC、TiDB Binlog 各自的适用场景、优缺点和异同… |
| ask_131 | asktug | 召回缺失 | tikv-configuration-file | 无（两路 top-50 都没召回） | TiDB 7.1 集群卡住，报 lock txn not found,… |
| ask_132 | asktug | 召回缺失 | configure-store-limit | 无（两路 top-50 都没召回） | 节点上线 balance 速度怎么控制？ |
| ask_134 | asktug | 排序压制 | partitioned-table | bm25:partitioned-table=21、vector:partitioned-table=3 | 有没有语句把一个表变成另一个表内的分区表？ |
| ask_135 | asktug | 排序压制 | partitioned-table | bm25:partitioned-table=13、vector:partitioned-table=29 | 为什么非分区表在修改分区时报错？ |
| ask_137 | asktug | 召回缺失 | garbage-collection-overview | 无（两路 top-50 都没召回） | TiKV truncate 后磁盘空间不释放，GC 相关问题怎么处理？ |
| ask_140 | asktug | 排序压制 | sql-non-prepared-plan-cache | bm25:sql-non-prepared-plan-cache=9 | TiDB 后续是否实现执行计划缓存功能？ |
| ask_141 | asktug | 排序压制 | expressions-pushed-down | bm25:expressions-pushed-down=24 | like 支持下推到各个 KV 节点计算吗？ |
| ask_142 | asktug | 召回缺失 | ticdc-open-api-v2 | 无（两路 top-50 都没召回） | TiCDC 已经有 WebUI 了吗？ |
| ask_143 | asktug | 召回缺失 | dm-query-status | 无（两路 top-50 都没召回） | DM 全量 dump 阶段日志信息少一个表，怎么排查？ |
| ask_144 | asktug | 排序压制 | ddl-introduction | bm25:ddl-introduction=5、vector:ddl-introduction=25 | Online DDL 语句执行流程中做了哪些事？ |
| ask_145 | asktug | 排序压制 | tiup-playground | vector:tiup-playground=7 | 修改服务器 IP 后，playground 需要改哪些配置？ |
| ask_146 | asktug | 召回缺失 | garbage-collection-configuration | 无（两路 top-50 都没召回） | 把表删了磁盘空间会立即释放吗，不释放怎么释放空间？ |
| ask_147 | asktug | 排序压制 | configure-memory-usage | vector:configure-memory-usage=20 | TiDB Server 执行 SQL 怎么判断 OOM 内存溢出？ |
| ask_148 | asktug | 召回缺失 | upgrade-tidb-using-tiup | 无（两路 top-50 都没召回） | 低版本 TiDB 迁移到高版本有什么方案？ |
| ask_149 | asktug | 召回缺失 | tiproxy-overview | 无（两路 top-50 都没召回） | 多个 TiDB Server 时，应用连接串怎么写，还是用 HAPro… |
| ask_151 | asktug | 召回缺失 | hardware-and-software-requirements | 无（两路 top-50 都没召回） | TiDB 最大能存储多少数据？ |
| ask_152 | asktug | 召回缺失 | mysql-compatibility | 无（两路 top-50 都没召回） | TiDB 支持 MySQL 的事件功能吗？ |
| ask_154 | asktug | 排序压制 | tidb-lightning-checkpoints | bm25:tidb-lightning-checkpoints=17 | lightning 增量数据导入报错，checkpoint 怎么处理？ |
| ask_155 | asktug | 召回缺失 | ddl-introduction | 无（两路 top-50 都没召回） | batch ddl job 执行时间差距巨大，可能原因有哪些？ |
| ask_156 | asktug | 召回缺失 | dev-guide-transaction-restraints | 无（两路 top-50 都没召回） | TiDB 是怎么判断大事务的？ |
| ask_158 | asktug | 召回缺失 | alert-rules | 无（两路 top-50 都没召回） | TiKV 节点数据盘在哪些情况下超过 90% 必须扩容？ |
| ask_159 | asktug | 排序压制 | optimizer-hints, use-tidb-to-read-tiflash | vector:use-tidb-to-read-tiflash=32 | 可以用 hint 强制使用 TiFlash 吗，怎么用？ |
| ask_160 | asktug | 召回缺失 | dm-arch | 无（两路 top-50 都没召回） | 什么是 DM 集群？ |
| doc_sql_1 | doc_fact | 排序压制 | sql-statement-add-index | bm25:sql-statement-add-index=5、vector:sql-statement-add-index=24 | TiDB 在线加索引（ADD INDEX）是怎么做到不锁表的？ |
| doc_sql_3 | doc_fact | 排序压制 | explain-overview | bm25:explain-overview=14、vector:explain-overview=15 | TiDB 执行计划里各算子（TableReader/IndexLook… |
| doc_sql_4 | doc_fact | 排序压制 | explain-joins | bm25:explain-joins=13 | TiDB 多表 JOIN 的执行计划里，join 顺序是怎么决定的？ |
| doc_sql_6 | doc_fact | 排序压制 | partitioned-table | bm25:partitioned-table=6、vector:partitioned-table=38 | TiDB 分区表的分区裁剪（partition pruning）是怎么… |
| doc_sql_7 | doc_fact | 排序压制 | sql-statement-alter-table | vector:sql-statement-alter-table=9 | TiDB ALTER TABLE 加列/改列时，哪些操作会重写整张表？ |
| doc_sql_8 | doc_fact | 召回缺失 | sql-statement-create-view | 无（两路 top-50 都没召回） | TiDB 的视图是物化的还是每次查询实时算的？ |
| doc_sql_9 | doc_fact | 排序压制 | generated-columns | bm25:generated-columns=17、vector:generated-columns=4 | TiDB 生成列（generated column）能不能建索引？ |
| doc_sql_10 | doc_fact | 排序压制 | sql-statement-analyze-table | bm25:sql-statement-analyze-table=30、vector:sql-statement-analyze-table=36 | TiDB 手动 ANALYZE TABLE 和自动 analyze 有… |
| doc_sql_14 | doc_fact | 排序压制 | sql-statement-kill | bm25:sql-statement-kill=6、vector:sql-statement-kill=12 | TiDB KILL 语句能终止正在执行的查询吗，和 MySQL 有什么… |
| doc_sql_16 | doc_fact | 排序压制 | json-functions | bm25:json-functions=19 | TiDB 的 JSON 函数和 MySQL 兼容吗，常用 JSON 函… |
| doc_sql_23 | doc_fact | 排序压制 | sql-statement-savepoint | bm25:sql-statement-savepoint=10 | TiDB 事务里 SAVEPOINT 怎么用，和 MySQL 兼容吗？ |
| doc_sql_24 | doc_fact | 召回缺失 | sql-statement-prepare | 无（两路 top-50 都没召回） | TiDB 的 PREPARE 预处理语句和直接执行有什么区别，为什么要… |
| doc_sql_30 | doc_fact | 排序压制 | sql-statement-import-into | vector:sql-statement-import-into=25 | TiDB IMPORT INTO 和 LOAD DATA 有什么区别，… |
| doc_sql_33 | doc_fact | 排序压制 | information-functions | bm25:information-functions=6、vector:information-functions=45 | TiDB 里怎么查看当前版本、连接 ID 等信息函数？ |
| doc_sql_35 | doc_fact | 召回缺失 | numeric-functions-and-operators | 无（两路 top-50 都没召回） | TiDB 数值计算里 DECIMAL 精度是怎么保证的？ |
| doc_sql_36 | doc_fact | 排序压制 | miscellaneous-functions | vector:miscellaneous-functions=16 | TiDB 有哪些 MySQL 没有的特有函数？ |
| doc_config_40 | doc_fact | 排序压制 | system-variables | vector:system-variables=3 | TiDB 系统变量在哪里查看和设置，session 级和 global… |
| doc_config_43 | doc_fact | 排序压制 | sql-mode | bm25:sql-mode=15 | TiDB 修改 sql_mode 会影响哪些已有查询的行为？ |
| doc_config_44 | doc_fact | 召回缺失 | tidb-configuration-file | 无（两路 top-50 都没召回） | TiDB 配置文件里的参数和系统变量有什么关系，优先级谁高？ |
| doc_config_45 | doc_fact | 排序压制 | tikv-configuration-file | vector:tikv-configuration-file=16 | TiKV 配置文件里最重要的几个参数是什么（如 raftstore、r… |
| doc_config_46 | doc_fact | 排序压制 | pd-configuration-file | bm25:pd-configuration-file=23 | PD 配置里 leader 调度相关的参数有哪些？ |
| doc_config_49 | doc_fact | 召回缺失 | configure-placement-rules | 无（两路 top-50 都没召回） | TiDB 怎么用 SQL 语句（CREATE PLACEMENT PO… |
| doc_config_52 | doc_fact | 排序压制 | time-to-live | bm25:time-to-live=2、vector:time-to-live=15 | TiDB 建表时怎么设置 TTL 让数据自动过期，有哪些参数？ |
| doc_config_54 | doc_fact | 排序压制 | auto-increment | vector:auto-increment=11 | TiDB 主键用 AUTO_INCREMENT 还是 AUTO_RAN… |
| doc_config_55 | doc_fact | 排序压制 | auto-random | bm25:auto-random=25 | TiDB AUTO_RANDOM 是什么，为什么能避免热点？ |
| doc_config_56 | doc_fact | 排序压制 | clustered-indexes | bm25:clustered-indexes=9、vector:clustered-indexes=9 | TiDB 聚簇索引和非聚簇索引在存储上有什么区别？ |
| doc_config_58 | doc_fact | 召回缺失 | garbage-collection-configuration | 无（两路 top-50 都没召回） | TiDB GC 清理太慢，应该调哪些参数？ |
| doc_config_59 | doc_fact | 召回缺失 | garbage-collection-overview | 无（两路 top-50 都没召回） | TiDB 删掉的数据要多久才真正从磁盘释放，GC 生命周期是什么？ |
| doc_tikv_65 | doc_fact | 召回缺失 | tikv-overview | 无（两路 top-50 都没召回） | TiKV 的整体架构里，Raft 层、存储引擎、事务层是怎么分工的？ |
| doc_tikv_66 | doc_fact | 排序压制 | rocksdb-overview | bm25:rocksdb-overview=23、vector:rocksdb-overview=10 | TiKV 底层 RocksDB 的 LSM-tree 是怎么工作的，写… |
| doc_tikv_69 | doc_fact | 排序压制 | tikv-in-memory-engine | bm25:tikv-in-memory-engine=11、vector:tikv-in-memory-engine=5 | TiKV 内存引擎（MEM）和 RocksDB 有什么区别，什么场景用？ |
| doc_tikv_72 | doc_fact | 排序压制 | tikv-control | bm25:tikv-control=24、vector:tikv-control=3 | tikv-ctl 工具能做什么，常见用法有哪些？ |
| doc_tikv_74 | doc_fact | 排序压制 | tune-region-performance | vector:tune-region-performance=3 | TiKV Region 的性能调优有哪些手段（如 hotspot 处理… |
| doc_tikv_76 | doc_fact | 排序压制 | command-line-flags-for-tikv-configuration | bm25:command-line-flags-for-tikv-configuration=26、vector:command-line-flags-for-tikv-configuration=18 | TiKV 启动时有哪些关键命令行参数？ |
| doc_tikv_78 | doc_fact | 排序压制 | pd-control | bm25:pd-control=16 | pd-ctl 工具能做哪些运维操作？ |
| doc_tikv_80 | doc_fact | 排序压制 | pd-microservices | bm25:pd-microservices=5 | TiDB 的 PD 微服务（tso/调度分离）是什么，有什么好处？ |
| doc_tikv_81 | doc_fact | 召回缺失 | command-line-flags-for-pd-configuration | 无（两路 top-50 都没召回） | PD 的调度参数里，leader-schedule-limit 等限制… |
| doc_tikv_82 | doc_fact | 排序压制 | grafana-pd-dashboard | vector:grafana-pd-dashboard=30 | PD 监控里，哪些指标反映调度是否均衡？ |
| doc_tikv_83 | doc_fact | 排序压制 | tune-operating-system | bm25:tune-operating-system=5 | 部署 TiKV 前，操作系统层面（内核、文件系统、I/O 调度）要做哪… |
| doc_tikv_84 | doc_fact | 排序压制 | massive-regions-best-practices | vector:massive-regions-best-practices=18 | TiDB 集群 Region 数量很多时，有哪些成熟的调优手段？ |
| doc_tikv_85 | doc_fact | 召回缺失 | rocksdb-overview | 无（两路 top-50 都没召回） | TiKV RocksDB 的 compaction 有哪几种类型，分别… |
| doc_tikv_86 | doc_fact | 召回缺失 | tikv-overview | 无（两路 top-50 都没召回） | TiKV 的 Raft 日志是怎么复制到多个副本的？ |
| doc_tikv_87 | doc_fact | 排序压制 | pd-scheduling-best-practices | vector:pd-scheduling-best-practices=49 | PD 的调度器有哪些类型（balance/hot/leader），各自… |
| doc_ticdc_90 | doc_fact | 排序压制 | ticdc-overview | vector:ticdc-overview=42 | TiCDC 的整体架构是什么样的，由哪些组件组成？ |
| doc_ticdc_91 | doc_fact | 排序压制 | deploy-ticdc | bm25:deploy-ticdc=27、vector:deploy-ticdc=11 | TiCDC 怎么部署，和 TiDB 集群是什么关系？ |
| doc_ticdc_93 | doc_fact | 召回缺失 | ticdc-changefeed-config | 无（两路 top-50 都没召回） | TiCDC Changefeed 的 start-ts、checkpo… |
| doc_ticdc_94 | doc_fact | 排序压制 | ticdc-sink-to-kafka | vector:ticdc-sink-to-kafka=26 | TiCDC 同步数据到 Kafka 是怎么配置的，支持哪些输出格式？ |
| doc_ticdc_95 | doc_fact | 排序压制 | ticdc-sink-to-mysql | vector:ticdc-sink-to-mysql=3 | TiCDC 同步到 MySQL 兼容库，DDL 是怎么同步的？ |
| doc_ticdc_99 | doc_fact | 排序压制 | ticdc-filter | bm25:ticdc-filter=17、vector:ticdc-filter=14 | TiCDC 怎么过滤不需要同步的表或事件？ |
| doc_ticdc_103 | doc_fact | 排序压制 | ticdc-data-replication-capabilities | bm25:ticdc-data-replication-capabilities=29 | TiCDC 支持同步哪些数据变更，有哪些能力边界？ |
| doc_dm_105 | doc_fact | 召回缺失 | dm-overview | 无（两路 top-50 都没召回） | TiDB DM（Data Migration）是做什么的，和 TiCD… |
| doc_dm_106 | doc_fact | 排序压制 | dm-arch | bm25:dm-arch=4 | DM 的 DM-master、DM-worker 分别负责什么？ |
| doc_dm_108 | doc_fact | 召回缺失 | dm-create-task | 无（两路 top-50 都没召回） | DM 怎么创建一个迁移任务，任务配置里有哪些关键项？ |
| doc_dm_110 | doc_fact | 召回缺失 | dm-task-configuration-guide | 无（两路 top-50 都没召回） | DM 任务配置里，全量导出和增量同步是怎么衔接的？ |
| doc_dm_111 | doc_fact | 排序压制 | dm-block-allow-table-lists | bm25:dm-block-allow-table-lists=12、vector:dm-block-allow-table-lists=11 | DM 的黑白名单（block-allow-list）怎么配置？ |
| doc_dm_112 | doc_fact | 排序压制 | dm-table-routing | bm25:dm-table-routing=14、vector:dm-table-routing=8 | DM 表路由规则能实现表名映射吗，怎么配？ |
| doc_dm_113 | doc_fact | 排序压制 | dm-shard-merge | bm25:dm-shard-merge=14、vector:dm-shard-merge=41 | DM 分库分表合并迁移是怎么工作的，有什么前提？ |
| doc_dm_116 | doc_fact | 排序压制 | dm-query-status | bm25:dm-query-status=8、vector:dm-query-status=15 | DM 怎么查询迁移任务的状态和进度？ |
| doc_dm_117 | doc_fact | 排序压制 | dm-ddl-compatible | bm25:dm-ddl-compatible=38、vector:dm-ddl-compatible=14 | DM 对上游 DDL 的兼容性处理是怎样的？ |
| doc_tiflash_120 | doc_fact | 排序压制 | tiflash-overview | vector:tiflash-overview=23 | TiFlash 是什么，解决什么场景的问题？ |
| doc_tiflash_121 | doc_fact | 排序压制 | create-tiflash-replicas | bm25:create-tiflash-replicas=27、vector:create-tiflash-replicas=5 | 怎么给表添加 TiFlash 副本，副本数怎么设？ |
| doc_tiflash_122 | doc_fact | 召回缺失 | use-tidb-to-read-tiflash | 无（两路 top-50 都没召回） | TiDB 查询怎么走 TiFlash，是自动还是手动指定？ |
| doc_tiflash_123 | doc_fact | 排序压制 | use-tiflash-mpp-mode | bm25:use-tiflash-mpp-mode=18、vector:use-tiflash-mpp-mode=10 | TiFlash 的 MPP 模式是什么，怎么开启？ |
| doc_tiflash_126 | doc_fact | 排序压制 | tiflash-supported-pushdown-calculations | bm25:tiflash-supported-pushdown-calculations=12、vector:tiflash-supported-pushdown-calculations=11 | TiFlash 支持哪些计算下推，和 TiKV 下推有什么区别？ |
| doc_tiflash_131 | doc_fact | 排序压制 | tiflash-data-validation | bm25:tiflash-data-validation=35、vector:tiflash-data-validation=18 | TiFlash 的数据校验是怎么做的，如何验证一致性？ |
| doc_tiflash_132 | doc_fact | 排序压制 | tune-tiflash-performance | bm25:tune-tiflash-performance=26 | TiFlash 查询慢时，有哪些调优手段？ |
| doc_ops_135 | doc_fact | 排序压制 | backup-and-restore-overview | bm25:backup-and-restore-overview=16、vector:backup-and-restore-overview=4 | TiDB 备份恢复有哪几种方式，各适合什么场景？ |
| doc_ops_136 | doc_fact | 排序压制 | br-snapshot-guide | vector:br-snapshot-guide=5 | TiDB BR 快照备份怎么操作，备份到哪里？ |
| doc_ops_137 | doc_fact | 排序压制 | br-pitr-guide | bm25:br-pitr-guide=23、vector:br-pitr-guide=3 | TiDB BR 日志备份（PITR）是什么，能恢复到任意时间点吗？ |
| doc_ops_139 | doc_fact | 排序压制 | br-log-architecture | bm25:br-log-architecture=17、vector:br-log-architecture=3 | TiDB BR 日志备份的架构和快照备份有什么不同？ |
| doc_ops_140 | doc_fact | 排序压制 | dumpling-overview | bm25:dumpling-overview=19、vector:dumpling-overview=14 | Dumpling 和 mysqldump 有什么区别，为什么推荐 Du… |
| doc_ops_141 | doc_fact | 排序压制 | tidb-lightning-overview | bm25:tidb-lightning-overview=45、vector:tidb-lightning-overview=4 | TiDB Lightning 有哪几种导入模式，各适合什么场景？ |
| doc_ops_142 | doc_fact | 排序压制 | upgrade-tidb-using-tiup | vector:upgrade-tidb-using-tiup=8 | TiDB 用 TiUP 升级的完整流程和注意事项是什么？ |
| doc_ops_143 | doc_fact | 召回缺失 | tiup-component-cluster-deploy | 无（两路 top-50 都没召回） | TiDB 用 TiUP 部署集群，拓扑文件怎么配置？ |
| doc_ops_144 | doc_fact | 召回缺失 | tiup-component-cluster-scale-out | 无（两路 top-50 都没召回） | TiDB 用 TiUP 扩容节点，怎么操作？ |
| doc_ops_145 | doc_fact | 排序压制 | tiup-component-cluster-scale-in | bm25:tiup-component-cluster-scale-in=32 | TiDB 用 TiUP 缩容节点，下线流程是什么？ |
| doc_ops_148 | doc_fact | 排序压制 | high-concurrency-best-practices | bm25:high-concurrency-best-practices=36、vector:high-concurrency-best-practices=4 | TiDB 高并发写入场景有哪些优化建议？ |
| doc_ops_151 | doc_fact | 排序压制 | uuid | vector:uuid=4 | TiDB 用 UUID 做主键有什么问题，怎么解决？ |
| doc_other_159 | doc_fact | 排序压制 | dev-guide-optimistic-and-pessimistic-transaction | bm25:dev-guide-optimistic-and-pessimistic-transaction=22、vector:dev-guide-optimistic-and-pessimistic-transaction=13 | TiDB 乐观事务和悲观事务的区别是什么，怎么选？ |
| doc_other_160 | doc_fact | 排序压制 | dev-guide-timeouts-in-tidb | vector:dev-guide-timeouts-in-tidb=1 | TiDB 有哪些超时参数，分别控制什么？ |
| doc_other_162 | doc_fact | 排序压制 | dev-guide-use-stale-read | bm25:dev-guide-use-stale-read=11、vector:dev-guide-use-stale-read=4 | TiDB Stale Read 是什么，怎么用？ |
| doc_other_163 | doc_fact | 排序压制 | dev-guide-use-follower-read | bm25:dev-guide-use-follower-read=38 | TiDB Follower Read 和 Stale Read 有什么… |
| doc_other_165 | doc_fact | 排序压制 | tiproxy-overview | vector:tiproxy-overview=4 | TiProxy 是什么，解决什么问题？ |
| mk_1 | miss_kill | 召回缺失 | sql-statement-add-index | 无（两路 top-50 都没召回） | TiDB 是不是不支持在线加索引？ |
| mk_3 | miss_kill | 召回缺失 | br-snapshot-guide | 无（两路 top-50 都没召回） | TiDB 是不是不能做全量备份？ |
| mk_5 | miss_kill | 排序压制 | partitioned-table | bm25:partitioned-table=2、vector:partitioned-table=22 | TiDB 好像不支持分区表裁剪吧？ |
| mk_6 | miss_kill | 召回缺失 | explain-overview | 无（两路 top-50 都没召回） | TiDB 是不是没有类似 EXPLAIN 的功能？ |
| mk_7 | miss_kill | 排序压制 | sql-statement-flashback-table | bm25:sql-statement-flashback-table=5、vector:sql-statement-flashback-table=24 | TiDB 的 FLASHBACK 恢复功能是不是只支持老版本？ |
| mk_8 | miss_kill | 排序压制 | create-tiflash-replicas | bm25:create-tiflash-replicas=20、vector:create-tiflash-replicas=23 | TiDB 是不是不能给表加多个 TiFlash 副本？ |
| mk_9 | miss_kill | 排序压制 | statistics | bm25:statistics=17、vector:statistics=11 | TiDB 的统计信息是不是只能手动收集，不能自动？ |
| mk_16 | miss_kill | 召回缺失 | sql-statement-select | 无（两路 top-50 都没召回） | TiDB 支持 SELECT INTO 导出查询结果吗？ |
| mk_19 | miss_kill | 排序压制 | sql-statement-query-watch | bm25:sql-statement-query-watch=3、vector:sql-statement-query-watch=23 | TiDB 有没有 QUERY WATCH 这种拦截慢查询的功能？ |

## 4. 拒答两侧清单（[B]=0.98 模拟口径，共 60 条）

- 不可答 40 条：**正确拒答 10 条**（una_20, una_23, una_24, una_25, una_28, una_29, una_31, una_32, una_34, una_35）；**该拒未拒 30 条**（见 2 节 id 列表）
- 误杀探针 20 条：**误杀 1 条**（mk_14）；其余 19 条正常作答

## 5. W3 真实 agent 链路实测（⚠️ 与第 4 节不是同一口径，禁止混报）

- 真实链路（qwen2.5:3b，走 `agent.run_query`）：正确拒答 **4/38**、漏拒 **34/38**、漏杀/误杀 **1/20**
- `status` 分布：{'answered': 53, 'insufficient_evidence': 5, 'invalid_output': 2}；其中 `invalid_output` ['una_21', 'una_22']（归因**模型稳定性**，不计分母）
- 结论：**拒答能力弱是真实短板、失准单向（该拒却答为主）**；与第 4 节的模拟门控同向，但绝对值不可互推。

## 6. 生成层低分样本（judge_100，W3）

- **judge_093**（均分 0.30）在安装部署 tidb 过程中，执行 init 过程失败
  - faithfulness 0.7｜relevancy 0.2｜context_recall 0.0
  - 裁判理由（context_recall）：上下文完全不包含参考答案中的任何信息
- **judge_039**（均分 0.37）TiDB 从哪个渠道可以获取每个版本的BUG列表？
  - faithfulness 0.7｜relevancy 0.4｜context_recall 0.0
  - 裁判理由（context_recall）：完全未提及如何获取每个版本的BUG列表
- **judge_011**（均分 0.47）使用tidb 4.0版本，如何提升性能？
  - faithfulness 0.7｜relevancy 0.7｜context_recall 0.0
  - 裁判理由（context_recall）：上下文完全未提及硬件要求相关信息
- **judge_046**（均分 0.47）在使用tidbbinlog组件做数据同步时，pump会产生日志，但是drainer无法解析
  - faithfulness 0.7｜relevancy 0.7｜context_recall 0.0
  - 裁判理由（context_recall）：上下文完全不包含参考答案中的任何信息
- **judge_074**（均分 0.47）哪里有tidb的grafanna平台比较完整的使用说明
  - faithfulness 0.7｜relevancy 0.7｜context_recall 0.0
  - 裁判理由（context_recall）：完全未提及grafana overview面板的重要监控指标

## 7. 术语鸿沟（root_cause，抽检 50 条 seed=42）

- 抽检 50 条 → **术语鸿沟 27/50 = 54%**（by_source：{'asktug': {'n': 30, 'yes': 24, 'rate': 0.8}, 'doc_fact': {'n': 18, 'yes': 2, 'rate': 0.1111}, 'miss_kill': {'n': 2, 'yes': 1, 'rate': 0.5}}）
- 判定为是的 27 条 id：ask_127, ask_106, ask_30, ask_18, ask_14, ask_125, ask_46, ask_55, ask_77, ask_3, ask_16, ask_101, ask_86, ask_126, ask_13, ask_25, ask_38, ask_82, ask_31, ask_124, ask_50, ask_49, ask_87, ask_99, doc_dm_106, doc_sql_36, mk_1
- 这些 id 的 `failure_mode.root_cause` 已回填为 `术语鸿沟`；**它是抽检比例，不是全量自动判**。

## 8. 引用纪律（红线）

- 引用分布数字 → 第 1 节（375 条，五类合计 375，`unclassified` 0）。
- 引用拒答 → **必须写明口径**：第 4 节是 `[B]=0.98` 的模拟门控；第 5 节才是真实链路实测。两者**不得混成一句**。
- 引用版本混淆 → 第 2 节末与独立探针集口径，**不得与主集 Hit@5 混报**。
- 修改本文件的唯一正路：改 `failure_mode_spec.md` 的规则 → 重跑本脚本；手改会被下一次 `--check` 打回。
