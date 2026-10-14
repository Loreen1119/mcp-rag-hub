# 文档事实类选点清单（167 条 query 口径，slug 核验存在）

口径：167 条 query，文档数 ≤167，每篇最多 2 条。模块配额按 query 数算。
列：条数 | slug | 出题方向 | 单/多证据 | AskTUG重复(★=已用)


## sql(40条)
1条 | `sql-statement-add-index` → 在线加索引 DDL 机制 | 单 | 
1条 | `sql-statement-explain-analyze` → EXPLAIN ANALYZE 实际执行统计 | 单 | 
2条 | `explain-overview` → 执行计划算子解读 | 单 | ★
1条 | `explain-joins` → JOIN 执行计划与 join 顺序 | 单 | ★
1条 | `sql-statement-with` → CTE 递归查询 | 单 | 
2条 | `partitioned-table` → 分区表类型与裁剪 | 单 | ★
1条 | `sql-statement-alter-table` → ALTER 加列/改列 | 单 | 
1条 | `sql-statement-create-view` → 视图与物化 | 单 | 
1条 | `generated-columns` → 生成列+索引 | 单 | ★
1条 | `sql-statement-analyze-table` → ANALYZE 收集统计 | 单 | 
1条 | `optimizer-hints` → Hint 强制走索引 | 单 | ★
1条 | `sql-statement-load-data` → LOAD DATA 导入 | 单 | ★
1条 | `sql-statement-flashback-table` → FLASHBACK 误删恢复 | 单 | 
1条 | `sql-statement-kill` → KILL 终止查询 | 单 | 
1条 | `window-functions` → 窗口函数 | 单 | 
1条 | `json-functions` → JSON 函数 | 单 | 
1条 | `cast-functions-and-operators` → 类型转换 | 单 | 
1条 | `date-and-time-functions` → 日期函数 | 单 | 
1条 | `group-by-modifier` → GROUP BY 修饰符 | 单 | 
1条 | `string-functions` → 字符串函数 | 单 | 
1条 | `control-flow-functions` → CASE/IF 控制流 | 单 | 
1条 | `sql-statement-grant-privileges` → 授权 | 单 | ★
1条 | `sql-statement-savepoint` → SAVEPOINT | 单 | 
1条 | `sql-statement-prepare` → PREPARE | 单 | 
1条 | `expressions-pushed-down` → 下推表达式 | 单 | ★
1条 | `sql-statement-flashback-cluster` → FLASHBACK CLUSTER 集群级恢复 | 单 | 
1条 | `sql-statement-recover-table` → RECOVER TABLE | 单 | 
1条 | `aggregate-group-by-functions` → 聚合函数 | 单 | 
1条 | `window-functions` → 窗口函数(第2条：排名) | 单 | 
1条 | `join-reorder` → join 重排优化 | 单 | 
1条 | `sql-statement-import-into` → IMPORT INTO 导入 | 单 | 
1条 | `sql-statement-create-sequence` → 序列 SEQUENCE | 单 | 
1条 | `sql-statement-lock-tables-and-unlock-tables` → 锁表 | 单 | 
1条 | `information-functions` → 信息函数 | 单 | 
1条 | `locking-functions` → 锁函数 | 单 | 
1条 | `numeric-functions-and-operators` → 数值函数 | 单 | 
1条 | `miscellaneous-functions` → 杂项函数 | 单 | 
1条 | `sql-statement-query-watch` → QUERY WATCH 资源管控 | 单 | 
1条 | `sql-statement-calibrate-resource` → CALIBRATE RESOURCE | 单 | 

## config(25条)
2条 | `system-variables` → 系统变量总览 | 单 | ★
2条 | `sql-mode` → SQL 模式 | 单 | 
1条 | `tidb-configuration-file` → TiDB 配置字段 | 单 | ★
1条 | `tikv-configuration-file` → TiKV 配置字段 | 单 | ★
1条 | `pd-configuration-file` → PD 配置字段 | 单 | 
1条 | `configure-memory-usage` → 内存控制 | 单 | ★
1条 | `configure-time-zone` → 时区 | 单 | 
1条 | `configure-placement-rules` → Placement Rules | 单 | ★
1条 | `configure-store-limit` → Store Limit | 单 | ★
1条 | `character-set-and-collation` → 字符集排序规则 | 单 | ★
1条 | `time-to-live` → TTL | 单 | ★
1条 | `cached-tables` → 缓存表 | 单 | 
1条 | `auto-increment` → 自增列 | 单 | ★
1条 | `auto-random` → AUTO_RANDOM | 单 | 
1条 | `clustered-indexes` → 聚簇索引 | 单 | ★
1条 | `transaction-isolation-levels` → 隔离级别 | 单 | 
1条 | `garbage-collection-configuration` → GC 配置 | 单 | ★
1条 | `garbage-collection-overview` → GC 机制 | 单 | ★
1条 | `statistics` → 统计信息 | 单 | ★
1条 | `extended-statistics` → 扩展统计 | 单 | 
1条 | `coprocessor-cache` → 下推缓存 | 单 | ★
1条 | `sql-prepared-plan-cache` → Prepared 计划缓存 | 单 | ★
1条 | `sql-non-prepared-plan-cache` → 非 Prepared 计划缓存 | 单 | ★

## tikv(25条)
1条 | `tikv-overview` → TiKV 架构 | 单 | 
1条 | `rocksdb-overview` → RocksDB LSM | 单 | ★
1条 | `titan-overview` → Titan 大 value | 单 | 
1条 | `titan-configuration` → Titan 配置 | 单 | ★
1条 | `tikv-in-memory-engine` → 内存引擎 | 单 | 
1条 | `tune-tikv-memory-performance` → 内存调优 | 单 | ★
1条 | `tune-tikv-thread-performance` → 线程调优 | 单 | 
1条 | `tikv-control` → tikv-ctl 工具 | 单 | ★
1条 | `partitioned-raft-kv` → 分区 Raft KV | 单 | 
1条 | `tune-region-performance` → Region 调优 | 单 | 
1条 | `configure-load-base-split` → Load Base Split | 单 | 
1条 | `command-line-flags-for-tikv-configuration` → TiKV 命令行参数 | 单 | 
1条 | `grafana-tikv-dashboard` → TiKV 监控 | 单 | ★
1条 | `pd-control` → pd-ctl | 单 | ★
1条 | `pd-recover` → PD Recover | 单 | 
1条 | `pd-microservices` → PD 微服务 | 单 | 
1条 | `command-line-flags-for-pd-configuration` → PD 配置参数 | 单 | 
1条 | `grafana-pd-dashboard` → PD 监控 | 单 | 
1条 | `tune-operating-system` → OS 调优 | 单 | 
1条 | `massive-regions-best-practices` → 海量 Region | 单 | ★
1条 | `rocksdb-overview` → RocksDB compaction(第2条) | 单 | ★
1条 | `tikv-overview` → TiKV Raft 角色(第2条) | 单 | 
1条 | `pd-scheduling-best-practices` → PD 调度策略 | 单 | ★
1条 | `readonly-nodes` → 只读节点 | 单 | 
1条 | `three-nodes-hybrid-deployment` → 三节点混合部署 | 单 | 


前三模块合计 91 条 query（目标 167，余 76 条在 TiCDC/DM/TiFlash/运维/其他）

## ticdc(15条)
1条 | `ticdc-overview` → TiCDC 架构与原理 | 单 | 
1条 | `deploy-ticdc` → TiCDC 部署 | 单 | 
1条 | `ticdc-changefeed-overview` → Changefeed 概念 | 单 | 
1条 | `ticdc-changefeed-config` → Changefeed 配置参数 | 单 | 
1条 | `ticdc-sink-to-kafka` → 同步到 Kafka | 单 | 
1条 | `ticdc-sink-to-mysql` → 同步到 MySQL | 单 | 
1条 | `ticdc-open-protocol` → Open Protocol 输出格式 | 单 | 
1条 | `ticdc-avro-protocol` → Avro Protocol | 单 | 
1条 | `ticdc-canal-json` → Canal-JSON Protocol | 单 | 
1条 | `ticdc-filter` → 日志过滤器 | 单 | 
1条 | `ticdc-table-routing` → 表路由 | 单 | 
1条 | `ticdc-ddl` → DDL 同步 | 单 | 
1条 | `ticdc-bidirectional-replication` → 双向复制 | 单 | 
1条 | `ticdc-data-replication-capabilities` → 同步能力详解 | 单 | 
1条 | `ticdc-integrity-check` → 单行数据校验 | 单 | 

## dm(15条)
1条 | `dm-overview` → DM 架构 | 单 | 
1条 | `dm-arch` → DM 组件架构 | 单 | 
1条 | `dm-best-practices` → DM 最佳实践 | 单 | 
1条 | `dm-create-task` → 创建迁移任务 | 单 | 
1条 | `dm-source-configuration-file` → 上游数据源配置 | 单 | 
1条 | `dm-task-configuration-guide` → 任务配置向导 | 单 | 
1条 | `dm-block-allow-table-lists` → 黑白名单 | 单 | 
1条 | `dm-table-routing` → 表路由 | 单 | 
1条 | `dm-shard-merge` → 分库分表合并 | 单 | 
1条 | `dm-safe-mode` → 安全模式 | 单 | 
1条 | `dm-error-handling` → 故障处理 | 单 | 
1条 | `dm-query-status` → 查询任务状态 | 单 | 
1条 | `dm-ddl-compatible` → DDL 兼容 | 单 | 
1条 | `dm-high-availability` → 高可用 | 单 | 
1条 | `relay-log` → Relay Log | 单 | 

## tiflash(15条)
1条 | `tiflash-overview` → TiFlash 架构 | 单 | 
1条 | `create-tiflash-replicas` → 构建副本 | 单 | 
1条 | `use-tidb-to-read-tiflash` → TiDB 读 TiFlash | 单 | 
1条 | `use-tiflash-mpp-mode` → MPP 模式 | 单 | 
1条 | `tiflash-disaggregated-and-s3` → 存算分离架构 | 单 | 
1条 | `tiflash-configuration` → 配置参数 | 单 | 
1条 | `tiflash-supported-pushdown-calculations` → 计算下推 | 单 | 
1条 | `tiflash-spill-disk` → 数据落盘 | 单 | 
1条 | `tiflash-late-materialization` → 延迟物化 | 单 | 
1条 | `tiflash-pipeline-model` → Pipeline 执行模型 | 单 | 
1条 | `tiflash-mintso-scheduler` → MinTSO 调度器 | 单 | 
1条 | `tiflash-data-validation` → 数据校验 | 单 | 
1条 | `tune-tiflash-performance` → 性能调优 | 单 | 
1条 | `use-fastscan` → FastScan | 单 | 
1条 | `tiflash-compatibility` → 兼容性说明 | 单 | 

## ops(20条)
1条 | `backup-and-restore-overview` → 备份恢复概述 | 单 | 
1条 | `br-snapshot-guide` → 快照备份指南 | 单 | 
1条 | `br-pitr-guide` → 日志备份 PITR | 单 | 
1条 | `br-snapshot-architecture` → 快照备份架构 | 单 | 
1条 | `br-log-architecture` → 日志备份架构 | 单 | 
1条 | `dumpling-overview` → Dumpling 导出 | 单 | 
1条 | `tidb-lightning-overview` → Lightning 导入 | 单 | 
1条 | `upgrade-tidb-using-tiup` → TiUP 升级 | 单 | 
1条 | `tiup-component-cluster-deploy` → TiUP 部署集群 | 单 | 
1条 | `tiup-component-cluster-scale-out` → TiUP 扩容 | 单 | 
1条 | `tiup-component-cluster-scale-in` → TiUP 缩容 | 单 | 
1条 | `ddl-introduction` → DDL 原理与最佳实践 | 单 | 
1条 | `haproxy-best-practices` → HAProxy 负载均衡 | 单 | 
1条 | `high-concurrency-best-practices` → 高并发写入 | 单 | 
1条 | `index-management-best-practices` → 索引管理 | 单 | 
1条 | `multi-column-index-best-practices` → 多列索引 | 单 | 
1条 | `uuid` → UUID 主键最佳实践 | 单 | 
1条 | `tidb-partitioned-tables-best-practices` → 分区表最佳实践 | 单 | 
1条 | `three-dc-local-read` → 三中心就近读 | 单 | 
1条 | `dr-solution-introduction` → 容灾方案概述 | 单 | 

## other(12条)
1条 | `dev-guide-choose-driver-or-orm` → 驱动/ORM 选择 | 单 | 
1条 | `dev-guide-connection-parameters` → 连接池参数 | 单 | 
1条 | `dev-guide-implicit-type-conversion` → 隐式类型转换 | 单 | 
1条 | `dev-guide-index-best-practice` → 索引最佳实践 | 单 | 
1条 | `dev-guide-optimistic-and-pessimistic-transaction` → 乐观/悲观事务 | 单 | 
1条 | `dev-guide-timeouts-in-tidb` → 各种超时 | 单 | 
1条 | `dev-guide-unique-serial-number-generation` → 唯一序列号生成 | 单 | 
1条 | `dev-guide-use-stale-read` → Stale Read | 单 | 
1条 | `dev-guide-use-follower-read` → Follower Read | 单 | 
1条 | `dashboard-slow-query` → Dashboard 慢查询页 | 单 | 
1条 | `tiproxy-overview` → TiProxy 简介 | 单 | 
1条 | `tiproxy-load-balance` → TiProxy 负载均衡 | 单 | 


后五模块合计 77 条。全部模块 = 91 + 77 = 168 条 query。