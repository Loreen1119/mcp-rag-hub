# Golden 对齐复核底表

生成：2026-10-14。类别六类：slug不存在 / slug拼写 / sv兜底 / 语义硬错 / query模糊 / title-query不匹配（+ 语义错位-待判定）。
字段：id | 类别 | query | 当前 golden | 问题说明 | 人工判定 | 修正 golden | 备注
人工判定枚举：保留 / 替换 / 淘汰 / 待定。修正 golden 填替换后 slug，保留则留空。
判定顺序：第1轮 slug不存在(3) → 第2轮 slug拼写(1) → 第3轮 sv兜底(9) → 第4轮 语义硬错(9) → 第5轮 query模糊/title不匹配(5) → 第6轮 语义错位全量扫。

| id | 类别 | query | 当前 golden | 问题说明 | 人工判定 | 修正 golden | 备注 |
|---|---|---|---|---|---|---|---|
| 128 | slug不存在 | pump 怎么检查数据同步延迟？ | monitor-tidb-binlog-cluster | 废弃组件 binlog，需替换或淘汰 | 淘汰 |  | binlog 废弃，语料无 ground truth，改写=捏造 |
| 139 | slug不存在 | 哪里能看到 TiCDC 的历史版本和更新内容？ | release-notes | 索引页非具体文档，需替换 | 淘汰 |  | 导航类问题，golden 索引页无法对齐 |
| 148 | slug不存在 | 低版本 TiDB 迁移到高版本有什么方案？ | tidb-binlog-overview | 废弃 binlog，应挂升级/迁移文档 | 替换 | upgrade-tidb-using-tiup | 迁移/升级文档，非废弃 binlog |
| 133 | slug拼写 | TiUP Grafana 怎么配置匿名登录？ | customized-montior-in-tiup-environment | montior→monitor 拼写错 | 淘汰 |  | slug 本身不存在，Grafana 匿名登录文档无覆盖，弱 golden 比没有更污染 |
| 12 | sv兜底 | TiDB 有没有办法让当前会话只读，START TRANSACTION READ ONLY 无效？ | system-variables | system-variables 需核：会话只读 | 替换 | sql-statement-start-transaction | 会话只读语法在 START TRANSACTION 文档 |
| 37 | sv兜底 | TiDB 内存持续升高，间歇性 OOM，heapinuse 有什么参数能限制吗？ | system-variables | system-variables 需核：内存 heapinuse | 替换 | configure-memory-usage | 参数限制题 |
| 54 | sv兜底 | ANALYZE TABLE 报错 Primary key column not found，无法 analyze 表，怎么处理？ | system-variables | system-variables 需核：analyze 报错 | 淘汰 |  | bug 现象非文档，ground truth 是 issue/changelog，语料无 |
| 58 | sv兜底 | INSERT 语句的 select 里不会走 TiFlash 吗？ | system-variables | system-variables 需核：TiFlash | 替换 | use-tidb-to-read-tiflash | INSERT SELECT 走 TiFlash |
| 64 | sv兜底 | TiDB 执行 SQL 完成后内存不会立即释放，导致占用越来越高，怎么解决？ | system-variables | system-variables 需核：内存释放 | 替换 | configure-memory-usage | 内存机制题 |
| 102 | sv兜底 | 动态分区裁剪收集统计信息时 globalstats 合并耗时很久 hang 住，怎么处理？ | system-variables | system-variables 需核：globalstats 合并 | 替换 | partitioned-table | 动态裁剪 globalstats |
| 109 | sv兜底 | 基于 TiCDC 不能同步无有效索引表，怎么从源头解决无有效索引表的创建？ | system-variables | system-variables 需核：无有效索引表 | 替换 | sql-statement-create-index | 无有效索引表从建索引解决 |
| 122 | sv兜底 | 临时表插入大量数据报错 the table is full，怎么处理？ | system-variables | system-variables 需核：临时表 full | 替换 | dev-guide-use-temporary-tables | 临时表 tmp-table-size |
| 51 | 语义硬错 | TiDB 有办法从 6.5.3 降级到 6.5.1 吗？ | pd-scheduling-best-practices | 降级挂 PD 调度，应挂降级/upgrade 文档 | 淘汰 |  | 官方不支持降级，golden 只有一句脚注，不可验证 |
| 92 | 语义硬错 | TiKV 一个节点下线很慢，怎么加速？ | tiflash-configuration | TiKV 下线慢挂 TiFlash 配置，应挂扩容缩容 | 替换 | tiup-component-cluster-scale-in | TiKV 下线=缩容 |
| 107 | 语义硬错 | 怎么查询一个月前的 DML SQL 执行语句？ | backup-and-restore-overview | 查历史 SQL 挂 backup，应挂 statement-summary/slow-query | 替换 | statement-summary-tables | 历史 SQL 查询 |
| 155 | 语义硬错 | batch ddl job 执行时间差距巨大，可能原因有哪些？ | time-to-live | batch ddl 挂 time-to-live，应挂 DDL/系统变量 | 替换 | ddl-introduction | batch ddl 执行时间差 |
| 158 | 语义硬错 | TiKV 节点数据盘在哪些情况下超过 90% 必须扩容？ | pd-configuration-file | 磁盘超90%挂 PD 配置，应挂容量告警 | 替换 | alert-rules | 磁盘超 90% 扩容=告警规则题 |
| 44 | 语义硬错 | TiDB 表设置 TTL 后导致系统读 IO 飙升、慢查询增多，怎么处理？ | hybrid-deployment-topology, versioning | TTL 读IO飙升挂部署+versioning，应挂 time-to-live | 替换 | time-to-live | TTL 读 IO 飙升 |
| 137 | 语义硬错 | TiKV truncate 后磁盘空间不释放，GC 相关问题怎么处理？ | tikv-control | truncate 空间不释放挂 tikv-control，应挂 garbage-collection | 替换 | garbage-collection-overview | truncate 空间不释放=GC |
| 143 | 语义硬错 | DM 全量 dump 阶段日志信息少一个表，怎么排查？ | sync-diff-inspector-overview | DM dump 少表挂 sync-diff，应挂 DM 文档 | 替换 | dm-query-status | dump 少表=状态查询 |
| 159 | 语义硬错 | 可以用 hint 强制使用 TiFlash 吗，怎么用？ | system-variables | hint 强制 TiFlash 挂 system-variables 是挂错，应挂 optimizer-hints/use-tidb-to-read-tiflash | 替换 | optimizer-hints|use-tidb-to-read-tiflash | 多证据候选，待判定 |
| 130 | query模糊 | 主键值对空格的处理有什么疑问？ | character-set-and-collation | "有什么疑问"不是问题 | 替换query |  | golden 待复核，可能需补 sql-mode 相关 slug |
| 153 | query模糊 | 监控文档里没有 Flow Control 相关指标描述，怎么办？ | grafana-tikv-dashboard | 文档吐槽非技术问题 | 淘汰 |  | 文档吐槽，非技术问答 |
| 157 | query模糊 | 等保扫描漏洞，这些怎么解决？ | high-reliability-faq | "这些"指代不明 | 淘汰 |  | "这些"指代截图，无具体问题 |
| 23 | title/query不匹配 | TiDB 里的 TSO 是什么，代表什么意思？ | tso | 标题与主问脱节(备份 vs TSO) | 保留 |  | 标题与正文脱节，query 已按正文重写 |
| 149 | 语义错位-待判定 | 应用连接 TiDB 的 IP 连接串里可以使用多个 ip+port 组合吗？ | tiproxy-overview | tiproxy-overview 对 ip+port 组合不算完全错，待判定 | 保留 |  | golden 待复核，按原帖问 HAProxy 还是 TiProxy 定 |