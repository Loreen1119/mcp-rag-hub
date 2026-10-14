#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AskTUG 评测集改写脚本（v2，读原文逐条改写，非脑内映射）。

输入：data/eval_rewrite_src.json（含 id/title/slugs/qbody）
输出：data/eval_set_v1.jsonl（158 条，query + golden 候选 + 双标签占位）

改写规则（定稿）：
  1. 去口语缀饰（大佬/请教/环境说明模板）
  2. 只取主问（原帖多问只留一个）
  3. 保留口语化术语，制造 query 口语 vs golden 官方术语的术语鸿沟
  4. 不新增原帖没有的事实条件
  5. 保留可检索锚点（版本/错误码/组件/命令/参数/操作阶段/现象）

golden 解析：递归 docs-cn 文件名索引，slug 取 URL 最后一段；碎链（release-*/vN）标"待复核"。
"""
import re, json, glob, os, sys

BASE = os.path.dirname(os.path.abspath(__file__))  # scripts/
ROOT = os.path.dirname(BASE)

rows = json.load(open(os.path.join(ROOT, 'data', 'eval_rewrite_src.json'), encoding='utf-8'))
DROP = {20, 33}  # 几百万 group（无主语无 golden 指向）、造数脚本（名词非问题）

# 递归文件名索引
name_idx = {}
for fp in glob.glob(os.path.join(ROOT, 'corpora', 'tidb', 'docs-cn', '**', '*.md'), recursive=True):
    name_idx[os.path.basename(fp)[:-3]] = fp.replace('\\', '/')

BROKEN = re.compile(r'^release-\d|^relea$|^v\d$')

def resolve_slugs(slugs):
    out, broken = [], []
    for s in slugs:
        if BROKEN.match(s):
            broken.append(s)
        else:
            out.append(s)
    return sorted(set(out)), broken

# 逐条改写映射：id -> query。golden 从 rows 的 slugs 递归解析。
REWRITE = {
    1: "TiDB 集群被安全扫描出 Prometheus 和 ETCD 未授权访问漏洞，应该怎么修复？",
    2: "TiKV 触发 TiKV_approximate_region_size 告警，手动 split region 超时怎么办？",
    3: "往 TiDB 批量 insert 数据很慢，有什么优化方向？",
    4: "用什么工具对 TiDB 做性能压测？",
    5: "TiDB 大表按时间条件删历史数据 delete 很慢，怎么快速删除？",
    6: "TiDB 查询有没有缓存机制？",
    7: "有没有类似 mysqltest 的工具能批量测试 TiDB 的 SQL？",
    8: "TiFlash 副本从 1 个扩到 2 个的过程中，表还能正常用 TiFlash 查吗？",
    9: "TiDB 数据清理和插入后频繁触发 analyze table，怎么避免？",
    10: "TiDB 服务器内存耗尽被重启，怎么排查原因？",
    11: "一条 SQL 慢在 not(isnull(字段)) 上，是什么原因？",
    12: "TiDB 有没有办法让当前会话只读，START TRANSACTION READ ONLY 无效？",
    13: "tidb_auto_analyze_ratio 参数怎么优化设置？",
    14: "DM 没有 LOCK 权限能不能做全量备份？",
    15: "TiDB 节点的 tidb.log 和慢日志太大想清理，log.file 相关配置在哪里设置？",
    16: "怎么查看 BR 的历史备份信息（做过哪些备份、是全量还是增量）？",
    17: "TiDB 里 JSON 查询比 like 模糊查询慢很多，怎么优化？",
    18: "TiDB 多表关联时是怎么把关联下推到 TiKV 执行的？",
    19: "dumpling 导出大表时怎么限速，减少对生产的影响？",
    21: "删除一个 TiKV 挂载盘导致 IO 跌零 10 秒以上，怎么回事？",
    22: "表设置了 TiFlash 副本后，执行计划却没有走 TiFlash，怎么处理？",
    23: "TiDB 里的 TSO 是什么，代表什么意思？",
    24: "同一个 TiDB 集群里，上游下游都是这个集群，只是不同数据库的表，能不能用 DM 同步？",
    25: "TiDB MergeRegion 调度不工作，空 region 持续增加不合并，怎么处理？",
    26: "事务里用 Bulk Copy 插入数据后，未提交时其他事务也能读到，ROLLBACK 也无法回退，怎么回事？",
    27: "生产环境的 BR 全量备份和增量备份、恢复脚本，用 TSO 做增量还是用日志备份，各有什么优缺点？",
    28: "WAL 日志和 RocksDB 里的 Raft 日志有什么区别，各自作用是什么？",
    29: "快速把大量数据导入 TiDB，为什么建议用 DM 而不是直接 insert？",
    30: "怎么查看 TiKV 里 Blob 内容都属于哪些表？",
    31: "同一个 SQL 指纹为什么会产生不同的执行计划 id，但计划内容又一样？",
    32: "对视图分页查询期间，视图涉及的表数据变化导致查询结果不一致，怎么解决？",
    34: "dm-worker 同步上游 MySQL 的 DDL 时暂停了，但 query-status 显示正常，怎么排查？",
    35: "TiKV 三台机器压测写入时只有一台负载高，另外两台空闲，为什么？",
    36: "dumpling 导出数据到 csv，longtext 字段解析不符合预期，怎么处理？",
    37: "TiDB 内存持续升高，间歇性 OOM，heapinuse 有什么参数能限制吗？",
    38: "有两个 TiDB 节点的情况下，怎么做负载均衡？",
    39: "TiCDC 或 TiDB-to-MySQL 同步，能同步用户和数据库吗，还是只能同步表？",
    40: "region 和 peer 怎么理解，3 副本是指 3 个 peer 吗？",
    41: "并发查询导致其他查询变慢，从 1 秒变 50 秒，怎么排查？",
    42: "DM 向多副本容灾集群写入速度较慢（5 副本），怎么优化？",
    43: "AUTO_INCREMENT=1000001 且 AUTO_ID_CACHE=1 时主键 id 会跳号，为什么？",
    44: "TiDB 表设置 TTL 后导致系统读 IO 飙升、慢查询增多，怎么处理？",
    45: "用 TiCDC 把增量数据同步到 Kafka 给 BI 用，需要部署几台 TiCDC？",
    46: "三台物理机各部署 4 个 TiKV，同一副本的数据会落在同一台服务器上的不同 TiKV 节点吗？",
    47: "TiDB 从 v7.5.2 平滑升级到 v7.5.5 的最佳实践？",
    48: "用 DM 从 MySQL 迁移 1TB 数据到 TiDB，大概要多久？遇到 TiDB 不支持的数据类型怎么处理？",
    49: "TiDB 事务里每隔 10 秒 send TxnHeartBeat 更新 TTL，这个 TTL 怎么理解？",
    50: "RU 占用高，怎么找到占用 RU 较高的 SQL？",
    51: "TiDB 有办法从 6.5.3 降级到 6.5.1 吗？",
    52: "MySQL 有实例的概念，TiDB 有这个概念吗？",
    53: "TiDB v3.1.0 的 40T 数据迁移到新集群，有什么成熟的方案或迁移工具？",
    54: "ANALYZE TABLE 报错 Primary key column not found，无法 analyze 表，怎么处理？",
    55: "普通表改成分区表后，怎么快速把普通表的数据导入分区表？",
    56: "一个 TiKV 节点 region 数量近 20 万（官方建议不超 5 万），会影响性能吗？",
    57: "单表 2 亿条数据一次 select 返回全表，内存 24G 报溢出，TiDB 一次查询支持返回多少数据？",
    58: "INSERT 语句的 select 里不会走 TiFlash 吗？",
    59: "TiDB 在数据安全方面做了哪些保障？",
    60: "怎么在查询计划里看每个算子基于代价模型预估的代价？",
    61: "客户端终止长事务后，TiDB 数据库端不终止任务且最终提交，怎么解决？",
    62: "TiDB 程序执行和手动执行同一 SQL，执行计划不一致，为什么？",
    63: "TiFlash 存算分离架构有哪些优劣势，适合什么场景？",
    64: "TiDB 执行 SQL 完成后内存不会立即释放，导致占用越来越高，怎么解决？",
    65: "生产环境 TiDB 主库集群 OOM 内存溢出，怎么排查？",
    66: "TiFlash 查 1.5 亿数据的表，执行计划走了 TiFlash 却比平时慢 2 倍，为什么？",
    67: "事务内 delete 是作用在 buffer 上吗，commit 前不会真正删 KV 吗？",
    68: "TiDB 升级到 v7.1.1 后安全扫描出漏洞，怎么解决？",
    69: "region 心跳定时忽高忽低正常吗？",
    70: "TiDB 能用 mysqldump 工具备份吗，兼容吗？",
    71: "TiDB 的日志可以直接删吗，有没有自动删除策略？",
    72: "TiDB 插数据很占内存，为什么？",
    73: "TiDB 可以多表同时授权吗？",
    74: "TiFlash 服务器无 AVX2 指令集导致无法启动，怎么处理？",
    75: "使用聚簇索引时 drop primary key 报 Error 8200，怎么处理？",
    76: "TiKV 节点存储有大小限制吗？",
    77: "BR 快照备份时，怎么监控备份进度？",
    78: "怎么设置 GO 的环境变量 GOMEMLIMIT，在哪些组件设置？",
    79: "TiDB 生产环境建议使用查询缓存功能吗？",
    80: "TiDB v6.0 的 Collation 能改成 MySQL 8.0 的 utf8mb4_0900_ai_ci 吗？",
    81: "TiCDC 同步数据异常（多条数据合并），怎么排查？",
    82: "删除数据报 transaction is too large, size 104857608，怎么处理？",
    83: "怎么用 flyway 连接 TiDB？",
    84: "hash join 前先过滤数据，怎么提高 SQL 查询速度？",
    85: "TiKV 无法写入，磁盘使用限制参数在哪看，怎么修改？",
    86: "truncate table 是对表还是对行做删除标记？drop table 呢？",
    87: "TiDB 分区表怎么做全表分组查询？",
    88: "TiDB 与 Greenplum 7 的 OLTP 性能对比如何？",
    89: "dashboard 用户登录认证失败 authenticate failed，怎么处理？",
    90: "TiDB 有提供 C++ 的 API 吗？",
    91: "TiDB GC 清理异常导致空间不断上涨，怎么处理？",
    92: "TiKV 一个节点下线很慢，怎么加速？",
    93: "dumpling 导出 CSV 怎么输出到一个文件？",
    94: "归档产生 delete 事件导致 CDC 链路延迟，怎么处理？",
    95: "为什么语句的执行计划没有算子？",
    96: "TiDB 查询结果怎么导出到指定文件？",
    97: "TiDB 索引结构是怎么存储的，会在 RocksDB 新建一个列簇吗？",
    98: "主从两个库同一个 SQL 执行计划不一样，为什么？",
    99: "哪里有 TiDB 的 Grafana 比较完整的使用说明？",
    100: "/tmp/tidb 目录空间大小怎么计算？",
    101: "为什么简单的查询会产生慢查询？",
    102: "动态分区裁剪收集统计信息时 globalstats 合并耗时很久 hang 住，怎么处理？",
    103: "SQL 执行计划的顺序怎么看？",
    104: "表亿级数据，查询速度慢，怎么优化？",
    105: "TiKV 突然出错后自己退出，tikv-20160.service 主进程退出，怎么排查？",
    106: "如何高效修改大表 id 列改成递增数字？",
    107: "怎么查询一个月前的 DML SQL 执行语句？",
    108: "timestamp(0) 的默认值问题，怎么处理？",
    109: "基于 TiCDC 不能同步无有效索引表，怎么从源头解决无有效索引表的创建？",
    110: "TiDB 目前只支持 hash 和 range 分区吗？",
    111: "BR 备份从 TiKV 集群读数据是从 leader 还是所有 peer 读？",
    112: "Write conflict 写写冲突怎么处理？",
    113: "TiKV 里 SST 文件、Raft 日志和 KV 数据是怎么对应的？",
    114: "DM 不支持多库迁移吗？",
    115: "TiDB 单机离线部署怎么做？",
    116: "用 navicat 在 TiDB 建表时没办法添加自增 id，为什么？",
    117: "用 tiup 部署后，等保修改了 ssh 端口，还需要配合做什么吗？",
    118: "tikv-ctl 扫描数据的时候的警告是什么原因？",
    119: "配置冷热隔离分区策略后，新增 label disk=hhd 导致数据写入慢、cpu 高，怎么处理？",
    120: "TiCDC 同步任务可以暂停多长时间？",
    121: "TiDB 读取热点是什么意思，怎么消除读热点？",
    122: "临时表插入大量数据报错 the table is full，怎么处理？",
    123: "dumpling 导数据很慢，卡在 0% 不动，怎么处理？",
    124: "TiDB 支持表级别压缩吗？",
    125: "单机房 3 节点 TiKV 集群怎么在线扩容成 3 机房 9 节点集群？",
    126: "TiKV 之间的网络带宽占用看哪个监控图？",
    127: "DM、TiCDC、TiDB Binlog 各自的适用场景、优缺点和异同是什么？",
    128: "pump 怎么检查数据同步延迟？",
    129: "TiDB 统计信息收集的采样比例可以调整吗？",
    130: "主键值对空格的处理有什么疑问？",
    131: "TiDB 7.1 集群卡住，报 lock txn not found, lock has expired，怎么处理？",
    132: "节点上线 balance 速度怎么控制？",
    133: "TiUP Grafana 怎么配置匿名登录？",
    134: "有没有语句把一个表变成另一个表内的分区表？",
    135: "为什么非分区表在修改分区时报错？",
    136: "TiDB 7.5.1 有哪些有用的特性？",
    137: "TiKV truncate 后磁盘空间不释放，GC 相关问题怎么处理？",
    138: "怎么对 TiCDC 同步异常做报警？",
    139: "哪里能看到 TiCDC 的历史版本和更新内容？",
    140: "TiDB 后续是否实现执行计划缓存功能？",
    141: "like 支持下推到各个 KV 节点计算吗？",
    142: "TiCDC 已经有 WebUI 了吗？",
    143: "DM 全量 dump 阶段日志信息少一个表，怎么排查？",
    144: "Online DDL 语句执行流程中做了哪些事？",
    145: "修改服务器 IP 后，playground 需要改哪些配置？",
    146: "把表删了磁盘空间会立即释放吗，不释放怎么释放空间？",
    147: "TiDB Server 执行 SQL 怎么判断 OOM 内存溢出？",
    148: "低版本 TiDB 迁移到高版本有什么方案？",
    149: "应用连接 TiDB 的 IP 连接串里可以使用多个 ip+port 组合吗？",
    150: "告警 TiKV pending task too much type resolved-ts，怎么处理？",
    151: "TiDB 最大能存储多少数据？",
    152: "TiDB 支持 MySQL 的事件功能吗？",
    153: "监控文档里没有 Flow Control 相关指标描述，怎么办？",
    154: "lightning 增量数据导入报错，checkpoint 怎么处理？",
    155: "batch ddl job 执行时间差距巨大，可能原因有哪些？",
    156: "TiDB 是怎么判断大事务的？",
    157: "等保扫描漏洞，这些怎么解决？",
    158: "TiKV 节点数据盘在哪些情况下超过 90% 必须扩容？",
    159: "可以用 hint 强制使用 TiFlash 吗，怎么用？",
    160: "什么是 DM 集群？",
}

# 特殊备注
NOTE = {
    3: "golden可疑，人工复核（导入场景走 Lightning，insert 优化走事务/SQL）",
    7: "golden可疑，tiup-bench 是压测非 sqltest，人工复核/替换",
    9: "确认原帖是否问频繁自动执行",
    13: "golden 碎链 release-3，人工复核",
    88: "golden 碎链 v5，人工复核",
    113: "golden弱，可能补为多证据",
    136: "golden 碎链 release-7，人工复核",
    160: "简单基线题",
}

out = []
for r in rows:
    i = r['id']
    if i in DROP or i not in REWRITE:
        continue
    slugs, broken = resolve_slugs(r['slugs'])
    note = NOTE.get(i, '')
    if broken:
        note = (note + '；' if note else '') + f"碎链未解析: {','.join(broken)}"
    out.append({
        'id': i,
        'orig_group': r['group'],
        'title': r['title'],
        'query': REWRITE[i],
        'golden_slugs': slugs,
        'distinct_slug_count': None,
        'is_true_multi': None,
        'grade': r.get('grade', ''),
        'note': note,
    })

out.sort(key=lambda x: x['id'])
with open(os.path.join(ROOT, 'data', 'eval_set_v1.jsonl'), 'w', encoding='utf-8') as f:
    for o in out:
        f.write(json.dumps(o, ensure_ascii=False) + '\n')

print(f'落盘 {len(out)} 条 → data/eval_set_v1.jsonl')
from collections import Counter
print('分组:', dict(Counter(o['orig_group'] for o in out)))
print('带碎链 note 的条数:', sum(1 for o in out if '碎链' in o['note']))
print('golden 待复核条数:', sum(1 for o in out if '复核' in o['note'] or '碎链' in o['note']))
