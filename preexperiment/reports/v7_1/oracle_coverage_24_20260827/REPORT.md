# V7.1 Complete-MIP Incumbent / Stage-1 Oracle Coverage Audit

日期：2026-08-27  
算例：`scale_g024_s401`  
模型：`hierarchical_group_area_bay_pattern_v7`

## 1. 审计口径

- feasibility-only witness：10 秒上限；
- Stage 1：10 秒、8 个按 `Y[g,a]` 支持区分的 pool solutions；
- Complete MIP：120 秒、单线程、seed 0；
- Stage-1 cap：用同一份 Stage-1 pool 回放 cap 2 与 cap 3；
- 只读审计，不修改模型、目标、peak cap、pricing、RIM 或生产候选域。

Complete MIP 在 120 秒内得到 incumbent `0.045754036252225815`，bound
`0.04228348040231884`，gap `7.5852%`。因此本文审计的是同预算下的
best-known incumbent support，不声称它是已证明最优解。

## 2. 核心结果

Complete-MIP incumbent 使用 28 个正流量 `group-area` 组合，共 640 箱。

| 域 | 覆盖组合 | 组合覆盖率 | 覆盖箱量 | 箱量覆盖率 | 完整覆盖箱组 |
|---|---:|---:|---:|---:|---:|
| Stage-1 cap 2 | 4 / 28 | 14.29% | 85 / 640 | 13.28% | 3 / 24 |
| Stage-1 cap 3 | 4 / 28 | 14.29% | 85 / 640 | 13.28% | 3 / 24 |

被覆盖的 4 个组合为：

| 箱组 | 箱区 | MIP 箱量 | 进入方式 |
|---|---:|---:|---|
| `390121_E007` | 35 | 17 | Stage-1 best |
| `390131_E002` | 90 | 30 | Stage-1 best |
| `390131_E008` | 24 | 12 | pool rank 1 |
| `390131_E011` | 13 | 26 | Stage-1 best |

feasibility witness 与 28 个 incumbent 组合的交集为 0。Stage-1 best 只覆盖
其中 3 个；diversified pool 只额外覆盖 1 个。

## 3. 24 个遗漏组合

`metadata rank` 是该箱组完整合法箱区的 metadata 顺序；`minimum cap` 是在
保持当前“至少一个 metadata 槽位、其余优先给 pool”规则时，首次能够选中该
组合所需的 cap。

| 箱组 | 箱区 | MIP 箱量 | metadata rank | minimum cap |
|---|---:|---:|---:|---:|
| `390121_E006` | 24 | 26 | 6 | 6 |
| `390121_E010` | 24 | 26 | 6 | 6 |
| `390121_E012` | 24 | 26 | 6 | 6 |
| `390131_E005` | 46 | 35 | 7 | 7 |
| `390131_E007` | 13 | 17 | 8 | 8 |
| `390131_E009` | 13 | 36 | 8 | 8 |
| `390121_E008` | 50 | 7 | 9 | 9 |
| `390131_E006` | 50 | 28 | 9 | 9 |
| `390121_E001` | 50 | 18 | 10 | 11 |
| `390131_E001` | 50 | 28 | 10 | 11 |
| `390131_E010` | 46 | 35 | 10 | 12 |
| `390121_E009` | 15 | 23 | 14 | 15 |
| `390131_E012` | 15 | 39 | 15 | 15 |
| `390121_E002` | 21 | 23 | 15 | 16 |
| `390121_E004` | 21 | 22 | 15 | 16 |
| `390121_E011` | 15 | 18 | 14 | 16 |
| `390131_E004` | 51 | 34 | 16 | 17 |
| `390121_E003` | 41 | 24 | 19 | 19 |
| `390121_E005` | 26 | 19 | 20 | 20 |
| `390131_E003` | 22 | 12 | 26 | 26 |
| `390131_E003` | 20 | 15 | 28 | 28 |
| `390131_E008` | 26 | 16 | 29 | 29 |
| `390121_E008` | 48 | 16 | 30 | 30 |
| `390131_E007` | 12 | 12 | 31 | 31 |

这些组合合计 555 箱。按遗漏箱区聚合，影响最大的为：50 区 81 箱、15 区
80 箱、24 区 78 箱、46 区 70 箱、13 区 53 箱、21 区 45 箱。

## 4. 遗漏原因

### 4.1 不是 cap=2 对 pool 的简单截断

24 个遗漏组合在 8 个 Stage-1 pool solutions 中的出现频率全部为 0。因此 cap
从 2 增加到 3 时，新增槽位仍被分配给 pool channel，却没有任何 oracle 组合
可以加入，解释了两次实验的 root LP 和 UB 完全相同。

### 4.2 Stage-1 surrogate 主动把 oracle support 排在后面

将 Complete-MIP incumbent 的 group-area 数量代入 Stage-1 coarse objective，得到
`0.020256304953014455`；Stage-1 best 为 `0.01810389897131638`。在 Stage-1 看来，
MIP support 比其 best 差 11.89%。Stage-1 自身 gap 只有 0.63%，所以这不是 10 秒内
没有搜索到，而是 coarse objective / relaxation 对完整模型的联合布局质量发生误排。

Stage 1 对每个 group-area 使用该箱区内最乐观的单贝位距离成本，并只用 aggregate
capacity 与简单 bay-incidence proxy 表示资源。它无法表达多个箱组对同一批排/贝位的
竞争、实际可用贝位位置、span，以及为避免冲突而进行的跨箱区协调。因此 Stage 1 会
偏向少数表面上容量大、局部成本低的箱区；Complete MIP 则使用了较分散但联合可实现、
最终业务目标更好的组合。

### 4.3 盲目提高 cap 不可取

当前 ranking 下，完整纳入这些组合至少需要某些箱组的 cap 达到 31，会基本破坏
restricted-domain 的规模控制。审计不支持继续机械增加 cap。

## 5. 结论与下一步

本次 UB 退化的直接原因已经确认：Stage-1 surrogate 与完整 Bay/row 联合模型之间存在
结构性 ranking mismatch，导致 85.71% 的 Complete-MIP incumbent group-area 支持被
排除。feasibility witness、cap=2/3 和 RIM 时间都不是主因。

下一步应修改 Stage-1 候选评分或增加 conflict-aware / bay-feasibility-aware 候选通道，
使候选选择反映组间资源竞争和贝位级可实现性；不应恢复 business-quality witness，也
不应直接把 cap 提高到 20–31。

## 6. 数据文件

- `coverage_audit.json`：完整汇总及每个 oracle 组合的全部字段；
- `coverage_audit.csv`：28 个 incumbent 组合的平面明细，包括 pool frequency、pool / metadata rank、容量、贝位数、距离和 cap 选择原因。
