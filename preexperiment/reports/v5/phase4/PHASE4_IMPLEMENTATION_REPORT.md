# Phase 4 Dynamic Initial-MIP Stopping Report

## 1. 实现

按照V5总体指令实现了：

- `initial_mip_max_remaining_fraction = 0.50`；
- `initial_mip_min_total_fraction = 0.05`；
- `initial_mip_stagnation_total_fraction = 0.10`；
- `initial_mip_min_relative_improvement = 1e-4`；
- 没有feasible incumbent时不允许stagnation终止；
- 保存`stagnation/time_limit/optimal/no_solution`退出原因和meaningful-improvement时钟。

该callback只用于V5 restricted initial MIP，Complete MIP baseline保持隔离。目标、可行域、static pool、multi-start、pricing、root proof、F&O和recourse均未改变。

## 2. Tests

- 42 passed，0 failed，0 skipped。
- 覆盖：无首解保护、微小改善不重置、显著改善重置、配置校验、end-to-end diagnostics、Complete MIP isolation、双重验证。

## 3. 配对结果

固定使用`conflict_multi_round`，只比较旧75%策略与总体指令给出的dynamic默认值。

| Groups | Fixed UB | Dynamic UB | Dynamic improvement | Fixed initial/F&O (s) | Dynamic initial/F&O (s) | Dynamic termination |
|---:|---:|---:|---:|---:|---:|---|
| 24 | **0.134315** | 0.140749 | -4.790% | 42.70 / 11.23 | 14.98 / 38.83 | stagnation |
| 48 | 0.143515 | **0.143245** | +0.188% | 84.17 / 22.05 | 53.12 / 53.28 | time_limit |
| 96 | **0.170406** | 0.174514 | -2.411% | 64.52 / 15.50 | 39.99 / 39.98 | time_limit |

三个规模的root LB逐值相同，所有解均通过internal/external validation。

## 4. 诊断

Dynamic stopping确实释放了大量F&O时间，但没有稳定改善UB：

- 24的initial UB从`0.143844`变差到`0.145210`，额外27.6秒F&O仍只能到`0.140749`；
- 48的initial UB从`0.146007`变差到`0.147273`，额外31.2秒F&O勉强补偿并最终改善0.188%；
- 96的initial UB从`0.177338`变差到`0.183330`，F&O自身改善量虽从`0.006932`增到`0.008816`，仍远不足以补回initial损失。

因此问题不是“F&O时间太少”这一项。当前conflict neighborhood使用额外时间的效率不足，而restricted initial MIP在后段仍会产生重要incumbent。

24在约15秒触发stagnation，表明`10% total budget`窗口对本问题过于激进；96即使没有stagnation、只执行50% hard cap仍明显退化，说明不能仅通过放宽stagnation修复。

## 5. 决策

Phase 4功能保留为显式opt-in实验选项，但生产默认恢复为关闭，继续使用原75%基线。不得根据唯一改善的48箱组把dynamic设为默认，也暂不针对三个case调节50%/10%参数。

下一步按顺序进入conflict signal重构：先让F&O能够有效利用开放时间，再重新评估initial/F&O时间分配。本阶段不实施Dual Stabilization、Valid Inequalities、Local Branching或Branch-and-Price。
