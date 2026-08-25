# Phase 3.3 Meaningful Seed Rotation Report

## 1. 目的与隔离范围

Phase 3.1/3.2 诊断显示，只要某轮改善，原实现就清空 unsuccessful seed 集合，下一轮会再次选择目标贡献最高的同一箱组。24/48/96 的旧基线因此连续三轮均以 `390121_E007` 为 seed。

本阶段只新增 `rotating_conflict_multi_round`：每轮排除所有此前已尝试 seed，直到候选耗尽才重置。旧 `conflict_multi_round`、稀疏有向图、目标、static pool、multi-start、pricing、root proof、initial-MIP 时间、F&O 三轮预算、cuts 与 recourse 均不改变。

生产默认仍为 `conflict_multi_round`；轮换策略是显式实验选项。算法版本为 `integrated_zone_v5_seed_rotation_phase3_3`。

## 2. Tests

- 44 passed，0 failed，0 skipped。
- 新增覆盖：即使每轮都改善，三轮也排除已尝试 seed；记录 distinct seed 数和 rotation policy。
- 24/48/96 三次实验均通过 internal/external validation。

## 3. 等预算结果

基线使用刚完成的 Phase 3.2 legacy paired runs；两者固定相同输入、seed、`PYTHONHASHSEED=0`、`Threads=1`、固定 75% initial restricted MIP。24 使用 60 秒，48/96 使用 120 秒。

| Groups | Legacy UB | Rotation UB | Rotation improvement | Legacy gap | Rotation gap | Decision |
|---:|---:|---:|---:|---:|---:|---|
| 24 | **0.134315** | 0.134761 | -0.332% | **13.914%** | 14.199% | fail |
| 48 | 0.141992 | **0.140949** | +0.735% | 16.682% | **16.066%** | pass |
| 96 | 0.171137 | **0.170013** | +0.657% | 18.259% | **17.719%** | pass |

所有 paired root LB 相同。24/96 的 initial UB 完全相同；48 rotation 的 initial UB 比 legacy 好 0.000173，但 rotation 的 F&O 自身改善量也更大，结论不由该波动驱动。

| Groups | Legacy F&O gain | Rotation F&O gain | Legacy F&O seconds | Rotation F&O seconds |
|---:|---:|---:|---:|---:|
| 24 | **0.009529** | 0.009083 | 11.226 | 11.232 |
| 48 | 0.004188 | **0.005058** | 21.969 | 22.007 |
| 96 | 0.006200 | **0.007324** | 15.311 | 15.295 |

## 4. 逐轮诊断

- 24 legacy seeds：`E007 -> E007 -> E007`；rotation：`E007 -> E008 -> E010`。第二轮 rotation 已到 0.136892，显著优于 legacy 的 0.140601；但 legacy 第三轮继续深挖 E007 后到达更好的 0.134315。
- 48 rotation：`E007 -> E012 -> E008`，逐轮 UB 为 0.143852、0.142820、0.140949；最终比 legacy 改善 0.735%。
- 96 rotation：`E007 -> E012 -> E004`，逐轮 UB 为 0.175764、0.172397、0.170013；第二轮即明显优于 legacy 的 0.174180，最终改善 0.657%。

轮换不增加建模轮数或 zone budget，主要收益来自探索不同 incumbent 盆地。24 的轻微退化说明“每轮强制不同”会牺牲对高价值 seed 的持续深化，因此不能根据 2 胜 1 负直接替换生产默认。

## 5. 阶段结论

1. 有效 seed rotation 是真实、有成本效率的 UB 改进方向，在 48/96 同时改善 UB 和 gap。
2. hard rotation 缺少 exploitation 回访机制，24 退化 0.332%。
3. 保留为实验策略，但默认继续使用 legacy `conflict_multi_round`。
4. 不针对三个 case 调轮换冷却期或指定 seed 顺序。
5. 下一阶段按既定顺序研究 elite bundle columns：保留跨箱组共同形成优质 incumbent 的列组合，而不是只按单列通道排名；仍需独立分支与 24/48/96 阶段门槛。

## 6. 原始结果

- Legacy：Phase 3.2 三个 `v5_phase3_2_baseline_*` 输出目录。
- Rotation：
  - `preexperiment_outputs/v5_phase3_3_rotating_pilot_l301_60s_20260825`
  - `preexperiment_outputs/v5_phase3_3_rotating_scale_g048_120s_20260825`
  - `preexperiment_outputs/v5_phase3_3_rotating_scale_g096_120s_20260825`
