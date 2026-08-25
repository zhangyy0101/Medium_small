# Phase 3.2 Sparse Directed Conflict Graph Report

## 1. 目的与隔离范围

本阶段只重构 F&O 邻域的冲突信号。旧 `conflict_multi_round` 完整保留为基线，新增实验策略 `directed_conflict_multi_round`。以下内容均未改变：目标函数、可行域、Phase 2 static primal pool、multi-start、pricing、root proof、时间预算、三轮 12%/22%/35% zone 预算、cuts、dual stabilization、local branching、branch-and-price 与 exact recourse。

动态 initial-MIP stopping 继续默认关闭。因此本阶段比较的是固定 75% initial restricted MIP 下的邻域选择差异。

## 2. 实现

旧图将每个箱组的所有可达箱区和物理排位用于无向 Jaccard；在 24/48/96 三个基线中，非零边占比均为 100%。新图改为：

1. 对每个箱组按当前业务单位成本排序，保留 `ceil(sqrt(可达箱区数))` 个候选目的箱区；
2. 建立 `mover -> incumbent owner` 有向边，表示 mover 的优质候选区域/排位需要 owner 释放当前占用；
3. 对过度普遍的候选物理资源使用按可达箱组频次归一化的逆频率权重；
4. 四项分量仍为同航次、共享箱区、物理资源冲突、峰值利用率交换，继续等权 0.25，不调人工权重；
5. 每个 mover 只保留最强的 `ceil(sqrt(箱组数-1))` 条出边；
6. F&O 仍按原 candidate-zone budget 动态开放所选箱组的完整 zone 候选。

算法版本为 `integrated_zone_v5_directed_conflict_phase3_2`。生产默认仍为旧 `conflict_multi_round`，新策略仅可显式启用。

## 3. Tests

- `.venv/bin/python -m unittest discover -s tests -p 'test_*.py'`
- 43 passed，0 failed，0 skipped。
- 新增覆盖：配置校验、有向资源所有者关系、四分量范围、稀疏出边上界、候选前沿诊断。
- 24/48/96 共 6 次 paired runs 全部通过 internal/external validation。

## 4. 等预算配对结果

固定 `PYTHONHASHSEED=0`、`Threads=1`、相同输入与 seed。24 使用 60 秒，48/96 使用 120 秒。

| Groups | Legacy UB | Directed UB | Directed improvement | Legacy gap | Directed gap | Decision |
|---:|---:|---:|---:|---:|---:|---|
| 24 | **0.134315** | 0.135394 | -0.804% | **13.914%** | 14.600% | fail |
| 48 | **0.141992** | 0.143225 | -0.868% | **16.682%** | 17.399% | fail |
| 96 | 0.171137 | **0.170362** | +0.453% | 18.259% | **17.887%** | pass |

三个规模的 root LB 在各自配对内完全一致：24 为 0.115627，48 为 0.118305，96 为 0.139889。24/96 的 initial UB 完全相同；48 的 initial UB 有 0.000173 的 wall-clock MIP 路径波动，但 directed 的 initial UB 反而更好，不能解释其更差的 final UB。

F&O 自身改善量：

| Groups | Legacy F&O gain | Directed F&O gain | Legacy F&O seconds | Directed F&O seconds |
|---:|---:|---:|---:|---:|
| 24 | **0.009529** | 0.008449 | 11.226 | 11.220 |
| 48 | **0.004188** | 0.002782 | 21.969 | 22.044 |
| 96 | 0.006200 | **0.006976** | 15.311 | 15.520 |

## 5. 冲突图与邻域诊断

| Groups | Legacy nonzero / pairs | Directed retained / arcs | Directed density | Final-round legacy groups | Final-round directed groups |
|---:|---:|---:|---:|---:|---:|
| 24 | 276 / 276 | 120 / 552 | 21.739% | 10 | 7 |
| 48 | 1128 / 1128 | 336 / 2256 | 14.894% | 18 | 13 |
| 96 | 4560 / 4560 | 960 / 9120 | 10.526% | 35 | 23 |

新图成功消除了旧图的完全稠密问题，并且在相同 candidate-zone budget 下更偏向候选数较大的少量箱组。24 的前两轮 directed UB 分别为 0.139156、0.137783，优于 legacy 的 0.141857、0.140601；但 legacy 第三轮通过 10 箱组联合调整跃迁到 0.134315，而 directed 的 7 箱组邻域没有覆盖该关键组合。48 同样表现为最终联合箱组不足。96 中较小而定向的邻域反而提高了单位时间效率。

## 6. 阶段结论

该实现证明“候选目的资源与 incumbent 实际占用者的有向关系”能够形成可解释的稀疏图，并在 96 箱组上改善 UB 和 gap；但 hard top-k 截断在 24/48 上产生明显 coverage loss。三个规模仅 1 胜 2 负，未通过升级默认策略的阶段门槛。

因此：

1. 保留实现和完整诊断，作为后续消融选项；
2. 默认继续使用旧 `conflict_multi_round`；
3. 不在本阶段针对三个 case 调整 top-k、四项权重或候选前沿阈值；
4. 下一独立阶段按既定顺序评估“有效 seed rotation”，解决当前每轮改善后重复选择同一 objective seed 的问题；
5. 暂不进入 elite bundle、local branching、dual stabilization、valid inequalities 或 branch-and-price。

## 7. 原始结果

- `preexperiment_outputs/v5_phase3_2_baseline_pilot_l301_60s_20260825`
- `preexperiment_outputs/v5_phase3_2_directed_pilot_l301_60s_20260825`
- `preexperiment_outputs/v5_phase3_2_baseline_scale_g048_120s_20260825`
- `preexperiment_outputs/v5_phase3_2_directed_scale_g048_120s_20260825`
- `preexperiment_outputs/v5_phase3_2_baseline_scale_g096_120s_20260825`
- `preexperiment_outputs/v5_phase3_2_directed_scale_g096_120s_20260825`
