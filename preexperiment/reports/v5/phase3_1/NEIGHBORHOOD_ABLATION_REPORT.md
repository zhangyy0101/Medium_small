# Phase 3.1 Neighborhood Ablation Report

## 1. 目的

在不改变目标、Phase 2 static pool、multi-start、pricing、root proof、peak constraint 和 exact recourse 的前提下，比较：

1. `objective_single`：旧 objective-mass 单轮，35%候选 zone；
2. `conflict_single`：当前 conflict 单轮，35%候选 zone；
3. `conflict_multi`：当前 conflict 三轮，12%/22%/35%；
4. `hybrid_multi`：首轮 objective，后两轮 conflict，12%/22%/35%。

## 2. 正确性和配对

- 40 tests passed，0 failed，0 skipped。
- 24/48/96 共12次运行全部通过 internal/external validation。
- 每个规模使用相同 manifest、seed、总时限、`Threads=1`、`PYTHONHASHSEED=0`。
- root LB 在同规模四策略中完全一致。
- initial MIP 使用 wall-clock `TimeLimit`，48各次 initial UB 有最多 `1.76e-4` 波动；96的hybrid有 `4.94e-4`波动。结论同时比较 final UB 和从各自 initial UB 出发的 F&O improvement，不把前段波动误归因于邻域。

## 3. 最终结果

| Groups | Objective single | Conflict single | Conflict multi | Hybrid multi | Winner |
|---:|---:|---:|---:|---:|---|
| 24 | 0.138649 | 0.139606 | **0.134315** | 0.135005 | Conflict multi |
| 48 | 0.142642 | 0.143765 | 0.141992 | **0.141580** | Hybrid multi |
| 96 | 0.170322 | **0.170235** | 0.170406 | 0.172522 | Conflict single |

相对 `objective_single`：

| Groups | Conflict single | Conflict multi | Hybrid multi |
|---:|---:|---:|---:|
| 24 | -0.691% | +3.126% | +2.628% |
| 48 | -0.787% | +0.456% | +0.745% |
| 96 | +0.051% | -0.049% | -1.292% |

正值表示UB更低。

## 4. Selection 与时间碎片化判断

### Selection

`conflict_single` 在24/48均显著输给 `objective_single`，只在96微弱领先0.051%。因此当前 conflict graph 不是稳定优于 objective attribution 的选组信号，前一阶段识别出的稠密 conflict graph 问题得到进一步支持。

### Multi-round fragmentation

多轮不是单纯有害：

- 24：`conflict_multi` 比两种单轮都好；
- 48：`conflict_multi` 比两种单轮都好；
- 96：`conflict_multi` 比 `conflict_single` 差0.10%，表现出大规模下的拆分/重建损失。

所以“Phase 3 UB退化完全由三轮切碎时间造成”不成立。更准确的结论是：多轮能通过 incumbent 演化进入新区域，但其收益随规模和搜索路径变化，96上不足以覆盖重复建模与重优化成本。

### Hybrid

Hybrid在48最好、24第二，但96最差；96即使扣除较差initial UB，其F&O improvement `0.005310`仍明显低于objective/conflict的约`0.0070`。因此hybrid不能直接升级为默认策略。

## 5. 结论

1. 没有一个现有策略在24/48/96全部获胜。
2. 当前 conflict selection 缺乏跨规模稳定性。
3. Multi-round有真实价值，但存在规模相关的时间碎片化风险。
4. Objective安全轮能改善部分case，却不是可靠防退化机制。
5. 现阶段不应调人工权重，也不应把hybrid设为生产默认值。

## 6. 下一步

按照V5总体设计顺序，下一步单独实现 **dynamic initial-MIP stopping**，不同时修改 conflict graph。原因是 initial restricted MIP 当前消耗大量primal时间，而其bound不是全局LB。先测试释放时间能否改善现有F&O，再依据结果设计稀疏、定向conflict graph。

本阶段不实现Dual Stabilization、Valid Inequalities、Local Branching或Branch-and-Price。
