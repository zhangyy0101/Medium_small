# V5 研究分支与实验索引

本文是 V5 算法研究的统一入口。模型口径始终为
`integrated_zone_v4`；V5 表示求解算法升级，不表示数学模型版本改变。

## 1. 线性提交链

| 阶段 | 分支 | 归档提交 | 状态 | 结论 |
|---|---|---|---|---|
| V4 基线 | `feat/integrated-yard-objective` | `c97d7c3` | 冻结 | 五项目标、精确 RMQ pricing 和 exact recourse 基线 |
| Phase 1 | `feat/v5-integrality-aware-primal-search` | `0a3b02a` | 保留 | diagnostics、greedy 修复、repaired multiple starts |
| Phase 1.1 | `feat/v5-phase1-1-budget-baseline-fix` | `dd8b55c` | 保留 | 修正 start 预算与 baseline 隔离 |
| Phase 2/2.1 | `feat/v5-phase2-proof-primal-pool` | `8d4d918` | 结构保留、策略待替换 | Proof/Primal 分离成立；static pool 会裁掉关键组合列 |
| Phase 3 | `feat/v5-phase3-conflict-aware-multiround-fo` | `83ca38e` | 机制保留、性能未过门槛 | 三轮 F&O 可持续改善，但未稳定优于 Complete MIP |
| Phase 3.1 | `exp/v5-phase3-1-neighborhood-ablation` | `77fa90e` | 消融归档 | objective/conflict/hybrid 均无跨规模统一赢家 |
| Phase 4 | `feat/v5-phase4-dynamic-initial-mip-stopping` | `5c307ff` | 实验保留、默认关闭 | 提前停止释放时间，但 24/96 UB 退化 |
| Phase 3.2 | `feat/v5-directed-conflict-phase3-2` | `ebddc0c` | 实验保留、不作默认 | 稀疏有向图解决稠密性，但仅 96 改善 |
| Phase 3.3 | `feat/v5-meaningful-seed-rotation-phase3-3` | `e9e15a2` | 候选保留、不作默认 | 48/96 改善约 0.7%，24 轻微退化 |

这些分支处于同一条继承链上，不需要相互 merge。较新的分支已经包含较早
阶段的代码和报告；各阶段分支仅用于复现实验边界。

## 2. 当前冻结代码的默认行为

最新研究代码虽然包含多个显式实验开关，但安全默认仍为：

```text
fix_optimize_policy = conflict_multi_round
initial_mip_dynamic_stopping_enabled = false
```

因此：

- `directed_conflict_multi_round` 只是 Phase 3.2 消融；
- `rotating_conflict_multi_round` 只是 Phase 3.3 候选；
- dynamic initial-MIP stopping 只是 opt-in 实验；
- 算法版本字符串记录“代码包含到哪个研究阶段”，不表示该阶段实验策略已成为默认。

## 3. 已确认可继续复用的基础

以下内容已经通过测试和代表性实验，应作为下一阶段的不变量：

1. 完整合法 zone 集和容量规则；
2. exact prefix-RMQ pricing；
3. root closure certificate，且 root LB 等于 complete zone LP optimum；
4. Proof-oriented master 与 Primal-oriented master 物理分离；
5. repaired multiple starts 的联合可行性认证；
6. exact row recourse、internal validation 和 external validation；
7. Complete MIP baseline 的独立实现路径；
8. 五项目标、归一化权重和 peak-utilization epsilon constraint。

## 4. 已确认的主要瓶颈

1. Phase 2 static primal pool 的逐列五通道裁剪会丢失跨箱组协调所需的关键组合列；
2. 当前 conflict graph/seed 变体只能产生约零点几个百分点的波动，继续微调边际收益低；
3. fixed initial MIP 占用大量 primal 时间，但现有 F&O 不能稳定利用提前释放的时间；
4. F&O 主要改善 UB，不能解决独立 root gap 弱于 Complete MIP tree bound 的问题；
5. 目前只有代表性单 seed，不能形成正式论文统计结论。

## 5. 报告入口

- Phase 1：`phase1/V5_IMPLEMENTATION_REPORT.md`
- Phase 1.1：`phase1_1/PHASE1_1_IMPLEMENTATION_REPORT.md`
- Phase 2：`phase2/V5_IMPLEMENTATION_REPORT.md`
- Phase 2.1 oracle audit：`phase2/ORACLE_COVERAGE_AUDIT.md`
- Phase 3：`phase3/V5_IMPLEMENTATION_REPORT.md`
- Phase 3.1：`phase3_1/NEIGHBORHOOD_ABLATION_REPORT.md`
- Phase 4：`phase4/PHASE4_IMPLEMENTATION_REPORT.md`
- Phase 3.2：`phase3_2/DIRECTED_CONFLICT_REPORT.md`
- Phase 3.3：`phase3_3/SEED_ROTATION_REPORT.md`

## 6. 分支使用规则

- 复现历史阶段：切换到上表对应分支，不在该分支继续开发；
- 新算法开发：只从冻结标签 `baseline/v5-phase3-handoff-20260825` 开始；
- 不删除历史阶段分支，不把失败消融 merge 回旧基线；
- 新阶段按“一个结构变化、一个独立分支、一组配对实验、一份阶段报告”执行；
- 未经正式 multi-seed stage gate，不修改生产默认策略。
