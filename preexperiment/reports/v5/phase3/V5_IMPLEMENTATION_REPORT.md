# V5 实施结果

## 1. 实施状态

- 已完成：本轮授权的 **Conflict-aware Multi-round Fix-and-Optimize**。实现包括 incumbent-dependent 四分量 conflict graph、objective contribution seed、按 12%/22%/35% 候选 zone 预算递增的三轮邻域、动态完整 zone 开放、local MIP、优质列回注、primal master 重优化、失败 seed 轮换及逐轮 diagnostics。
- 已完成实验：24/48/96 箱组各一个固定种子 representative paired smoke；`Threads=1`、`PYTHONHASHSEED=0`，与 Phase 2/Complete MIP 使用相同 materialized manifest 和同一总时限。三个 Phase 3 解均通过内部与外部验证。
- 未完成：Phase 2 static pool、五通道、multi-start 的进一步调整；Dynamic Initial-MIP Stopping、Dual Stabilization、Valid Inequalities、Local Branching、Branch-and-Price；16/72 箱组、多开发种子和正式 holdout。
- 未完成原因：上述内容不属于本轮授权范围。本轮在 Phase 3 实现、测试和指定 representative experiments 后停止；当前单种子结果不能替代正式论文统计。

Phase 3 的实现正确且每个代表算例三轮均产生改善，但阶段门没有通过：24 箱组相对 Phase 2 明显改善，48/96 箱组却分别比 Phase 2 单轮 F&O 略差 0.39%/0.48%。

## 2. Git 状态

- base branch: `feat/v5-phase2-proof-primal-pool`
- base commit: `8d4d9185881717872ed75973f4a0d2b89fd49d74`
- working branch: `feat/v5-phase3-conflict-aware-multiround-fo`
- final implementation commit: `cffc7ca538fa7da1ed73614a383e40b6b62821dd`
- 报告与 compact results 另行提交；其提交哈希以本报告所在分支最终 `HEAD` 为准。

## 3. 代码修改

### `yard_planning/contiguous_zone_generation.py`

- 修改：算法版本更新为 `integrated_zone_v5_conflict_multiround_phase3`；新增三轮默认参数 `fix_optimize_max_rounds=3` 和 `(0.12, 0.22, 0.35)`，并校验轮数、区间和单调性。
- 修改：新增 `_objective_contribution_by_group()`，只把原 objective attribution 抽成可复用 seed signal，没有改变 attribution 数学定义。
- 修改：新增 `_incumbent_area_state()` 和 `_build_group_conflict_scores()`，依据同航次、候选箱区 Jaccard、原子候选物理资源冲突、incumbent peak-utilization 交换关系生成 `[0,1]` 分量和等权 conflict score。
- 修改：conflict graph 只使用 atomic placements、candidate area/resource sets 和 incumbent，不枚举或 materialize 全部 zone pair。
- 修改：新增 `_select_conflict_fix_optimize_neighborhood()`；每轮从当前 incumbent 重新计算 contribution/conflict，在本轮候选 zone 总预算内扩展 coupled groups。无改善时排除已失败 seed；有改善时允许重算后复用原 seed。
- 修改：将原单轮 `_run_objective_fix_optimize()` 升级为多轮 orchestrator。每轮按“剩余时间 / 剩余轮数”分配 outer budget，并保持 `fix_optimize_local_fraction=0.85`。
- 修改：新增 `_run_fix_optimize_round()` 和 `_apply_zone_master_start()`；只为本轮 neighborhood 动态 materialize 完整合法 zone，求解 local MIP，把 local incumbent 使用的 zone 以 `fix_opt_round_N` provenance 回注 primal master，再进行受时限约束的 master reoptimization。
- 修改：保存每轮 seed、groups、candidate budget/count、前后 UB、局部/重优化 UB、耗时、nodes、time-to-first/time-to-best、新列数和是否改善。
- 保留：显式 `objective` 单轮 policy 仅作为 Phase 2 消融兼容路径；Complete MIP 路径不调用 conflict graph 或多轮 F&O。

### `preexperiment/runner.py`

- 修改：归档 F&O policy、stop reason、attempted seeds、逐轮 diagnostics，以及带 `fix_opt_round_N` provenance 的 final master column counts。
- 原因：让动态开放、回注列和逐轮增益可以在实验产物中审计。

### `benchmark_contiguous_zones.py`

- 修改：CLI 默认切换到 `conflict_multi_round`，新增最大轮数和三轮候选 zone fractions 参数；保留 `disabled`/`objective` 消融选项。
- 原因：支持可复现运行 Phase 3，同时不破坏旧基线入口。

### `tests/test_contiguous_zone_generation.py`

- 修改：新增 conflict graph micro test，验证共享航次/箱区/物理资源且存在 peak exchange 的 group pair 得分高于不相关 pair，并验证四分量等权且未进行完整 zone-pair materialization。
- 修改：新增受控 multi-round test，验证第一轮失败后 seed 轮换、后续轮次仍可改善、预算逐轮扩大和最终 incumbent 单调不劣。
- 修改：更新配置、算法版本、end-to-end 三轮 diagnostics 和 Complete MIP 隔离断言。

### `MODEL_SCOPE.md`

- 修改：将算法流程说明从单轮 objective-mass F&O 更新为 conflict-aware 三轮动态邻域与列回注。

### `preexperiment/reports/v5/phase3/*`

- 修改：新增环境、算法配置、evaluation suite、ablation、stage-gate 机器可读文件和本报告。
- 原因：保留可追溯的 Phase 3 配置、输入、结果与负面阶段门结论。

### 数学模型变化检查

| 检查项 | 是否改变 | 证据 |
|---|---:|---|
| objective / 权重 | 否 | 五项目标及归一化权重原样保留 |
| integer feasible set / 业务约束 | 否 | 只改变启发式搜索和动态列开放顺序 |
| zone 定义 / 完整隐式 zone 集 | 否 | neighborhood 使用既有 `_ensure_complete_group_zones()` |
| peak utilization | 否 | 只读取 incumbent utilization 作为 conflict signal |
| Exact RMQ pricing | 否 | root generation 未修改 |
| Root LB 证明等级 | 否 | 24/48/96 LB 与 Phase 2 完全一致 |
| exact row recourse | 否 | fill/recourse 与双重验证未修改 |
| Complete MIP baseline | 否 | 隔离测试继续通过，比较结果复用已归档的相同输入运行 |

## 4. Tests

- passed: **39**
- failed: **0**
- skipped: **0**

执行结果：

```text
.venv/bin/python -m pytest -q
.......................................                                  [100%]
39 passed in 0.43s
```

关键 correctness：

- RMQ vs exhaustive: PASS，`test_rmq_pricing_matches_complete_enumeration`。
- root LP vs complete LP: PASS，`test_root_matches_complete_zone_lp`；三个 representative case 的 root LB 与 Phase 2 逐值一致。
- exact recourse: PASS，`test_exact_fill_is_independently_valid` 和 `test_complete_zone_mip_has_valid_exact_recourse`。
- conflict graph: PASS，`test_conflict_graph_prioritizes_shared_physical_resources`。
- multi-round later improvement: PASS，`test_multiround_rotates_seed_and_accepts_later_improvement`。
- Complete MIP isolation: PASS；Complete MIP 不调用 Phase 3 conflict graph。
- internal validation: PASS，24/48/96 全部通过。
- external validation: PASS，24/48/96 全部通过。
- `git diff --check`: PASS。

## 5. V4 → V5 分阶段结果

下表均为最终 UB，越小越好。`V5-full` 在此特指“完成本轮授权 Phase 3 后的版本”，不表示总体设计中尚未授权的后续方法已实现。Complete MIP 数值来自 Phase 1.1 的隔离运行；manifest、模型、时限和线程均与 Phase 3 配对一致。

| Case | V4 UB | V5-start UB | V5-pool UB | V5-full / Phase 3 UB | Complete MIP UB |
|---|---:|---:|---:|---:|---:|
| 24 / `pilot_l_301` | 0.135016 | 0.133714 | 0.138649 | **0.134315** | 0.131522 |
| 48 / `scale_g048_s601` | 0.139063 | 0.139310 | **0.142953** | 0.143515 | 0.142826 |
| 96 / `scale_g096_s801` | 0.166001 | 0.168837 | **0.170322** | 0.171137 | 0.171103 |

Phase 3 相对 Phase 2 的 UB 改善分别为 **+3.126%、-0.393%、-0.479%**。所以 24 箱组修复明显，但中大型没有复现 Phase 2 单轮 objective-mass neighborhood 的最好结果。

## 6. Proof / Primal Pool

`Reduction = (Proof pool - Primal pool) / Proof pool`。Proof pool 与 static primal pool 完全沿用 Phase 2；`Final master` 的增量是 Phase 3 三轮真正回注、去重后的新列。

| Case | Implicit zones | Proof pool | Primal pool | Final master | Reduction | Pool build (s) | Phase 3 new columns |
|---|---:|---:|---:|---:|---:|---:|---:|
| 24 | 40,054 | 1,065 | 994 | 1,017 | 6.67% | 0.120 | 23 |
| 48 | 79,892 | 4,487 | 2,111 | 2,136 | 52.95% | 0.242 | 25 |
| 96 | 159,907 | 19,801 | 4,620 | 4,693 | 76.67% | 0.492 | 73 |

### Static primal columns by origin

同一列可有多个 origin，因此横向合计可大于 pool size。

| Case | Root | Reduced cost | Capacity fit | Business efficiency | Spatial diversity | Greedy start |
|---|---:|---:|---:|---:|---:|---:|
| 24 | 127 | 378 | 475 | 487 | 219 | 90 |
| 48 | 278 | 773 | 991 | 997 | 414 | 278 |
| 96 | 627 | 1,190 | 1,751 | 1,787 | 805 | 887 |

### Final selected zones by origin

F&O round provenance 是非互斥标签：一列可以在多轮 local incumbent 中被使用；真正新增列数应以上表 `Phase 3 new columns` 为准。

| Case | Root | Reduced cost | Capacity fit | Business efficiency | Spatial diversity | Greedy start | R1 | R2 | R3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 24 | 19 | 24 | 7 | 6 | 5 | 6 | 39 | 44 | 59 |
| 48 | 19 | 21 | 23 | 21 | 5 | 26 | 76 | 82 | 97 |
| 96 | 86 | 30 | 21 | 21 | 13 | 105 | 204 | 219 | 265 |

Phase 2 的 mandatory start zones、overflow columns、overflow groups 分别仍为 24: `90/19/15`、48: `278/164/47`、96: `887/726/92`；本轮没有调整 `K_g`、五通道或 mandatory policy。

## 7. Multi-start

| Case | Generated | Deduplicated | Repaired | Feasible repaired | Submitted | Best start UB |
|---|---:|---:|---:|---:|---:|---:|
| 24 | 20 | 4 | 4 | 4 | 4 | 0.212761 |
| 48 | 6 | 6 | 6 | 6 | 6 | 0.228434 |
| 96 | 7 | 7 | 5 | 5 | 5 | 0.248737 |

这些数值和 Phase 2 相同；本轮没有调整 greedy candidates、repair、multi-start 数量或提交策略。Initial primal MIP 的 first incumbent 就是对应 best submitted start；最终 initial-MIP incumbent 在进入 F&O 前分别为 0.143844、0.146007、0.177338。

## 8. F&O

`Seconds` 包括 selection、完整 neighborhood zone build、local MIP 和 master reoptimization。`New` 是该轮实际注入 master 的新列数。

| Case | Round | Seed | Groups | Candidate zones | Before | After | Improvement | Seconds | New |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 24 | 1 | `390121_E007` | 3 | 4,022 | 0.143844 | 0.141857 | 0.001987 (1.38%) | 3.720 | 2 |
| 24 | 2 | `390121_E007` | 5 | 7,966 | 0.141857 | 0.140601 | 0.001256 (0.89%) | 3.718 | 8 |
| 24 | 3 | `390121_E007` | 10 | 13,286 | 0.140601 | 0.134315 | 0.006286 (4.47%) | 3.713 | 13 |
| 48 | 1 | `390121_E007` | 6 | 9,030 | 0.146007 | 0.143852 | 0.002154 (1.48%) | 7.310 | 12 |
| 48 | 2 | `390121_E007` | 11 | 17,430 | 0.143852 | 0.143847 | 0.000005 (0.004%) | 7.309 | 3 |
| 48 | 3 | `390121_E007` | 18 | 26,970 | 0.143847 | 0.143515 | 0.000332 (0.23%) | 7.297 | 10 |
| 96 | 1 | `390121_E007` | 11 | 18,958 | 0.177338 | 0.175764 | 0.001574 (0.89%) | 5.001 | 16 |
| 96 | 2 | `390121_E007` | 22 | 34,846 | 0.175764 | 0.174180 | 0.001584 (0.90%) | 4.990 | 22 |
| 96 | 3 | `390121_E007` | 35 | 55,930 | 0.174180 | 0.171137 | 0.003043 (1.75%) | 4.980 | 35 |

逐轮性质：

- 三个 case 都执行 3 轮、3 轮均严格改善，stop reason 都是 `max_rounds`；后续轮次确实继续贡献 UB，而非重复空跑。
- 由于每轮都改善 incumbent，按设计允许重新计算后复用同一 seed；所以三个 case 的 seed 都保持 `390121_E007`。失败 seed 轮换已由受控单元测试覆盖，但这些实际 case 没有触发。
- 24 箱组第三轮贡献了本 case 总 F&O 改善的 66.0%；96 箱组第三轮贡献 49.1%，说明递增邻域有实际作用。
- 48 箱组第二轮增益几乎为零，第三轮虽恢复改善，最终仍没有追上 Phase 2 单轮 objective neighborhood。

## 9. Final paired comparison

相对改善定义为 `(Complete MIP UB - Phase 3 UB) / Complete MIP UB`，正值表示 Phase 3 更好。Independent gap 分别使用各算法自己的 LB；common-bound gap 对两者采用较强的共同 LB。

| Case | Phase 3 UB | Complete UB | UB improvement | Phase 3 LB | Phase 3 gap | Complete LB | Complete gap | Common-bound gap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 24 | 0.134315 | 0.131522 | -2.124% | 0.115627 | 13.914% | 0.120199 | 8.609% | 10.510% |
| 48 | 0.143515 | 0.142826 | -0.482% | 0.118305 | 17.566% | 0.119838 | 16.095% | 16.498% |
| 96 | 0.171137 | 0.171103 | -0.020% | 0.139889 | 18.259% | 0.140273 | 18.018% | 18.035% |

### 16/24 groups

- 16 groups：未运行，不能给出数值或结论。
- 24 groups：Phase 3 比 Phase 2 pool 改善 3.126%，但仍比 Complete MIP 差 2.124%；independent gap 高 5.305 percentage points。单个 case 不足以判断“持续性”，但已运行结果没有达到 1% 容忍目标。

### 48+ groups

- case 数：2，仅 48/96 各一个 representative seed。
- median UB improvement: **-0.251%**。
- mean UB improvement: **-0.251%**。
- minimum / maximum: **-0.482% / -0.020%**。
- win / tie / loss rate: **0% / 0% / 100%**。
- Phase 3 mean/median independent gap: **17.913%**；Complete MIP mean gap: **17.057%**。
- independent gap difference: Phase 3 平均高 **0.856 percentage points**，没有实现低约 2 percentage points 的目标。
- mean time-to-best ratio `Phase3 / Complete`: **1.000**，总体近似相同。

三个 manifest SHA-1 与 Phase 2 及 Complete MIP 配对输入完全一致：

```text
24: 3f29840d61575f75ae0647dddd82e2e41c54e7f5
48: b04b34d1fd51e7267c3b22bc63fe3d455d876c8a
96: 7d20bab411ab58baf0ed1f1a439fa56c3d3e00ab
```

## 10. Anytime performance

时间均从整个算法入口开始计，不只计 Gurobi local MIP。

| Case | Phase 3 first (s) | MIP first (s) | Phase 3 best (s) | MIP best (s) | Root (s) |
|---|---:|---:|---:|---:|---:|
| 24 | 8.041 | 3.004 | 56.104 | 49.603 | 1.267 |
| 48 | 13.337 | 6.360 | 111.519 | 114.002 | 4.008 |
| 96 | 45.864 | 32.290 | 112.888 | 110.576 | 26.230 |

- time-to-first：Phase 3 三例都晚于 Complete MIP；Phase 3 仍需先完成 exact root、pool rebuild 和 repaired-start preparation，本轮没有修改这些前置阶段。
- time-to-best：Phase 3 只在 48 箱组更早 2.48s；24/96 分别晚 6.50s/2.31s。48+ 平均 ratio 为 0.9996，不能声称明显提前。
- Complete MIP 追平/超过 Phase 3 最终 UB：24 箱组约在全局 23.91s 已超过；48 箱组约在 114.00s 才超过 Phase 3 于 111.52s 找到的最终 UB；96 箱组约在 110.58s 已超过。
- root time 与 Phase 2 基本相同，root LB 完全相同；本轮 UB 差异来自 primal search，而不是下界被削弱。

## 11. 当前主要瓶颈

1. **Conflict score 过密，区分度不足。** 第一轮 conflict graph 的非零 pair 比例在 24/48/96 都是 100%；平均 `resource_conflict` 为 0.911/0.972/0.880，平均 `peak_exchange` 为 0.706/0.927/0.949。候选集合很宽时，大多数 group 都看起来冲突，排序容易被 same-voyage block 主导。
2. **中大型邻域没有覆盖 Phase 2 单轮的最佳组合。** Phase 3 的轮 3 已开放 18/35 个 group 和 26,970/55,930 个 zone，但 48/96 UB 仍比 Phase 2 高 0.39%/0.48%。这不是“未开放任何关键列”，而是固定时间内所选联合 group 组合和搜索路径不如旧 objective-mass neighborhood。
3. **同一 seed 连续改善但会形成局部聚焦。** 实际三例每轮都略有改善，因此合法复用 `390121_E007`；48 箱组第二轮仅改善 0.004%，说明“只要有任意改善就复用”可能在密集 conflict graph 下过于保守。不过本轮按总体设计实现，没有临时加入阈值或针对种子调参。
4. **Initial MIP 仍占主要 primal 时间。** 24/48/96 initial MIP 分别使用 42.58s、84.22s、64.67s；留给三轮 F&O 的总时间约 11.15s、21.92s、14.97s。本轮被要求冻结 `zone_mip_time_fraction=0.75`，因此没有实现 dynamic stopping。
5. **LB integrality gap 仍在，但不是本轮回归来源。** Phase 3 不改变 root，48/96 LB 与 Phase 2 相同；当前阶段首先失败于 UB 邻域质量，不能用 stabilization/cuts 解释或修复这次 UB regression。
6. **统计证据不足。** 只有三个单种子 representative smoke，不能评估稳定 win rate，也不能据此对正式 holdout 特调 conflict 权重或轮次预算。

## 12. 是否达到 Stage Gate

- correctness: **PASS**。39 tests 全通过，root LB 无 mismatch，三个解均通过独立双重验证，Complete MIP 保持隔离。
- 16/24: **FAIL / incomplete evidence**。16 未运行；24 比 Complete MIP 差 2.124%，超过 1% 容忍线。
- 48+ UB: **FAIL**。median improvement -0.251%、win rate 0%，远低于 median ≥3%、win rate ≥80%。
- independent gap: **FAIL**。48+ Phase 3 平均 gap 比 Complete MIP高 0.856 percentage points，而非低约 2 points。
- scalability: **PASS**。48/96 static primal pool 分别为 proof pool 的 47.05%/23.33%；动态回注后 final master 仍仅为 47.61%/23.70%。
- Phase 3 mechanism: **PASS（功能）/ FAIL（性能）**。三轮均产生有效新列和单调 UB 改善，但中大型没有稳定优于 Phase 2 单轮，更没有达到论文阶段门。

## 13. 下一步建议

依据当前结果，选择总体设计中的 **D：下一阶段评估 Local Branching**，但只能在 Phase 3 结果审阅并用额外 development seeds 确认现象后单独授权实现。

理由：当前主要问题是 incumbent 周围的联合变量交换没有被粗粒度 group conflict neighborhood 稳定表达；三轮 local MIP 一直有效，说明 primal 邻域搜索仍有潜力，而不是 root proof、目标函数或可行域错误。Local Branching 可以直接限制相对 incumbent 的组合变化距离，作为 conflict-aware F&O 的补充或替代消融。

本轮不建议直接选择 B/C/E：

- 暂不进入 Dual Stabilization 或 Valid Inequalities，因为本轮退化发生在 UB，root LB 与 Phase 2 完全一致。
- 暂不进入 Branch-and-Price，因为代表性实验尚未满足先做轻量 primal improvement 的阶段门，也没有证据表明必须引入完整树上定价。
- 不继续针对本组三个 seeds 调整 conflict 权重、Phase 2 static pool、multi-start、objective、pricing 或 time split；应先保留本阶段负面结果，等待下一阶段明确授权。

尚未运行的正式评估命令应在后续单独阶段建立 multi-seed development suite 与冻结 holdout 后再确定；本报告不虚构其数值。
