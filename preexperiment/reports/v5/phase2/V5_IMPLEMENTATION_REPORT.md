# V5 实施结果

## 1. 实施状态

- 已完成：Phase 2 的 `RootSnapshot`、Proof/Primal Master 物理分离、按 group 计算的列预算、五通道 diversified column pool、round-robin 合并、repaired-start mandatory columns、column provenance、LP warm start，以及相应 diagnostics、测试和 24/48/96 箱组 representative paired experiments。
- 未完成：Multi-round F&O、Conflict Graph、Dynamic Initial-MIP Stopping、Dual Stabilization、Valid Inequalities、Local Branching、Branch-and-Price；也未运行 16/72 箱组、多开发种子和正式 holdout。
- 未完成原因：这些内容不属于本轮 Phase 2 授权范围；本轮按要求在 Proof/Primal Pool + Diversified Columns 完成后停止。当前实验只是单种子 representative smoke，不构成正式论文统计结论。

## 2. Git 状态

- base commit: `dd8b55c2d482afbfa4edcc7138891a6c008c16f2`
- final implementation commit: `a3af4e8b37c949a715bb352dbee11edef2c61c05`
- branch: `feat/v5-phase2-proof-primal-pool`
- 说明：实现提交和本报告提交分开保存；报告提交哈希以仓库最终 `HEAD` 为准。

## 3. 代码修改

### `yard_planning/contiguous_zone_generation.py`

- 修改：新增 `RootSnapshot`，在 exact root closed 后冻结 root objective、dual、各类 LP 变量值、LP warm start 和 proof-zone indices；销毁 proof master 后重新构建只含 primal columns 的新 master。
- 修改：新增 group-specific 预算

  ```text
  K_g = min(integer_pool_columns_per_group,
            max(1, ceil(sqrt(possible_zone_count_by_group[g]))))
  ```

- 修改：从 `root_support`、`reduced_cost`、`capacity_fit`、`business_efficiency`、`spatial_diversity` 五个通道生成候选，并按固定顺序 deterministic round-robin 选列，没有增加新的加权评分。
- 修改：提交给 Gurobi 的 repaired starts 所需列全部标记为 `greedy_start` mandatory columns；超过 nominal `K_g` 时保留全部列并记录 overflow。
- 修改：保存 primal column 的多来源 provenance，并统计 primal-pool origins、最终解 selected origins、proof/primal 比例、pool build time 和 warm-start 状态。
- 修改：在新 primal master 上按变量名回填 root LP start；现有 Phase 1.1 repaired multiple MIP starts 和单轮 F&O 得到保留。
- 原因：root proof pool 的职责是证明 LB，integer primal pool 的职责是寻找 UB；两者需要物理分离，且整数池要保留容量匹配、业务效率和空间交换能力。

### `preexperiment/runner.py`

- 修改：将 proof/primal pool size、reduction、build time、mandatory overflow、origin counts、selected-origin counts 和 LP warm-start diagnostics 写入实验归档。
- 原因：让 pool 的规模、构成和实际被选中的列可审计，并为后续消融提供机器可读数据。

### `tests/test_contiguous_zone_generation.py`

- 修改：新增 proof pool 大于 primal pool 的 micro test、Complete MIP 隔离 test、diversified multi-origin 与 deterministic test、LP warm-start on/off 等价 test、mandatory overflow 不删 start columns test；更新算法版本断言。
- 原因：覆盖 Phase 2 指定的模型分离、正确性、确定性、warm-start 安全性和 mandatory-column 保留要求。

### `preexperiment/reports/v5/phase2/*`

- 修改：新增环境、算法配置、evaluation suite、ablation、stage-gate 机器可读文件及本报告。
- 原因：保证实验输入、配置、结果和结论可以复核。

### 数学模型变更检查

| 项目 | 是否改变 |
|---|---:|
| 目标函数及权重 | 否 |
| 可行域与业务约束 | 否 |
| zone 定义与完整隐式 zone 集 | 否 |
| peak-utilization 处理 | 否 |
| exact recourse | 否 |
| Complete MIP baseline 建模路径 | 否 |

本阶段改变的是 root 求证之后提交给整数 master 的列集合和求解流程，不是数学模型本身。

## 4. Tests

- passed: **37**
- failed: **0**
- skipped: **0**

执行结果：

```text
python -m pytest tests/test_contiguous_zone_generation.py -v  -> 19 passed
python -m pytest tests/test_complete_mip_baseline.py -v       -> 3 passed
python -m pytest -q                                           -> 37 passed
```

关键 correctness：

- RMQ vs exhaustive: PASS，既有 `test_rmq_pricing_matches_complete_enumeration` 继续通过。
- root LP vs complete LP: PASS，既有 `test_root_matches_complete_zone_lp` 继续通过；三个 representative case 的 root LB 与 Phase 1.1 完全一致。
- exact recourse: PASS，既有 exact-recourse regression 继续通过，模型与实现未修改。
- internal validation: PASS，24/48/96 三个 Phase 2 解全部通过。
- external validation: PASS，24/48/96 三个 Phase 2 解全部通过。
- Complete MIP isolation: PASS，spy test 证明 Complete MIP 不调用 Phase 2 primal-pool builder。
- LP warm start: PASS，on/off 得到相同 UB/LB。

## 5. V4 → V5 分阶段结果

下表均为最终 UB，越小越好。`V5-pool` 保留 Phase 1.1 已存在的 repaired multi-start 和单轮 F&O；`V5-full` 因不属于本阶段而未实现。Complete MIP 使用同一 materialized manifest、相同总时限和 `Threads=1` 的 Phase 1.1 隔离运行结果，未为了 Phase 2 重跑或修改 baseline。

| Case | V4 | V5-start | V5-pool | V5-full | Complete MIP |
|---|---:|---:|---:|---:|---:|
| 24 / `pilot_l_301` | 0.135016 | 0.133714 | 0.138649 | 未实现 | 0.131522 |
| 48 / `scale_g048_s601` | 0.139063 | 0.139310 | 0.142953 | 未实现 | 0.142826 |
| 96 / `scale_g096_s801` | 0.166001 | 0.168837 | 0.170322 | 未实现 | 0.171103 |

V5-pool 相对 V5-start 的 UB 分别恶化 3.69%、2.61%、0.88%；相对 V4，三个 case 均恶化超过 1%。因此 Phase 2 已达到结构实现目标，但当前 pool policy 存在明确的 algorithmic regression，不能据此宣称整体算法改善。

## 6. Proof / Primal Pool

`Reduction = (Proof pool - Primal pool) / Proof pool`。`Primal pool` 包含 mandatory repaired-start columns，`Final master` 还包括单轮 F&O 后新增的列。

| Case | Implicit zones | Proof pool | Primal pool | Final master | Reduction | Pool build (s) |
|---|---:|---:|---:|---:|---:|---:|
| 24 | 40,054 | 1,065 | 994 | 1,006 | 6.67% | 0.120 |
| 48 | 79,892 | 4,487 | 2,111 | 2,135 | 52.95% | 0.242 |
| 96 | 159,907 | 19,801 | 4,620 | 4,696 | 76.67% | 0.494 |

48/96 的 primal/proof 比例分别为 47.05% 和 23.33%，达到 `primal pool <= 50% proof pool` 的 scalability target。24 箱组只有 6.67% reduction，原因是 proof pool 本身较小，而 mandatory starts 占据较大比例。

### Primal pool columns by origin

同一列可有多个 origin，因此各行合计可以大于 primal-pool size。

| Case | Root | Reduced cost | Capacity fit | Business efficiency | Spatial diversity | Greedy start |
|---|---:|---:|---:|---:|---:|---:|
| 24 | 127 | 378 | 475 | 487 | 219 | 90 |
| 48 | 278 | 773 | 991 | 997 | 414 | 278 |
| 96 | 627 | 1,190 | 1,751 | 1,787 | 805 | 887 |

### Final selected zones by origin

| Case | Root | Reduced cost | Capacity fit | Business efficiency | Spatial diversity | Greedy start | Single-round F&O |
|---|---:|---:|---:|---:|---:|---:|---:|
| 24 | 18 | 24 | 8 | 8 | 4 | 7 | 12 |
| 48 | 19 | 19 | 20 | 19 | 7 | 29 | 24 |
| 96 | 84 | 34 | 23 | 24 | 15 | 101 | 74 |

### Mandatory overflow

| Case | Mandatory start zones | Overflow columns | Overflow groups |
|---|---:|---:|---:|
| 24 | 90 | 19 | 15 |
| 48 | 278 | 164 | 47 |
| 96 | 887 | 726 | 92 |

所有已提交 feasible repaired starts 所需列均被保留，没有为了满足 nominal `K_g` 删除 start support。

## 7. Multi-start

| Case | Generated | Feasible repaired | Submitted | Best start UB |
|---|---:|---:|---:|---:|
| 24 | 20 | 4 | 4 | 0.212761 |
| 48 | 6 | 6 | 6 | 0.228434 |
| 96 | 7 | 5 | 5 | 0.248737 |

这里沿用并验证 Phase 1.1 的 repaired multiple MIP starts；Phase 2 的新增职责是确保所有 submitted starts 的支持列进入重建后的 primal master。

## 8. F&O

| Case | Round | Groups | Candidate zones | Before | After | Improvement | Seconds |
|---|---:|---:|---:|---:|---:|---:|---:|
| 24 | 1 | 7 | 8,002 | 0.143844 | 0.138649 | 0.005195 (3.61%) | 11.203 |
| 48 | 1 | 16 | 27,370 | 0.146007 | 0.142953 | 0.003053 (2.09%) | 22.070 |
| 96 | 1 | 30 | 47,830 | 0.177338 | 0.170322 | 0.007016 (3.96%) | 15.543 |

这只是 Phase 1.1 已有的单轮 F&O。未实现 multi-round、conflict graph 或自适应 neighborhood；表中不能被解读为后续阶段结果。

## 9. Final paired comparison

相对改善统一定义为 `(Complete MIP UB - V5-pool UB) / Complete MIP UB`，正值表示 V5-pool 更好。

### 16/24 groups

- 16 groups：未运行，不能给出数值或结论。
- 24 groups：V5-pool UB 为 0.138649，Complete MIP UB 为 0.131522，相对改善为 **-5.42%**；独立 gap 分别为 16.60% 和 8.61%。单个 24-group representative case 明显未通过“小规模不持续劣于 1%”目标。

### 48+ groups

- case 数：2（48/96 各一个 representative seed；样本过少，不是正式统计）。
- median UB improvement: **0.184%**
- mean UB improvement: **0.184%**
- win rate: **50%**
- independent gap difference: V5-pool 平均 gap 比 Complete MIP **高 0.498 percentage points**，即更差。
- 48 groups：-0.089%，近似持平但略输。
- 96 groups：+0.457%，小幅胜出。

用于配对的 materialized manifests 与 Phase 1.1 完全一致，SHA-1 分别为：

```text
24: 3f29840d61575f75ae0647dddd82e2e41c54e7f5
48: b04b34d1fd51e7267c3b22bc63fe3d455d876c8a
96: 7d20bab411ab58baf0ed1f1a439fa56c3d3e00ab
```

尚未运行、因此不得虚构结果的正式实验：16-group representative、72-group representative、multi-seed development suite、`paper_eval_suite_v5` holdout。

## 10. Anytime performance

| Case | V5 first (s) | MIP first (s) | V5 best (s) | MIP best (s) |
|---|---:|---:|---:|---:|
| 24 | 8.002 | 3.004 | 48.076 | 49.603 |
| 48 | 13.366 | 6.360 | 101.410 | 114.002 |
| 96 | 45.878 | 32.290 | 113.051 | 110.576 |

- time-to-first：V5-pool 在三个 case 都更慢，主要受 exact root proof 和 primal-master 重建之前不能产生整数解影响。
- time-to-best：24/48 更快，96 略慢；48+ 的平均 `V5 / Complete MIP` time-to-best ratio 为 0.956。
- root time：24/48/96 分别为 1.250s、4.030s、26.278s；96 的 root 占 120s 总预算约 21.9%，尚未触及总设计中 25% 的高占比信号，但已明显压缩整数阶段时间。
- LP warm start on/off correctness test 得到相同 UB/LB，但这三个 representative case 不足以证明其稳定的性能收益。

## 11. 当前主要瓶颈

1. **列池缩小成功，但 UB 退化。** 48/96 的 pool reduction 已达目标，然而三个 case 的 V5-pool UB 都比 V5-start 差；说明当前 round-robin 的 nominal group budget 与 mandatory overflow 虽控制了规模，却没有稳定保留足够的整数协同列。
2. **单列 provenance 多样，不等于组合多样。** 五个通道都进入 pool，但最终被选中的 `capacity_fit`、`business_efficiency`、`spatial_diversity` 列数量仍较少；现有 channel ranking 是逐 group 的，不能显式发现多个 group 之间的共享资源冲突和可交换组合。
3. **Mandatory overflow 很普遍。** 48/96 分别有 47/92 个 group overflow，新增 164/726 列。这保证了 start 可行性，但削弱了 `K_g` 的实际控制力，也说明 start 支持与 diversified base pool 的重合仍不足。
4. **初始整数解等待 root。** time-to-first 在三个 case 都落后 Complete MIP；Proof/Primal 分离减少了整数变量数，却没有消除 exact root 的串行前置成本。
5. **一次 F&O 有价值但不充分。** 单轮 F&O 在三个 case 均改善 2.09%–3.96%，且最终解有 12/24/74 个 F&O-origin selected zones；这表明主要剩余机会可能来自跨 group 的联合换列，而不是继续单纯扩大每组静态 pool。
6. **证据量不足。** 当前只有 24/48/96 各一个 seed；不能判断退化是否稳定，也不能据此调整面向 holdout 的参数。

## 12. 是否达到 Stage Gate

- correctness: **PASS**。root LB 无 mismatch，全部内部/外部验证通过，Complete MIP 保持隔离。
- 16/24: **FAIL / incomplete evidence**。16 未运行；已运行的 24 箱组 V5-pool 比 Complete MIP 差 5.42%，超过 1% 容忍目标。
- 48+: **FAIL**。代表性两例 median/mean UB improvement 均只有 0.184%，win rate 50%，未达到 median ≥3%、win rate ≥80%。
- gap: **FAIL**。48+ 的 V5-pool 平均 independent gap 反而高 0.498 percentage points，未达到低约 2 percentage points 的目标。
- scalability: **PASS**。48/96 primal pool 分别为 proof pool 的 47.05%/23.33%，均不高于 50%。
- algorithmic regression check: **触发**。相对 V4，24/48/96 的 UB 均恶化超过 1%，因此不应在未定位原因前盲目扩展或宣称 V5 改善。

## 13. 下一步建议

后续 oracle/coverage audit 已确认：Phase 2 final master 只覆盖三个已知更优 oracle support 的 49.12%、40.19%、47.37%；只补回缺失的 29、64、140 个 signatures 后，当前同一数学模型精确恢复 oracle objective，误差不超过 `2.23e-16`。详细证据见 `ORACLE_COVERAGE_AUDIT.md`。

因此本阶段结论更新为：**Phase 2 的结构、正确性和 pool scalability 已完成，但当前 diversified pool 会裁掉已知更优整数组合所需的关键 support，足以解释 UB regression，Stage Gate 未通过。Phase 2 到此关闭。**

- 当前不进入 conflict-aware multi-round F&O；
- 当前不继续调整 `K_g`、channel 顺序或 pool 参数；
- 不实施 Dual Stabilization、Valid Inequalities 或 Branch-and-Price；
- 保留当前实现和负面审计结果，等待是否另行授权重新设计 primal-pool policy。
