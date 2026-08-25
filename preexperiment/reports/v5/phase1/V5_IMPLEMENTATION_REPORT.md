# V5 实施结果

## 1. 实施状态

- 已完成：第一阶段 Diagnostics、Greedy Candidate early-return 修复、联合可行性 repair、Repaired Multiple MIP Starts，以及 24/48/96 箱组代表性配对实验。
- 未完成：Proof/Primal Pool Separation、Integrality-aware Primal Pool、Multi-round F&O、Dual Stabilization、Valid Inequalities、Branch-and-Price 和 Local Branching。
- 未完成原因：这些内容属于后续阶段，本轮按用户要求明确禁止提前实现。

模型口径保持为 `integrated_zone_v4`；算法实现单独标识为 `integrated_zone_v5_start_phase1`。目标项、权重、归一化尺度、峰值 headroom、排区容量规则、精确定价、根节点 LB 和 exact row recourse 均未改动。

## 2. Git 状态

- base commit: `c97d7c3615921d438ac2a5305609c7085486eb21`
- final commit: 以本报告提交后的 `git log -1` 和最终回复为准
- branch: `feat/v5-integrality-aware-primal-search`

## 3. 代码修改

### `yard_planning/gurobi_backend.py`

- 修改：增加 MIP progress recorder、callback-aware `optimize()`、`terminate()`、multiple-start facade，以及 runtime/node 读取。
- 原因：统一记录首解、改进解、有效 bound 变化和最终状态，并真实提交多个 start；不声称 Gurobi 接受了某个 start。

### `yard_planning/contiguous_zone_generation.py`

- 修改：增加独立算法版本；贪心阶段完整遍历 4 种 ordering × 5 种 import protection；按 zone support 去重并排序；增加等价 zone 缓存和既有 peak cap 检查。
- 修改：将 objective reconstruction 从 solver certificate 中解耦。
- 修改：固定候选 zone support 后，联合求解出口流、匿名进口、激活变量和峰值约束；只把严格可行 repair 作为 MIP start，最多 6 个，repair 总预算不超过 8%。
- 修改：为 initial MIP、单轮 F&O local MIP、最终 primal-master reoptimization 和 complete-zone MIP 记录 anytime 轨迹。
- 原因：修复原先“首个完整候选立即返回”和“未经认证的 partial start 交给主 MIP 静默修复”两个问题。

### `preexperiment/runner.py`

- 修改：输出算法版本、Proof/Primal 当前池规模、多 Start 数量、candidate/repair 明细、首解/最好解时间、F&O 贡献和 selected-zone 来源统计。
- 原因：使实验结果能够定位 UB、gap 和首解速度的瓶颈。

### `preexperiment/complete_mip_baseline.py`

- 修改：保存 complete-zone MIP 的算法版本、MIP progress、anytime 轨迹和 start diagnostics。
- 原因：支持相同环境、相同预算下的配对 anytime 比较。

### `tests/test_contiguous_zone_generation.py`

- 修改：新增 greedy 全策略遍历、联合不可行 reject、2+ repaired starts 提交、最终 incumbent 不劣于最佳 start，以及 diagnostics/progress 回归测试。
- 原因：覆盖第一阶段的关键 correctness gate。

## 4. Tests

- passed: 31
- failed: 0
- skipped: 0

关键 correctness：

- RMQ vs exhaustive: passed
- root LP vs complete LP: passed
- exact recourse: passed
- internal validation: 24/48/96 与三项 Complete MIP 全部 passed
- external validation: 24/48/96 与三项 Complete MIP 全部 passed

实际执行：

```text
python -m pytest tests/test_contiguous_zone_generation.py -v  -> 13 passed
python -m pytest tests/test_complete_mip_baseline.py -v        -> 3 passed
python -m pytest -q                                            -> 31 passed
```

## 5. V4 → V5 分阶段结果

下表为 UB（越小越好）。`V5-pool` 和 `V5-full` 本轮未实现、未运行。

| Case | V4 | V5-start | V5-pool | V5-full | Complete MIP |
|---|---:|---:|---:|---:|---:|
| 24 / `pilot_l_301` | 0.135016 | 0.133714 | 未实施 | 未实施 | 0.131690 |
| 48 / `scale_g048_s601` | 0.139063 | 0.138250 | 未实施 | 未实施 | 0.140395 |
| 96 / `scale_g096_s801` | 0.166001 | 0.169993 | 未实施 | 未实施 | 0.179106 |

V5-start 相对 V4 的 UB 改善依次为 `+0.96%`、`+0.58%`、`-2.40%`。独立 root gap 依次由 `14.36% → 13.53%`、`14.93% → 14.43%`、`15.73% → 17.71%`。因此第一阶段在 24/48 箱组略有改善，但 96 箱组出现明确回退。

## 6. Proof / Primal Pool

本轮没有实现 Proof/Primal Pool Separation；`Primal pool` 只是现有 root pool、legacy enrichment、repaired-start mandatory zones 和单轮 F&O 新增 zones 的当前 active pool。两者不可被解读为已经分离的 V5 pool。

| Case | Implicit zones | Proof pool | Primal pool | Reduction |
|---|---:|---:|---:|---:|
| 24 | 40,054 | 1,065 | 2,074 | 94.82% |
| 48 | 79,892 | 4,487 | 6,528 | 91.83% |
| 96 | 159,907 | 19,801 | 23,850 | 85.09% |

当前 Primal pool 均大于 Proof pool，96 箱组约为 Proof pool 的 120.4%。这不是后续 primal-pool stage gate 的实现结果。

## 7. Multi-start

| Case | Generated | Feasible repaired | Submitted | Best start UB |
|---|---:|---:|---:|---:|
| 24 | 20 | 4 | 4 | 0.212761 |
| 48 | 20 | 8 | 6 | 0.227516 |
| 96 | 20 | 8 | 6 | 0.248324 |

所有 submitted start 都经过固定 zone support 下的联合 MIP repair；主 MIP 最终 UB 均不劣于最佳 repaired start。未使用“Gurobi accepted”表述，因为接口没有提供严格的逐 start 接受证明。

候选生成耗时分别为 `4.33 s`、`9.23 s`、`21.72 s`；repair 实际耗时分别为 `0.72 s`、`2.41 s`、`4.48 s`，均低于对应 8% 总预算上限。

## 8. F&O

本轮只保留并诊断 V4 的单轮 F&O，没有实现 Multi-round F&O。

| Case | Round | Groups | Zones | Before | After | Improvement | Seconds |
|---|---:|---:|---:|---:|---:|---:|---:|
| 24 | 1 | 7 | 9,583 | 0.139403 | 0.133714 | 0.005689 | 11.27 |
| 48 | 1 | 15 | 24,556 | 0.143471 | 0.138250 | 0.005221 | 22.17 |
| 96 | 1 | 30 | 47,432 | 0.176515 | 0.169993 | 0.006522 | 15.73 |

三项均有改进，但 96 箱组的 initial incumbent 已明显弱于 V4，单轮 F&O 无法完全追回。

## 9. Final paired comparison

### 16/24 groups

- 本轮按指定范围只运行 24 groups，没有运行 16 groups。
- 24 groups：V5-start UB 比 Complete MIP 差 `1.54%`；用 Complete MIP LB 计算时，V5 gap 为 `10.14%`，Complete MIP gap 为 `8.76%`。
- 结论：单个 24-group 代表 case 尚未达到“不持续出现 >1% 劣势”的目标；是否“持续”仍需后续多 seed 验证。

### 48+ groups

- median UB improvement: `3.31%`（仅 48、96 两个代表 case，双样本 median 等于 mean）
- mean UB improvement: `3.31%`
- win rate: `100%`（2/2，相对本轮 Complete MIP）
- independent gap difference: 使用同一 Complete MIP LB 后，V5 平均低 `2.76 percentage points`

这只是 development smoke test，不是 holdout 结论。尤其 96 箱组 V5-start 相对 V4 退化 2.40%，不能据 paired MIP 优势宣称 V5 已整体改进。

## 10. Anytime performance

| Case | V5 time-to-first | MIP time-to-first | V5 time-to-best | MIP time-to-best |
|---|---:|---:|---:|---:|
| 24 | 8.00 s | 7.64 s | 55.14 s | 48.95 s |
| 48 | 19.05 s | 15.70 s | 105.87 s | 111.91 s |
| 96 | 59.34 s | 37.43 s | 111.15 s | 110.94 s |

V5-start 在三项上都没有更快得到首解；96 箱组慢约 21.9 秒。48 箱组更早达到最终最好解，但 24、96 箱组没有 anytime 速度优势。

## 11. 当前主要瓶颈

1. Greedy candidates 虽能产生严格可行 starts，但 repaired start UB 很弱，和最终 UB 相差较大。
2. 候选生成随规模增长明显：4.33 s → 9.23 s → 21.72 s，直接推迟主 MIP 首解。
3. 96 箱组 initial MIP 只探索到很少节点，首解直到 59.34 s；多 Start 的质量不足以抵消时间开销。
4. 当前 primal active pool 仍由 proof pool 和 legacy RC enrichment 主导，没有做到 Proof/Primal 分离，整数结构仍然偏大。
5. 单轮 F&O 能稳定下降 UB，但无法修复 96 箱组相对 V4 的整体退化。

## 12. 是否达到 Stage Gate

- 16/24: 未达到；24-group 对 Complete MIP 仍有 1.54% UB 劣势，且尚无 16-group 数据。
- 48+: 代表性双样本的 UB median 3.31%、win rate 100%，数值上达到 development target，但样本量不足，不能作为正式结论。
- gap: 48+ 使用 Complete MIP LB 后平均改善 2.76 percentage points，代表性结果达到目标；V5 自身 closed-root gap 在 96 箱组却比 V4 差 1.98 points。
- scalability: 未达到；96 箱组相对 V4 UB 退化 2.40%，候选生成和首解延迟仍明显。

## 13. 下一步建议

先审阅第一阶段结果，不继续自动实现。若批准进入下一阶段，应按 V5 设计单独实施 Proof/Primal Pool Separation 与 Integrality-aware Primal Pool，并重点验证：

1. mandatory repaired-start zones 能否保留，同时让 72/96 groups 的 primal pool 明显小于 proof pool；
2. 更小的 primal master 能否把 96-group time-to-first 从 59.34 s 明显提前；
3. 在不改变模型、目标和 root LB 的前提下，能否消除 V5-start 相对 V4 的 96-group UB 回退。

本轮在此停止，不实现任何后续阶段功能。
