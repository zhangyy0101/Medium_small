# Phase 1.1 实施结果

## 1. Git

- Phase-1 base commit: `0a3b02ade7f9130384e4429e7f88444cd3612ff3`
- Phase-1.1 final commit: `dfb68df0d4b43d8f1536496f47273a3a755d2cfe`（代码与测试；本报告随后单独提交）
- branch: `feat/v5-phase1-1-budget-baseline-fix`

本轮没有实现 Proof/Primal Pool Separation、Integrality-aware Primal Pool、Multi-round F&O、Dual Stabilization、Valid Inequalities、Local Branching 或 Branch-and-Price，也没有修改目标函数、权重、峰值 headroom、排区候选规则、RMQ 定价和 exact row recourse。

## 2. Baseline 调用链审计

### V5 solve()

实际调用链：

```text
solve()
→ exact root generation
→ restricted primal-pool enrichment
→ _integerize_zone_master(start_policy="v5_multi_start")
→ shared 10% start deadline
→ round-robin lazy candidate generation
→ joint fixed-support repair
→ repaired multiple MIP starts
→ initial restricted MIP
→ single-round F&O
→ exact row recourse
```

### Complete MIP solve_complete_zone_mip()

实际调用链：

```text
solve_complete_zone_mip()
→ materialize all zones
→ complete LP initialization
→ _integerize_zone_master(start_policy="complete_mip_baseline")
→ V4 single partial native Gurobi-repair start
→ fully enumerated same-zone MIP
→ exact row recourse
```

Phase 1 Complete MIP 是否执行 V5 expensive multi-start：

- yes
- 证据：Phase 1 的 `solve()`、`_solve_complete_zone_mip()`、`analyze_complete_zone_mip()` 和 `solve_complete_zone_mip()` 都无条件调用同一个 `_integerize_zone_master()`；该函数无条件调用 Phase 1 `_greedy_zone_mip_start()`。三个归档 baseline 分别记录了 20 个候选、联合 repair 和 4/4/6 个 submitted starts。
- Phase 1.1 证据：三项隔离 baseline 均记录 `integer_search_policy=complete_mip_baseline`、`v5_multi_start_enabled=false`、`candidate_generation_executed=false`、`repair_executed=false`；spy 测试确认 V5 candidate-generation 调用次数为 0，MIP progress 仍存在。

## 3. Complete MIP baseline 数值变化解释

| Case | V4 historical | Phase 1 | Phase 1.1 isolated | 原因 |
|---|---:|---:|---:|---|
| `pilot_l_301` | 0.130387 | 0.131690 | 0.131522 | Phase 1 加入 20 候选、4 次 repair 和 4 starts，改变了初始化及可用于主 MIP 的时间；Phase 1.1 恢复单个 V4 native start。 |
| `scale_g048_s601` | 0.145105 | 0.140395 | 0.142826 | Phase 1 的 20 候选、4 次 repair 和 4 starts 在该例上偶然改善 incumbent，因此污染后的 baseline 显著偏好；隔离后回到更接近历史 V4 的区间。 |
| `scale_g096_s801` | 0.167890 | 0.179106 | 0.171103 | Phase 1 baseline 的候选生成和 repair 共占 23.22 s，并提交 6 starts，压缩了完整 MIP 搜索；隔离后仅 0.215 s 的 V4 native-start 准备，UB 明显恢复。 |

核对结论：

- 三项 V4、Phase 1、Phase 1.1 的 generation manifest SHA-1 分别逐案完全一致：`3f29840d...`、`b04b34d...`、`7d20bab...`。
- scenario seed 为 301/601/801；`Threads=1`，`PYTHONHASHSEED=0`；总预算为 60/120/120 s；exact recourse reserve 均为 5%。
- 模型 schema 均为 `integrated_zone_v4`；目标的五项权重、自然尺度、峰值 epsilon cap 和 objective certificate 逐案一致。
- V5 与 Complete MIP 的 zone master 使用相同 Gurobi 参数。`ColumnGenerationConfig` 虽接收 scenario seed，但 zone master 历史路径没有显式设置 Gurobi `Seed`，所以三个版本实际都使用 Gurobi 默认固定 seed 0；该事实不影响配对公平性，但 scenario seed 与 solver seed 必须区分。
- Phase 1.1 没有完全复现 V4 historical UB 的剩余原因已定位：V4 historical 主 MIP 没有 Python `MipProgressRecorder` callback；Phase 1 起为满足 anytime trajectory 诊断加入 callback，Phase 1.1 按要求保留。固定 wall-clock 下 callback 改变可完成的 solver 工作量，但没有改变模型、目标或可行域。

结论：Phase 1 Complete MIP 的数值变化主要来自错误继承 expensive V5 multi-start；Phase 1.1 已隔离该路径。V4 historical 与 Phase 1.1 的剩余差异来自可验证的 instrumentation path 差异，而不是输入、目标、预算或线程不一致。

## 4. Start Budget 实现

- total fraction: `mip_start_total_time_fraction = 0.10`
- hard deadline implementation: `start_deadline = min(integer-search deadline, start_phase_begin + total_limit × 0.10)`；candidate 的 group/while/zone enumeration、repair、zone 注入、start materialization 和 Gurobi submission 都检查同一 deadline。
- repair sub-budget: `min(0.08 × total_limit, remaining start budget)`，包含在 10% 外层预算内，不与其相加。
- lazy generation: 沿 Phase 1 已有 4 orderings × 5 protections，以 protection round-robin 覆盖 ordering family；每个 candidate 完成后立即去重、评估、repair，并立即保留认证 start。
- interruption behavior: 中途超时记录 `budget_exhausted`，计入 `candidate_generation_interrupted_count` 和 `budget_interrupted_candidate_count`，不计入 `infeasible_candidate_count`。已认证 start 在后续候选超时时不会丢失；0–5 个 start 均允许进入主 MIP。
- termination reasons: `max_starts_reached`、`budget_exhausted`、`all_strategies_exhausted`、`no_feasible_start`，并保留已有 repair-candidate 子上限对应的明确原因。

Phase 1.1 V5 结果：

| Case | Start budget | Candidate gen | Repair | Starts | Time-to-first | Initial UB | Final UB |
|---|---:|---:|---:|---:|---:|---:|---:|
| `pilot_l_301` | 6.000 | 4.012 | 0.794 | 4 | 7.763 | 0.139403 | 0.133714 |
| `scale_g048_s601` | 12.000 | 3.405 | 1.873 | 6 | 12.761 | 0.142126 | 0.139310 |
| `scale_g096_s801` | 12.000 | 8.641 | 2.792 | 5 | 43.794 | 0.178805 | 0.168837 |

96 箱组的 start preparation 实际为 12.064 s；超出名义 12 s 的 64 ms 是当前 candidate 中断和 diagnostics 清理浮动，不再出现 Phase 1 的 26.19 s。5 个已认证 starts 被保留并提交。

## 5. Tests

- passed: 34
- failed: 0
- skipped: 0

关键 correctness：

- RMQ vs exhaustive: passed
- root LP vs complete LP: passed
- exact recourse: passed
- internal validation: 24/48/96 V5 与三项 isolated Complete MIP 全部 passed
- external validation: 24/48/96 V5 与三项 isolated Complete MIP 全部 passed
- baseline isolation: passed；Complete MIP 的 V5 candidate-generation spy call count = 0
- start-budget deadline: passed；使用 fake clock 验证 candidate 与 repair 共享 deadline
- budget interruption semantics: passed；interrupted 不记为 infeasible
- first-feasible early-return regression: passed；首个 repair feasible 后仍尝试第二个 structural family
- zero-start fallback: passed；0 starts 时主 MIP 正常求解并通过独立验证

实际执行：

```text
.venv/bin/python -m pytest tests/test_contiguous_zone_generation.py -v  -> 16 passed
.venv/bin/python -m pytest tests/test_complete_mip_baseline.py -v        -> 3 passed
.venv/bin/python -m pytest -q                                            -> 34 passed
```

## 6. Phase 1 → Phase 1.1

| Case | Metric | Phase 1 | Phase 1.1 | Change |
|---|---|---:|---:|---:|
| 24 | candidate generation seconds | 4.335 | 4.012 | -0.323 |
| 24 | repair seconds | 0.717 | 0.794 | +0.077 |
| 24 | total start-preparation seconds | 5.052 | 4.871 | -0.180 |
| 24 | submitted starts | 4 | 4 | 0 |
| 24 | time-to-first | 8.004 | 7.763 | -0.241 |
| 24 | time-to-best | 55.135 | 54.937 | -0.198 |
| 24 | initial MIP UB | 0.139403 | 0.139403 | 0.000000 |
| 24 | final UB | 0.133714 | 0.133714 | 0.000000 |
| 48 | candidate generation seconds | 9.225 | 3.405 | -5.821 |
| 48 | repair seconds | 2.411 | 1.873 | -0.539 |
| 48 | total start-preparation seconds | 11.637 | 5.434 | -6.202 |
| 48 | submitted starts | 6 | 6 | 0 |
| 48 | time-to-first | 19.046 | 12.761 | -6.285 |
| 48 | time-to-best | 105.871 | 112.852 | +6.981 |
| 48 | initial MIP UB | 0.143471 | 0.142126 | -0.001345 |
| 48 | final UB | 0.138250 | 0.139310 | +0.001060 |
| 96 | candidate generation seconds | 21.717 | 8.641 | -13.076 |
| 96 | repair seconds | 4.476 | 2.792 | -1.684 |
| 96 | total start-preparation seconds | 26.193 | 12.064 | -14.129 |
| 96 | submitted starts | 6 | 5 | -1 |
| 96 | time-to-first | 59.338 | 43.794 | -15.544 |
| 96 | time-to-best | 111.154 | 106.044 | -5.110 |
| 96 | initial MIP UB | 0.176515 | 0.178805 | +0.002291 |
| 96 | final UB | 0.169993 | 0.168837 | -0.001156 |

UB 变化不是 Phase 1.1 的通过条件。本轮没有据 48 箱组回退或 96 箱组改善继续调整 ordering、protection、repair、ranking、candidate 数或 start 数。

## 7. 96-group 时间分解

- root: 25.044 s
- candidate generation: 8.641 s
- repair: 2.792 s
- primal model setup: 7.293 s，其中 integerization 前的模型准备/池 enrichment 为 6.662 s，认证 start 的 zone 注入、materialization、submission 与其他 start work 为 0.631 s
- first-incumbent search: 0.025 s（从 start preparation 结束到 submitted start 形成首个 incumbent 的残差）
- initial MIP: 66.230 s，包含 12.064 s start preparation；扣除 start preparation 后的 presolve/主搜索约 54.167 s
- F&O: 16.067 s
- recourse: 1.206 s

前五项相加对应全局 time-to-first 43.794 s。F&O 与 recourse 根据各 MIP progress 的 global offset、wall time、总 solve time 和阶段计时交叉重建。

## 8. Phase 1.1 是否通过

- baseline isolation: PASS
- baseline explanation: PASS
- start hard cap: PASS（96 箱组 12.064 s，含 64 ms 中断/清理浮动）
- correctness: PASS
- diagnostics: PASS

## 9. 是否进入 Phase 2

- YES

Phase 2 应首先验证 Proof / Primal Pool Separation 与 Integrality-aware Primal Pool 能否在不改变模型、目标、可行域和 closed-root LB 的条件下，显著缩小 72/96 箱组 primal master，并改善 time-to-first、UB 和 gap。Phase 1.1 到此停止。
