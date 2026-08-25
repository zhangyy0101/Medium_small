# Phase 2 Oracle / Coverage Audit

## 结论

**已确认：Phase 2 的 UB 退化确实与关键整数解组合列被裁掉有关。**

这不是仅凭“列池变小、UB 变差”作出的相关性判断。审计捕获了修改前独立复现出的三个已知更优可行解的真实 zone signatures，逐项检查它们在 Phase 2 各阶段列池中的覆盖，并只补回缺失 signatures 做固定决策可行性证书。三个 addback 模型均求得 `optimal`，且精确恢复 oracle 目标值。

按照本轮要求，Phase 2 到此结束：不实施 conflict-aware multi-round F&O，也不继续调整 pool 参数。

## 1. 审计问题

需要区分两个解释：

1. Phase 2 仍包含足够列，只是 Gurobi 在限时内搜索波动，未找到原来的优质解；
2. Phase 2 实际删掉了已知更优解需要的 zone support，因此受限 master 根本不能表达这些组合。

本审计检验第二种解释。

## 2. Oracle 定义

原 Phase 1.1 归档只保存了 UB 和最终行分配，没有保存最终 zone signatures，无法事后从行流量唯一反推出完整预留 zone。因此使用 base commit `dd8b55c2d482afbfa4edcc7138891a6c008c16f2`，在相同 materialized input、seed、时限、`Threads=1`、`PYTHONHASHSEED=0` 下独立复现，并在不改变求解逻辑的子类钩子中捕获最终 selected zone signatures。

| Case | 原归档 Phase 1.1 UB | 本次 oracle UB | Phase 2 audit UB | Phase 2 相对 oracle 更差 |
|---|---:|---:|---:|---:|
| 24 | 0.133714 | 0.137151 | 0.138649 | 1.09% |
| 48 | 0.139310 | 0.139289 | 0.141824 | 1.82% |
| 96 | 0.168837 | 0.169681 | 0.170322 | 0.38% |

24/96 的限时复现没有重现归档中的最优 incumbent，因此本审计没有把更好的归档 UB 冒充为已捕获 support；只使用本次实际捕获、实际验证且仍优于 Phase 2 的解。三个 oracle 的 root LB 均与 Phase 2 完全相同，数学模型、目标、输入和 exact recourse 未改变。

一个 zone 的一致性键为：

```text
(group_id, area_no, row_no, candidate_indices)
```

## 3. Coverage 结果

| Case | Oracle zones | Base covered | Mandatory 后 | Final master | Final coverage | 完整覆盖箱组 |
|---|---:|---:|---:|---:|---:|---:|
| 24 | 57 | 27 | 28 | 28 | 49.12% | 9 / 24 |
| 48 | 107 | 35 | 42 | 43 | 40.19% | 10 / 48 |
| 96 | 266 | 69 | 125 | 126 | 47.37% | 33 / 96 |

这里的 `Final master` 已包括 repaired-start mandatory columns 和 Phase 1.1 已有的单轮 F&O 补列。即使到最终 master，仍有：

| Case | 缺失 oracle zones | 涉及箱组 | 完全零覆盖箱组 | 缺失箱组在 F&O 外 |
|---|---:|---:|---:|---:|
| 24 | 29 | 15 | 7 | 8 |
| 48 | 64 | 38 | 17 | 25 |
| 96 | 140 | 63 | 15 | 39 |

mandatory starts 只额外补回了 1、7、56 个 oracle zones；单轮 F&O 在此基础上只额外补回 0、1、1 个 oracle zones。Phase 2 最终解与 oracle 最终解的 zone overlap 也只有 7/57、10/107、49/266。

这说明当前问题不是“五种 channel 是否都出现”，而是它们没有保留已知优质解所需的整套跨箱组 support。大量 oracle 箱组只覆盖一部分甚至完全没有对应 zone，逐组 round-robin 的来源多样性不能保证组合覆盖。

## 4. Oracle addback 证书

对每个 case：

1. 使用本次捕获的 Phase 2 final master；
2. 只加入其中缺失的 oracle signatures；
3. 固定 oracle 的 zone、export-flow 和 anonymous-import decisions；
4. 在当前 Phase 2 代码的同一 master 上求解到最优。

| Case | 补回列数 | 合并后 master | 状态 | Oracle objective | Addback objective | 绝对误差 |
|---|---:|---:|---|---:|---:|---:|
| 24 | 29 | 1,035 | optimal | 0.137150612516 | 0.137150612516 | 5.55e-17 |
| 48 | 64 | 2,209 | optimal | 0.139289217428 | 0.139289217428 | 5.55e-17 |
| 96 | 140 | 4,836 | optimal | 0.169680864608 | 0.169680864608 | 2.22e-16 |

固定决策 addback 的作用不是比较求解速度，而是做 representability certificate：原优质组合在 Phase 2 final master 中因 signatures 缺失而不能原样表达；补回这些 signatures 后，当前同一数学模型立即、精确地接受该组合。因此“仅仅是 Gurobi 本次没搜到”的解释不成立。

## 5. 可得出的结论和边界

可以确认：

- Phase 2 的裁池删除了三个已知更优可行解中的大量 support；
- 缺失发生在 15/38/63 个箱组，具有跨箱组组合性质；
- mandatory starts 和现有单轮 F&O 没有恢复这些组合；
- 只补回缺失列即可在未改模型下精确恢复三个 oracle objective；
- 因而 coverage failure 足以解释本次观察到的 UB regression，是当前最直接的主因。

不能过度解释为：

- 每一条缺失列都具有不可替代性；可能存在等价 support；
- 已证明所有 seed 都会发生相同退化；本次仍只有三个 representative cases；
- 已确定下一版最优的列池策略；本审计没有调参或比较替代策略。

## 6. Phase 2 处置

状态：**CLOSED AFTER CONFIRMED ORACLE-COVERAGE FAILURE**。

- 不继续实现 conflict-aware multi-round F&O；
- 不实施 Dual Stabilization、Valid Inequalities 或 Branch-and-Price；
- 不继续深挖或调节 Phase 2 的 `K_g`、channel 顺序和 pool 参数；
- 保留当前分支、代码、实验和负面结果，作为后续是否重新设计 primal-pool policy 的依据。

本次审计只增加报告与机器可读结果，没有修改生产算法。
