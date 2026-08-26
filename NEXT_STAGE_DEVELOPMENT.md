# V6 Root Proof + Compact Primal Coverage V1

> 当前数学契约已升级为 V6.1：在场历史违规状态采用“不增加违反程度”规则，
> 尺寸仍严格一致，贝位箱高与排内精确箱组采用已有集合成员关系。下文记录的
> V6.0 运行时间、LB、UB 和 gap 仅保留为历史基线；在 V6.1 上重新运行前不能
> 继续用于算法优劣判断。

## 已完成基线

当前已经具备：

1. 独立于 V5 的完整小规模 row-aware zone universe；
2. V6-native 解析峰值 cap 与 compact feasibility witness；
3. 共用固定 `rho_cap` 的完整 V6 业务 MIP，以及可选 exact `rho*` oracle；
4. solver 目标与 `V6ModelEvaluator` 独立重构目标一致性检查；
5. 未裁剪 Complete MIP 安全边界：zone 数超过阈值时失败，不返回伪完整结果。

独立入口为：

```text
benchmark_v6_complete_mip.py
```

## 本阶段已完成

已经建立与 Complete MIP 保持同一整数语义的 V6 projected restricted master 和 exact pricing，且没有加入 primal heuristics。

完成项：

1. 主问题使用全局 `q[group,bay]`，zone 列只承载连续贝位、排资源和预留容量；
2. 生产流程用已验证 compact witness zones 直接启动业务 RMP；无 witness 的独立调用仍可从零 zone 用临时 Phase-I deficit 构造可行列；
3. exact pricing 用贝位容量 knapsack DP 和连续区间标签合并，允许每个贝位选择不同的非空排集合，不设置旧的固定排 strip 或需求上限；
4. exhaustive pricing 与非枚举 exact pricing 在 Phase-I 和业务目标下返回相同最小 reduced cost；
5. exact pricing 无负 reduced cost 时，closed root LP 与完整枚举 projected V6 LP 目标一致；
6. 根节点结果明确标记为 LP lower bound，不伪装成整数 UB。

## Restricted integer 已完成

1. LP 和整数主问题复用完全相同的建模代码，只切换变量域；
2. 通过同箱组同贝位不重叠约束，把正的 `q_gb` 唯一恢复为 `q_zb`；
3. restricted incumbent 已通过 `V6ModelEvaluator` 的可行性与目标独立重构；
4. 完整 zone pool 的 projected integer master 与 Complete MIP 在微型算例上目标一致；
5. 纯根节点列池的局限也已确认：它可能给出较差 UB，甚至没有整数可行组合。

这不是 pricing 错误。负 reduced-cost pricing 的职责是闭合 LP；对 LP 非必要的列仍可能是整数协调所必需的，因此 proof pool 不能自动充当 primal pool。

### 根列池微型审计结果

| 算例 | Complete MIP | closed root LB | 根列数 | 根列池整数结果 |
|---|---:|---:|---:|---|
| 1 箱组、1 箱、1 贝位 | 0.0000 | -0.1750 | 1 | 可行，UB=0.2125，未达到完整最优 |
| 2 箱组、3 个出口箱、1 个进口箱 | 0.0000 | -0.5250 | 2 | infeasible，缺少跨箱组协调列 |

同一批算例在输入完整合法 zone universe 后，projected integer master 均与 Complete MIP 目标一致且通过 evaluator。因此直接原因已经定位为根 proof pool 的整数覆盖不足，而不是整数主问题、流量恢复或 evaluator 不一致。

## Compact primal coverage V1 已完成

V1 没有调整 exact pricing，而是增加一个不枚举 zone 的紧凑 row-atom MIP：

1. 直接选择 group-specific row atoms、实际 `q_gb` 和匿名进口预留；
2. 在一个联合 MIP 中处理物理排冲突、尺寸/箱高状态、进口整贝互斥和 peak cap；
3. 把同箱组同箱区的连续已用贝位合并为合法 V6 zone；
4. 从多个联合 incumbent 中提取并去重协调列；
5. 与 root proof pool 合并后重新求 restricted integer master；
6. root LP 和 exact pricing 完全不变，完整 zone enumeration 不进入该流程。

修复后的同一组微型结果为：

| 算例 | root LB | coverage 后 UB | Complete MIP | 结果 |
|---|---:|---:|---:|---|
| 1 箱组、1 箱、1 贝位 | -0.1750 | 0.0000 | 0.0000 | 达到完整最优 |
| 2 箱组、3 个出口箱、1 个进口箱 | -0.5250 | 0.0000 | 0.0000 | 从整数 infeasible 修复并达到完整最优 |

## 下一阶段目标

旧流程曾要求 compact row-atom min-max MIP严格证明 `rho*`。单种子测量表明，
96 箱组为了缩小极小的 peak proof gap 会额外消耗 60 秒，而 `h=0.5` 后对
最终 cap 的影响很小。因此生产流程已经改为：解析计算 V6 可达负载下界、
生成固定 cap、compact MIP只认证一个 cap 内可行 witness，并把 witness 作为
后续 compact primal warm start。exact compact/Complete min-max 只保留为
oracle 和敏感性诊断。

24/48/96 单种子 root 已用重写后的 exact pricing 重新测量：分别在
5.48/15.95/41.85 秒闭合，Phase-I 均由已验证 witness 安全跳过。旧实现的
24 箱组同口径案例约 50.57 秒，48 箱组在 60 秒内未闭合 business phase，
96 箱组在 60 秒内未完成 Phase-I。因此 root closure gate 已通过。接下来：

已恢复 compact primal 与 final RIM 各 30 秒预算，并修复 compact incumbent
未作为完整 MIP start 传给 RIM 的交接缺陷。修复后 24/48/96 均在 0.007 秒内
获得已验证 RIM 首解；final UB 分别为 0.07028/0.11259/0.18633，相对 exact
root LB 的常规 gap 仍为 38.94%/26.15%/43.21%。RIM 相对 compact incumbent
的 UB 改善分别为 0%/2.86%/14.72%。因此接下来：

1. 冻结当前 exact root 和 repaired RIM start；
2. 做 `5/25、10/20、20/10、30/0` 等 compact/RIM 交接预算消融，寻找规模自适应规则；
3. 让独立 compact MIP baseline 获得与混合流程相同的端到端 wall-clock 预算；
4. 只有 UB/anytime 优势稳定后才运行 24/48/96 多种子正式预实验。

## 暂不迁移

- V5 static pool、multi-start、F&O、Adaptive Expansion、Local Branching；
- V5 fixed-row strip 的 RMQ pricing；
- dual stabilization、valid inequalities、Branch-and-Price；
- 任何 V6 预实验或与 V5 数值的直接比较。

## Correctness gate

```text
restricted-master integer semantics == complete V6 constraints
restricted-master objective coefficients == V6 evaluator coefficients
exhaustive row-aware pricing == exact row-aware pricing
closed root LP == fully enumerated V6 LP
integer incumbent passes V6ModelEvaluator
full-zone projected integer objective == Complete MIP objective
Complete MIP remains independently runnable
```

以上 correctness gate 和 primal coverage V1 微型 gate 已通过。旧的单种子
smoke 已定位出 root closure 和 exact peak proof 耗时；exact peak 已从
生产流程移除，重写后的 exact root 已通过 24/48/96 单种子 gate。该结果仍只
证明 root 可扩展性，不能替代恢复正常 primal/RIM 预算后的 UB/gap 正式实验；
也不得将完整枚举或旧 V5 static pool 隐藏为默认补列机制。
