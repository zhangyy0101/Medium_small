# V7.1 Pricing 与根节点效率优化记录

## 结论

本轮保留不改变模型、restricted domain 或闭合判据的精确优化；不引入
dual stabilization、近似 reduced-cost 容差或其他会改变根节点证明语义的
机制。

最终代码保留：

- 按 anchor 建立 atom 与已有列签名索引；
- 删除由 anchor convexity 行隐含的 LP `lambda <= 1` 冗余上界，使 RMP
  中已有列在返回对偶下结构性 dual-feasible；
- 生产 pricing 不再为已有列建立不断增长的 no-good 约束；
- 对主问题系数完全相同的 Bay Pattern 去重；
- 使用安全的逐状态 reduced-cost 松弛下界跳过可证明非负的 pricing 状态；
- 持久化相同 `(anchor, size, height, allowed atoms)` 的 pricing MIP，后续
  CG 轮次只更新对偶目标系数并重新求解；
- 默认仍为每个 bay 每轮最多返回 3 列。

已撤回：

- 强制 RMP 使用 dual simplex：在 96 箱组上没有加速；
- 将每 bay 每轮列数提高到 8：列池膨胀且 LP 收敛更慢。

## 96 箱组单种子诊断

算例均为 `scale_g096_s801`，根节点预算 60 秒。

| 版本 | 状态 | 轮数 | Pattern 数 | 最后 restricted LP | pricing / s | master / s |
|---|---:|---:|---:|---:|---:|---:|
| 原始生产实现 | 未闭合 | 34 | 7,105 | 0.102484417 | 41.47 | 14.38 |
| anchor/signature 索引 | 未闭合 | 48 | 7,535 | 0.102458399 | 35.27 | 19.45 |
| 去除已有列 no-good | 未闭合 | 73 | 8,145 | 0.102446732 | 20.06 | 34.06 |
| 主问题等价列去重 | 未闭合 | 49 | 7,480 | 0.102444657 | 30.11 | 24.25 |
| 安全筛选 + 强制 dual simplex | 未闭合 | 49 | 6,953 | 0.102444243 | 28.91 | 25.75 |
| K=8 消融 | 未闭合 | 27 | 10,947 | 0.102471236 | 37.79 | 16.95 |

安全筛选在对应 K=3 实验中筛掉 295 / 10,290 个状态。它保持 exactness，
但该实例的下界较松，单独收益有限。K=8 的最后 LP 值更高且列数增加约
57%，因此不作为默认设置。

持久化 pricing 的最终 96 箱组运行按用户要求中止，未形成可用于比较的
根节点 checkpoint，因此本报告不声称它已经让 96 箱组在 60 秒内闭合。
它的正确性由重复改写目标系数并与微型穷举 oracle 对照的测试覆盖。

## 正确性边界

- pricing MIP 的变量、约束、top-K solution-pool 与 reduced-cost 校验未变；
- 安全筛选使用放松后的乐观下界，只有下界非负时才跳过状态；
- 每次持久化重优化后仍要求 Gurobi 返回 `optimal`，并用 master 的
  reduced-cost 函数复核回收列；
- 最终闭合仍要求完整 restricted-domain sweep 中不存在低于
  `-1e-8` 的列；
- restricted LP 仍不冒充全域 lower bound。

## 测试

最终完整测试：`143 passed, 2 subtests passed`。

新增覆盖包括：

- 安全下界筛选非负 pricing 状态；
- 同一持久化 pricing MIP 连续使用两套不同目标系数；
- 两次结果分别与微型穷举 pricing 一致；
- 生产 pricing 不生成 no-good 约束；
- restricted CG 与 restricted-domain 穷举 LP 一致。

