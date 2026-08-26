# V6 根节点重写与单种子扩展性报告

日期：2026-08-26
模型：`row_aware_bay_zone_v6`
范围：只评估 exact root column generation；不是 UB/MIP 正式预实验

## 1. 修改内容

根节点保持 V6 数学模型和列生成框架不变，只重写求解实现：

1. 生产流程把已经由 compact 模型验证过的可行 witness zones 直接作为业务
   RMP 初始列，因此不再重复执行人工变量 Phase-I；无 witness 的独立调用仍保留
   原 Phase-I 作为安全后备。
2. 删除“每个箱组、每个箱区连续段反复建立 Gurobi pricing MIP”的实现。
   新 pricing 先用容量动态规划求每个贝位的精确 k-best 合法排子集，再用区间
   标签合并求连续贝位 zone；pricing MIP 数为 0。
3. 一轮对每个箱组批量加入最多 512 列，而不是每组一列。目标停滞后，逐步把
   每个区间保留的排组合深度从 1 提高到 2/4/8，用来清除对偶退化尾部。
4. LP 中删除显式 `x_z <= 1` 上界。该上界已被每个非空 zone 必然进入的
   `same_group_bay_nonoverlap <= 1` 隐含；删除冗余上界后，定价不需要为已在
   上界的列维护不断增长的 no-good 集合。
5. 已删除实验中被证明更慢的 persistent group pricing MIP 及旧 run-level
   pricing MIP，没有保留双轨死代码。

动态规划仍是 exact pricing：微型算例中最小 reduced cost 与完整 zone 枚举
一致，闭合根 LP 与完整枚举 projected LP 目标一致。

## 2. 最终配置结果

统一设置：Python 3.13.15、Gurobi、单线程、seed 0、root 总预算 60 秒；
解析峰值 cap 与 compact feasibility witness 使用当前默认流程。

| 算例 | 箱组 | row atoms | Phase-I | 业务 CG 轮数 | 根列数 | root 时间 | exact LB |
|---|---:|---:|---:|---:|---:|---:|---:|
| `scale_g024_s401` | 24 | 18,474 | 跳过 | 10 | 9,961 | 5.48s | 0.04291185 |
| `scale_g048_s601` | 48 | 36,946 | 跳过 | 14 | 25,653 | 15.95s | 0.08315040 |
| `scale_g096_s801` | 96 | 73,890 | 跳过 | 17 | 61,323 | 41.85s | 0.10582754 |

三者最后一轮最小 reduced cost 的绝对值均小于 `1e-16`，并标记
`closed_by_exact_pricing=true`。所以 24/48/96 单种子 root closure gate 已通过，
不是用目标停滞代替 exact certificate。

运行分解如下：

| 算例 | pricing 总时间 | RMP 总时间 | 其他/建列开销 |
|---|---:|---:|---:|
| 24 | 4.25s | 0.36s | 0.87s |
| 48 | 12.76s | 1.13s | 2.05s |
| 96 | 32.15s | 4.57s | 5.13s |

当前剩余主成本仍是 exact DP pricing，但已从大量子 MIP 的不可控证明时间，变成
随箱组数和迭代数较稳定增长的纯组合动态规划时间。

## 3. 与重写前的直接比较

同一 `pilot_l_301`、同一解析 cap 下，重写前 root 为 50.57 秒、64 个业务轮次、
LB `0.07691384507862174`；重写后为 6.97 秒、13 个业务轮次，LB 完全相同。
时间减少 86.2%，约 7.25 倍加速。

重写前 `scale_g048_s601` 在 60.23 秒后仍停在 business phase；重写后 15.95 秒
闭合。重写前 `scale_g096_s801` 在 60.45 秒后仍停在 Phase-I；重写后跳过
Phase-I 并在 41.85 秒闭合业务根 LP。

这些比较只说明根节点实现改善。统一复跑时，compact primal 和 final RIM 被
故意限制为 1 秒，因此结果文件中的 final RIM 失败不能解释为新根节点失败，
也不能用于比较 UB 或最终 gap。

## 4. 消融结论

- 每区间固定深度 1：24 箱组最快，但 48 箱组出现 70 轮对偶退化尾部。
- 自适应最大深度 8：48 箱组由 34.07 秒降到约 16 秒；总体最稳。
- 最大深度 16：pricing 单轮成本过高，48 箱组回升至 26.26 秒。
- barrier 且不 crossover：RMP 时间显著增加，48 箱组为 29.45 秒。
- persistent pricing MIP 和重复 top-k no-good MIP：60 秒内不能闭合，已删除。

## 5. 结论和边界

根本问题已经从“V6 模型不能做列生成”缩小为“row-aware pricing 必须利用其
贝位容量和连续区间结构”。本次重写保留 exact CG 骨架，而且 24/48/96 均在
60 秒内给出严格根闭合证明，因此已经具备继续研究 primal improvement 的基础。

但 96 箱组仍需 41.85 秒，尚未达到所有生产规模都在数秒内闭合的理想目标。
进一步压缩它应作为性能工程优化，不应重新引入固定排 strip、裁剪合法 zone 或
放弃 exact certificate。下一阶段应在固定本根算法后，单独恢复合理的 UB 时间
预算，评估 compact primal/RIM，并开发不会改变根下界的 primal 改善组件。

机器可读的通过结果位于
`preexperiment_outputs/v6_root_rewrite_final_root_only_scale_24_48_96_s1_20260826/`。
