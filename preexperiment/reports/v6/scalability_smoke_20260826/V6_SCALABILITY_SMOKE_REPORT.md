# V6 单种子 Scalability / Anytime 体检报告

> 历史说明：本报告记录的是旧的 exact-`rho*` 前置流程。该前置阶段已被
> V6-native 解析 cap＋compact feasibility witness 替代；本文数值只用于
> 解释改动原因，不代表当前默认流程。

日期：2026-08-26
模型：`row_aware_bay_zone_v6`
算法：compact `rho*` → exact root CG → compact primal coverage → merged RIM

## 1. 本轮目的与口径

本轮不是正式论文实验，而是 V6 首轮生产规模体检。场景只复用
`preexperiment` 的确定性输入配方，不调用冻结的 V5 planner。所有模型均为
同一个 V6 契约；不使用完整 zone 枚举，也不把未闭合的 `rho*` 或 root CG
结果冒充有效证明。

统一设置：Python 3.13.15、Gurobi、单线程、solver seed 0；compact primal
与 final RIM 各 30 秒，pricing 子问题最多 10 秒。root CG 总预算为 60 秒。
pilot 的 compact `rho*` 预算为 30 秒；96 箱组另加一次 60 秒诊断。

这里的“箱组数”是实际 `ProblemData.export_groups` 数量；进口箱仍按
流向与尺寸进行匿名容量预留。

## 2. 阶段结果

| 算例 | 箱组 | 出口箱 | 匿名进口箱 | row atoms | compact peak | exact root CG | compact primal | final RIM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `pilot_s_101` | 4 | 80 | 168 | 3,460 | 0.99s，optimal | 5.12s，closed | 30.29s，timelimit | 0.36s，optimal |
| `pilot_m_201` | 16 | 360 | 332 | 13,840 | 8.20s，optimal | 31.74s，closed | 30.61s，timelimit | 0.89s，optimal |
| `pilot_l_301` | 24 | 640 | 498 | 18,474 | 2.52s，optimal | 49.95s，closed | 30.79s，timelimit | 2.50s，optimal |
| `scale_g048_s601` | 48 | 1,280 | 499 | 36,946 | 26.60s，optimal | 60.24s，未闭合 | 未执行 | 未执行 |
| `scale_g096_s801` | 96 | 2,560 | 497 | 73,890 | 61.71s，未证明 optimal | 未执行 | 未执行 | 未执行 |

48 箱组在 root business phase 完成了 22 轮，已有 1,504 列；最后一个完整轮次
的 master objective 为 `0.0958617`，最小 reduced cost 仍为
`-0.00154983`，所以不能声称根节点闭合，也不能把该值当作最终 root LB。

96 箱组在 60 秒 compact peak 诊断中得到：incumbent `0.35203095`、bound
`0.35148955`、solver gap `0.1538%`。首解约 0.98 秒出现，最好 incumbent
约 33.52 秒出现。数值已很接近，但 V6 策略要求 `rho*` 严格证明，因此流程
正确停止。

## 3. 已完整通过算例的 LB / UB

| 算例 | `rho*` | exact root LB | final UB | 绝对目标差 `UB-LB` | 相对 UB 的差 |
|---|---:|---:|---:|---:|---:|
| `pilot_s_101` | 0.0109375 | -0.00485354 | 0.01202302 | 0.01687656 | 140.37% |
| `pilot_m_201` | 0.0373832 | 0.00944448 | 0.03095067 | 0.02150619 | 69.49% |
| `pilot_l_301` | 0.0737179 | 0.07690321 | 0.10307687 | 0.02617366 | 25.39% |

旧 diagnostics 中的 `relative_root_gap` 使用
`(UB-LB)/max(1, |UB|)`。由于 V6 目标通常小于 1，它实际等于绝对目标差，
不能解释为常规 MIP relative gap。本轮 runner 已增加明确的
`final_scale_normalized_root_gap` 与 `final_incumbent_relative_root_gap` 字段；
论文报告应优先给出绝对归一化目标差，并明确相对差的分母。

三个算例中，final RIM 的 UB 均与 compact primal incumbent 相同。proof
columns 扩大了列池，但没有在这些单种子上产生更优整数组合。这说明当前
root columns 的证明作用有效，primal 改善作用尚不明显。

## 4. Anytime 与目标贡献

compact primal 的首解时间分别约为 0.002s、0.006s、1.159s，最好解时间
分别约为 0.270s、15.305s、22.460s；30 秒结束时 compact MIP 自身 gap
分别为 23.84%、37.63%、15.88%。最终 RIM 在 0.36–2.50 秒内证明其缩小列池
最优，因此主要 UB 时间消耗在 compact primal，而不是 RIM。

final UB 的三类加权贡献如下：

| 算例 | 空间集中 | 泊位运输 | 预留容量效率 |
|---|---:|---:|---:|
| `pilot_s_101` | 63.77% | 14.14% | 22.09% |
| `pilot_m_201` | 44.48% | 51.71% | 3.81% |
| `pilot_l_301` | 70.89% | 28.46% | 0.64% |

权重本身固定，但“预留容量效率”的实际贡献随规模迅速减小。本轮三个解的
未利用预留容量都很小，因此这首先说明该项在当前数据上的边际区分能力弱，
不能仅凭单种子直接判定权重错误；正式结论仍需多种子和消融。

## 5. 结论与下一开发门槛

V6 的四阶段生产流程已经在 4/16/24 箱组真实堆场场景上闭环，模型验证、
proof/primal column 合并和 final RIM 都工作正常。这比只在微型 oracle 上
验证前进了一步。

但当前流程还不能称为 production-scalable 或达到正式论文实验条件：

1. 48 箱组 exact root CG 在 60 秒内无法闭合；
2. 96 箱组 compact `rho*` 在 60 秒内仍差一个很小但不可忽略的证明 gap；
3. 4–24 箱组的 root relaxation 相对 UB 仍弱，尤其小算例；
4. compact primal 用满预算，而回注 proof columns 没有改善这三个 UB；
5. 目标类别的实际贡献随规模明显漂移。

因此下一步不应立刻扩展到多数种子，也不应先加 Local Branching。优先顺序应为：

1. 缩短并强化 compact `rho*` 的证明，先让 96 箱组在可接受预算内闭合；
2. 优化 exact pricing/root CG 的规模行为，使 48 箱组能够闭合；
3. 在不破坏 exact root 的前提下，改善 primal columns 的跨箱组组合价值；
4. 达到上述门槛后，再做 24/48/96 多种子与完整 MIP 的同预算正式对比。

原始机器可读结果位于 `preexperiment_outputs/v6_scalability_*_20260826/`。
本目录的 `summary.csv` 和 `summary.json` 保存了本报告使用的精简数据。
