# V6 解析 Peak Cap 迁移报告

日期：2026-08-26
模型：`row_aware_bay_zone_v6`
当前默认算法：解析 cap → compact feasibility witness → exact root CG → warm-started compact primal → merged RIM

## 1. 修改结论

默认流程已经不再先求解 compact exact `rho*`。现在先依据 V6 合法 row atom、
出口与匿名进口工作量以及可达箱区容量计算解析下界 `L`，再按
`rho_cap = L + h(1-L)`（当前 `h=0.5`）得到固定 cap。随后只求一个满足该
cap 的 compact 可行解并经独立 evaluator 认证；该 witness 同时作为业务
compact primal 的 MIP start。

`L` 不是 `rho*`，代码和输出均明确标记
`reference_role=analytic_workload_lower_bound` 与
`minimum_feasible_utilization_proven=false`。原 compact/Complete min-max
求解器仍保留为小规模 oracle 和敏感性诊断，但不再属于默认生产路径。

## 2. 24/48/96 单种子复测

统一设置：Python 3.13、Gurobi、单线程、solver seed 0；peak feasibility
预算 10 秒，exact root CG 总预算 60 秒。24 箱组在根节点闭合后继续执行
30 秒 compact primal 和 30 秒 RIM；48/96 箱组在 root gate 未通过后正确停止。

| 算例 | 箱组 | row atoms | 旧 exact peak | 新 peak 准备 | `L` | 固定 cap | witness peak | 后续结果 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `pilot_l_301` | 24 | 18,474 | 2.52s | 0.76s | 0.070615 | 0.535308 | 0.360577 | root 50.57s 闭合；final UB 0.103263 |
| `scale_g048_s601` | 48 | 36,946 | 26.60s | 1.87s | 0.140625 | 0.570313 | 0.569712 | root 60.23s 未闭合 |
| `scale_g096_s801` | 96 | 73,890 | 61.71s 仍未证明 | 3.84s | 0.280423 | 0.640212 | 0.639423 | root 60.45s 停在 Phase-I |

新 peak 准备相对旧 exact peak：24 箱组约快 3.3 倍，48 箱组约快 14.2 倍；
96 箱组则从 60 秒以上仍不能给出合法证明，变为 3.84 秒内完成 cap 可行性
认证。因此原先不必要的 `rho*` 证明瓶颈已经移除。

48 箱组完成 28 轮、生成 1,680 列后仍有 reduced cost
`-0.00158379`；96 箱组完成 18 轮、生成 1,728 列，但 Phase-I deficit 仍为
`549.1349`。当前首要扩展性瓶颈已转移为 exact root CG，而不是 peak 准备。

## 3. UB 与可比性

24 箱组新 final UB 为 `0.10326316`，旧 exact-`rho*` 流程为
`0.10307687`，表面高约 `0.18%`。两者不能解释为同一模型下算法 UB 退化：
新解析 cap 为 `0.53530752`，旧 exact-`rho*` cap 为 `0.53685897`，新 cap
略严格，两个运行的可行域并不完全相同。

正式比较论文算法和 Complete MIP 时，两者必须外部接收同一解析 policy；
当前 CLI 和 solver 接口已经按此处理。解析 witness 的 warm start 在 24 箱组
业务 compact primal 中成功应用，首个 incumbent 约 0.00085 秒出现，最好
incumbent 约 25.34 秒出现。单一种子只能证明 warm start 链路有效，尚不能
声称它稳定改善最终 UB。

## 4. 当前默认流程

```text
V6 input / row atoms
        ↓
analytic workload lower bound L + fixed epsilon cap
        ↓
compact fixed-cap feasibility certificate
        ├── fail: stop，不放松或删除 peak 约束
        └── witness
              ↓
exact root column generation（LP lower bound / proof columns）
              ↓
compact business primal（witness warm start / coordination columns）
              ↓
proof + primal columns merge
              ↓
restricted integer master + independent evaluator
```

Complete MIP 基线跳过列生成，但接收完全相同的外部 peak policy。只有显式
oracle 模式才运行 compact 或 Complete exact min-max `rho*`。

## 5. 下一步边界

本次修改达到了“参考 V5、不让 exact `rho*` 阻塞默认流程”的目的，且没有
迁移 V5 的 fixed-row zone 语义。下一开发点应聚焦 root CG：先改善 Phase-I
初始覆盖和批量列生成，再处理 business phase 的尾部负 reduced-cost 收敛。
在 48/96 箱组 root closure gate 通过前，不应把资源投入到 Local Branching、
正式多种子 UB 比较或 Branch-and-Price。

原始结果位于：

- `preexperiment_outputs/v6_analytic_peak_pilot_l301_20260826/`
- `preexperiment_outputs/v6_analytic_peak_scale_g048_s601_20260826/`
- `preexperiment_outputs/v6_analytic_peak_scale_g096_s801_20260826/`
