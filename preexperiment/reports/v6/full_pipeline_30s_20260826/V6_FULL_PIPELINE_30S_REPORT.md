# V6 30 秒 Primal / RIM 单种子复测报告

日期：2026-08-26
模型：`row_aware_bay_zone_v6`
范围：24/48/96 箱组各一个种子；不是正式论文多种子实验

## 1. 实验口径

本轮固定已经通过 root closure gate 的 exact CG，不再调整 pricing、批量扩列、
目标函数或模型。统一设置为 Python 3.13.15、Gurobi、单线程、seed 0：

- exact root 总预算 60 秒；
- compact primal coverage 30 秒，最多提取 4 个 pool solutions；
- 合并 root/primal columns 后的 restricted integer master（RIM）30 秒；
- 所有阶段使用同一个解析峰值 cap 和同一个 V6 objective config。

## 2. 首轮发现并修复的交接缺陷

首轮全流程复测中，compact primal 的合法 incumbent zones 已经进入合并列池，
但 RIM 没有收到相应的 zone 选择、`q[group,bay]`、匿名进口预留和辅助状态
MIP start。结果是：24/48 箱组 RIM 在 30 秒内得到的 UB 反而差于已知 compact
incumbent，96 箱组没有找到任何可行解。

现已增加以 atom signature 映射、与临时 zone id 无关的完整 RIM warm start，
并同时设置出口尺寸/箱高、进出口贝位使用、进口尺寸和航次—箱区辅助状态。
修复后：

- 三个 RIM 都在 0.007 秒内接受已验证 incumbent；
- 三个结果均满足 `warm_start_objective_preserved=true`；
- 最终 incumbent 均通过 `V6ModelEvaluator` 独立验证；
- 不改变 root、模型可行域或目标函数。

## 3. 最终 LB / UB / Gap

| 算例 | root 时间 | exact LB | compact UB | final RIM UB | RIM 对 compact 改善 | `UB-LB` | `(UB-LB)/UB` |
|---|---:|---:|---:|---:|---:|---:|---:|
| `scale_g024_s401` | 5.41s | 0.04291185 | 0.07027657 | 0.07027657 | 0.00% | 0.02736472 | 38.94% |
| `scale_g048_s601` | 15.69s | 0.08315040 | 0.11590294 | 0.11259110 | 2.86% | 0.02944069 | 26.15% |
| `scale_g096_s801` | 41.44s | 0.10582754 | 0.21849124 | 0.18633384 | 14.72% | 0.08050630 | 43.21% |

结论很清楚：root 已能闭合，但整体 gap 并不小。96 箱组中 root columns 对
primal 协调有明显价值；24 箱组没有额外改善，48 箱组只有小幅改善。

## 4. Anytime 与阶段瓶颈

| 算例 | compact 首解 | compact 最好解 | compact 结束 gap | RIM 首解 | RIM 最好解 | RIM 结束 gap |
|---|---:|---:|---:|---:|---:|---:|
| 24 | 0.0008s | 29.01s | 22.95% | 0.0015s | 0.0015s | 17.23% |
| 48 | 0.0015s | 28.22s | 24.87% | 0.0033s | 28.34s | 18.55% |
| 96 | 0.0028s | 6.25s | 50.85% | 0.0069s | 26.18s | 41.33% |

- 24 箱组的 compact primal 一直到预算末端仍在改善，RIM 主要用于加强列池内
  bound，没有改善 UB。
- 48 箱组的 compact primal 和 RIM 都在预算末端产生最好解，当前 30/30 分配
  仍可能改变结果。
- 96 箱组的 compact primal 在 6.25 秒后停滞，而 RIM 利用 root proof columns
  将 UB 再改善 14.72%；对大规模而言，较短 compact warm-start 阶段和较长 RIM
  阶段更可能合理。

因此不存在一个由本轮即可确认的统一固定时间比例。下一轮应做少量预算交接消融，
而不是直接扩展多数种子。

## 5. 最终 UB 的目标贡献

| 算例 | 空间集中度 | 泊位运输 | 预留容量效率 |
|---|---:|---:|---:|
| 24 | 55.70% | 42.41% | 1.89% |
| 48 | 62.91% | 34.29% | 2.80% |
| 96 | 67.61% | 27.40% | 4.99% |

三类目标均有非零贡献，但预留容量效率仍明显较弱。这里只是一个种子，不能据此
重新调权或删除目标；正式判断仍需多种子和目标消融。

## 6. 论文比较口径的重要边界

compact primal coverage 本身是一个联合 row-atom MIP：它同时处理所有箱组、
物理排冲突、匿名进口和峰值约束。因此当前流程应准确描述为：

`exact CG root + compact-MIP primal start + merged RIM`

它不是“不调用 MIP 的纯列生成算法”。若论文对比完整 MIP，必须把 compact 阶段
计入论文算法总时间，并让独立 compact MIP baseline 获得相同总 wall-clock 预算。
否则不能用当前 UB 声称列生成优于 MIP。

## 7. 下一 gate

1. 冻结 exact root 和本轮 repaired RIM start；
2. 在 24/48/96 单种子上做 compact/RIM 预算交接消融，例如
   `5/25、10/20、20/10、30/0`，先寻找规模自适应规则；
3. 同时运行独立 compact MIP baseline，使用与混合算法相同的端到端时间预算；
4. 只有混合算法在多数规模上稳定改善 UB 或 anytime 后，才扩展多个种子；
5. 若 24/48 仍无明显优势，应把后续研究重点放在轻量 primal column expansion，
   而不是继续堆叠 root proof 组件。

机器可读结果位于：

- `preexperiment_outputs/v6_full_pipeline_30s_warmstart_smoke_g024_s401_20260826/`
- `preexperiment_outputs/v6_full_pipeline_30s_warmstart_scale_48_96_s1_20260826/`
