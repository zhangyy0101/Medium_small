# 出口资料箱贝位—排资源规划

当前开发主线为 V7.1 Hierarchical Restricted Column Generation。V7.1 保持 V7 业务模型与 Bay Pattern 不变，强化 Stage 1 并把 Stage 2 冻结在 restricted area domain。论文算法的前置 witness 当前只验证 `rho_cap` 下的整数可行性，不再优化业务目标；其 proof patterns 只作为 master 可行性保底列，不再扩张 Stage-1/pricing 搜索域。针对 oracle/coverage audit 暴露的 coarse surrogate 错排，Stage 1 已增加与正式目标一致的 bay-local placement ranking。24 箱组单种子验证显示方向有效，但尚不能替代多种子正式实验。V6.1 及 V5 均已冻结，仅用于历史复现。

## V7 模型摘要

- 出口箱组固定为 `航次 + 尺寸 + 箱高 + 卸货港`。zone 已从正式变量、可行域、目标函数和 pricing column 中删除。
- 行原子保留尺寸、20/40/45 footprint、45 英尺边缘资格、逐排容量、箱高不混、精确箱组不混排和历史状态“不增加违反程度”语义。
- 同一物理贝位最多由三个新出口箱组使用；40/45 英尺箱会在 footprint 覆盖的每个物理贝位计数。
- 进口箱继续按流向与尺寸匿名预留，并与新出口在物理贝位上互斥。
- peak utilization 是 data-derived hard epsilon constraint，不设箱区均衡或 peak 次级目标。
- 目标只含空间集中度（少箱区、少贝位、短 span）、靠近在场同组箱和泊位运输距离；当前权重仅是待实验标定的开发基线。
- Stage 1 使用 Stage-1 best、按 `Y[g,a]` 支持去重的 near-optimal solution pool 和 bay-local placement 候选构造冻结的 `A_restricted[g]`，并使用简单 bay-incidence packing proxy。bay-local 通道按真实兼容 anchor 容量，以正式目标中的 flow、额外贝位和 span 成本为每个箱组独立排序；默认额外 cap 为 5，其中粗 MIP pool 至多占 1 个名额。best support 不受 cap 限制，Stage-1 数量不固定 Stage 2，feasibility witness support 不参与候选域构造。
- Stage 2 只在 frozen restricted domain 内生成新的 Bay Pattern；witness proof patterns 可以作为显式登记的固定域外列留在 master，但不会开放对应箱区或允许 pricing 在域外生成新列。full-domain pricing 只是不改变求解状态的可选审计。
- 生产算法默认共享一个 120 秒总预算：feasibility-only witness、Stage 1 和 restricted root 使用软上限，RIM 使用全部剩余时间；可选全域审计和 Complete MIP 对照均不占该生产预算。
- restricted LP 不是完整模型的全局下界；未做全域证书时，`global_lower_bound` 与 `global_gap` 明确为 `None`。

完整定义见 [V7 模型契约](docs/V7_MODEL_CONTRACT.md)，当前实施边界见 [MODEL_SCOPE.md](MODEL_SCOPE.md)。

## 当前代码边界

- `yard_planning/v7_atoms.py`：无 zone 的合法 row-atom 候选域；
- `yard_planning/v7_model.py`：V7 版本、目标与尺度、peak 策略和独立 evaluator；
- `yard_planning/v7_complete_mip.py`：zone-free compact V7 MIP 与 peak 可行 witness；
- `yard_planning/v7_stage1_area.py`：强化的 group-to-area Stage 1、packing proxy 和 restricted-domain 构造；
- `yard_planning/v7_bay_patterns.py`：Bay Pattern、support-size 1/2/3 精确枚举和 pricing oracle；
- `yard_planning/v7_column_generation.py`：restricted RMP/CG 与只读 full-domain pricing audit；
- `yard_planning/v7_integer.py`：restricted integer master、解恢复和独立验证；
- `yard_planning/v7_pipeline.py`：V7 代码级端到端流程；
- `yard_planning/v6_*`、`yard_planning/row_aware_zones.py`：冻结的 V6.1 历史实现；
- `yard_planning/contiguous_zone_generation.py`、`yard_planning/direct_milp.py`：冻结的 V5 历史实现。

## 环境与测试

项目使用 Python 3.13，并需要在进入求解阶段时具备有效的 Gurobi 许可证：

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -p 'test_*.py'
```

V7.1 当前提供 Python API `yard_planning.v7_pipeline.solve_v7`，尚未声明为经过生产规模验证的 CLI。当前 24 箱组单种子得到 UB `0.045080`，比保存的同预算 Complete-MIP incumbent `0.045754` 低 1.47%；oracle support 覆盖由 4/28 提升到 22/28。48 箱组 UB `0.079309` 比旧 V7 改善 5.20%，但仍比保存的 MIP incumbent 高 1.73%；96 箱组 restricted root 未在 60 秒内闭合，因而没有正式 UB。以上结果只验证修正方向和规模瓶颈，不能替代正式多种子实验。V5 命令行入口默认会拒绝运行；只有复现历史结果时才显式确认：

```bash
python benchmark_contiguous_zones.py --allow-legacy-v5-model --input example/input_data.json
python -m preexperiment --allow-legacy-v5-model --paper-time-limit 30
```

这些命令的结果仍属于 V5。下列 V6 入口也仅用于历史复现，不代表 V7。

V6 Complete MIP 只适用于可完整枚举 zone 的小算例：

```bash
python benchmark_v6_complete_mip.py \
  --input PATH/TO/SMALL_INPUT.json \
  --peak-feasibility-time-limit 10 \
  --business-time-limit 60 \
  --solver-threads 1 \
  --output PATH/TO/v6_complete_result.json
```

默认先解析计算 ε cap，再用 compact MIP找一个 cap 内可行 witness；Complete MIP 与论文算法共用该 cap。只有显式传入 `--peak-method exact` 时才求解 exact `rho*` oracle。`--maximum-zone-count` 是执行安全阈值，超过时直接失败，不会裁剪合法 zone。

V6 根节点列生成的小算例入口为：

```bash
python benchmark_v6_root_cg.py \
  --input PATH/TO/SMALL_INPUT.json \
  --peak-feasibility-time-limit 10 \
  --pricing-time-limit 60 \
  --solver-threads 1 \
  --output PATH/TO/v6_root_cg_result.json
```

该入口默认使用 `--peak-method analytic`。`--peak-method compact` 和 `--peak-method complete` 分别保留非枚举/完整枚举 exact `rho*` 诊断。其 `root_lp_objective` 是下界，不是可行 UB。

在同一入口中增加以下参数，可继续求解“严格只使用根节点生成列”的 restricted integer master：

```bash
python benchmark_v6_root_cg.py \
  --input PATH/TO/SMALL_INPUT.json \
  --restricted-integer-time-limit 60 \
  --restricted-integer-mip-gap 0 \
  --output PATH/TO/v6_root_and_integer_result.json
```

该模式不会偷偷加入完整枚举列或 V5 候选池。若根列池没有整数可行组合，会明确报出 infeasible；因此它目前是 correctness/coverage 诊断入口，而不是完整论文算法入口。

启用 compact primal coverage V1：

```bash
python benchmark_v6_root_cg.py \
  --input PATH/TO/SMALL_INPUT.json \
  --peak-method analytic \
  --primal-coverage-time-limit 60 \
  --primal-coverage-pool-solutions 4 \
  --restricted-integer-time-limit 60 \
  --output PATH/TO/v6_covered_result.json
```

该流程不会枚举完整 zone universe；compact MIP 的多个联合整数解被恢复为连续 zone 候选后与 root proof pool 合并，exact root bound 不会改变。
