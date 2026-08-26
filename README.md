# 出口资料箱贝位—排资源规划

当前开发主线为 V7 两阶段 group-area / bay-pattern 模型。V7 已完成模型与算法代码以及微型正确性验证；24/48/96 箱组、性能对比、规模性和正式 UB/LB/gap 评价尚未运行，统一留到下一实验轮。V6.1 及 V5 均已冻结，仅用于历史复现。

## V7 模型摘要

- 出口箱组固定为 `航次 + 尺寸 + 箱高 + 卸货港`。zone 已从正式变量、可行域、目标函数和 pricing column 中删除。
- 行原子保留尺寸、20/40/45 footprint、45 英尺边缘资格、逐排容量、箱高不混、精确箱组不混排和历史状态“不增加违反程度”语义。
- 同一物理贝位最多由三个新出口箱组使用；40/45 英尺箱会在 footprint 覆盖的每个物理贝位计数。
- 进口箱继续按流向与尺寸匿名预留，并与新出口在物理贝位上互斥。
- peak utilization 是 data-derived hard epsilon constraint，不设箱区均衡或 peak 次级目标。
- 目标只含空间集中度（少箱区、少贝位、短 span）、靠近在场同组箱和泊位运输距离；当前权重仅是待实验标定的开发基线。
- 算法先求 group-to-area 的 Stage 1，再求全局 bay-pattern RMP；Stage 1 配额不固定 Stage 2，初始候选箱区上限也不是业务硬约束。根节点关闭前必须执行全域 exact pricing certification，并动态恢复被初始 active set 排除的改进箱区。

完整定义见 [V7 模型契约](docs/V7_MODEL_CONTRACT.md)，当前实施边界见 [MODEL_SCOPE.md](MODEL_SCOPE.md)。

## 当前代码边界

- `yard_planning/v7_atoms.py`：无 zone 的合法 row-atom 候选域；
- `yard_planning/v7_model.py`：V7 版本、目标与尺度、peak 策略和独立 evaluator；
- `yard_planning/v7_complete_mip.py`：zone-free compact V7 MIP 与 peak 可行 witness；
- `yard_planning/v7_stage1_area.py`：group-to-area Stage 1、solution pool 和 active-set 构造；
- `yard_planning/v7_bay_patterns.py`：Bay Pattern、support-size 1/2/3 精确枚举和 pricing oracle；
- `yard_planning/v7_column_generation.py`：全局 RMP、active/full-domain pricing certification 和动态箱区激活；
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

V7 当前提供 Python API `yard_planning.v7_pipeline.solve_v7`，尚未声明为经过生产规模验证的 CLI。V5 命令行入口默认会拒绝运行；只有复现历史结果时才显式确认：

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
