# 出口资料箱贝位—排资源规划

当前开发主线为 V6 row-aware bay-zone 模型。V6 已完成数学模型契约、小规模完整 zone oracle、解析峰值 cap 与 compact 可行性认证、完整业务 MIP、restricted master + exact pricing、restricted integer master，以及 V6-native compact primal coverage V1。当前默认流程先用 V6 可达容量下界确定 ε cap，并用短时 compact MIP认证可行性；exact CG 提供根下界，row-atom primal MIP生成整数协调列并回注 restricted integer master。exact `rho*` 只保留为 oracle/诊断，不再阻塞生产流程。

## V6 模型摘要

- 出口箱组固定为 `航次 + 尺寸 + 箱高 + 卸货港`；仅使用尚未进场的确定性出口资料箱，不使用预测箱、TOPS plan 或上游大计划目标。
- 一个 zone 是一个箱组在同一箱区的一段连续锚定贝位，以及每个贝位中选择的非空排集合；相邻贝位的排号可以不同。
- 不同箱组不可共排；尺寸和箱高相同的不同箱组可以通过不同排共享一个贝位。
- 在场箱采用“不增加违反程度”规则：尺寸兼容保持严格；历史混箱高贝位只允许新箱选择一个已有箱高，历史混箱组排只允许新箱选择一个已有精确箱组；新分配箱之间仍保持箱高不混贝、箱组不混排。
- 进口箱按流向和尺寸做匿名贝位容量预留，并满足 footprint、尺寸状态和物理容量；新进口与新出口不可共享物理贝位。
- 目标分为三类：空间集中度、泊位运输距离、预留容量效率。峰值利用率采用 V6 可达容量解析下界导出的 ε 约束，并由 compact 可行解认证。
- V6 没有 row dispersion、group-area target、large-plan guidance、shortage 或“需求量 + 一个原子排容量”的 zone 上限。

完整模型定义见 [V6 模型契约](docs/V6_MODEL_CONTRACT.md)，当前实施边界见 [MODEL_SCOPE.md](MODEL_SCOPE.md)，下一步见 [NEXT_STAGE_DEVELOPMENT.md](NEXT_STAGE_DEVELOPMENT.md)。

## 当前代码边界

- `yard_planning/v6_model.py`：V6 版本、三类目标配置、自然尺度、峰值策略、模型元数据和独立解验证/目标重构；
- `yard_planning/row_aware_zones.py`：小规模完整 row-aware zone 枚举，是未来 MIP 与 pricing 的正确性 oracle；
- `yard_planning/v6_complete_mip.py`：共用生产 ε cap 的完整枚举业务 MIP，以及仅供小规模 oracle 使用的 exact min-max MIP；
- `yard_planning/v6_column_generation.py`：投影 restricted master、可行 witness 起点与后备 Phase-I、基于贝位容量/连续区间动态规划的 exact pricing 和根 LP 批量列生成；
- `yard_planning/v6_restricted_integer.py`：restricted integer master、以 atom signature 传递 compact incumbent 的完整 MIP start、`q[g,b]` 到 `q[z,b]` 的恢复及独立 incumbent 校验；
- `yard_planning/v6_primal_coverage.py`：不枚举 zone 的 compact row-atom primal MIP、连续 zone 恢复、多 incumbent 候选列和根列池合并；
- `benchmark_v6_complete_mip.py`：独立 V6 Complete MIP 小算例入口；
- `benchmark_v6_root_cg.py`：独立 V6 根节点 correctness 入口；
- `yard_planning/models.py`：固定箱组身份和输入数据结构；
- `yard_planning/output_validator.py`：物理结果外部校验；
- `tests/test_v6_model_contract.py`：V6 契约测试；
- `tests/test_v6_column_generation.py`：pricing oracle 和全列 LP 等价测试；
- `tests/test_v6_restricted_integer.py`：全列整数等价、解恢复和根列池覆盖不足测试；
- `tests/test_v6_primal_coverage.py`：根列池修复、Complete MIP 微型对照和 40 英尺 footprint 测试；
- `yard_planning/contiguous_zone_generation.py`、`yard_planning/direct_milp.py`：冻结的 V5 历史实现，不是 V6 求解器；
- `paper_large_plan`：历史参考，不进入 V6；
- `preexperiment/reports/v5`：历史实验记录，不可与 V6 比较。

## 环境与测试

项目使用 Python 3.13，并需要在进入求解阶段时具备有效的 Gurobi 许可证：

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -p 'test_*.py'
```

V5 命令行入口默认会拒绝运行，避免误生成“看似 V6”的结果。只有复现历史结果时才显式确认：

```bash
python benchmark_contiguous_zones.py --allow-legacy-v5-model --input example/input_data.json
python -m preexperiment --allow-legacy-v5-model --paper-time-limit 30
```

这些命令的结果仍属于 V5；V6 使用下面的独立 Complete MIP 入口，不会改名复用上述入口。

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
