# 出口资料箱排级堆场规划

本项目实现面向 TRE 论文的排级堆场规划模型。详细位置分配只针对已申报、尚未入场的出口资料箱，且所有出口资料箱必须完成分配；出口预测箱不参与模型，也不形成额外预留。

进口航次不参与具体排级分配。模型只读取大计划的 `new_qty`，为即将卸船的进口箱建立匿名容量预留；`planned_qty` 已包含在场箱，不能作为新增需求。进口预留仅服从箱区功能、贝位尺寸和共享物理容量，不继承出口箱的航次、目的港、箱高等不混约束。

大计划箱型必须为 `20` 或 `40`，其中 `40` 表示物理 40/45 英尺大箱类别；`ALL`、空值、`45` 和未知值会触发输入错误。45 英尺出口箱只能放在箱区首尾的可用边缘大贝，同时仍须满足尺寸、配对足迹、排容量、箱高、航次和目的港等普适约束；边缘位置仍可安排满足约束的非 45 英尺箱。

## 求解器

仓库保留三套相互独立的求解器：

- `cg`：论文算法——完整航次方案列生成、航次内部箱区局部模式协调、严格定价认证，以及一次受限排级整数恢复；
- `direct`：同一数学模型的完整排位置 MILP，即对照基线 M0；
- `lbbd`：实验性替代算法——强箱区容量割、排级不混冲突图割，以及按箱区精确求解的逻辑 Benders 分解。它不调用 `cg`；仅在首轮没有完整上界时复用共享紧凑模型构造器生成一次短时 primal seed，并丢弃该模型的下界。

论文算法的外层一列表示“一个出口航次跨全部可行箱区的完整排级方案”。外层主问题只为每个航次选择一列，并统一协调共享排/贝容量、跨航次不混状态、进口预留和大计划偏差，因此凸性块数量等于航次数，而不是航次数乘箱区数。

每个单航次定价内部再按箱区组织局部整数模式。内层主问题协调各箱区模式以满足该航次全部箱组需求，箱区定价 MIP 负责排级容量、不混、足迹、堆栈和集中堆存状态。局部 MIP 下界先形成单航次有效定价下界，再用于修正外层主问题下界；当内层下界不足以严格认证时，完整航次排级 MIP 执行精确定价认证。算法不会把启发式下界当作全局证明。

根节点结束后只执行一次受限排级整数恢复。候选位置来自外层 LP 活跃航次方案、每个航次最近的八个方案，并按需补足每个箱组的可达容量。恢复模型严格执行完整数学模型，但只负责产生可行上界；其受限模型下界不参与全局证明。

## 业务目标

全部目标先按实例自然尺度直接归一化，再使用经验权重：

| 目标 | 权重 |
|---|---:|
| 减少箱组跨箱区分散 | 0.290 |
| 减少箱区内部排级分散 | 0.240 |
| 靠近已有同类箱 | 0.070 |
| 减少大计划箱区偏差 | 0.270 |
| 减少按分配箱量加权的泊位—箱区距离 | 0.130 |

出口需求平衡、容量、贝位尺寸、配对足迹、45 英尺边缘大贝、同排箱高不混、航次不混、目的港不混及配置的其他属性规则均为硬约束。泊位—箱区距离缺失时直接报错。

## 运行

```bash
python -m pip install -r requirements.txt
python -X utf8 -B run_yard_plan.py --solver cg --run-name voyage_plan_cg
python -X utf8 -B run_yard_plan.py --solver direct --run-name m0_direct
python -X utf8 -B run_yard_plan.py --solver lbbd --run-name strengthened_lbbd
```

主要参数：

- `--total-time-limit`、`--solver-threads`：总时限和线程数；
- `--mip-gap`：M0 的停止 gap；CG 的受限整数恢复仍请求零 gap，但受总时限约束；
- `--max-pricing-iterations`：外层和内层列生成的最大迭代数；
- `--plans-per-pricing`：一次定价最多返回的候选方案/局部模式数。

生成并运行多样化 3 倍航次压力算例：

```bash
python -X utf8 -B example/generate_diverse_voyages_case.py --copies 3 --overwrite
python -X utf8 -B benchmark_voyage_plans.py --input example/diverse_voyages_3x/input_data.json --large-plan example/diverse_voyages_3x/large_plan.csv --total-time-limit 120 --solver-threads 1
python -X utf8 -B benchmark_logic_benders.py --input example/diverse_voyages_3x/input_data.json --large-plan example/diverse_voyages_3x/large_plan.csv --total-time-limit 120 --solver-threads 1 --compare-direct
```

## 输出

- `export_row_plan.csv`：出口资料箱排级分配；
- `import_capacity_reservation.csv`：匿名进口容量预留；
- `bay_summary.csv`：由排级结果聚合的贝位汇总；
- `selected_row_locations.csv`：受限排级整数恢复得到的排位置；
- `declared_export_demand.csv`：输入的出口资料箱需求；
- `diagnostics.json`、`run_summary.json`：上下界、gap、定价和运行时间；
- `output_validation.json`：独立硬约束复核结果。

## 代码边界

- `yard_planning/planner.py`：两套求解器共享的数据索引、业务规则、目标与输出校验；
- `yard_planning/voyage_plan_column_generation.py`：完整航次外层列生成、内层局部模式定价、严格认证和整数恢复；
- `yard_planning/direct_milp.py`：M0 紧凑排位置模型；
- `yard_planning/gurobi_backend.py`：统一 Gurobi 接口；
- `adapters/planning_input.py`：业务输入清洗与模型数据构造；
- `yard_planning/output_validator.py`：独立于求解模型的结果复核。

数学模型与算法边界见 [MODEL_SCOPE.md](MODEL_SCOPE.md)，多样化算例生成规则见 [example/diverse_voyages_3x/README.md](example/diverse_voyages_3x/README.md)，当前对照结果见 [example/diverse_voyages_3x/benchmark_120s.md](example/diverse_voyages_3x/benchmark_120s.md)。
