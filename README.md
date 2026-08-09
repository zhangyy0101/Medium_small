# 出口资料箱排级堆场规划

本项目实现面向 TRE 论文的排级堆场规划模型。详细位置分配只针对已申报、尚未入场的出口资料箱，且所有出口资料箱必须完成分配；出口预测箱不参与模型，也不形成额外预留。

进口航次不参与具体排级分配。模型只读取大计划的 `new_qty`，为即将卸船的进口箱建立匿名容量预留；`planned_qty` 已包含在场箱，不能作为新增需求。进口预留仅服从箱区功能、贝位尺寸和共享物理容量，不继承出口箱的航次、目的港、箱高等不混约束。

大计划箱型必须为 `20` 或 `40`，其中 `40` 表示物理 40/45 英尺大箱类别；`ALL`、空值、`45` 和未知值会触发输入错误。45 英尺出口箱只能放在箱区首尾的可用边缘大贝，同时仍须满足尺寸、配对足迹、排容量、箱高、航次和目的港等普适约束；边缘位置仍可安排满足约束的非 45 英尺箱。

## 求解器

仓库保留五套相互独立的求解器：

- `cg`：论文算法——完整航次方案列生成、航次内部箱区局部模式协调、严格定价认证，以及一次受限排级整数恢复；
- `direct`：同一数学模型的完整排位置 MILP，即对照基线 M0；
- `lbbd`：实验性替代算法——主问题以“箱组—箱区整数箱量 + 航次—排足迹—行不混类别二元状态 + 连续候选排流”协调进口预留、物理排冲突和业务目标；每个出口航次由一个持久化精确排级子问题独立验证并恢复整数落位，必要时生成 Hall 容量割、IIS 核逻辑可行性割和条件最优性割。算法不调用 M0 修复，也没有备用求解链。
- `lbbd_profile`：独立的严格资源类型聚合 LBBD——把容量、尺寸能力、20/40/45 英尺足迹兼容性相同的具体排聚合为整数资源类型；主问题使用部分重叠足迹池的 Hall 型容量界和资源类型—不可混类容量约束。候选先经全局物理足迹匹配和精确航次子问题快速验证：匹配被证明不可行时生成打包割或条件匹配可行性割；匹配可行但航次解聚失败、或精确排代价高于 `theta` 时，按需调用联合选择全部具体足迹与排落位的精确 oracle，并生成 IIS 核条件可行性割或条件最优性割后重解主问题。只有主问题最优且精确 recourse 已认证时才报告收敛。初始化骨架只有通过精确解聚才可作为 MIP Start；该版本不覆盖或调用 `lbbd`、`cg`、`direct`。
- `lbbd_selective`：与上述求解器并列的冲突引导自适应资源状态 LBBD。箱组—贝位数量保持整数，箱组—箱区激活使用候选容量上界收紧的 Big-M 和最少启用箱区覆盖割；初始整数核心以航次均衡的稀缺性评分为主体，并加入规模为冲突超边数平方根量级的物理冲突种子。初始化仅临时放松排数以取得候选数量点，随后立即恢复正式主问题类型；联合精确排级 oracle 只建立当前正数量支持集上的等价稀疏模型。正式主问题点若被 IIS 证明不可行，先单向、小批量提升 IIS 邻域中的 profile 和排计数整数状态；无状态可提升时，才生成类型化物理容量割或紧凑单调 IIS 割。可行 recourse 对低估的航次排级代价生成局部数量条件最优性割。冲突修复与航次均衡排级邻域作为独立上界助手，不参与下界证明。全部阶段共享一个总截止时间，按总时限比例和实时剩余量动态分配，未使用时间自动回流。该入口不调用 M0、`lbbd_profile` 或备用求解链。

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
python -X utf8 -B run_yard_plan.py --solver lbbd_profile --run-name profile_lbbd
python -X utf8 -B run_yard_plan.py --solver lbbd_selective --run-name selective_lbbd
```

主要参数：

- `--total-time-limit`、`--solver-threads`：总时限和线程数；`lbbd_selective` 只使用该总时限，主问题、稀疏 oracle、可行修复和邻域额度均按比例及实时剩余量自适应分配；
- `--mip-gap`：M0 的停止 gap；CG 的受限整数恢复仍请求零 gap，但受总时限约束；
- `--max-pricing-iterations`：外层和内层列生成的最大迭代数；
- `--plans-per-pricing`：一次定价最多返回的候选方案/局部模式数；
- `--lbbd-max-iterations`：`lbbd`、`lbbd_profile` 和 `lbbd_selective` 的主问题—逻辑割最大迭代轮数；
- `--lbbd-master-feasibility-time-limit`：`lbbd` 与 `lbbd_profile` 的初始可行阶段上限；`lbbd_selective` 不使用固定阶段秒数；
- `--lbbd-master-time-limit`：`lbbd` 与 `lbbd_profile` 生成新割后的主问题重优化上限；
- `--lbbd-voyage-time-limit`：`lbbd` 与 `lbbd_profile` 的单航次子问题上限。

生成并运行多样化 3 倍航次压力算例：

```bash
python -X utf8 -B example/generate_diverse_voyages_case.py --copies 3 --overwrite
python -X utf8 -B benchmark_voyage_plans.py --input example/diverse_voyages_3x/input_data.json --large-plan example/diverse_voyages_3x/large_plan.csv --total-time-limit 120 --solver-threads 1
python -X utf8 -B benchmark_logic_benders.py --input example/diverse_voyages_3x/input_data.json --large-plan example/diverse_voyages_3x/large_plan.csv --total-time-limit 120 --solver-threads 1 --compare-direct
python -X utf8 -B benchmark_profile_benders.py --input example/many_groups_6v_12g/input_data.json --large-plan example/many_groups_6v_12g/large_plan.csv --total-time-limit 120 --solver-threads 1
python -X utf8 -B benchmark_selective_benders.py --input example/many_groups_6v_12g/input_data.json --large-plan example/many_groups_6v_12g/large_plan.csv --total-time-limit 120 --solver-threads 1
python -X utf8 -B example/generate_natural_conflict_case.py --overwrite
python -X utf8 -B benchmark_selective_benders.py --input example/natural_conflict_peak/input_data.json --large-plan example/natural_conflict_peak/large_plan.csv --total-time-limit 90 --solver-threads 1 --output outputs/selective_natural_conflict_peak_90s.json
python -X utf8 -B benchmark_row_configurations.py --root-only --compare-m0-lp --input example/many_groups_6v_12g/input_data.json --large-plan example/many_groups_6v_12g/large_plan.csv --total-time-limit 120 --solver-threads 1
python -X utf8 -B benchmark_template_flow_benders.py --total-time-limit 30 --solver-threads 1 --compare-template-direct --output outputs/template_flow_base_30s.json
python -X utf8 -B benchmark_template_flow_benders.py --input example/many_groups_6v_12g/input_data.json --large-plan example/many_groups_6v_12g/large_plan.csv --total-time-limit 120 --solver-threads 1 --compare-template-direct
```

`benchmark_row_configurations.py` is an isolated research prototype for
physical-row-track Dantzig--Wolfe decomposition. It implements exact root
pricing, a Lagrangian lower bound, integer-friendly group consolidation, and
the restricted configuration MIP. The Branch-and-Price tree is deliberately
not implemented because the root-LP and restricted-MIP stage gates were not
met; the prototype does not alter any production solver entry point.

`benchmark_template_flow_benders.py` is a second isolated experiment.  Its
master selects hard no-mix row templates, integer capacity on each selected
template, the associated integer stack use, and anonymous import capacity.
Its persistent recourse LP assigns declared export quantities and produces
classical feasibility/optimality cuts.  Before root separation, the same
integer master is allowed to find one feasible template seed using a fixed
fraction of the post-build remaining budget; `SolutionLimit=1` makes this
stage stop immediately on easy instances.  The
large-plan shares are apportioned to integral declared-box area targets so
that an integer template capacity yields an integral transportation flow.
`--compare-template-direct` solves the same redefined model compactly; this is
the scientifically valid direct reference, while `--compare-direct` retains
the original M0 comparison.  Neither solver is registered in `run_yard_plan`.

`benchmark_contiguous_zones.py` is an independent two-level experiment. Its
primary master sends the actual declared quantity of each export group through
dedicated contiguous row zones. Exact prefix/RMQ top-K pricing avoids scanning
all intervals; adaptive batches accelerate degenerate rounds. Capacity-cover
and zone-support inequalities strengthen the root relaxation. An integer
restricted master uses a square-root adaptive enrichment pool. Its proportional
Branch-and-Price probe shares generated columns and returns time immediately
when its first node cannot close; rebuilt child LPs inherit the parent
primal/dual start. The selected
group-bay export flows and anonymous import reservation are then fixed in a
compact row recourse model. A constructive capacity certificate proves that
the chosen disjoint zones can realize every export flow, and the solver also
reconstructs the primary objective independently before validating the final
row plan.
This experiment deliberately changes the paper model: a selected zone reserves
the full compatible capacity of its physical rows, so its objective and
feasible set are not those of M0. It is not registered in `run_yard_plan` and
does not modify any existing solver entry point.

```bash
python -X utf8 -B benchmark_contiguous_zones.py --root-only --compare-complete-zone-lp --total-time-limit 60 --solver-threads 1
python -X utf8 -B benchmark_contiguous_zones.py --input example/many_groups_6v_12g/input_data.json --large-plan example/many_groups_6v_12g/large_plan.csv --total-time-limit 120 --solver-threads 1
python -X utf8 -B benchmark_contiguous_zones.py --complete-zone-mip-only --input example/many_groups_6v_12g/input_data.json --large-plan example/many_groups_6v_12g/large_plan.csv --total-time-limit 120 --solver-threads 1
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
- `yard_planning/voyage_resource_benders.py`：航次—排资源 LBBD 主问题、航次子问题与逻辑割；
- `yard_planning/profile_resource_benders.py`：排资源类型聚合主问题、共享物理资源池、快速足迹解聚、全局精确反聚合子问题及条件逻辑割；
- `yard_planning/selective_resource_benders.py`：冲突超图初始化、IIS 驱动单向状态提升、选择性整数资源状态与排数主问题、联合精确排级 recourse、物理容量证书与单调 IIS 后备割、条件最优性割，以及独立的冲突修复与受限排级邻域上界改进；
- `yard_planning/row_configuration_generation.py`：独立的物理排轨道配置列生成实验，包括精确定价、Lagrangian 根节点下界、集中配置初始列和整数受限主问题；
- `yard_planning/template_flow_benders.py`：独立的硬属性排模板—整数容量主问题、连续箱量流子问题、Phase-I/对偶 Benders 割及同模型紧凑直接对照；
- `yard_planning/contiguous_zone_generation.py`：独立的连续区段根节点列生成、整数区段支持选择和受限排级精确填充实验；
- `yard_planning/logic_benders.py`：LBBD 的稳定公共导入入口；
- `yard_planning/direct_milp.py`：M0 紧凑排位置模型；
- `yard_planning/gurobi_backend.py`：统一 Gurobi 接口；
- `adapters/planning_input.py`：业务输入清洗与模型数据构造；
- `yard_planning/output_validator.py`：独立于求解模型的结果复核。

数学模型与算法边界见 [MODEL_SCOPE.md](MODEL_SCOPE.md)，多样化算例生成规则见 [example/diverse_voyages_3x/README.md](example/diverse_voyages_3x/README.md)，自然高峰冲突算例见 [example/natural_conflict_peak/README.md](example/natural_conflict_peak/README.md)，当前对照结果见 [example/diverse_voyages_3x/benchmark_120s.md](example/diverse_voyages_3x/benchmark_120s.md)。
