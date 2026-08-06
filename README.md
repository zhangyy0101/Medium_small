# 出口资料箱排级堆场规划

本项目实现面向 TRE 论文的排级堆场规划模型。详细位置分配只针对已申报、尚未入场的出口资料箱；所有出口资料箱必须完成分配。出口预测箱不参与本模型，也不形成额外预留。

进口航次不参与具体排级分配。模型只读取大计划 `new_qty`，为即将卸船的进口箱建立匿名容量预留；`planned_qty` 已包含在场箱，不能作为新增需求。进口预留仅服从箱区功能、贝位尺寸和物理容量约束，不继承出口箱的航次、目的港、箱高等不混约束。

大计划箱型必须为 `20` 或 `40`；其中 `40` 表示物理 40/45 英尺大箱类别。`ALL`、空值、`45` 或其他未知值会触发输入错误。45 英尺出口箱只能放在箱区首尾的可用边缘大贝，同时仍须满足尺寸、配对足迹、排容量、箱高、航次和目的港等普适约束；边缘位置仍可安排满足约束的非 45 英尺箱。

## 求解器

仓库只保留两套求解器：

- `cg`：论文算法——嵌套箱区配置 Branch-and-Price；
- `direct`：同一数学模型的完整排位置 MILP，即对照基线 M0。

论文算法的一列表示一个箱区的完整整数配置。受限主问题为每个箱区选择一个配置，并在全局层面满足出口箱组需求、进口预留总量和大计划偏差关系。

箱区定价根据实例结构自动选择：

- 候选位置较少或只有一个物理连通块时，直接求解完整箱区定价 MIP；
- 候选位置较多且含多个物理块时，启用嵌套精确定价。每个物理块用整数 MIP 生成局部配置，持久化协调 LP 负责跨块箱组数量和一次性的箱组—箱区启用成本；
- 对物理数据、候选语义和约束结构完全同构的块，只求解一个代表定价 MIP，并把小型解池严格映射到同构块；出现排级或贝位级分支时自动停用共享；
- 协调 LP 值加各块负约化成本下界修正构成严格有效的完整箱区定价下界；协调整数模型只用于寻找可行负列；无法完成认证时回退到完整箱区 MIP。

选择性定价轮次只搜索上一轮仍有生产力的箱区，默认每两轮进行一次全部箱区的精确定价。选择性轮次不能更新全局下界或宣告收敛；必要时会立即执行完整认证。

根节点排级重组只负责产生可行上界。它使用当前主问题中取正值的箱区配置、每个箱区最近两个外层配置及每个嵌套物理块最近两个配置所暴露的排位置，不参与下界证明。根节点闭合后，算法依次采用箱组—箱区数量、箱组—排数量、排使用状态和进口预留数量分支。

## 业务目标

全部目标先按实例自然尺度直接归一化，再使用经验权重：

| 目标 | 权重 |
|---|---:|
| 减少箱组跨箱区分散 | 0.290 |
| 减少箱区内部排级分散 | 0.240 |
| 靠近已有同类箱 | 0.070 |
| 减少大计划箱区偏差 | 0.270 |
| 减少按分配箱量加权的泊位—箱区距离 | 0.130 |

出口需求平衡、容量、贝位尺寸、配对足迹、45 英尺边缘大贝、同排箱高不混、航次不混、目的港不混和配置的其他属性规则均为硬约束。

## 运行

```bash
python -m pip install -r requirements.txt
python -X utf8 -B run_yard_plan.py --solver cg --run-name nested_bp
python -X utf8 -B run_yard_plan.py --solver direct --run-name m0_direct
```

论文算法常用参数：

- `--total-time-limit`、`--mip-gap`、`--solver-threads`：总时限、全局相对 gap 和线程数；
- `--max-pricing-iterations`、`--max-branch-nodes`：单节点最大定价轮数和最大分支节点数；
- `--full-pricing-frequency`：完整箱区定价认证频率；
- `--direct-candidate-limit`：直接定价与嵌套定价的结构阈值；
- `--nested-max-iterations`、`--nested-time-fraction`：内层块配置生成轮数和箱区定价时间占比；
- `--complex-area-pool-size`、`--simple-area-pool-size`：每轮复杂/简单箱区最多加入的配置数。

压力算例可以按当前输入格式生成：

```bash
python -X utf8 -B example/generate_more_voyages_case.py --copies 3 --overwrite
python -X utf8 -B benchmark_area_configuration.py --algorithm branch-price --total-time-limit 120 --solver-threads 1
```

## 输出

- `export_row_plan.csv`：出口资料箱排级分配；
- `import_capacity_reservation.csv`：匿名进口容量预留；
- `bay_summary.csv`：由排级结果聚合的贝位汇总；
- `selected_row_locations.csv`：最终整数箱区配置展开后的排位置；
- `declared_export_demand.csv`：输入的出口资料箱需求；
- `diagnostics.json`、`run_summary.json`：上下界、gap、定价和运行时间统计；
- `output_validation.json`：独立硬约束复核结果。

## 代码边界

- `yard_planning/planner.py`：两套求解器共享的数据索引、业务规则、目标和输出校验；
- `yard_planning/area_configuration.py`：箱区配置主问题、直接定价和嵌套块定价；
- `yard_planning/area_branch_price.py`：论文算法的节点求解、分支树和排级上界；
- `yard_planning/direct_milp.py`：M0 紧凑排位置模型；
- `yard_planning/gurobi_backend.py`：统一 Gurobi 接口；
- `adapters/planning_input.py`：业务输入清洗与模型数据构造；
- `yard_planning/output_validator.py`：与求解模型独立的结果复核。

数学模型范围见 [MODEL_SCOPE.md](MODEL_SCOPE.md)，当前 3 倍航次算例结果见 [example/more_voyages_3x/benchmark_120s.md](example/more_voyages_3x/benchmark_120s.md)。
