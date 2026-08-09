# 出口资料箱连续排区规划

这是当前论文模型与算法的最小化代码库。模型只详细分配已申报、尚未入场的出口资料箱，并要求全部完成分配；出口预测箱不进入模型。进口航次不做具体排级落位，只根据大计划 `new_qty` 形成匿名容量预留。`planned_qty` 已包含在场箱，不作为新增需求读取。

当前论文算法把同一出口箱组在同一箱区、同一物理排上的连续贝位集合定义为一个专用排区（zone）。选中排区会保留其完整兼容容量，实际箱量在所选排区内流动；最终通过精确排级子问题恢复具体分配并独立校验。

## 模型边界

- 出口需求：仅资料箱，必须全部分配；
- 进口需求：仅按 `new_qty` 预留匿名容量，服从箱区功能、贝位尺寸和共享物理容量；
- 大计划箱型只接受 `20`、`40`，其中 `40` 表示物理大箱类别；`ALL`、空值、`45` 和未知值直接报错；
- 45 英尺出口箱只能进入箱区首尾的可用边缘大贝，但不会排斥满足普适约束的非 45 英尺箱；
- 泊位—箱区距离必须完整，缺失即报错；
- 航次、目的港、箱高等不混规则只施加于出口排级分配，不施加于匿名进口预留。

主要硬约束包括完整需求平衡、箱区功能、贝位和排容量、尺寸容量、40/45 英尺配对足迹、堆栈资源、45 英尺边缘大贝、航次不混、目的港不混、同排箱高不混及配置的其他不可混属性。

## 目标函数

六个业务目标按实例自然尺度归一化后加权求和：

| 目标 | 权重 |
|---|---:|
| 减少箱组跨箱区分散 | 0.25 |
| 减少不连续排区数量 | 0.22 |
| 靠近已有同类箱 | 0.08 |
| 减少出口分配与进口预留的大计划偏差 | 0.22 |
| 减少排区未利用的保留容量 | 0.13 |
| 减少按实际分配箱量加权的泊位—箱区距离 | 0.10 |

首个必需箱区和首个必需排区不计入分散惩罚。排级精确回填只承担可行性认证与次级质量诊断，不改变上述主目标、上下界或 gap。

## 算法流程

1. 构建原子排位置与连续候选排区；单个排区容量不超过箱组需求加该物理排上的一个最大原子排容量。
2. 在根节点执行严格列生成。精确 prefix/RMQ top-k 定价隐式搜索全部合法连续区间，直到不存在负检验数列，得到完整排区 LP 的全局下界。
3. 对列池自适应扩充并求解受限整数主问题，得到排区选择和实际箱量流。
4. 按当前解的目标贡献选择箱组邻域：默认覆盖至少 60% 可归因目标，同时将候选排区规模控制在全体的 35% 左右；对邻域执行 Fix-and-Optimize。
5. 固定箱组—贝位流和进口预留，求解精确排级 recourse；随后重构主目标并独立复核全部硬约束。

同模型的完整排区 MIP 可用于小规模校验。`DirectMilpPlanner` 是排级 recourse 的共享实现，也可作为不同模型的 M0 结构参考；其目标值不能与连续排区模型直接相减。

## 安装与运行

需要 Python 3.10+ 和可用的 Gurobi 许可证：

```bash
python -m pip install -r requirements.txt
python -X utf8 -B benchmark_contiguous_zones.py --total-time-limit 60 --solver-threads 1
```

严格根节点与完整 LP 自动对照：

```bash
python -X utf8 -B benchmark_contiguous_zones.py --root-only --compare-complete-zone-lp --total-time-limit 60 --solver-threads 1
```

72 箱组压力算例：

```bash
python -X utf8 -B benchmark_contiguous_zones.py --input example/many_groups_6v_12g/input_data.json --large-plan example/many_groups_6v_12g/large_plan.csv --total-time-limit 120 --solver-threads 1
```

同模型完整排区 MIP：

```bash
python -X utf8 -B benchmark_contiguous_zones.py --complete-zone-mip-only --input example/many_groups_6v_12g/input_data.json --large-plan example/many_groups_6v_12g/large_plan.csv --total-time-limit 120 --solver-threads 1
```

使用 `--output PATH` 保存 JSON 结果；`--compare-row-m0` 仅增加不同模型的排级 M0 结构参考。

## 当前基准结果

在一线程、统一求解器时限下：

| 算例 | 时限 | 全局下界 | 最终上界 | gap |
|---|---:|---:|---:|---:|
| 基础算例（9 组） | 60 s | 0.14508715 | 0.15182511 | 4.44% |
| 72 箱组算例 | 120 s | 0.17924039 | 0.19188545 | 6.59% |

72 箱组算例中，自适应邻域把受限主问题上界从 `0.19938177` 改善至 `0.19188545`，改善 3.76%；全部 2,241 个出口箱和 624 个进口预留箱均通过精确 recourse 认证。

## 代码结构

- `benchmark_contiguous_zones.py`：唯一算法入口与同模型对照入口；
- `yard_planning/contiguous_zone_generation.py`：严格定价、整数主问题、自适应 Fix-and-Optimize 和精确回填；
- `yard_planning/direct_milp.py`：共享排级紧凑模型与不同模型 M0 参考；
- `yard_planning/planner.py`：公共索引、业务规则、目标基础和结果构造；
- `adapters/planning_input.py`：输入清洗与模型数据构造；
- `yard_planning/output_validator.py`：独立结果校验；
- `example/many_groups_6v_12g`：72 箱组压力算例；
- `tests`：输入预处理、精确定价、完整 LP 对照、容量规则与 recourse 测试。

详细数学边界与上下界解释见 [MODEL_SCOPE.md](MODEL_SCOPE.md)。
