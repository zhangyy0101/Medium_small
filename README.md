# 出口资料箱连续排区规划

这是当前论文模型与算法的最小化代码库。论文主链已经改为集成模型，不再读取上游大计划文件。需求采用确定性已知箱口径：输入中的预测字段不进入规划需求，TOPS plan 也不进入模型。模型详细分配尚未入场的出口资料箱并要求全部完成分配；进口箱不做航次级、箱组级或排级落位，而是从进口资料箱中扣除已在场箱后，按流向和物理箱型形成匿名容量预留。

当前论文算法把同一出口箱组在同一箱区、同一物理排上的连续贝位集合定义为一个专用排区（zone）。选中排区会保留其完整兼容容量，实际箱量在所选排区内流动；最终通过精确排级子问题恢复具体分配并独立校验。

## 模型边界

- 出口需求：仅资料箱，必须全部分配；
- 进口需求：从进口资料箱直接汇总，预留匿名容量，服从箱区功能、贝位尺寸、共享物理容量和峰值利用率约束；
- 45 英尺出口箱只能进入箱区首尾的可用边缘大贝，但不会排斥满足普适约束的非 45 英尺箱；
- 泊位—箱区距离必须完整，缺失即报错；
- 航次、目的港、箱高等不混规则只施加于出口排级分配，不施加于匿名进口预留。

主要硬约束包括完整需求平衡、箱区功能、贝位和排容量、尺寸容量、40/45 英尺配对足迹、堆栈资源、45 英尺边缘大贝、航次不混、目的港不混、同排箱高不混及配置的其他不可混属性。

## 目标函数

六个出口业务目标按实例自然尺度归一化后加权求和：

| 目标 | 权重 |
|---|---:|
| 减少航次跨箱区分散 | 0.12 |
| 减少箱组跨箱区分散 | 0.20 |
| 减少不连续排区数量 | 0.28 |
| 靠近已有同类箱 | 0.10 |
| 减少出口排区未利用的保留容量 | 0.17 |
| 减少按出口分配箱量加权的泊位—箱区距离 | 0.13 |

每个航次的首个必需箱区、每个箱组的首个必需箱区和首个必需排区不计入分散惩罚。进口匿名预留不参与这六项目标。峰值利用率不作为额外目标，而是采用数据驱动的 ε 约束：先根据本算例负载与可达剩余容量计算下界，再按可配置余量生成上限；默认余量系数为 0.50。该参数不是码头给定阈值，正式实验必须做敏感性分析。排级精确回填只承担可行性认证与次级质量诊断，不改变主目标、上下界或 gap。

## 算法流程

1. 构建原子排位置与连续候选排区；单个排区容量不超过箱组需求加该物理排上的一个最大原子排容量。
2. 在根节点执行严格列生成。精确 prefix/RMQ top-k 定价隐式搜索全部合法连续区间，直到不存在负检验数列，得到完整排区 LP 的全局下界。
3. 对列池自适应扩充并求解受限整数主问题，得到排区选择和实际箱量流。
4. 按当前解的目标贡献选择箱组邻域：默认覆盖至少 60% 可归因目标，同时将候选排区规模控制在全体的 35% 左右；对邻域执行 Fix-and-Optimize。
5. 固定箱组—贝位流和进口预留，求解精确排级 recourse；随后重构主目标并独立复核全部硬约束。

同模型的完整排区 MIP 可用于小规模校验。`DirectMilpPlanner` 是排级 recourse 的共享实现，也可作为不同模型的 M0 结构参考；其目标值不能与连续排区模型直接相减。

## 安装与运行

当前环境使用 Python 3.13，并需要可用的 Gurobi 许可证：

```bash
python -m pip install -r requirements.txt
python -X utf8 -B benchmark_contiguous_zones.py --input example/input_data.json --total-time-limit 60 --solver-threads 1
```

峰值 ε 约束的默认余量系数为 `0.50`；可用 `--peak-utilization-headroom-fraction` 显式修改，并在正式实验中至少报告一组敏感性分析。

首批可复现预实验算例可用场景配方批量生成并分层校验：

```bash
python -m preexperiment --threads 1
python -m preexperiment --cases pilot_m_201 --paper-time-limit 30 --threads 1
```

具体数据维度、存储方式和校验层次见 [preexperiment/README.md](preexperiment/README.md)。

严格根节点与完整 LP 自动对照：

```bash
python -X utf8 -B benchmark_contiguous_zones.py --root-only --compare-complete-zone-lp --total-time-limit 60 --solver-threads 1
```

72 箱组压力算例：

```bash
python -X utf8 -B benchmark_contiguous_zones.py --input example/many_groups_6v_12g/input_data.json --total-time-limit 120 --solver-threads 1
```

同模型完整排区 MIP：

```bash
python -X utf8 -B benchmark_contiguous_zones.py --complete-zone-mip-only --input example/many_groups_6v_12g/input_data.json --total-time-limit 120 --solver-threads 1
```

使用 `--output PATH` 保存 JSON 结果；`--compare-row-m0` 仅增加不同模型的排级 M0 结构参考。

## 历史阶段门结果

此前数值使用“先大计划、再详细排区”的旧目标，只保留作开发追溯，不能与当前集成模型的 UB、LB 和 gap 比较。当前模型必须使用 schema v3 场景重新运行预实验、消融实验和完整排区 MIP 对照。

## 代码结构

- `benchmark_contiguous_zones.py`：唯一算法入口与同模型对照入口；
- `paper_large_plan`：保留的历史顺序式大计划参考实现，不进入当前论文算法主链；
- `preexperiment`：基于固定业务快照生成多种子场景，并自动校验集成论文算法输入/输出；
- `yard_planning/contiguous_zone_generation.py`：严格定价、整数主问题、自适应 Fix-and-Optimize 和精确回填；
- `yard_planning/direct_milp.py`：共享排级紧凑模型与不同模型 M0 参考；
- `yard_planning/planner.py`：公共索引、业务规则、目标基础和结果构造；
- `adapters/planning_input.py`：输入清洗与模型数据构造；
- `yard_planning/output_validator.py`：独立结果校验；
- `example/many_groups_6v_12g`：72 箱组压力算例；
- `tests`：输入预处理、精确定价、完整 LP 对照、容量规则与 recourse 测试。

详细数学边界与上下界解释见 [MODEL_SCOPE.md](MODEL_SCOPE.md)。
