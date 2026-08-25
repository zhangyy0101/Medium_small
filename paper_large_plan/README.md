# Paper large plan

该目录是为论文实验提取的独立大计划。原目录 `flat_full_yard_plan_1.5_scip/` 仅作业务参考，不被导入，也不在其中修改代码。

## 范围

- 与论文算法读取同一份 `InputAdapterGd` JSON；
- 采用确定性已知箱口径：只统计堆场现箱和已申报资料箱，并按箱号去重；
- 输入中即使保留 `predict_cntrs` 或旧 `cntr_volume` 字段，大计划也完全忽略；仅有预测而没有资料箱的航次不会进入计划；
- 进口需求来自现箱和单证箱，现箱固定在当前箱区；
- 保留箱区功能、封场、20/40 尺寸能力、共享物理容量和泊位距离；
- 依次最小化缺量、航次跨箱区分散、最高新增容量利用率和泊位距离；
- 不读取也不预留 TOPS plan；
- 不含旧版人工指定、历史滚动修正、港口白名单、进口聚类、作业路限制、泊位冲突、中小计划、接口和可视化代码；
- 求解器只使用 Gurobi，不含 SCIP/PySCIPOpt 依赖。

## 输入与输出

输入是论文算法使用的同一个 `input_data.json`。输出 `large_plan.csv` 包含：

`voy_id, flow, area_no, size, planned_qty, snapshot_qty, new_qty`

其中论文算法把出口 `new_qty` 当作已知资料箱的箱区分布引导，把进口 `new_qty` 当作匿名容量预留。上下游使用同一已知箱需求口径。

## 运行

```bash
python -m paper_large_plan \
  --input example/input_data.json \
  --output paper_large_plan_outputs/large_plan.csv \
  --time-limit 120 \
  --threads 1 \
  --seed 0
```

同目录还会生成 `large_plan_diagnostics.json`，记录需求口径、目标值、缺量和求解状态。若缺量不为 0，应先检查输入容量和功能配置，不应直接进入论文算法实验。
