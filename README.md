# 出口资料箱排级堆场分配模型

本分支是面向 TRE 论文场景整理后的模型。详细决策只针对已申报但尚未入场的出口资料箱；进口航次仅形成容量预留，不参与具体贝位或排分配。出口预测余量不分配、也不预留。

大计划长表只读取 `new_qty`。进口 `new_qty` 用于容量预留；出口 `new_qty` 仅归一化为资料箱的箱区比例参考。`planned_qty` 包含在场箱，因此不作为本模型的需求、目标总量或预留量。完整边界见 [MODEL_SCOPE.md](MODEL_SCOPE.md)。

大计划箱型字段必须明确为 `20` 或 `40`，其中大计划口径的 `40` 包含真实40英尺和45英尺箱。`ALL`、空值、`45` 及其他未知值会直接触发输入错误。

目标采用两阶段词典序：先最小化未分配资料箱，再固定该最小值优化大计划转移箱量、箱组分散、同类箱邻近、按箱量加权的泊位距离和大箱配对损失。进口大箱预留与大箱配对损失共用同一配对状态。

参与详细分配的出口航次必须具有泊位映射，所有候选箱区必须具有对应的正有限距离；缺失或非法距离直接报输入错误，不做填补。

## 代码结构

- `run_medium_small.py`：运行入口（文件名暂保留，避免破坏已有调用脚本）。
- `medium_small/column_generation_planner.py`：出口资料箱排级分配的列生成求解器。
- `block_bay_planning/models.py`：核心数据结构。
- `adapters/input_adapter_gd.py`：原始 JSON 输入对象。
- `adapters/input_adapter_standard.py`：构造出口详细需求、容量和聚合预留。
- `example/`：完整示例输入及上游大计划结果。

## 运行

```bash
python -m pip install -r requirements.txt
python -X utf8 -B run_medium_small.py --run-name example_full
```

默认使用 Gurobi，需要可用的 Gurobi 许可证；可用 `--no-gurobi` 运行内置启发式回退流程。

主要输出：

- `export_row_plan.csv`：出口资料箱排级分配结果；
- `area_bay_summary.csv`：由排级结果聚合得到的箱区—贝位汇总；
- `declared_export_demand.csv`：出口资料箱需求；
- `unplaced_boxes.csv`：未分配量；
- `generated_columns.csv`：生成的可行列；
- `diagnostics.json`、`run_summary.json`：求解与一致性诊断。

可用 `--total-time-limit`、`--mip-time-limit` 调整求解时间，用 `--voyages` 选择调试航次。
