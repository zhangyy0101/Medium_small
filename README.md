# 1.5 中小计划独立精简版

本目录只保留 `flat_full_yard_plan_1.5_scip` 中运行“中计划 + 小计划”所需的核心代码和一个完整算例。它不包含大计划求解器、完整大中小流程、API、可视化、网页代码，也不包含单独小计划入口。

## 目录内容

- `run_medium_small.py`：唯一运行入口。
- `medium_small/column_generation_planner.py`：SCIP 列生成中小计划核心算法。
- `block_bay_planning/models.py`：核心数据模型。
- `adapters/input_adapter_gd.py`：JSON 输入对象。
- `adapters/input_adapter_standard.py`：从完整输入和大计划构建中小计划问题所需的数据处理与约束构建。
- `example/input_data.json`：1.5 的完整示例输入，未裁剪。
- `example/large_plan.csv`：原大计划算法对该算例真实运行后生成的完整 `allocation.csv`。
- `example/large_plan_diagnostics.json`：上述大计划运行的诊断信息，用于说明算例来源。

## 安装和运行

```bash
python -m pip install -r requirements.txt
python -X utf8 -B run_medium_small.py --run-name example_full
```

默认会读取 `example/input_data.json` 和 `example/large_plan.csv`，并对大计划中的全部航次运行中小计划。结果写入 `outputs/example_full/`。

主要结果为：

- `medium_plan.csv`
- `small_plan.csv`
- `unplaced_boxes.csv`
- `generated_columns.csv`
- `medium_demand_by_port.csv`
- `diagnostics.json`
- `run_summary.json`

可用 `--total-time-limit` 和 `--mip-time-limit` 调整求解时间；可用 `--voyages 航次1 航次2` 仅做调试，但完整算例的默认命令不筛选航次。
