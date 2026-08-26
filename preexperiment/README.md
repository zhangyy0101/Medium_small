# 预实验数据生成与校验

> V6 边界：默认 `python -m preexperiment` 仍是输入校验/冻结 V5 runner；
> V6 使用独立的 `benchmark_v6_scalability.py`，不会调用 V5 planner。
> 当前模型契约为 V6.1；下列 2026-08-26 报告由旧 V6.0 历史状态兼容规则生成，
> 只可用于实现追溯，必须在 V6.1 上复跑后才能作为当前 LB/UB/gap 证据。
> V6 已把生产峰值流程改为解析 cap＋compact feasibility witness；旧的
> exact-`rho*` smoke 只保留为诊断。24/48/96 单种子复测已经确认 peak
> 准备不再是瓶颈；根节点重写后 24/48/96 的 exact root closure gate 均已
> 在 60 秒内通过。下一 gate 是恢复合理的 primal/RIM 预算并验证 UB 与 gap。

## V6 分阶段 scalability runner

下列命令依次运行解析 cap 与可行性认证、exact root CG、compact primal coverage
和 final RIM，并在每个阶段后写入 checkpoint。任何必需证明超时都会立即
停止，不会继续产生口径不合法的 LB/UB：

```bash
python benchmark_v6_scalability.py \
  --suite preexperiment/scale_suite.json \
  --cases scale_g048_s601 \
  --output-root preexperiment_outputs/v6_scalability_g048 \
  --peak-feasibility-time-limit 10 \
  --root-total-time-limit 60 \
  --pricing-time-limit 10 \
  --coverage-time-limit 30 \
  --integer-time-limit 30 \
  --solver-threads 1 \
  --solver-seed 0
```

结果包含 feasibility/compact MIP 与 final RIM 的 incumbent/bound trajectory、root CG
逐轮 reduced cost、列池规模、LB/UB 和三类目标实际贡献。旧体检见
`preexperiment/reports/v6/scalability_smoke_20260826/`，根节点重写结果见
`preexperiment/reports/v6/root_rewrite_20260826/`，恢复 30 秒 primal/RIM 预算及
RIM warm-start 修复结果见
`preexperiment/reports/v6/full_pipeline_30s_20260826/`。

只测 exact root 时增加 `--root-only`；runner 会在根闭合后把算例标记为成功，
不再用极短的 primal/RIM 预算制造无关的全流程失败状态。

这里保存的是可复现的“场景配方”，而不是重复六份约 178 MB 的堆场快照。每个场景由经过 SHA-256 固定的基础输入、参数和随机种子唯一确定；运行时只物化一个 `InputAdapterGd` 对象并直接交给集成论文模型，不再生成或读取大计划 CSV。

## Pilot v1

`pilot_suite.json` 包含小、中、大三档，每档两个种子，共六个算例。当前控制维度包括：

- 出口航次数；
- 每航次箱组数；
- 每航次资料箱量；
- 进口资料箱抽样比例。

堆场快照、箱区功能、封场信息、泊位和距离均继承基础业务数据。场景只生成已知资料箱，不生成预测箱；TOPS 不进入物化后的输入模式。

## 分层校验

默认命令会对六个算例执行：

1. 按随机种子生成出口资料箱和进口抽样；
2. 检查箱号唯一性、航次一致性和 manifest 数量；
3. 直接从资料箱构建出口详细需求和进口匿名预留需求；
4. 核对出口箱组数、出口资料箱量和扣除已在场箱后的进口资料箱量；
5. 可选运行论文算法，并对排级输出与匿名进口预留执行内部、外部双重校验。

```bash
python -m preexperiment \
  --suite preexperiment/pilot_suite.json \
  --output-root preexperiment_outputs/pilot_v1 \
  --threads 1
```

先选择一个中型算例执行完整论文算法和双重独立输出校验：

```bash
python -m preexperiment \
  --cases pilot_m_201 \
  --paper-time-limit 30 \
  --threads 1
```

输出目录包含每个算例的 `generation_manifest.json` 和 `validation.json`；运行完整论文算法时还会产生出口排级结果和匿名进口容量预留 CSV。汇总结果位于 `suite_summary.csv/json`。

## 算法消融与正确性对照

在已经完成 60 秒完整算法 pilot 后，运行下列命令可用相同输入、随机种子、线程数和总时间预算比较“完整算法”与“关闭 F&O”，并对两个小算例执行全枚举 LP/MIP 同模型校验：

```bash
python -m preexperiment.compare \
  --full-results-root preexperiment_outputs/pilot_v1_full_60s \
  --output-root preexperiment_outputs/pilot_v1_comparison_60s \
  --method-time-limit 60 \
  --correctness-time-limit 120 \
  --threads 1
```

结果汇总在 `comparison_summary.csv/json` 和 `comparison_report.md`。F&O improvement 为正表示完整算法取得了更小的目标值；全枚举 LP/MIP 仅与论文的连续箱区模型比较，不与目标定义不同的旧模型混为一谈。

已有 `preexperiment_outputs` 中采用旧目标定义的结果仅保留作开发记录，不能作为当前集成模型的实验结论。新实验必须使用模型 schema `integrated_zone_v4` 重新生成。

未利用排区容量目标消融使用 `--disable-unused-capacity-objective`。该开关把对应权重置零，并按比例重新归一化其余目标；应与默认目标使用相同场景、seed、线程和时间预算，比较未利用容量、排区数量、跨区、邻近和泊位距离等原始 KPI，而不是直接相减两套不同定义的目标值。

在中大型算例上运行同模型的全枚举完整排区 MIP：

```bash
python -m preexperiment.complete_mip \
  --full-results-root preexperiment_outputs/pilot_v1_full_60s \
  --paired-inputs-root preexperiment_outputs/pilot_v1_comparison_60s \
  --output-root preexperiment_outputs/pilot_v1_complete_mip_60s \
  --time-limit 60 \
  --threads 1
```

公平对比为双方都预留 5% 总时间做精确排级 recourse，并对最终 CSV 执行内部和外部独立校验。需求口径调整后应使用新的输出目录重新运行，不能与旧预测场景的 UB 直接混用。
