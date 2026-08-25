# 预实验数据生成与校验

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

已有 `preexperiment_outputs` 中的结果来自旧的大计划引导目标，仅保留作开发记录，不能作为当前集成模型的实验结论。新实验必须使用本目录 schema v3 配方重新生成。

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
