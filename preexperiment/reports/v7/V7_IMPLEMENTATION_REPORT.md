# V7 Two-Stage Model Redesign — Implementation Report

## 1. Git

- base branch: `refactor/v7-model-redesign`
- base commit: `621a6eb1f0eecab2e989819ca8a304af937fd411`
- implementation branch: `refactor/v7-model-redesign`（按用户要求直接在当前分支修改，未另建分支）
- final commit: `NOT CREATED`（本轮未获提交授权；当前实现保留在工作树中）

## 2. Implemented model changes

- zone removed from V7 formal model: implemented。V7 的变量、可行域、目标、pricing column 和 evaluator 均不依赖 zone；连续 segment 只允许在结果展示层恢复。
- height no-mix retained: implemented。新出口分配在每个物理贝位只能选择一个箱高；历史混箱高按 V6.1 non-worsening 语义只允许复用一个已有箱高。
- size/footprint retained: implemented。20/40/45 尺寸、完整物理 footprint、45 英尺边缘大贝资格、anchor/physical capacity 均保留。
- legacy non-worsening retained: implemented。历史尺寸严格一致；历史 row 有箱时，新箱必须匹配一个已有的精确联合箱组键，不能把航次集合与卸货港集合任意组合。
- max 3 groups / physical bay implemented: implemented，并在 Complete MIP、global RMP、pattern validity 和独立 evaluator 中一致执行；40/45 英尺 footprint 覆盖到的每个物理贝位均计数。
- peak utilization kept as hard epsilon constraint: implemented。使用完整合法 atom 域推导解析 cap，并要求 zone-free compact feasibility witness；不把解析下界冒充 exact `rho*`。
- no area-balance secondary objective: implemented。V7 不含箱区负载均衡、利用率方差或 peak minimization 次级目标。
- old zone-dependent objective terms removed: implemented。没有 zone dispersion、unused zone capacity 或 voyage-area dispersion。
- new objective terms implemented: implemented。目标包括 extra group areas、extra group bays、group-area 内归一化 bay span、在场精确同组邻近度和泊位运输距离；权重明确标记为 provisional development baseline。
- anonymous imports retained: implemented。进口按流向和物理尺寸匿名预留，满足需求、尺寸/footprint/容量约束，并与新出口在物理贝位上互斥。

## 3. New V7 modules

- `v7_model.py`: implemented。定义 schema/objective version、暂定权重、全域尺度、peak policy、模型契约和独立 evaluator。
- `v7_atoms.py`: implemented。基于冻结 V6.1 业务兼容规则构造 V7 专用、无 zone 的 row atoms，并提供 group-bay/group-area 索引。
- `v7_complete_mip.py`: implemented。直接构造 zone-free compact V7 MIP，可求业务目标或仅求 peak feasibility witness；未完成时显式报错。
- `v7_stage1_area.py`: implemented。实现 coarse group-to-area MIP、solution pool、active-set 构造和 full/active graph diagnostics。
- `v7_bay_patterns.py`: implemented。定义不固定实际箱量的 Bay Pattern，实现 support size 1/2/3 精确 row-partition 枚举、微型 exhaustive oracle 和 exact reduced-cost pricing。
- `v7_column_generation.py`: implemented。实现 global pattern master、active-domain pricing、mandatory full-domain certification、动态 group-area 激活及微型 full-pattern oracle。
- `v7_integer.py`: implemented。实现 restricted integer master、atom/q/import 恢复、incumbent trajectory 字段和独立 evaluator 检查。
- `v7_pipeline.py`: implemented。串联完整 atom 域、解析 cap、compact witness、Stage 1、Stage 2 root CG、restricted integer 和最终独立验证；没有隐藏 Complete-MIP 或 exhaustive-pattern fallback。

上述八个模块均为 implemented；没有模块被标记为 partially implemented 或 not implemented。这里的“implemented”只表示代码与微型正确性门槛完成，不表示生产规模性能已经验证。

## 4. Stage 1 implementation

- variables: 整数 `Q[g,a]` 表示箱组在箱区的粗粒度数量，二元 `Y[g,a]` 表示箱组是否使用箱区。
- constraints: 箱组 exact demand、`Q-Y` 联动、基于完整合法 atoms 的 group-area reachable capacity，以及 aggregate area peak capacity。
- objective: 只使用 Stage 1 能诚实表达的 extra group areas、箱区级 existing proximity 和 berth distance；不伪造 bay count、bay span 或 row-level objective。
- solution-pool interface: 读取最优解及 Gurobi solution pool，返回每个候选解的 `Q/Y`、目标与 pool 元数据。
- active-set construction: 先保留最优解所用箱区，再按 solution-pool 频率、分配质量、coarse cost 与确定性箱区顺序扩充。
- candidate cap semantics: 默认 cap 为 4，但只控制初始算法 active set；若 Stage-1 最优解需要超过 cap 的箱区，这些箱区全部保留，不会被裁掉。
- proof that quota is not fixed in Stage 2: `Q*` 只用于 seed/active-set guidance；Stage 2 自己保留 `q[g,b]` 和 exact group demand，没有 `q` 等于 Stage-1 `Q*` 的约束。接口 `quota_fixed_in_stage2()` 返回 `False`，并有 `test_stage1_returns_guidance_not_fixed_quota` 覆盖。

## 5. Stage 2 implementation

- Bay Pattern definition: 一列描述一个 anchor bay 的完整合法出口排容量结构，保存 group capacities、group rows、physical rows、physical bays、size、height 和可恢复 atom indices；不保存或固定最终 `q[g,b]`。
- height no-mix: pattern 内只允许一个新箱高，global master 也保留物理贝位箱高状态约束。
- <=3 groups: pattern 内 support size 最多为 3；global master 对所有重叠 footprint 再施加每个物理贝位最多三个新出口箱组，避免相邻 40/45 英尺 anchor 绕过限制。
- `q[group,bay]` projected-flow design: 实际整数/连续箱量仍是 master 变量，通过 pattern 提供的 group capacity 与 pattern 选择变量联动。
- exact pricing interface: `V7ExactBayPricing` 对 support size 1/2/3 枚举行分配，返回 top-K negative patterns 和全局 minimum reduced cost 诊断。
- active-domain pricing: 常规迭代只允许当前 `A_active[g]` 中的 group-area pairs，以缩小定价域。
- full-domain certification: active 域没有负 reduced cost 后，强制在完整合法域精确定价；只有完整域 minimum reduced cost 不小于容差才设置 `root_closed=True`。
- dynamic area activation: 全域 certification 若发现改进 pattern，会激活其此前被排除的 group-area pairs、加入列并继续迭代。

## 6. Evaluator

`V7ModelEvaluator` 独立于求解模型重建可行性与目标，检查：

- 每个出口箱组 exact demand，以及 `q[g,b]` 与所选 row atoms 的容量关系；
- 物理 row 独占、anchor/physical capacity、20/40/45 footprint 和 45 英尺资格；
- 新出口尺寸唯一、箱高不混、每个物理贝位最多三个新出口箱组；
- 历史尺寸、历史箱高集合和历史精确 row-group 的 non-worsening compatibility；
- 匿名进口 exact demand、尺寸状态、容量、footprint 及新进口/新出口物理贝位互斥；
- hard peak cap；
- extra group areas、extra group bays、within-area span、existing exact-group proximity 和 berth distance 的 raw/normalized/weighted objective reconstruction。

Complete MIP 和 restricted integer incumbent 均与 evaluator 的重构目标进行数值一致性检查。

## 7. Fast correctness tests actually run

最终 V7 定向测试命令运行 28 项，全部通过。`unittest` 报告耗时 0.087 秒，wall time 0.15 秒。逐项结果如下：

| Test name | Result |
|---|---|
| `test_45ft_atoms_are_limited_to_edge_large_bays` | PASS |
| `test_atoms_are_zone_free_and_preserve_physical_resources` | PASS |
| `test_exact_existing_row_group_is_required` | PASS |
| `test_existing_mixed_height_is_nonworsening` | PASS |
| `test_area_existing_and_berth_raw_terms_are_separately_observable` | PASS |
| `test_contract_removes_zone_objective_and_declares_three_categories` | PASS |
| `test_new_height_no_mix_is_independent_of_legacy_mixed_state` | PASS |
| `test_objective_reconstructs_area_bay_span_existing_and_berth_terms` | PASS |
| `test_three_groups_are_feasible_but_four_are_rejected` | PASS |
| `test_analytic_peak_policy_has_zone_free_integer_witness` | PASS |
| `test_40ft_assignment_reserves_both_footprint_members` | PASS |
| `test_complete_mip_allows_three_but_rejects_four_groups_in_one_bay` | PASS |
| `test_height_no_mix_and_import_export_exclusivity_are_in_model` | PASS |
| `test_solver_objective_matches_independent_evaluator` | PASS |
| `test_cap_never_removes_areas_required_by_best_solution` | PASS |
| `test_graph_diagnostics_distinguish_full_and_active_domains` | PASS |
| `test_stage1_returns_guidance_not_fixed_quota` | PASS |
| `test_40ft_pattern_records_both_physical_bays` | PASS |
| `test_four_group_pattern_and_mixed_height_pattern_do_not_exist` | PASS |
| `test_support_enumeration_equals_independent_exhaustive_oracle` | PASS |
| `test_active_group_filter_is_not_part_of_full_domain_pricing` | PASS |
| `test_exact_support_pricing_matches_exhaustive_subset_pricing` | PASS |
| `test_full_domain_certification_recovers_excluded_area` | PASS |
| `test_full_pattern_integer_oracle_equals_compact_mip` | PASS |
| `test_global_three_group_limit_catches_overlapping_40ft_anchors` | PASS |
| `test_witness_patterns_make_initial_master_feasible` | PASS |
| `test_integer_incumbent_is_independently_validated` | PASS |
| `test_tiny_two_stage_pipeline_returns_valid_integer_incumbent` | PASS |

另外实际运行了以下快速检查：

- 8 个 V7 模块 `py_compile`: PASS，wall time 0.07 秒。
- `python -m unittest discover -s tests -p 'test_*.py'`: 125/125 PASS，测试框架耗时 0.391 秒、wall time 0.74 秒；该快速回归集包含冻结 V6/V5 单元测试，没有运行 benchmark 或 preexperiment。

## 8. Explicitly deferred experiments

DEFERRED:
- 24/48/96 benchmarks
- multi-seed experiments
- scalability
- performance comparison
- sensitivity
- ablation
- weight calibration
- formal UB/LB/gap evaluation
- runtime tuning

同样未进入 F&O、Adaptive Expansion、Local Branching、Dual Stabilization、Valid Inequalities 或 Branch-and-Price。

## 9. Known implementation limitations

- exact Bay Pattern pricing 当前通过 support size 1/2/3 的组合与 row partition 精确枚举实现；微型 oracle 已验证正确，但在真实候选图上的运行时间和内存尚未测量。
- Gurobi Stage-1 solution pool 的多样性、初始 active-set cap=4 的实际覆盖率以及动态 area activation 的频率尚未在正式数据上验证。
- 当前目标类别权重 `0.60/0.20/0.20` 和空间内部权重 `0.35/0.40/0.25` 只是开发基线，尚未标定，也没有敏感性证据。
- 解析 peak cap 已有 compact integer feasibility witness 门槛，但 headroom `0.50` 尚未做业务/敏感性论证；本轮没有求 exact `rho*`。
- restricted integer pool 目前由 root、peak witness 与 Stage-1-guided patterns 组成；真实规模下的整数覆盖质量和 time-to-first-incumbent 未验证。
- full-pattern universe 和 exhaustive pattern oracle 只适用于 tiny correctness cases；生产 pipeline 不调用它们。
- 尚无 V7 命令行 benchmark、实验结果 CSV、生产规模日志或论文级对比表。
- 没有证据支持 V7 在 UB、LB、gap、时间或规模性上优于完整 MIP 或历史 V6/V5。

## 10. Next experiment-round recommendations

1. 先对单个 24/48/96 箱组 seed 做受控 smoke run，记录 atom/pattern 数、Stage-1 图覆盖、active/full pricing 时间、动态激活次数、root closure 和 integer time-to-first。
2. 只有 smoke run 证明实现能在预算内完成后，再开展固定时间预算下的多 seed V7 与同模型 Complete MIP 对比。
3. 分开报告 root LB 质量、restricted integer UB 质量和总 wall time，避免把 root proof 与 primal performance 混为一谈。
4. 在基础性能稳定后再进行 candidate-area cap、pricing batch size、peak headroom 和目标权重的敏感性/消融。
5. 若定价成为主要瓶颈，先基于诊断决定是否需要算法级重写；不要在没有证据前直接加入 F&O、local branching、stabilization、cuts 或 branch-and-price。

结论仅为：**V7 code implementation complete**。

不得据此声称：V7 performance validated、V7 scalability validated 或 V7 experimentally superior。
