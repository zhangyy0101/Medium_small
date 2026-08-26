# V7.1 强化 Stage 1 / Restricted Stage 2 模型契约

状态：V7.1 代码实现、微型正确性验证和一个 24 箱组单种子候选域诊断已完成；性能、规模性和论文数值结论尚未验证。
V7.1 不改变 V7 业务模型、目标或 Bay Pattern 定义，只改变两阶段职责边界。

版本：

- 模型：`hierarchical_group_area_bay_pattern_v7`
- 算法：`hierarchical_v7_1_restricted_area_bay_cg`
- 目标：`v7_spatial_existing_berth_normalized_v1`

## 1. 正式决策对象

V7 正式删除 zone。模型直接使用：

- `q[g,b]`：箱组在 anchor bay 的实际整数箱量；
- `u[g,b]`：箱组是否使用 anchor bay；
- `y[g,a]`：箱组是否使用箱区；
- `z[i]`：合法物理 row atom 是否被选择；
- `U_phy[g,beta]`：箱组 footprint 是否使用物理贝位；
- `p[flow,size,b]`：匿名进口预留量。

相邻贝位 segment 只能在报告层恢复，不能影响可行域、目标或 pricing。

## 2. 固定业务约束

箱组严格定义为“航次＋尺寸＋箱高＋卸货港”。出口需求精确满足，不设置 shortage。

每个 V7 row atom 在生成时已经验证：

- `OF` 箱区功能；
- 20/40/45 尺寸与完整 footprint；
- 45 英尺边缘大贝资格；
- 正的逐排剩余容量；
- 历史尺寸严格一致；
- 新箱高属于历史已有箱高集合；
- 历史 row 存在箱组时必须匹配精确联合键，不能使用航次集合与港口集合的笛卡尔积。

最终模型继续执行：物理排最多分给一个新出口箱组；一个物理贝位的新出口尺寸唯一、
箱高唯一；历史违规状态固定且不得增加新类型。

对每个物理贝位：

```text
sum_g U_phy[g,beta] <= 3
```

这里只计本次新规划使用的出口箱组。40/45 英尺箱在 footprint 的每个物理贝位计数。

匿名进口按流向和物理尺寸聚合，满足 exact demand、箱区功能、footprint、anchor/物理
容量、已有尺寸状态、单一进口尺寸状态，并与新出口在物理贝位上互斥。

## 3. Peak hard epsilon constraint

V7 用完整合法 atom 域和匿名进口候选计算 reachable-capacity 解析下界 `L`，再用：

```text
rho_cap = L + h * (1 - L)
```

形成 hard cap。当前 development baseline 为 `h=0.50`。解析下界本身不声称是
`rho*`；必须由包含箱高不混、物理 footprint、进口互斥和每物理贝最多三个新箱组的
zone-free compact MIP 找到整数可行 witness。当前 witness 使用零目标并在找到第一个经
独立验证的整数可行解后结束，只负责证明 `rho_cap` 下的整数可行性并向后续阶段提供
固定 proof patterns；它不优化论文业务目标、不报告业务目标 gap，也不向 Stage 1 注入
mandatory candidate areas。若 witness 失败，
流程显式失败。

V7 不包含 min-max、利用率方差、箱区负载方差或箱区均衡 secondary objective。

## 4. 归一化目标

V7 最小化三类目标：

```text
spatial consolidation
+ existing exact-group proximity
+ berth transport distance
```

空间集中度内部为：

- `sum_g max(0, used_areas_g - 1)`；
- `sum_g max(0, used_bays_g - 1)`；
- 每个 group-area 内已用 bay order 的归一化 `max-min` span。

已有同组距离和泊位距离均按实际 `q[g,b]` 加权。没有可达同组历史锚点的箱组在该项取
中性 0。

默认类别权重为 `0.60 / 0.20 / 0.20`；空间内部为 `0.35 / 0.40 / 0.25`。
这些值只标记为 `provisional_development_baseline`，不是论文最终权重。

area-count、bay-count、span、existing proximity 和 berth distance 分别计算确定性正
scale；所有 scale 均从完整合法候选域得到，不依赖 Stage-1 restricted set。

## 5. Stage 1

Stage 1 使用整数 `Q[g,a]`、`N[g,a]` 和二元 `Y[g,a]`。`Q` 的上界来自合法 row
atoms 的可达容量，并满足箱组 exact demand 与箱区 aggregate peak capacity。`N` 是
最小 compatible anchor-bay incidence：`Q <= max_single_anchor_capacity * N`，并受
compatible anchor 数量限制。area 级 `sum footprint_width*N <= 3*usable physical bays`
只是一条必要 packing proxy，不替代详细可行性。

Stage-1 目标只含可粗粒度表达的 group-area count、area existing proximity 与泊位距离；
不伪造 bay count 或 span。

Stage 1 正式构造冻结的 `A_restricted[g]`：

- Stage-1 best support 全部 mandatory；feasibility witness support 不参与候选域构造；
- Gurobi solution pool 只按 `Y[g,a]` 支持区分解，`Q/N` 差异不会制造重复候选；
- `pool_gap` 作为归一化 Stage-1 目标的绝对近优包络，pool frequency、quantity mass 和首次出现排名用于选择多样化 alternatives；
- 对每个 `group-area` 用真实 compatible anchor 容量构造单箱组 bay-local placement；在三种确定性 anchor 顺序中选择 shortage 最小、正式 flow/额外贝位/span 成本最低的方案，并据此产生 bay-local alternatives；
- `additional_candidate_area_cap` 默认 5，只控制 Stage-1 best 之外的额外候选；`maximum_pool_candidate_areas` 默认 1，使其余名额优先来自 bay-local ranking，且不能裁 best support；
- `Q*[g,a]` 从不作为 Stage-2 quota 或等式。

## 6. Stage 2 Bay Pattern

一列是一种 anchor bay 完整出口排容量结构，记录 group capacities、group rows、物理
rows、物理 bays、尺寸和箱高。列不固定最终实际箱量。

pattern 内部严格保证逐排独占、尺寸/箱高一致、footprint、容量和最多三个
groups。Global RMP 显式保留 `q/u/y/span/U_phy`、imports、peak 和全部跨贝位约束。

pricing 按一个 anchor bay 枚举 support size 1/2/3，并精确求解 row allocation。生产
Stage 2 只能在冻结的 `A_restricted[g]` 内生成新列，不得自动激活其他箱区。feasibility
witness 恢复出的 pattern 以显式登记的 proof column 身份保留在 master 中，即使位于
restricted domain 外也只提供可行性保底，不会开放相应箱区的 pricing。对 restricted domain
完成一次无负列的 exact sweep 后标记 `restricted_root_closed=True`；默认
`global_root_certified=False`。

full-domain exact pricing 仅通过独立 `audit_full_domain_pricing` 接口执行。审计只报告全域
minimum reduced cost、排除域负列和受影响 group-area pairs，不加入列、不修改 restricted
mapping，也不继续 CG。

## 7. Integer 与独立认证

restricted root 闭合后，Restricted Integer Master 合并 root patterns、显式登记的 peak
witness proof patterns、Stage-1-guided patterns 和 deterministic one-group patterns。整数 incumbent
被恢复为 selected row atoms、整数 `q[g,b]` 和
匿名进口预留，并由 `V7ModelEvaluator` 独立检查需求、排/贝/footprint、历史状态、
箱高、最多三组、进口、peak 和完整目标。

V7.1 不包含 F&O、adaptive expansion、local branching、dual stabilization、valid
inequalities 或 branch-and-price。

restricted LP objective 不是完整 V7 MIP 的全局下界。正式结果只输出
`restricted_lp_bound`、`restricted_integer_ub` 和 `restricted_mip_gap`；没有独立全域
证书时，`global_lower_bound=None`、`global_gap=None`。整数解通过完整 evaluator，因此仍是
完整业务问题的合法 feasible UB。

默认生产计时采用统一总预算 `algorithm_time_limit=120s`。feasibility-only witness、Stage 1 和
restricted root 分别保留软上限 `10/10/60s`，并为后续必要阶段保留最低启动时间；RIM
没有默认固定上限，而是在模型构建完成后按统一 wall-clock deadline 使用全部剩余时间。
`integer_time_limit` 只保留为消融实验的可选额外 ceiling。只读全域 pricing 审计在 RIM
之后执行且不计入生产预算；Complete MIP 使用独立的同等对照预算。

## 8. 证据边界

当前完成 141 项 unit/micro tests 及 2 个 subtests。原 proof/search separation 的 24 箱组
单种子诊断中，cap 2/3 只覆盖 Complete-MIP incumbent 28 个 group-area 组合中的 4 个
（85/640 箱），论文算法 UB 为 `0.060389`。引入 bay-local ranking、采用额外 cap 5 且
pool 最多占 1 个名额后，覆盖提升到 22/28（534/640 箱），restricted root 在 9.37 秒、
23 轮内闭合，120 秒总预算下 UB 为 `0.045080`。该 UB 比保存的同预算 Complete-MIP
incumbent `0.045754` 低 1.47%，本轮未重跑 Complete MIP。restricted gap 2.92% 仍不是
global gap，因为尚无 full-domain lower-bound certificate。48 箱组 UB `0.079309` 比旧
V7 改善 5.20%，但仍比保存的 MIP incumbent 高 1.73%；96 箱组 restricted root 在
60 秒上限后仍有负列，因此没有正式 UB。当前结果只能支持修正方向有效并暴露 pricing
规模瓶颈，不能支持正式性能结论；多种子、正式规模性、敏感性、消融、权重标定和运行参数调优仍
属于后续实验。
