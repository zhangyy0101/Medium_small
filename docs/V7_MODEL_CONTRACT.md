# V7 两阶段模型契约

状态：V7 代码实现与微型正确性验证完成；性能、规模性和论文数值结论尚未验证。
V6.1 保持冻结，V7 不修改任何 V6 schema 或历史求解语义。

版本：

- 模型：`hierarchical_group_area_bay_pattern_v7`
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
zone-free compact MIP 找到整数可行 witness。若 witness 失败，流程显式失败。

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
scale；所有 scale 均从完整合法候选域得到，不依赖 Stage-1 active set。

## 5. Stage 1

Stage 1 使用整数 `Q[g,a]` 和二元 `Y[g,a]`。`Q` 的上界来自合法 row atoms 的可达
容量，并满足箱组 exact demand 与箱区 aggregate peak capacity。

Stage-1 目标只含可粗粒度表达的 group-area count、area existing proximity 与泊位距离；
不伪造 bay count 或 span。

solution pool 用于构造 `A_active[g]`。初始 cap 默认 4，但：

- cap 只是 active-set 算法参数；
- Stage-1 best 解使用超过 4 个箱区时必须全部保留；
- `Q*[g,a]` 从不作为 Stage-2 quota 或等式。

## 6. Stage 2 Bay Pattern

一列是一种 anchor bay 完整出口排容量结构，记录 group capacities、group rows、物理
rows、物理 bays、尺寸和箱高。列不固定最终实际箱量。

pattern 内部严格保证逐排独占、尺寸/箱高一致、footprint、容量和最多三个 active
groups。Global RMP 显式保留 `q/u/y/span/U_phy`、imports、peak 和全部跨贝位约束。

pricing 按一个 anchor bay 枚举 support size 1/2/3，并精确枚举行分配。常规迭代只允许
active areas；准备闭合根节点时，必须恢复全部合法 group-bay pairs 做 full-domain exact
pricing。若发现负 reduced-cost pattern，则激活对应 group-area、加入列并继续。只有全域
minimum reduced cost 不小于容差时，`root_closed=True`。

## 7. Integer 与独立认证

根闭合后，restricted integer pool 至少合并 root patterns、peak witness patterns 和
Stage-1-guided patterns。整数 incumbent 被恢复为 selected row atoms、整数 `q[g,b]` 和
匿名进口预留，并由 `V7ModelEvaluator` 独立检查需求、排/贝/footprint、历史状态、
箱高、最多三组、进口、peak 和完整目标。

第一版不包含 F&O、adaptive expansion、local branching、dual stabilization、valid
inequalities 或 branch-and-price。

## 8. 证据边界

当前只完成 unit/micro correctness。24/48/96、多种子、规模性、性能对比、正式
UB/LB/gap、敏感性、消融、权重标定和运行参数调优全部推迟到下一实验轮。
