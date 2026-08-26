# V6 模型契约与实施边界

状态：V6.1 数学模型、小规模完整枚举 MIP、V6-native 解析峰值 cap 与 compact 可行性认证、投影 restricted master、exact pricing、restricted integer master 和 compact primal coverage V1 已实现；exact compact min-max 保留为 oracle。V6.1 将历史违规状态的兼容语义固定为“不增加违反程度”；V6.0 与 V5 的旧实验结果只用于历史复现，不能作为 V6.1 的算法或对比结果。

代码唯一契约入口为 `yard_planning/v6_model.py`，小规模完整 zone oracle 为 `yard_planning/row_aware_zones.py`。后续完整 MIP、restricted master、pricing 和结果重构必须共用其中的模型版本、目标配置和独立 evaluator。

## 1. 输入与固定业务语义

### 1.1 出口箱组

一个出口箱组唯一对应：

`航次 + 尺寸 + 箱高 + 卸货港`

仅规划尚未进场的确定性出口资料箱（`OF`）。相同四元组必须在优化前聚合；不同四元组必须是不同箱组。预测箱、TOPS plan、上游大计划箱区目标和缺箱变量均不进入 V6。

### 1.2 贝位、排与 footprint

- 一个物理排资源由 `(bay_key, row_no)` 唯一标识。
- 不同箱组不可共用同一物理排；一个已选 zone 也不能重复占用其他已选 zone 的物理排。
- 同一贝位允许尺寸、箱高相同的不同箱组使用不同排。
- 20 英尺箱占一个锚定贝位；40/45 英尺箱占锚定贝位及其完整配对 footprint 的同号排。
- 45 英尺出口箱只能使用箱区边缘的大贝。
- 现有箱采用“历史状态不恶化”兼容规则：尺寸约束仍严格保持原样；若贝位已有一个或多个箱高，新出口箱高必须属于已有箱高集合；若物理排已有一个或多个箱组，新出口箱组必须精确属于该排已有的四元组集合。
- 对新分配箱本身仍执行绝对不混规则：同一物理贝位的新出口箱尺寸、箱高唯一，同一物理排的新出口箱组唯一。历史混合状态不会被优化变量重排或修复，也不能引入新的尺寸、箱高或排箱组类型。
- 排内历史箱组按精确联合键保存，不能把已有航次集合与卸货港集合做笛卡尔积；现有进口或未知方向箱也不能伪装成可匹配的新出口箱组。

V6 直接使用物理排独占约束，不再用“固定排号 strip”或多组间接属性约束表达新出口箱的排独占。旧 stack 辅助变量也不再单独保留：zone 原子已经按实际物理排、尺寸和剩余层容量建立，排独占与逐排容量共同给出相同的可堆放性。

### 1.3 Zone

一个 zone 是：

`一个箱组 + 一个箱区 + 一段连续锚定贝位 + 每个贝位选取的非空排集合`

- 连续性只沿贝位轴定义，不要求排号连续或相同。
- 相邻贝位可选择完全不同的排；一个贝位也可为同一 zone 选择多个排。
- 每个已选 zone 的每个锚定贝位都必须分配至少一箱，禁止用零流量贝位“搭桥”制造虚假连续。
- zone 在某贝位的容量等于该贝位被选排原子的尺寸兼容剩余容量之和；总容量为各贝位容量之和。
- 候选 zone 可以相互重叠；最终选择通过物理排独占和同箱组贝位不重叠约束协调。
- V6 不再设置“需求量 + 一个原子排容量”的 zone 长度/容量上限。它不是业务约束，也没有被证明为支配规则；过大的保留区由未利用容量目标惩罚。

### 1.4 匿名进口预留

进口箱按 `流向 + 尺寸` 聚合，只决定贝位级匿名容量，不决定航次、卸货港、箱高、箱组或具体排。预留必须满足：

- 精确满足每个流向—尺寸的需求；
- 箱区功能兼容；
- 20/40 英尺 footprint、锚定贝位尺寸容量和所有 footprint 贝位物理容量；
- 每个被进口使用的物理贝位只有一个进口尺寸状态；
- 进口尺寸与已有贝位尺寸状态一致；
- 新进口预留与新出口分配不能共享任何物理贝位；
- 不参与出口集中度和距离目标。

## 2. 决策变量与硬约束

令 `x_z` 表示是否选择 group-specific zone，`q_zb` 表示 zone `z` 在其锚定贝位 `b` 的出口箱量，`p_fsb` 表示进口流向 `f`、尺寸 `s` 在锚定贝位 `b` 的匿名预留量。

核心约束为：

1. `x_z <= q_zb <= C_zb x_z`：选中 zone 后每贝正流量，未选中时零流量；
2. `sum(z,b in g) q_zb = demand_g`：出口箱组需求精确满足；
3. 任一物理排最多属于一个已选 zone；
4. 同一箱组在同一锚定贝位最多由一个已选 zone 覆盖；
5. 同一物理贝位上的新出口箱组尺寸、箱高一致；
6. zone 的箱区功能、footprint、45 英尺边缘资格及已有状态“不恶化”兼容；
7. `sum_b p_fsb = import_demand_fs`，并满足进口 footprint、尺寸和物理容量；
8. 新进口与新出口的物理贝位使用状态互斥；
9. 每个箱区的计划 slot load 不超过数据驱动峰值利用率上限。

以上约束对完整 MIP 和列生成主问题保持同一整数语义。为使新增 zone 只进入固定约束行，restricted master 将 `q_zb` 精确投影为全局 `q_gb`：`sum_z x_z <= q_gb <= sum_z C_zb x_z`，并保留同箱组同贝位至多一个 zone 的约束。完整 MIP 不能再以 row-location M0 替代，因为 M0 的集中度表示和可行域与 V6 zone 模型不同。

## 3. 三类归一化目标

V6 使用一次求解的归一化加权和，以保持目标可分解并适用于列生成。目标分为三类：

| 类别 | 类别权重 | 类内构成 |
|---|---:|---|
| 空间集中度 | 0.6250 | zone 分散 0.56；航次跨箱区 0.24；靠近已有同类箱 0.20 |
| 泊位运输距离 | 0.1625 | 按实际出口箱量加权的泊位—箱区归一化距离 |
| 预留容量效率 | 0.2125 | 已选 zone 容量减去实际出口箱量 |

等价的五个原始项权重为 `0.3500 / 0.1500 / 0.1250 / 0.1625 / 0.2125`。这里的“五项”只是三类目标的可分解展开，不是五个并列业务目标。

原始指标和自然尺度为：

- zone 分散：`sum_g(max(0, selected_zones_g - 1))`，尺度为各箱组可达到的额外 zone 数上界；
- 航次跨箱区：`sum_v(max(0, used_areas_v - 1))`，尺度为各航次可达额外箱区数上界；
- 已有同类箱距离：按分配箱量加权。锚点所在箱区内用贝位序号距离除以该箱区跨度；使用无锚点箱区记为 1；没有任何可达锚点的箱组记为中性 0。尺度为有可达锚点的出口需求；
- 泊位距离：在每个航次的可达箱区范围内做 min-max 归一化，再按箱量加权；尺度为总出口需求；
- 未利用容量：`sum_z(C_z x_z) - total_export_demand`；尺度为总出口需求。

所有尺度至少取 1，避免退化实例除零。默认权重保留上一轮已校准的原始比例，但只是 V6 初始标定值；正式论文仍需进行权重敏感性和目标消融。箱组跨箱区、排号分散、上游大计划偏差、短缺和 TOPS 均不在目标中。

## 4. 峰值利用率 ε 约束

由于没有码头给定的最高利用率，V6 不把峰值作为第四类目标。生产流程为：

1. 根据合法 V6 row atoms 与匿名进口候选，分别计算全局、出口、进口、航次、箱组和进口流向—尺寸的可达负载/容量下界，并包含箱区整数容量断点；
2. 取最大解析下界 `L`，对给定余量系数 `h` 设置 `rho_cap = L + h(1-L)`；
3. 在完整 compact row-atom 可行域上只寻找一个满足该 cap 的整数 witness，不要求证明 min-max optimal；
4. root CG、compact primal 和 Complete MIP 都使用完全相同的 `rho_cap`；feasibility witness 同时作为 compact primal warm start；
5. 主模型约束每个箱区的出口与匿名进口 footprint slot load 不超过 `rho_cap`。

默认 `h=0.50`。`L` 是可复现的解析工作量下界，不得称为 `rho*`；只有 feasibility witness 通过独立 evaluator 后才接受该 cap。若给定 headroom 下无法取得 witness，流程必须失败并调整声明的 headroom，不能静默删除约束。

compact 与完整枚举 exact min-max MIP均保留为小规模 oracle、敏感性分析和诊断入口，但不再是生产算法的强制前置阶段。

## 5. 实施边界与准入顺序

当前已完成：固定输入契约、完整小规模 row-aware zone oracle、三类目标与自然尺度、解析 ε cap、compact 可行性认证、可行 witness warm start、可选 exact min-max oracle、固定 ε 的完整业务 MIP、独立可行性/目标 evaluator、投影 restricted master、临时 Phase-I、非枚举 exact pricing、根 LP 列生成、restricted integer master 和 compact primal coverage。完整枚举 MIP 只作为 oracle，且只接受未裁剪的合法 zone universe。

旧 exact-`rho*` 单种子 smoke 已定位其不必要的证明耗时，现已被解析 cap＋feasibility witness 生产流程替代。根节点进一步改为用已验证 witness 直接启动业务 RMP，并用贝位容量 knapsack＋连续区间标签动态规划进行 exact pricing；24/48/96 单种子均已在 60 秒内闭合。Phase-I 仍是无 witness 独立调用时的后备机制，其 deficit 只是寻找 LP 可行列的临时算法变量，不属于 V6 业务决策，也不会进入最终业务主问题。该单种子 gate 不是正式多种子 UB/gap 实验。

完整枚举 MIP 当前已经通过：

1. MIP 解由 `V6ModelEvaluator` 独立复核，solver 目标与重构目标一致；
2. 小算例覆盖跨贝换排、同贝不同组分排、40/45 footprint、已有状态、进口互斥和峰值；
3. 完整 MIP 稳定后，列生成只能调用同一套 zone、目标系数和 evaluator；
4. solver 目标与独立 evaluator 重构目标在微型算例上一致。

当前已经证明非枚举 pricing 与 exhaustive pricing 的最小 reduced cost 一致、闭合根 LP 与完整枚举投影 LP 目标一致，以及完整 zone pool 下的 projected integer master 与 Complete MIP 目标一致。restricted integer incumbent 可以无歧义恢复为 `q_zb` 并通过 evaluator。纯根 proof pool 的次优/整数不可行已经由 compact row-atom primal coverage V1 修复：该模块联合考虑所有箱组和进口冲突，将多个 incumbent 恢复为 zone 候选并回注，同时保持 exact root bound。compact 最好 incumbent 通过 atom signature、全局箱量、进口预留及辅助状态完整传入 RIM，不能依赖会在合并列池后变化的临时 zone id。该流程不调用完整 zone enumeration，也没有迁移 V5 static pool。
