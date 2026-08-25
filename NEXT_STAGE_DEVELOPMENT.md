# 下一阶段开发准备：Adaptive Primal Search

## 1. 开发目标

保留 V5 的顶层论文骨架：

```text
Exact proof-oriented root CG
→ compact primal integer search
→ adaptive coordinated improvement
→ exact row recourse
```

下一阶段只重构 primal search，不修改数学模型和 root proof。直接目标是解决
oracle/coverage audit 已确认的问题：静态逐列裁剪无法保留跨箱组协调所需的
关键组合列。

## 2. 开发分支和基线

```text
frozen tag:
baseline/v5-phase3-handoff-20260825

working branch:
feat/v5-adaptive-primal-search-next
```

冻结标签包含 Phase 1–4、Phase 3.1–3.3 的代码开关、测试和报告；默认仍为
`conflict_multi_round`，dynamic initial stopping 关闭。

## 3. 第一批允许修改的范围

1. 将 static primal pool 改成“核心列池 + 按 incumbent/冲突反馈动态扩展”的列管理；
2. 以完整可行 support/bundle 为单位保存和回注列，不只按单列排名；
3. 保证任何新增 primal 列不改变 proof master、root LB 和 complete legal zone set；
4. 保存每次扩展的触发原因、bundle 来源、列数、首次使用轮次和最终选中情况；
5. 保留原 incumbent 作为 fallback，任何局部搜索都不得让最终 UB 变差。

## 4. 暂不同时实现

- 目标权重或归一化修改；
- peak-utilization cap 修改；
- pricing 或 root closure 修改；
- dual stabilization；
- valid inequalities；
- Branch-and-Price；
- 针对单个 seed 调整冲突权重、top-k 或时间比例。

Local Branching 是下一阶段的候选搜索器，但应在 adaptive column coverage
机制有独立消融结果后单独接入，避免无法判断收益来源。

## 5. Correctness gate

每次结构修改至少保证：

```text
RMQ pricing == exhaustive pricing
root LP == complete zone LP
internal validation == PASS
external validation == PASS
exact row recourse == PASS
Complete MIP path remains isolated
final UB <= accepted incumbent UB
```

## 6. 性能 stage gate

开发阶段先使用多个 development seeds，不直接在 holdout 上调参：

- 24 groups：不能出现持续超过 1% 的 UB 劣势；
- 48 groups 以上：相对 Complete MIP 的 UB improvement median 目标至少 3%；
- 48 groups 以上：win rate 目标至少 80%；
- 报告 mean/median/min/max、win/tie/loss 和 time-to-best；
- gap 改善与 UB 改善分开判断。

如果 adaptive primal search 仍只能带来不足 1% 的波动，应停止继续堆叠
primal heuristics，重新评估论文定位；如果 UB 已稳定改善但 independent gap
仍弱，再单独进入 valid inequalities，而不是继续调 F&O。

## 7. 开始编码前的命令

```bash
git status --short --branch
git describe --tags --always
.venv/bin/python -m unittest discover -s tests -p 'test_*.py'
```

任何新实现都应先在 micro tests 中证明 bundle 不丢失、incumbent 单调安全，
再运行 24/48/96 代表性实验。
