# Complete-voyage plan benchmark

The diversified instance contains six detailed export voyages, 27 export groups, 2,241 declared export containers, and the unchanged 624-container anonymous import commitment.

All methods use one Gurobi thread, the same mathematical model, and a 120-second optimization limit.

| Method | Upper bound | Valid lower bound | Relative gap |
|---|---:|---:|---:|
| Complete-voyage plan CG with inner local patterns | 0.25368780 | 0.13084315 | 48.42% |
| M0 direct MILP | **0.25170219** | **0.24450638** | **2.86%** |
| Strengthened conflict-graph LBBD | 0.26700085 | 0.24217315 | 9.30% |

The hierarchical decomposition improves the former voyage-area master incumbent (`0.25820039`) and reduces its gap (`50.21%` to `48.42%`), but the lower bound remains weak. The outer master has only six voyage blocks instead of 300 voyage-area blocks, yet complete-plan coordination still tails off: six business rounds produce 150 outer plans within the root allowance.

On the base instance, the current 60-second result is an incumbent of `0.19270516`, a valid lower bound of `0.18736580`, and a `2.77%` gap. The small complete-column regression instance closes exactly and matches M0 at `0.132`.

These results validate the implementation and expose the remaining algorithmic limitation, but they do not establish a computational advantage over M0. They should be treated as a development diagnosis rather than a paper performance claim.

The alternative LBBD closes the base instance exactly in 9.1 seconds at `0.19268294`. On the diversified instance it creates 80 initial conflict-clique cuts, 80 initial Hall cuts, 15 dynamic Hall cuts, 17 IIS-core capacity cuts, 20 logic feasibility cuts, and 19 logic optimality cuts. Its compact primal seed supplies the reported upper bound but never contributes to the lower bound. The stronger lower bound is substantially better than column generation's, but both its incumbent and gap remain worse than M0 on this stress case.

Reproduce with:

```bash
python -X utf8 -B benchmark_voyage_plans.py \
  --input example/diverse_voyages_3x/input_data.json \
  --large-plan example/diverse_voyages_3x/large_plan.csv \
  --total-time-limit 120 \
  --solver-threads 1 \
  --output outputs/vpcg_final_diverse_3x_120s.json

python -X utf8 -B benchmark_logic_benders.py \
  --input example/diverse_voyages_3x/input_data.json \
  --large-plan example/diverse_voyages_3x/large_plan.csv \
  --total-time-limit 120 \
  --solver-threads 1 \
  --output example/diverse_voyages_3x/lbbd_120s.json
```
