# 72-group scale case

This retained development case contains six detailed export voyages and twelve positive size-height-port groups per voyage, for 72 export groups in total. It preserves the base yard snapshot and import commitments while expanding declared export demand to 2,241 boxes.

Files:

- `input_data.json`: complete yard and vessel input;
- `large_plan.csv`: historical sequential-model artifact, unused by the integrated model;
- `manifest.json`: deterministic case dimensions and group profiles.

Run the current algorithm with one thread and a 120-second end-to-end budget:

```bash
python -X utf8 -B benchmark_contiguous_zones.py \
  --input example/many_groups_6v_12g/input_data.json \
  --total-time-limit 120 \
  --solver-threads 1
```

The former committed LB/UB/gap used a large-plan-guidance objective and is not comparable with the integrated model. Regenerate the result before using this case as evidence.
