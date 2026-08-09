# 72-group scale case

This retained development case contains six detailed export voyages and twelve positive size-height-port groups per voyage, for 72 export groups in total. It preserves the base yard snapshot and import commitments while expanding declared export demand to 2,241 boxes.

Files:

- `input_data.json`: complete yard and vessel input;
- `large_plan.csv`: export guidance and anonymous import-reservation input;
- `manifest.json`: deterministic case dimensions and group profiles.

Run the current algorithm with one thread and a 120-second end-to-end budget:

```bash
python -X utf8 -B benchmark_contiguous_zones.py \
  --input example/many_groups_6v_12g/input_data.json \
  --large-plan example/many_groups_6v_12g/large_plan.csv \
  --total-time-limit 120 \
  --solver-threads 1
```

The committed stage-gate result is `LB=0.1792403938`, `UB=0.1918854490`, and a 6.59% certified gap. Exact row recourse realizes all 2,241 export boxes and all 624 boxes of anonymous import reservation.
