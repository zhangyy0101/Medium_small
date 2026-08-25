from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from adapters.input_adapter_gd import InputAdapterGd

from .input_builder import build_large_plan_data
from .models import LargePlanData, LargePlanSolution
from .solver import solve_large_plan


OUTPUT_COLUMNS = (
    "voy_id",
    "flow",
    "area_no",
    "size",
    "planned_qty",
    "snapshot_qty",
    "new_qty",
    "planning_time",
    "status_name",
    "objective_value",
)


def allocation_frame(data: LargePlanData, solution: LargePlanSolution) -> pd.DataFrame:
    if not solution.has_solution:
        raise RuntimeError(f"Large plan has no solution (status={solution.status_name}).")
    rows: list[dict[str, Any]] = []
    for size, snapshot, new_values in (
        ("20", data.snapshot20, solution.new20),
        ("40", data.snapshot40, solution.new40),
    ):
        keys = sorted(set(snapshot) | set(new_values))
        for voyage, flow, area in keys:
            snapshot_qty = int(snapshot.get((voyage, flow, area), 0))
            new_qty = int(new_values.get((voyage, flow, area), 0))
            planned_qty = snapshot_qty + new_qty
            if planned_qty <= 0:
                continue
            rows.append(
                {
                    "voy_id": voyage,
                    "flow": flow,
                    "area_no": area,
                    "size": size,
                    "planned_qty": planned_qty,
                    "snapshot_qty": snapshot_qty,
                    "new_qty": new_qty,
                    "planning_time": data.planning_time,
                    "status_name": solution.status_name,
                    "objective_value": solution.objective_value,
                }
            )
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS).sort_values(
        ["voy_id", "flow", "area_no", "size"],
        ignore_index=True,
    )


def run_large_plan_file(
    input_path: str | Path,
    output_path: str | Path,
    *,
    time_limit: float = 120.0,
    mip_gap: float = 0.001,
    threads: int = 1,
    seed: int = 0,
    verbose: bool = True,
) -> tuple[pd.DataFrame, LargePlanData, LargePlanSolution]:
    input_path = Path(input_path).resolve()
    output_path = Path(output_path).resolve()
    adapter = InputAdapterGd.load_from_json(str(input_path))
    data = build_large_plan_data(adapter)
    solution = solve_large_plan(
        data,
        time_limit=time_limit,
        mip_gap=mip_gap,
        threads=threads,
        seed=seed,
        verbose=verbose,
    )
    output = allocation_frame(data, solution)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False, encoding="utf-8-sig")
    diagnostics = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "planning_time": data.planning_time,
        "status": solution.status_name,
        "runtime": solution.runtime,
        "mip_gap": solution.mip_gap,
        "objective_components": solution.objective_components,
        "shortage20": {"|".join(key): value for key, value in solution.shortage20.items()},
        "shortage40": {"|".join(key): value for key, value in solution.shortage40.items()},
        "input_diagnostics": data.diagnostics,
    }
    diagnostics_path = output_path.with_name(f"{output_path.stem}_diagnostics.json")
    diagnostics_path.write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")
    return output, data, solution
