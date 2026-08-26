"""Run the V7.1 Complete-MIP incumbent versus Stage-1 coverage audit."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pandas as pd

from adapters.input_adapter_gd import InputAdapterGd
from adapters.planning_input import load_planning_inputs
from preexperiment.scenario_generator import load_suite, materialize_scenario
from preexperiment.v7_oracle_coverage import audit_stage1_incumbent_coverage
from yard_planning.v7_atoms import build_v7_row_atoms
from yard_planning.v7_complete_mip import V7CompleteMipConfig, V7CompleteMipSolver
from yard_planning.v7_model import V7ObjectiveConfig, derive_v7_analytic_peak_policy
from yard_planning.v7_stage1_area import V7Stage1AreaSolver, V7Stage1Config


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite", type=Path, default=ROOT / "preexperiment" / "scale_suite.json"
    )
    parser.add_argument("--case", default="scale_g024_s401")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT
        / "preexperiment"
        / "reports"
        / "v7_1"
        / "oracle_coverage_24_20260827",
    )
    parser.add_argument("--peak-witness-time-limit", type=float, default=10.0)
    parser.add_argument("--stage1-time-limit", type=float, default=10.0)
    parser.add_argument("--complete-mip-time-limit", type=float, default=120.0)
    parser.add_argument("--caps", nargs="+", type=int, default=[2, 3])
    parser.add_argument("--solver-threads", type=int, default=1)
    parser.add_argument("--solver-seed", type=int, default=0)
    parser.add_argument("--verbose-solver", action="store_true")
    return parser.parse_args()


def write_outputs(output_root: Path, audit: dict[str, object]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "coverage_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    rows = audit["rows"]
    if rows:
        with (output_root / "coverage_audit.csv").open(
            "w", newline="", encoding="utf-8-sig"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def main() -> None:
    args = parse_args()
    base_path, specs = load_suite(args.suite.resolve())
    matches = [spec for spec in specs if spec.case_id == args.case]
    if len(matches) != 1:
        raise ValueError(f"unknown or duplicate case: {args.case}")
    base = InputAdapterGd.load_from_json(str(base_path))
    adapter, manifest = materialize_scenario(base, matches[0])
    planning_time = pd.Timestamp(adapter.planning_time)
    inputs = load_planning_inputs(
        adapter,
        planning_time=planning_time.to_pydatetime(),
        voyages=manifest["selected_export_voyages"],
    )
    problem = inputs.problem
    atoms, _limits = build_v7_row_atoms(problem)
    objective = V7ObjectiveConfig()
    peak_policy, _diagnostics = derive_v7_analytic_peak_policy(
        problem, objective, atoms=atoms
    )

    print(f"[{args.case}] solving feasibility-only witness", flush=True)
    witness = V7CompleteMipSolver(
        problem,
        V7CompleteMipConfig(
            time_limit=args.peak_witness_time_limit,
            mip_gap=0.0,
            solver_threads=args.solver_threads,
            solver_seed=args.solver_seed,
            verbose=args.verbose_solver,
            require_business_optimality=False,
            objective=objective,
        ),
    ).solve(peak_policy, feasibility_only=True)

    print(f"[{args.case}] solving Stage 1 and collecting rankings", flush=True)
    stage1 = V7Stage1AreaSolver(
        problem,
        peak_policy,
        atoms,
        V7Stage1Config(
            time_limit=args.stage1_time_limit,
            maximum_pool_solutions=8,
            pool_gap=0.10,
            additional_candidate_area_cap=min(args.caps),
            maximum_pool_candidate_areas=1,
            solver_threads=args.solver_threads,
            solver_seed=args.solver_seed,
            verbose=args.verbose_solver,
            objective=objective,
        ),
    ).solve()

    print(f"[{args.case}] solving 120-second Complete MIP incumbent", flush=True)
    complete = V7CompleteMipSolver(
        problem,
        V7CompleteMipConfig(
            time_limit=args.complete_mip_time_limit,
            mip_gap=0.0,
            solver_threads=args.solver_threads,
            solver_seed=args.solver_seed,
            verbose=args.verbose_solver,
            require_business_optimality=False,
            objective=objective,
        ),
    ).solve(peak_policy)

    audit = audit_stage1_incumbent_coverage(
        problem,
        atoms,
        stage1,
        complete,
        witness=witness,
        caps=args.caps,
        maximum_pool_candidate_areas=1,
        objective=objective,
    )
    audit["case_id"] = args.case
    audit["configuration"] = {
        "peak_witness_time_limit": args.peak_witness_time_limit,
        "stage1_time_limit": args.stage1_time_limit,
        "complete_mip_time_limit": args.complete_mip_time_limit,
        "caps": sorted(set(args.caps)),
        "solver_threads": args.solver_threads,
        "solver_seed": args.solver_seed,
    }
    write_outputs(args.output_root.resolve(), audit)
    print(
        json.dumps(
            {
                "case_id": args.case,
                "complete_mip": audit["complete_mip"],
                "caps": audit["caps"],
                "output_root": str(args.output_root.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
