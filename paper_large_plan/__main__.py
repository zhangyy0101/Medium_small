from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import run_large_plan_file


ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the paper large plan with Gurobi.")
    parser.add_argument("--input", type=Path, default=ROOT / "example" / "input_data.json")
    parser.add_argument("--output", type=Path, default=ROOT / "paper_large_plan_outputs" / "large_plan.csv")
    parser.add_argument("--time-limit", type=float, default=120.0)
    parser.add_argument("--mip-gap", type=float, default=0.001)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output, data, solution = run_large_plan_file(
        args.input,
        args.output,
        time_limit=args.time_limit,
        mip_gap=args.mip_gap,
        threads=args.threads,
        seed=args.seed,
        verbose=not args.quiet,
    )
    print(f"status: {solution.status_name}")
    print(f"runtime: {solution.runtime:.3f}s")
    print(f"shortage: {solution.shortage_total}")
    print(f"demand policy: {data.diagnostics['demand_policy']}")
    print(f"allocation rows: {len(output)}")
    print(f"output: {args.output.resolve()}")


if __name__ == "__main__":
    main()
