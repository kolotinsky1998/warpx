#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import replace

from jax_smirnov_pic import run_simulation, smirnov_default_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--poisson-solver",
        choices=("direct_inverse", "fft_capacitance", "cg", "jacobi"),
        help="Poisson solver backend to use.",
    )
    parser.add_argument("--trace-dir", help="Directory for JAX trace output.")
    parser.add_argument("--trace-start-step", type=int, default=0, help="Step to start JAX tracing from.")
    parser.add_argument("--trace-num-steps", type=int, default=0, help="How many steps to capture in the JAX trace.")
    parser.add_argument("--profile", action="store_true", help="Enable detailed per-step profiling.")
    parser.add_argument(
        "--profile-summary-only",
        action="store_true",
        help="Collect timing summary without writing per-step profiling rows.",
    )
    args = parser.parse_args()
    config = smirnov_default_config()
    if args.poisson_solver:
        config = replace(config, poisson_solver=args.poisson_solver)
    run_simulation(
        config,
        profile=args.profile,
        profile_summary_only=args.profile_summary_only,
        trace_dir=args.trace_dir,
        trace_start_step=args.trace_start_step,
        trace_num_steps=args.trace_num_steps,
    )


if __name__ == "__main__":
    main()
