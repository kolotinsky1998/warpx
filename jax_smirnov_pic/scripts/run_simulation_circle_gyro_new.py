#!/usr/bin/env python3
from __future__ import annotations

import argparse

from jax_smirnov_pic import run_simulation, smirnov_default_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", action="store_true", help="Enable detailed per-step profiling.")
    parser.add_argument(
        "--profile-summary-only",
        action="store_true",
        help="Collect timing summary without writing per-step profiling rows.",
    )
    args = parser.parse_args()
    config = smirnov_default_config()
    run_simulation(config, profile=args.profile, profile_summary_only=args.profile_summary_only)


if __name__ == "__main__":
    main()
