#!/usr/bin/env python3
from __future__ import annotations

from jax_smirnov_pic import run_simulation, smirnov_default_config


def main() -> None:
    config = smirnov_default_config()
    run_simulation(config)


if __name__ == "__main__":
    main()
