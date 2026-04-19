# jax_smirnov_pic

JAX/CUDA reimplementation of Vladimir Smirnov's `SimulationCircleGyroNEW` from
`CPP_2D_PIC_GYRO-main`.

The project aims to reproduce the original physical model while using a
GPU-friendly functional architecture:

- fixed-capacity particle pools
- `alive` masks instead of `append/pop`
- bilinear deposit/gather
- weighted Jacobi Poisson solver on the circular plasma domain
- gyrokinetic electrons
- explicit source/sink bookkeeping

The repository is intentionally compact. The code favors clarity first, but the
core timestep is structured so it can be `jax.jit`-compiled and later optimized.

## Current status

This repository contains the full project scaffold and a first end-to-end code
path for `SimulationCircleGyroNEW`. Since JAX/CUDA is not available in the
current workspace, the code was written from the Smirnov reference and arranged
to be runnable once the target environment is prepared.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For CUDA-enabled JAX, install the wheel matching your CUDA version instead of
the CPU default from `requirements.txt`.

## Run

```bash
python scripts/run_simulation_circle_gyro_new.py
```

The default configuration mirrors the Smirnov scenario and writes outputs to:

```text
outputs/simulation_circle_gyro_new/
```

## Outputs

- `counters.csv`
- `rho_e_*.txt`
- `rho_i_*.txt`
- `phi_*.txt`
- metadata JSON

## Layout

- `SPEC.md` freezes the transferred physics/modeling choices
- `ROADMAP.md` contains the implementation roadmap
- `jax_smirnov_pic/` contains the simulation code
- `scripts/` contains CLI entry points
- `tests/` contains focused numerical checks
