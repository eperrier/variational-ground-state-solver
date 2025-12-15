# Getting started

## Prerequisites

You can run PSITest WebApp in one of two ways:

- **Docker (recommended)**: simplest and closest to how the repo is set up.
- **Local Python environment**: useful if you want to hack on code or iterate quickly.

### Hardware

- CPU-only machines work.
- For GPU acceleration, you need an NVIDIA GPU and the NVIDIA Container Toolkit (for Docker) or a CUDA-enabled PyTorch install (for local).

## What the app does

The UI exposes four experiment modules plus a trace browser:

- **Ground state**: train a parameterized wavefunction \(\psi_\theta(x)\) by minimizing the variational energy.
- **Excited states**: sequentially train \(\psi_0, \psi_1, \dots\) with orthogonality penalties.
- **Hyperparameter sweep**: run many short trainings with randomized settings and compare outcomes.
- **Eigensolver**: compute reference eigenvalues/eigenfunctions with a sparse finite-difference solver.
- **Trace**: browse saved runs (config, environment, logs, code snapshot).

## Where outputs go

All outputs are written under:

- `1 - RESEARCH/PSITest/runs/<timestamp>_<kind>_<tag>/...`

You can override the base directory with the environment variable:

- `PSITEST_OUTPUT_DIR`

The Docker setup mounts `./outputs/` on your host machine to the container path used for outputs, so that runs persist after you stop the container.
