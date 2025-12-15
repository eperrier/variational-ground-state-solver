# Outputs and traceability

One design goal of PSITest WebApp is that results are **verifiable**. Every run is persisted with enough metadata to reproduce and audit it.

## Output base directory

The base output directory defaults to:

- `1 - RESEARCH/PSITest`

You can override this with an environment variable:

- `PSITEST_OUTPUT_DIR`

In Docker, the app sets this variable to `1 - RESEARCH/PSITest` and mounts `./outputs/` to that path.

## Run directory layout

Each experiment creates a run folder:

- `.../runs/<timestamp>_<kind>_<tag>/`

Inside are standard subfolders:

- `data/`: config and arrays
- `logs/`: CSV logs
- `figures/`: PNG plots
- `checkpoints/`: PyTorch checkpoints
- `code_snapshot/`: a copy of the code used for the run

## Standard artifacts

### data/config.json

A JSON record of:

- experiment kind (ground/excited/sweep/fd)
- potential key and parameters
- grid settings (L, N, dx)
- model key and parameters
- training config
- dtype and device

### data/env.json

Environment capture:

- Python version
- platform
- torch version
- CUDA availability + CUDA version
- GPU name (if available)
- pip freeze (best-effort)
- git commit (best-effort)

### logs/*.csv

- Ground state: `logs/training_log.csv`
- Excited: `logs/state_00_train_log.csv`, ...
- Sweep: results table as CSV

### figures/*.png

All plots produced by the module.

### checkpoints/*.pt

Model checkpoints and final weights.

### code_snapshot/

A “frozen” copy of the relevant repo files for that run:

- `app/streamlit_app.py`
- `src/psitest/` library
- Docker / compose files, requirements

This allows internal mechanics to be audited even if the repo evolves.

## Run zips

After a run completes, a zip archive is created next to the run folder:

- `.../runs/<run_name>.zip`

The UI provides a download button for the zip.

## Recommended verification workflow

1. Inspect `data/config.json` and confirm the settings.
2. Inspect `data/env.json` and confirm the runtime environment.
3. Inspect logs for convergence.
4. Inspect figures.
5. If needed, use `scripts/reproduce_from_config.py` to rerun from config.
