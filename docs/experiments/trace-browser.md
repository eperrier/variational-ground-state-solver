# Trace browser

The Trace tab is the “verifiability” and reproducibility module of the webapp.

It lets you inspect the artifacts saved for each run:

- config.json (all experiment settings)
- env.json (Python, Torch/CUDA, pip freeze, git commit)
- training logs (CSV)
- a snapshot of the code used for the run

## How runs are stored

All runs are stored under:

- 1 - RESEARCH/PSITest/runs/

Each run folder is named:

- <timestamp>_<kind>_<tag>

For details see [Outputs and traceability](../reference/outputs-and-trace.md).

## Using the Trace tab

1. **Select a run** from the dropdown.
2. Inspect `config.json` and `env.json` directly in the UI.
3. Browse log CSVs and view the last ~200 rows.
4. Browse the code snapshot and view files inline.
5. Download the run zip (or create it if it doesn’t exist).

## Why the code snapshot matters

Saving a copy of the exact source files used for each run allows you to:

- Verify the internal mechanics (objective function, discretization, model definition)
- Confirm that later code changes did not affect historical runs
- Reproduce a run using the CLI reproducer
