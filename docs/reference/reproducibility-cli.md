# Reproduce runs (CLI)

The repo includes a small reproduction tool:

- `scripts/reproduce_from_config.py`

It re-runs an experiment described by a saved `data/config.json` and writes results to a new run folder.

## Usage

```bash
python scripts/reproduce_from_config.py "1 - RESEARCH/PSITest/runs/<run_name>/data/config.json"
```

A new run folder is created with a kind prefix like `repro_ground`, `repro_excited`, etc.

## What is reproduced

- Ground state: re-trains the configured model and writes `final_results.npz` + logs
- Excited states: retrains sequential states and writes `variational_states.npz`
- Sweep: re-runs the sweep using the same RNG seed (1234) and the same choice sets recorded in the config
- FD eigensolver: re-runs `eigsh` with the saved grid and k

## Notes and limitations

- The reproduced run will not be bit-identical in general if you use GPU (non-determinism) or change hardware.
- Use float64 for best numerical stability.
- The goal is **verifiable mechanics**, not perfect determinism.
