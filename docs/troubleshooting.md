# Troubleshooting

## The app is slow

Common causes:

- Large N (8192) and float64 (more expensive)
- Many steps or many excited states
- CPU-only execution

Actions:

- Reduce N to 2048 or 4096
- Reduce steps
- Use float32 (at the cost of stability)
- Enable GPU (Docker overlay or local CUDA)

## GPU not detected in Docker

- Confirm NVIDIA Container Toolkit is installed.
- Run:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

In the sidebar, the app should report device=cuda and show a GPU name.

## Training diverges or produces NaNs

Actions:

- Reduce lr by 3–10x
- Increase grad_clip (or keep it on) and consider a smaller envelope a
- Switch to float64
- Increase N if derivatives look too coarse

## Boundary mass is large

This usually means the domain is too small.

Actions:

- Increase L
- Optionally increase envelope a
- Re-run and check boundary mass again

## Ground-state energy does not match FD reference

Some gap is expected (variational is an upper bound), but large gaps can indicate:

- Underpowered model (increase hidden/depth or change architecture)
- Optimization not converged (increase steps or enable LBFGS refinement)
- Domain too small (increase L)
- Resolution too coarse (increase N)

## Excited states collapse to the ground state

Actions:

- Increase lam_ortho
- Use Fourier-MLP or SIREN
- Increase steps per state

## Verifying internal mechanics

To audit a run:

1. In Trace tab, inspect `code_snapshot/` and confirm the objective and discretization.
2. Inspect `data/config.json` for exact parameters.
3. Reproduce from config:

```bash
python scripts/reproduce_from_config.py "1 - RESEARCH/PSITest/runs/<run>/data/config.json"
```

This will create a new run folder and rerun the experiment.
