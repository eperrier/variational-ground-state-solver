# PSITest WebApp

A local web application (Streamlit) that runs variational Schrödinger experiments.

## What it can do

- **Core experiment**: variational ground state energy and wavefunction for a 1D Hamiltonian
  \(H = -\tfrac{1}{2}\tfrac{d^2}{dx^2} + V(x)\).
- **Excited states**: sequential training with orthogonality penalties.
- **Different potentials**: harmonic, double-well, quartic families (including the assignment potential \(V(x)=x^4-2x^2+2x\)).
- **Different architectures / ansätze**:
  - MLP + envelope (Dirichlet-ish boundary enforcement)
  - Fourier-feature MLP (often helps oscillatory modes)
  - SIREN-style sinusoidal network
  - RBF mixture basis
- **Hyperparameter sweeps**: random sweeps over learning rate, width, depth, envelope strength, etc.
- **Numerical eigensolver**: sparse finite-difference eigensolver (Dirichlet BC) to benchmark the variational results.

## Output / traceability

All experiment outputs are written under:

- `1 - RESEARCH/PSITest/` (by default) and subfolders per-run.

Each run saves:

- `data/config.json` (all parameters)
- `data/env.json` (Python, torch/CUDA info, OS, package versions)
- `logs/training_log.csv` (per-step metrics)
- `figures/*.png` (plots)
- `checkpoints/*.pt` (model checkpoints)
- `code_snapshot/` (a copy of the exact source files used to run the experiment)

This makes the internal mechanics and results verifiable.

## Quick start (Docker)

### 1) Build + run

```bash
docker compose up --build
```

Then open:

- `http://localhost:8501`

### 2) Persist outputs on your host machine

By default, `docker-compose.yml` mounts `./outputs` on the host to `/workspace/1 - RESEARCH/PSITest` inside the container.

### 3) GPU (optional)

If you have an NVIDIA GPU and the NVIDIA Container Toolkit installed, run:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

## Local (non-Docker) run

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

## Repo structure

- `app/streamlit_app.py` : Streamlit UI
- `src/psitest/` : reusable experiment library
- `assets/` : static assets

