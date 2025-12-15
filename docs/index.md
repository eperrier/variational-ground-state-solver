# PSITest WebApp

PSITest WebApp is a local Streamlit application for running and validating **variational** (neural) and **finite-difference** solvers for 1D Schrödinger eigenproblems.

The app was built to support a take-home task: use gradient-based optimization to approximate the ground state wavefunction of

\[
H = -\tfrac{1}{2}\tfrac{d^2}{dx^2} + V(x),
\qquad
V(x)=x^4-2x^2+2x.
\]

It also includes extensions:

- Excited states via sequential training + orthogonality penalties
- Multiple potentials (harmonic, double well, quartic families)
- Multiple model architectures / ansätze
- Hyperparameter sweeps
- A numerical eigensolver benchmark (sparse finite differences)
- Traceability: configs, logs, figures, checkpoints, and a code snapshot per run

## How the project is structured

- `app/streamlit_app.py`: the Streamlit UI
- `src/psitest/`: the experiment library used by the UI
  - `potentials.py`: potential definitions and suggested domain sizes
  - `models.py`: wavefunction ansätze
  - `trainer.py`: variational training + sweeps + comparison utilities
  - `solver_fd.py`: finite-difference eigensolver (reference)
  - `trace.py`: run folders, environment capture, code snapshots, zip export
  - `plotting.py`: matplotlib plots
- `docker-compose.yml`: local deployment with persisted output volume

## What you should read first

- **Getting started**: how to run the app (Docker or local)
- **Ground state (variational)**: the core experiment and settings
- **Outputs and traceability**: where results are stored and how to verify a run

