# Ground state experiment (variational)

This module trains a parameterized wavefunction \(\psi_\theta(x)\) to approximate the **ground state** of the 1D Hamiltonian

\[
H = -\tfrac{1}{2}\tfrac{d^2}{dx^2} + V(x).
\]

It uses a standard variational objective:

\[
E[\psi] = \frac{\langle \psi | H | \psi \rangle}{\langle \psi | \psi \rangle}.
\]

In code, the energy is computed in a numerically stable form (no second derivatives):

\[
\langle \psi|H|\psi\rangle = \int \left(\tfrac{1}{2}|\partial_x\psi|^2 + V(x)|\psi|^2\right)dx.
\]

## How to run

1. Select a potential and (if needed) adjust its parameters.
2. Choose the domain and grid resolution.
3. Choose a wavefunction architecture/ansatz and its parameters.
4. Set training hyperparameters.
5. Click **Run ground-state experiment**.

A run folder is created under `1 - RESEARCH/PSITest/runs/...` containing config/logs/figures/checkpoints and a code snapshot.

## Settings

### Potential

- **Potential**: dropdown of available potentials.
- **Potential parameters**: depends on the selected potential (see [Potentials](../reference/potentials.md)).
- The app displays a **Suggested L** for the selected potential.

### Grid

- **Use suggested L**: if enabled, uses the potential-specific recommended half-width \(L\).
- **L (domain half-width)**: defines the computational domain \([-L, L]\).
- **N (grid points)**: number of uniform grid points. Options: 1024, 2048, 4096, 8192.

Notes:

- Larger \(L\) reduces boundary artifacts but increases compute.
- Larger \(N\) improves derivative/integration accuracy but increases compute.

### Architecture / ansatz

- **Architecture / ansatz**: choose a model family (MLP, Fourier-MLP, SIREN, RBF).
- **Model parameters**: shown below the architecture selector. See [Models and ansätze](../reference/models.md).

The ground state run constructs the model with `positive=True`, which applies a Softplus to the model output to bias toward a **nodeless** wavefunction.

### Training

- **steps**: training steps (default 4000)
- **lr**: learning rate (default 2e-3)
- **weight_decay**: AdamW weight decay (default 1e-6)
- **grad_clip**: gradient norm clipping (default 1.0)
- **LBFGS refinement**: optional second-stage optimizer for final polish
- **lbfgs_steps**: number of LBFGS iterations (default 250)
- **lam_ortho**: orthogonality penalty strength
  - For ground state this should usually be **0** (default in this tab).

For details and tuning guidance see [Training settings](../reference/training.md).

### Other controls

- **Compute FD reference (k=6)**: runs the finite-difference eigensolver (first 6 eigenpairs) and displays \(E_0\) as a benchmark.
- **dtype**: float64 (default) or float32.
  - float64 is more numerically stable for this physics-style optimization.
- **run tag**: becomes part of the run folder name.

## Visualizations and outputs

After the run completes, the UI shows:

- **Potential plot**: \(V(x)\) over \([-L, L]\)
- **Training curves**: energy `E` and training `loss` vs step
- **Boundary mass**: probability mass near the boundaries (outer 10% on each side). This should be small if \(L\) is large enough.
- **Wavefunction comparison**: \(\psi_\text{nn}(x)\) overlaid with \(\psi_\text{ref}(x)\) if reference was computed.

The run folder contains:

- `data/config.json`, `data/env.json`
- `logs/training_log.csv`
- `figures/potential.png`, `figures/training_curves.png`, `figures/boundary_mass.png`
- `checkpoints/model_final.pt` and intermediate checkpoints
- `code_snapshot/` (exact code used)

## Verifying the run

Recommended verification steps:

1. Inspect `data/config.json` for the exact settings.
2. Inspect `logs/training_log.csv` for energy convergence.
3. Confirm the boundary mass is small.
4. If FD reference is enabled, compare `E` vs `E_ref`.
5. Use the **Trace** tab to inspect the code snapshot.

