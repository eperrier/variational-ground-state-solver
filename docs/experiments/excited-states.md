# Excited states

This module finds approximate excited states \(\psi_1, \psi_2, \dots\) by training wavefunctions sequentially with an **orthogonality penalty**.

## Method

State 0 (ground state) is trained by minimizing the energy:

\[
\psi_0 = \arg\min_{\psi} E[\psi].
\]

For state \(n>0\), the training objective is:

\[
\mathcal{L}[\psi_n] = E[\psi_n] + \lambda \sum_{m < n} \langle \psi_n | \psi_m \rangle^2.
\]

- The \(\langle\cdot|\cdot\rangle\) overlap is computed with numerical integration on the grid.
- Each \(\psi_n\) is normalized each step before computing energy and overlaps.
- The penalty strength is `lam_ortho`.

This is simple and robust, but it is still an approximation: the quality depends on the architecture, \(L\), \(N\), and optimization settings.

## Settings

### Potential

Same as the ground state tab: choose potential + parameters, optionally accept the suggested \(L\).

### Grid

- **Use suggested L** (default true)
- **L**
- **N**: 1024, 2048, 4096, 8192

### Architecture / ansatz

Same architecture selection as ground state. For excited states:

- State 0 is built with `positive=True` (Softplus output) to bias toward node-less ground state.
- States \(n>0\) are built with `positive=False` to allow sign changes and nodes.

### Number of states

- **# states** (`K_states`): number of sequential states to train (2 to 8).

### Training

Same fields as [Training settings](../reference/training.md), but note:

- `lam_ortho` defaults to **80** in this tab.
- You may need to tune `lam_ortho`:
  - Too small: excited states may collapse back to ground state.
  - Too large: optimization can become stiff and slow.

### Reference comparison

- **Compute FD reference**: computes reference eigenpairs using the FD eigensolver (`k = max(6, K_states)`).
- After training, the app computes an **overlap matrix** between NN states and reference states and matches states by maximum overlap.

## Visualizations and outputs

The UI shows:

- Potential plot
- A table of trained energies \(E_n\)
- Plot of the trained wavefunctions \(\psi_n(x)\)
- If reference is computed:
  - a comparison table (NN energy vs reference energy, deltas, overlaps)
  - an overlap matrix plot

The run folder includes:

- `logs/state_00_train_log.csv`, `logs/state_01_train_log.csv`, ...
- `data/variational_states.npz` (all \(\psi_n\) and \(E_n\))
- If reference is computed: `data/reference_fd_k.npz` and `data/comparison_to_reference.csv`

## Practical tips

- If the boundary mass is not small, increase \(L\).
- If excited states collapse, increase `lam_ortho` or use an architecture that can represent oscillations (Fourier-MLP or SIREN).
- If training is noisy, reduce `lr` or enable LBFGS refinement.
