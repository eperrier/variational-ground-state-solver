# Training settings

Training settings are exposed in the UI and map to `TrainConfig` in `src/psitest/trainer.py`.

The optimizer is AdamW by default, with optional LBFGS refinement.

## Objective

At each step the model produces ψ(x). The trainer normalizes ψ on the grid:

- ψₙ(x) = ψ(x) / sqrt(∫|ψ|² dx)

Then computes energy:

- E = ∫ [ 0.5 |dψₙ/dx|² + V(x) |ψₙ|² ] dx

For excited states, an additional orthogonality penalty is used:

- loss = E + lam_ortho Σ⟨ψₙ|ψ_prev⟩²

## UI parameters

### steps
Number of AdamW update steps.

Typical guidance:

- Ground state: 2000–8000
- Excited states: often 4000–12000 per state depending on architecture

### lr
AdamW learning rate.

- Start around 1e-3 to 3e-3.
- If E oscillates or diverges, reduce lr.

### weight_decay
AdamW weight decay.

- Default 1e-6 (usually mild). Can be set to 0 if you want pure Adam.

### grad_clip
Global gradient norm clipping.

- Helps prevent unstable spikes.

### LBFGS refinement
If enabled, after AdamW the run performs an LBFGS pass.

- Useful for squeezing out the final few digits of convergence.
- If enabled, set `lbfgs_steps` (default 250).

### lam_ortho
Orthogonality penalty strength.

- Ground state tab defaults to 0.
- Excited state tab defaults to 80.

Tuning tips:

- If state n collapses to state 0, increase lam_ortho.
- If training becomes very slow or fails to reduce energy, decrease lam_ortho.

## Diagnostics recorded

The trainer logs per-step:

- E: energy
- K: kinetic contribution
- U: potential contribution
- loss: total objective including penalties
- ortho_pen: Σ⟨ψ|ψ_prev⟩²
- boundary_mass: probability mass near boundaries
- walltime_s: elapsed wall time

These are saved to CSV in `logs/`.
