# Excited states and orthogonality

The excited-state module uses a sequential variational strategy:

1. Train ψ0 as the ground state.
2. Train ψ1 with a penalty that enforces ⟨ψ1|ψ0⟩ ≈ 0.
3. Train ψ2 with penalties against ψ0 and ψ1, and so on.

## Why orthogonality matters

Eigenfunctions of a Hermitian Hamiltonian form an orthonormal basis.

For the n-th eigenstate, we require:

- ⟨ψn|ψm⟩ = 0 for m < n

Without this, minimizing energy would always collapse to the ground state.

## Penalty objective

The trainer uses:

- loss = E[ψn] + λ Σ_m<n ⟨ψn|ψm⟩²

Where λ is `lam_ortho`.

## Practical tuning of lam_ortho

- Too small: states collapse to lower states.
- Too large: optimization becomes stiff and may get stuck.

A good starting point is 50–150, but it depends on the potential, grid, and architecture.

## Architecture considerations

Excited states require the model to represent nodes and oscillations.

In practice:

- Fourier-MLP and SIREN typically handle oscillations best.
- Plain MLP can work for low-lying states but may struggle for higher n.
- RBF mixtures can work but may need large M for higher states.
