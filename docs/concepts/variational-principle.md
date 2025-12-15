# Variational method

The core idea of the webapp is the variational characterization of the ground state:

- For any normalizable trial wavefunction ψ, the energy expectation value
  E[ψ] = ⟨ψ|H|ψ⟩ / ⟨ψ|ψ⟩
  is an upper bound on the true ground energy E0.

Equality is achieved when ψ is the true ground eigenfunction.

## The Hamiltonian

We consider a 1D Hamiltonian of the form:

- H = -(1/2) d²/dx² + V(x)

## Stable energy computation

Rather than computing second derivatives, we use the identity:

- ⟨ψ|H|ψ⟩ = ∫ [ 0.5 |dψ/dx|² + V(x) |ψ|² ] dx

This is numerically stable on a grid.

## Why normalization is handled inside training

The trainer normalizes ψ on every step before evaluating the objective. This ensures:

- The optimizer cannot “cheat” by shrinking ψ to reduce numerator and denominator together.
- The scale of ψ does not affect gradient magnitudes as strongly.

## Connection to neural wavefunctions

The webapp chooses a parameterized family ψθ(x) (e.g. an MLP) and performs gradient descent on θ.

This produces:

- an energy estimate (upper bound)
- a wavefunction approximation

The finite-difference eigensolver provides an independent benchmark.
