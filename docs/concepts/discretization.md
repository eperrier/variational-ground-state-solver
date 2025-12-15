# Discretization and numerics

All variational experiments discretize the 1D domain on a uniform grid.

## Grid

The grid is defined by:

- Half-width L (domain is [-L, L])
- N grid points
- dx = spacing = (2L)/(N-1)

The helper is `make_grid` in `src/psitest/grid.py`.

## Integrals

Integrals are computed with the trapezoidal rule on a uniform grid:

- ∫ f(x) dx ≈ dx · [0.5 f0 + f1 + ... + f_{N-2} + 0.5 f_{N-1}]

This is implemented as `trapz_uniform`.

## Derivatives

Derivatives use central differences on interior points and one-sided differences at boundaries:

- ψ'(x_i) ≈ (ψ_{i+1} - ψ_{i-1})/(2 dx)

This is implemented as `grad_central`.

## Boundary conditions

The finite-difference eigensolver uses Dirichlet boundaries:

- ψ(-L)=ψ(L)=0

The variational models enforce a similar behavior via the envelope factor (see [Models and ansätze](../reference/models.md)).

## Boundary mass diagnostic

A common failure mode is using a domain that is too small. The app tracks:

- boundary mass = probability mass in the outer 10% of the domain

If this is not small, the wavefunction is “pushing into” the boundary and the computed energy can be distorted.

## Choosing L and N

- Increase L if boundary mass is large.
- Increase N if training seems stable but the wavefunction looks jagged or the FD reference changes with resolution.

A typical baseline for confining potentials in this repo is:

- L ≈ 8
- N ≈ 4096
