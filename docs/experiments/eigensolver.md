# Eigensolver (finite difference reference)

This module computes reference eigenvalues and eigenfunctions using a sparse **finite-difference** discretization of the Hamiltonian:

- H = -(1/2) d²/dx² + V(x)

## Discretization

- Uniform grid on [-L, L] with spacing dx.
- Dirichlet boundary conditions: ψ(-L)=ψ(L)=0.
- The second derivative uses a central difference stencil on interior points.

The solver builds a sparse matrix for H on interior points and calls SciPy’s `eigsh` (ARPACK) to compute the smallest k eigenpairs.

Each returned eigenfunction is normalized so that:

- ∫ |ψ(x)|² dx = 1

## Settings

- **Potential** + potential parameters
- **Use suggested L** / **L**
- **N**: grid points (1024, 2048, 4096, 8192, 16384)
- **k eigenstates**: number of eigenpairs to compute (2 to 12)
- **run tag**: output naming

## Visualizations and outputs

The UI shows:

- Potential plot V(x)
- A table of energies E₀, E₁, ...
- Plots of the first few eigenfunctions (up to 6 are plotted)

The run folder includes:

- `data/reference_fd_k.npz` with arrays: x, V, evals, psis
- `figures/potential.png`
- `figures/wavefunctions.png`

## When to use this module

- To benchmark variational runs (ground state or excited states)
- To validate that an excited-state training run has found distinct eigenfunctions
- To sanity-check domain and resolution choices
