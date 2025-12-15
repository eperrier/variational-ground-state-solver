# Potentials

Potentials are defined in `src/psitest/potentials.py`.

In the UI, potentials are selected by key, and some potentials expose additional parameters.

Each potential also has a “suggested L” heuristic that tries to pick a safe domain half-width for the box [-L, L].

## Available potentials

### 1) quartic_tilted (assignment)

Key: `quartic_tilted`

Formula:

- V(x) = x⁴ - 2 x² + 2 x

Parameters:

- none (fixed)

Suggested L:

- 8.0

### 2) harmonic

Key: `harmonic`

Formula:

- V(x) = 0.5 (ω²) x²

Parameters:

- ω (omega): frequency (default 1.0)

Suggested L heuristic:

- max(5.0, 6.0 / sqrt(omega))

### 3) double_well

Key: `double_well`

Formula:

- V(x) = x⁴ - a x²

Parameters:

- a: controls well separation/depth (default 2.0)

Suggested L heuristic:

- max(7.0, 6.0 + 0.8 * abs(a))

### 4) quartic_family

Key: `quartic_family`

Formula:

- V(x) = α x⁴ + β x² + γ x + δ

Parameters (defaults shown):

- α (alpha): x⁴ coefficient (default 1.0)
- β (beta): x² coefficient (default -2.0)
- γ (gamma): x coefficient (default 2.0)
- δ (delta): constant offset (default 0.0)

Suggested L heuristic:

- max(6.0, 8.0 / (alpha^(1/4) + 1e-9))

## Notes on choosing L

Even with a suggested L, you should validate domain size with the **boundary mass** diagnostic:

- boundary mass = probability mass in the outer 10% of the domain.

If the boundary mass is not small:

- increase L
- and/or increase the envelope strength `a`

A too-small domain can artificially distort the wavefunction and energy.
