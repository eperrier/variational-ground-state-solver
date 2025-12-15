# Models and ansätze

Wavefunction models are defined in `src/psitest/models.py`.

All models implement the same high-level ansatz:

- ψ(x) = envelope(x) · amplitude(x)

## The envelope

All architectures share a “Dirichlet-ish” envelope:

- envelope(x) = exp(-a x²) · clamp(1 - (x/L)², 0, +∞)

Properties:

- Enforces ψ(±L)=0 via (1 - (x/L)²)
- Encourages localization via exp(-a x²)

Parameter:

- a (envelope strength): larger means stronger decay

## Ground state vs excited states

The wrapper class `WaveAnsatz` has a `positive` flag:

- `positive=True`: amplitude(x) = Softplus(raw(x))
  - biases toward nodeless ψ (good for ground state)
- `positive=False`: amplitude(x) = raw(x)
  - allows sign changes and nodes (necessary for excited states)

In the webapp:

- Ground state run uses positive=True
- Excited states: state 0 uses positive=True, states 1..K use positive=False

## Available architectures

### 1) plain_mlp

Label: “MLP + envelope”

Core:

- Standard MLP from 1D input to scalar output.

Parameters:

- hidden: width (32, 64, 128, 256)
- depth: number of hidden layers (2..5)
- act: activation {tanh, silu, gelu}
- a: envelope strength

Use cases:

- Baseline ground state
- Usually sufficient for smooth, low-oscillation eigenfunctions

### 2) fourier_mlp

Label: “Fourier features + MLP + envelope”

Core:

- Maps x to random Fourier features, then runs an MLP.
- Often improves representation of oscillatory functions.

Parameters:

- hidden, depth, act, a
- num_fourier: number of random Fourier frequencies
- ff_scale: frequency scale (larger yields higher-frequency features)
- include_x: 0/1 (whether to concatenate raw x to Fourier features)

Use cases:

- Excited states (oscillations)
- Potentials where eigenfunctions have fine structure

### 3) siren

Label: “SIREN (sin activations) + envelope”

Core:

- Sinusoidal activations with SIREN-style initialization.

Parameters:

- hidden, depth, a
- omega0: base frequency for sine layers

Use cases:

- Excited states and highly oscillatory functions

### 4) rbf

Label: “RBF mixture + envelope”

Core:

- amplitude(x) = Σ w_i exp(-0.5 ((x - μ_i)/σ_i)²)

Parameters:

- M: number of RBF centers
- init_span: initial span for μ_i placement
- init_sigma: initial σ_i
- a: envelope strength

Use cases:

- Interpretable basis-like approximation
- Often good for smooth states; may need more components for high oscillations
