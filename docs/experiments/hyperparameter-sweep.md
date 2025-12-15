# Hyperparameter sweep

This module runs multiple short **ground-state** trainings with randomized hyperparameters and compares the resulting energies.

It is useful for:

- Understanding optimization sensitivity
- Finding stable learning rates and model sizes
- Comparing architectures on the same potential

## How the sweep works

Each trial:

1. Samples a training config (AdamW settings) and a model parameter set from the sweep ranges.
2. Builds a fresh ground-state model (positive amplitude bias).
3. Trains for the configured number of steps.
4. Records final energy and metadata into a results table.

The sampler uses a fixed RNG seed (`1234`) so that a given sweep configuration is reproducible.

## Settings

### Potential and grid

Same as other tabs:

- Potential + potential parameters
- Use suggested L / L
- N (grid points). In sweep mode, N options are: 512, 1024, 2048, 4096

### Model selection

- **Architecture / ansatz**: selects which model family is used.
- The sweep UI does not expose every architecture parameter.

What changes per-trial is controlled by the sweep ranges. Some architecture-specific parameters are held fixed inside the sweep runner:

- `plain_mlp`: activation fixed to tanh
- `fourier_mlp`: activation tanh, num_fourier=24, ff_scale=2.5, include_x=1
- `siren`: omega0=30.0
- `rbf`: M=48, init_span=3.0, init_sigma=0.7

### Sweep controls

- **n_trials**: number of trials (default 12)
- **steps per trial**: training steps for each trial (default 1500)
- **dtype**: float64 or float32
- **run tag**: becomes part of the output folder name

### Sweep ranges

These are the value sets the sampler draws from:

- **lr choices**: learning rate choices for AdamW
- **hidden choices**: network width choices (or used as a proxy size parameter)
- **depth choices**: network depth choices
- **envelope a choices**: envelope Gaussian strength

## Outputs

The sweep run folder includes:

- `data/config.json`: sweep configuration
- `data/sweep_results.csv`: full results table with final energy per trial

The UI shows:

- A table of the top trials (lowest energies)
- Plotly charts:
  - Energy vs learning rate (colored by hidden, symbol by depth)
  - Energy vs hidden (colored by depth)

## How to interpret results

- Lower energy is better for ground state (variational upper bound).
- Watch for trials with large boundary mass; they may be artificially low or unstable.
- If energies are noisy across trials, reduce the lr options or increase steps per trial.

