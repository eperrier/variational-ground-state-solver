"""Reproduce a PSITest run from a saved config.json.

Usage:
  python scripts/reproduce_from_config.py "1 - RESEARCH/PSITest/runs/<run>/data/config.json"

Notes:
- This is a simple reproduction utility intended for verification.
- It re-runs the configured experiment and writes results to a new run folder.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

from psitest.grid import GridConfig, make_grid
from psitest.models import ModelConfig, build_model
from psitest.potentials import PotentialConfig, build_potential
from psitest.solver_fd import fd_eigensolve_k
from psitest.trainer import TrainConfig, train_wavefunction, train_states_sequential, random_sweep
from psitest.trace import RunContext


def main(cfg_path: Path):
    cfg = json.loads(cfg_path.read_text())
    kind = cfg.get("experiment_kind")

    # dtype
    dtype_choice = cfg.get("dtype", "float64")
    dtype = torch.float64 if dtype_choice == "float64" else torch.float32
    torch.set_default_dtype(dtype)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pot_cfg = PotentialConfig(key=cfg["potential"]["key"], params=cfg["potential"].get("params") or {})
    V_torch, V_numpy, _L_suggested, _pfull = build_potential(pot_cfg)

    grid = cfg["grid"]
    L = float(grid["L"])
    N = int(grid["N"])

    if kind in ["ground", "excited", "sweep"]:
        x_np, x, dx = make_grid(GridConfig(L=L, N=N), device=device, dtype=dtype)
        V_np = V_numpy(x_np)
        Vx = V_torch(x)

    ctx = RunContext.create(kind=f"repro_{kind}", tag=cfg_path.parents[1].name)
    ctx.save_env()
    ctx.snapshot_repo()
    ctx.save_config(cfg)

    if kind == "fd":
        x_np = np.linspace(-L, L, N)
        dx = float(x_np[1] - x_np[0])
        V_np = V_numpy(x_np)
        k = int(cfg.get("k", 6))
        evals, psis = fd_eigensolve_k(x_np, V_np, dx, k=k)
        np.savez(ctx.data_dir / "reference_fd_k.npz", x=x_np, V=V_np, evals=evals, psis=psis)
        print("Reproduced FD eigensolver. E0:", evals[0])
        return

    if kind == "ground":
        train_cfg = TrainConfig(**cfg["train"])
        model_cfg = ModelConfig(key=cfg["model"]["key"], params=cfg["model"].get("params") or {})
        model = build_model(model_cfg, L=L, positive=True).to(device)
        res = train_wavefunction(model, x, Vx, dx, train_cfg, orthonormal_to=None)
        res["df"].to_csv(ctx.logs_dir / "training_log.csv", index=False)
        np.savez(ctx.data_dir / "final_results.npz", x=x_np, V=V_np, psi=res["psi"].detach().cpu().numpy(), E=res["E"], K=res["K"], U=res["U"], boundary_mass=res["boundary_mass"])
        print("Reproduced ground state. E:", res["E"])
        return

    if kind == "excited":
        train_cfg = TrainConfig(**cfg["train"])
        model_cfg = ModelConfig(key=cfg["model"]["key"], params=cfg["model"].get("params") or {})
        K_states = int(cfg.get("K_states", 4))

        def build_model_fn(n: int):
            positive = (n == 0)
            return build_model(model_cfg, L=L, positive=positive).to(device)

        states = train_states_sequential(K_states, build_model_fn, x, Vx, dx, train_cfg)
        energies = [s["E"] for s in states]
        psis = np.stack([s["psi"].detach().cpu().numpy() for s in states], axis=0)
        np.savez(ctx.data_dir / "variational_states.npz", x=x_np, V=V_np, E=np.array(energies), psis=psis)
        print("Reproduced excited states. Energies:", energies)
        return

    if kind == "sweep":
        # This is a minimal reproduction: rerun the sampler choices recorded in config.
        # In the webapp we used randomness; here we reproduce the same random seed.
        sw = cfg["sweep"]
        n_trials = int(sw["n_trials"])
        steps = int(sw["steps"])
        lr_choices = sw["lr_choices"]
        hidden_choices = sw["hidden_choices"]
        depth_choices = sw["depth_choices"]
        a_choices = sw["a_choices"]

        model_key = cfg["model"]["key"]
        rng = np.random.default_rng(1234)

        def sampler(t: int):
            lr = float(rng.choice(lr_choices))
            hidden = int(rng.choice(hidden_choices))
            depth = int(rng.choice(depth_choices))
            a = float(rng.choice(a_choices))
            model_params = {"hidden": hidden, "depth": depth, "a": a}
            if model_key in ["plain_mlp", "fourier_mlp"]:
                model_params["act"] = "tanh"
            if model_key == "fourier_mlp":
                model_params.update({"num_fourier": 24, "ff_scale": 2.5, "include_x": 1})
            if model_key == "siren":
                model_params.update({"omega0": 30.0})
            if model_key == "rbf":
                model_params.update({"M": 48, "init_span": 3.0, "init_sigma": 0.7})
            train_cfg = TrainConfig(steps=steps, lr=lr, weight_decay=1e-6, grad_clip=1.0, print_every=10_000, ui_update_every=max(10, steps // 10), ckpt_every=10_000, use_lbfgs=False, lam_ortho=0.0)
            return train_cfg, model_params

        def build_model_from_params(model_params):
            return build_model(ModelConfig(key=model_key, params=model_params), L=L, positive=True).to(device)

        df = random_sweep(n_trials, sampler, build_model_from_params, x, Vx, dx)
        df.to_csv(ctx.data_dir / "sweep_results.csv", index=False)
        print("Reproduced sweep. Best E:", df["E"].min())
        return

    raise ValueError(f"Unknown experiment_kind: {kind}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python scripts/reproduce_from_config.py <path/to/config.json>")
    main(Path(sys.argv[1]))
