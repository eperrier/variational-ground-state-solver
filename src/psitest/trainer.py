from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .physics import boundary_mass, energy_from_normalized_psi, overlap, trapz_uniform, normalize_psi


@dataclass
class TrainConfig:
    """Training hyperparameters."""

    steps: int = 4000
    lr: float = 2e-3
    weight_decay: float = 1e-6
    grad_clip: float = 1.0
    print_every: int = 250
    ui_update_every: int = 25
    ckpt_every: int = 2000

    use_lbfgs: bool = False
    lbfgs_steps: int = 250

    lam_ortho: float = 80.0  # orthogonality penalty strength for excited states


ProgressCallback = Callable[[int, Dict[str, float]], None]


def train_wavefunction(
    model: nn.Module,
    x: torch.Tensor,
    Vx: torch.Tensor,
    dx: float,
    cfg: TrainConfig,
    orthonormal_to: Optional[List[torch.Tensor]] = None,
    progress: Optional[ProgressCallback] = None,
    checkpoint_fn: Optional[Callable[[int, nn.Module, Dict[str, Any]], None]] = None,
    tag: str = "model",
) -> Dict[str, Any]:
    """Train a wavefunction model.

    Objective:
      loss = E[psi] + lam_ortho * Σ_m <psi|psi_m>^2

    where psi is normalized internally each step.

    Returns dict with:
      - psi (torch tensor, normalized)
      - final metrics
      - df (training log)
    """

    device = x.device
    orthonormal_to = orthonormal_to or []

    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))

    rows: List[Dict[str, float]] = []
    t0 = time.time()

    for step in range(1, int(cfg.steps) + 1):
        opt.zero_grad(set_to_none=True)

        psi = model(x)
        psi_n = psi / (torch.sqrt(trapz_uniform(psi**2, dx)) + 1e-18)

        comps = energy_from_normalized_psi(psi_n, Vx, dx)
        E = comps["E"]
        K = comps["K"]
        U = comps["U"]

        ortho_pen = torch.tensor(0.0, device=device, dtype=psi_n.dtype)
        overlaps: List[torch.Tensor] = []
        for psi_prev in orthonormal_to:
            ov = overlap(psi_n, psi_prev, dx)
            overlaps.append(ov)
            ortho_pen = ortho_pen + ov**2

        loss = E + float(cfg.lam_ortho) * ortho_pen
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.grad_clip))
        opt.step()

        bm = float(boundary_mass(psi_n, dx).detach().cpu())

        row: Dict[str, float] = {
            "step": float(step),
            "E": float(E.detach().cpu()),
            "K": float(K.detach().cpu()),
            "U": float(U.detach().cpu()),
            "loss": float(loss.detach().cpu()),
            "ortho_pen": float(ortho_pen.detach().cpu()),
            "boundary_mass": bm,
            "walltime_s": float(time.time() - t0),
        }
        for j, ov in enumerate(overlaps):
            row[f"overlap_with_{j}"] = float(ov.detach().cpu())

        rows.append(row)

        if progress is not None and (step % int(cfg.ui_update_every) == 0 or step == 1 or step == int(cfg.steps)):
            progress(step, row)

        if step % int(cfg.print_every) == 0 or step == 1:
            dt = time.time() - t0
            print(f"[{tag}] step={step}/{cfg.steps} E={row['E']:.8f} loss={row['loss']:.8f} bm={bm:.2e} t={dt:.1f}s")

        if checkpoint_fn is not None and step % int(cfg.ckpt_every) == 0:
            checkpoint_fn(step, model, {"train_cfg": cfg.__dict__})

    # Optional LBFGS refinement for final polish
    if cfg.use_lbfgs:
        lbfgs = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=int(cfg.lbfgs_steps), line_search_fn="strong_wolfe")

        def closure():
            lbfgs.zero_grad(set_to_none=True)
            psi = model(x)
            psi_n = psi / (torch.sqrt(trapz_uniform(psi**2, dx)) + 1e-18)
            comps = energy_from_normalized_psi(psi_n, Vx, dx)
            E = comps["E"]
            ortho_pen = torch.tensor(0.0, device=device, dtype=psi_n.dtype)
            for psi_prev in orthonormal_to:
                ov = overlap(psi_n, psi_prev, dx)
                ortho_pen = ortho_pen + ov**2
            loss = E + float(cfg.lam_ortho) * ortho_pen
            loss.backward()
            return loss

        lbfgs.step(closure)

    with torch.no_grad():
        psi_final = normalize_psi(model(x), dx)
        comps = energy_from_normalized_psi(psi_final, Vx, dx)
        bm_final = float(boundary_mass(psi_final, dx).detach().cpu())

    df = pd.DataFrame(rows)
    return {
        "model": model,
        "psi": psi_final.detach(),
        "E": float(comps["E"].detach().cpu()),
        "K": float(comps["K"].detach().cpu()),
        "U": float(comps["U"].detach().cpu()),
        "boundary_mass": bm_final,
        "df": df,
    }


def train_states_sequential(
    K_states: int,
    build_model_fn: Callable[[int], nn.Module],
    x: torch.Tensor,
    Vx: torch.Tensor,
    dx: float,
    cfg: TrainConfig,
    progress: Optional[Callable[[str, int, int, Dict[str, float]], None]] = None,
    checkpoint_fn: Optional[Callable[[str, int, nn.Module, Dict[str, Any]], None]] = None,
) -> List[Dict[str, Any]]:
    """Train states sequentially.

    State 0: ground state (typically set build_model_fn(0) to be positive amplitude).
    State n>0: excited (orthogonality penalty against previous states).
    """

    states: List[Dict[str, Any]] = []
    prev_psis: List[torch.Tensor] = []

    for n in range(int(K_states)):
        model = build_model_fn(n)

        def prog(step: int, row: Dict[str, float]):
            if progress is not None:
                progress(f"state_{n}", n, step, row)

        def ckpt(step: int, m: nn.Module, meta: Dict[str, Any]):
            if checkpoint_fn is not None:
                checkpoint_fn(f"state_{n}", step, m, meta)

        res = train_wavefunction(
            model=model,
            x=x,
            Vx=Vx,
            dx=dx,
            cfg=cfg,
            orthonormal_to=prev_psis,
            progress=prog,
            checkpoint_fn=ckpt,
            tag=f"state_{n}",
        )
        states.append(res)
        prev_psis.append(res["psi"])  # already normalized

    return states


def random_sweep(
    n_trials: int,
    sampler: Callable[[int], Tuple[TrainConfig, Dict[str, Any]]],
    build_model_from_params: Callable[[Dict[str, Any]], nn.Module],
    x: torch.Tensor,
    Vx: torch.Tensor,
    dx: float,
    progress: Optional[Callable[[int, int, Dict[str, Any]], None]] = None,
) -> pd.DataFrame:
    """Run a random hyperparameter sweep.

    sampler(trial_index) returns:
      - TrainConfig (for training)
      - model_params dict (for model construction)

    build_model_from_params(model_params) returns nn.Module for ground state.
    """

    rows: List[Dict[str, Any]] = []

    for t in range(int(n_trials)):
        cfg, model_params = sampler(t)
        model = build_model_from_params(model_params)

        def prog(step: int, row: Dict[str, float]):
            # only report coarse progress to UI
            if progress is not None and (step == 1 or step == cfg.steps or step % max(1, cfg.steps // 10) == 0):
                progress(t, step, {"trial": t, "step": step, **row, **model_params, **cfg.__dict__})

        res = train_wavefunction(model, x, Vx, dx, cfg, orthonormal_to=None, progress=prog, checkpoint_fn=None, tag=f"trial_{t}")

        rows.append({
            "trial": t,
            **model_params,
            **cfg.__dict__,
            "E": res["E"],
            "K": res["K"],
            "U": res["U"],
            "boundary_mass": res["boundary_mass"],
        })

    return pd.DataFrame(rows)

# --- Comparison utilities (variational vs reference eigensolver) ---

from scipy.optimize import linear_sum_assignment


def overlap_matrix_np(psi_a: np.ndarray, psi_b: np.ndarray, dx: float) -> np.ndarray:
    """Compute |⟨a_i|b_j⟩| for two sets of wavefunctions.

    psi_a: (KA, N), psi_b: (KB, N)
    """
    return np.abs(dx * (psi_a @ psi_b.T))


def match_states_by_overlap(psi_nn: np.ndarray, psi_ref: np.ndarray, dx: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Match NN states to reference states by maximum overlap.

    Returns (O, nn_indices, ref_indices).
    """
    O = overlap_matrix_np(psi_nn, psi_ref, dx)
    nn_idx, ref_idx = linear_sum_assignment(-O)
    return O, nn_idx, ref_idx


def compare_states_to_reference(
    psi_nn: np.ndarray,
    E_nn: np.ndarray,
    psi_ref: np.ndarray,
    E_ref: np.ndarray,
    dx: float,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Return a comparison DataFrame and overlap matrix.

    - Matches by overlap.
    - Reports per-state energy deltas and overlaps.
    """
    # Normalize defensively
    psi_nn = psi_nn / np.sqrt(dx * np.sum(psi_nn**2, axis=1, keepdims=True) + 1e-30)
    psi_ref = psi_ref / np.sqrt(dx * np.sum(psi_ref**2, axis=1, keepdims=True) + 1e-30)

    K = min(psi_nn.shape[0], psi_ref.shape[0])
    psi_nn = psi_nn[:K]
    psi_ref = psi_ref[:K]
    E_nn = np.array(E_nn[:K], dtype=np.float64)
    E_ref = np.array(E_ref[:K], dtype=np.float64)

    O, nn_idx, ref_idx = match_states_by_overlap(psi_nn, psi_ref, dx)

    rows = []
    for i, j in zip(nn_idx, ref_idx):
        raw_ov = dx * np.sum(psi_nn[i] * psi_ref[j])
        ov = float(abs(raw_ov))
        rows.append({
            "nn_state": int(i),
            "ref_state": int(j),
            "E_nn": float(E_nn[i]),
            "E_ref": float(E_ref[j]),
            "delta": float(E_nn[i] - E_ref[j]),
            "abs_overlap": ov,
        })

    df = pd.DataFrame(rows).sort_values("ref_state")
    return df, O
