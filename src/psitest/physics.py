from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch


def trapz_uniform(y: torch.Tensor, dx: float) -> torch.Tensor:
    """Trapezoidal integral on a uniform grid (differentiable w.r.t. y)."""
    return dx * (0.5 * y[0] + y[1:-1].sum() + 0.5 * y[-1])


def grad_central(psi: torch.Tensor, dx: float) -> torch.Tensor:
    """Central difference dpsi/dx with one-sided boundary approximations."""
    d = torch.empty_like(psi)
    d[1:-1] = (psi[2:] - psi[:-2]) / (2.0 * dx)
    d[0] = (psi[1] - psi[0]) / dx
    d[-1] = (psi[-1] - psi[-2]) / dx
    return d


@torch.no_grad()
def normalize_psi(psi: torch.Tensor, dx: float) -> torch.Tensor:
    """Normalize so that ∫|psi|^2 dx = 1."""
    nrm = torch.sqrt(trapz_uniform(psi**2, dx))
    return psi / (nrm + 1e-18)


def norm2(psi: torch.Tensor, dx: float) -> torch.Tensor:
    """Compute ⟨psi|psi⟩ = ∫|psi|^2 dx."""
    return trapz_uniform(psi**2, dx)


def energy_components_from_psi(psi: torch.Tensor, Vx: torch.Tensor, dx: float) -> Dict[str, torch.Tensor]:
    """
    Compute energy functional for a possibly unnormalized psi:

      E = (K + U) / ⟨psi|psi⟩
      K = ∫ 0.5 |dpsi/dx|^2 dx
      U = ∫ V(x) |psi|^2 dx

    This form avoids second derivatives and is numerically stable.
    """
    psi2 = psi**2
    nrm = trapz_uniform(psi2, dx)
    dpsi = grad_central(psi, dx)
    K = 0.5 * trapz_uniform(dpsi**2, dx)
    U = trapz_uniform(Vx * psi2, dx)
    E = (K + U) / (nrm + 1e-18)
    return {"E": E, "K": K / (nrm + 1e-18), "U": U / (nrm + 1e-18), "norm": nrm}


def energy_from_normalized_psi(psi_n: torch.Tensor, Vx: torch.Tensor, dx: float) -> Dict[str, torch.Tensor]:
    """Energy components for a normalized psi (∫|psi|^2 dx = 1)."""
    dpsi = grad_central(psi_n, dx)
    K = 0.5 * trapz_uniform(dpsi**2, dx)
    U = trapz_uniform(Vx * (psi_n**2), dx)
    E = K + U
    return {"E": E, "K": K, "U": U}


def overlap(psi_a: torch.Tensor, psi_b: torch.Tensor, dx: float) -> torch.Tensor:
    """Compute ⟨a|b⟩ = ∫ psi_a(x) psi_b(x) dx."""
    return trapz_uniform(psi_a * psi_b, dx)


def boundary_mass(psi_n: torch.Tensor, dx: float, frac: float = 0.10) -> torch.Tensor:
    """Probability mass in the outer frac of the domain (both ends)."""
    n_edge = max(1, int(frac * psi_n.numel()))
    return trapz_uniform(psi_n[:n_edge] ** 2, dx) + trapz_uniform(psi_n[-n_edge:] ** 2, dx)


@dataclass
class EnergyTracePoint:
    step: int
    E: float
    K: float
    U: float
    loss: float
    boundary_mass: float
    ortho_pen: float
