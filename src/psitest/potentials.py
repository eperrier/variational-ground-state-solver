from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import numpy as np
import torch


@dataclass(frozen=True)
class PotentialConfig:
    """User-facing potential selection.

    key: one of available_potentials()
    params: dictionary of numeric parameters for that potential.
    """

    key: str = "quartic_tilted"
    params: Dict[str, float] = None

    def with_defaults(self) -> "PotentialConfig":
        spec = POTENTIALS[self.key]
        p = dict(spec.default_params)
        if self.params:
            p.update(self.params)
        return PotentialConfig(key=self.key, params=p)


@dataclass(frozen=True)
class PotentialSpec:
    key: str
    label: str
    default_params: Dict[str, float]
    # function to suggest a safe half-domain width L
    suggest_L: Callable[[Dict[str, float]], float]
    # functions to evaluate V(x)
    torch_fn: Callable[[torch.Tensor, Dict[str, float]], torch.Tensor]
    numpy_fn: Callable[[np.ndarray, Dict[str, float]], np.ndarray]


def _suggest_L_quartic(_: Dict[str, float]) -> float:
    return 8.0


def _suggest_L_harmonic(params: Dict[str, float]) -> float:
    # For harmonic oscillator, most mass is within a few sigmas.
    omega = float(params.get("omega", 1.0))
    return max(5.0, 6.0 / np.sqrt(omega))


def _suggest_L_double_well(params: Dict[str, float]) -> float:
    a = float(params.get("a", 2.0))
    # Larger |a| means wells further out; choose a slightly larger box.
    return float(max(7.0, 6.0 + 0.8 * abs(a)))


def _suggest_L_quartic_family(params: Dict[str, float]) -> float:
    alpha = float(params.get("alpha", 1.0))
    return float(max(6.0, 8.0 / (alpha ** 0.25 + 1e-9)))


def _V_quartic_tilted_torch(x: torch.Tensor, _: Dict[str, float]) -> torch.Tensor:
    return x**4 - 2.0 * x**2 + 2.0 * x


def _V_quartic_tilted_numpy(x: np.ndarray, _: Dict[str, float]) -> np.ndarray:
    return x**4 - 2.0 * x**2 + 2.0 * x


def _V_harmonic_torch(x: torch.Tensor, params: Dict[str, float]) -> torch.Tensor:
    omega = float(params.get("omega", 1.0))
    return 0.5 * (omega**2) * x**2


def _V_harmonic_numpy(x: np.ndarray, params: Dict[str, float]) -> np.ndarray:
    omega = float(params.get("omega", 1.0))
    return 0.5 * (omega**2) * x**2


def _V_double_well_torch(x: torch.Tensor, params: Dict[str, float]) -> torch.Tensor:
    a = float(params.get("a", 2.0))
    return x**4 - a * x**2


def _V_double_well_numpy(x: np.ndarray, params: Dict[str, float]) -> np.ndarray:
    a = float(params.get("a", 2.0))
    return x**4 - a * x**2


def _V_quartic_family_torch(x: torch.Tensor, params: Dict[str, float]) -> torch.Tensor:
    # V(x) = alpha x^4 + beta x^2 + gamma x + delta
    alpha = float(params.get("alpha", 1.0))
    beta = float(params.get("beta", -2.0))
    gamma = float(params.get("gamma", 2.0))
    delta = float(params.get("delta", 0.0))
    return alpha * x**4 + beta * x**2 + gamma * x + delta


def _V_quartic_family_numpy(x: np.ndarray, params: Dict[str, float]) -> np.ndarray:
    alpha = float(params.get("alpha", 1.0))
    beta = float(params.get("beta", -2.0))
    gamma = float(params.get("gamma", 2.0))
    delta = float(params.get("delta", 0.0))
    return alpha * x**4 + beta * x**2 + gamma * x + delta


POTENTIALS: Dict[str, PotentialSpec] = {
    "quartic_tilted": PotentialSpec(
        key="quartic_tilted",
        label="Assignment: x^4 - 2 x^2 + 2 x",
        default_params={},
        suggest_L=_suggest_L_quartic,
        torch_fn=_V_quartic_tilted_torch,
        numpy_fn=_V_quartic_tilted_numpy,
    ),
    "harmonic": PotentialSpec(
        key="harmonic",
        label="Harmonic: 0.5 * (omega^2) * x^2",
        default_params={"omega": 1.0},
        suggest_L=_suggest_L_harmonic,
        torch_fn=_V_harmonic_torch,
        numpy_fn=_V_harmonic_numpy,
    ),
    "double_well": PotentialSpec(
        key="double_well",
        label="Double well: x^4 - a * x^2",
        default_params={"a": 2.0},
        suggest_L=_suggest_L_double_well,
        torch_fn=_V_double_well_torch,
        numpy_fn=_V_double_well_numpy,
    ),
    "quartic_family": PotentialSpec(
        key="quartic_family",
        label="Quartic family: alpha x^4 + beta x^2 + gamma x + delta",
        default_params={"alpha": 1.0, "beta": -2.0, "gamma": 2.0, "delta": 0.0},
        suggest_L=_suggest_L_quartic_family,
        torch_fn=_V_quartic_family_torch,
        numpy_fn=_V_quartic_family_numpy,
    ),
}


def available_potentials() -> Dict[str, str]:
    """Mapping key -> label."""
    return {k: v.label for k, v in POTENTIALS.items()}


def build_potential(cfg: PotentialConfig) -> Tuple[Callable[[torch.Tensor], torch.Tensor], Callable[[np.ndarray], np.ndarray], float, Dict[str, float]]:
    """Return (V_torch(x), V_numpy(x), suggested_L, params_with_defaults)."""
    cfg = cfg.with_defaults()
    spec = POTENTIALS[cfg.key]
    params = dict(cfg.params) if cfg.params else {}

    def V_t(x: torch.Tensor) -> torch.Tensor:
        return spec.torch_fn(x, params)

    def V_n(x: np.ndarray) -> np.ndarray:
        return spec.numpy_fn(x, params)

    L = float(spec.suggest_L(params))
    return V_t, V_n, L, params
