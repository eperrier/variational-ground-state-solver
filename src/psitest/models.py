from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ModelConfig:
    """User-facing model selection."""

    key: str = "plain_mlp"
    params: Dict[str, float] = None

    def with_defaults(self) -> "ModelConfig":
        spec = MODELS[self.key]
        p = dict(spec.default_params)
        if self.params:
            p.update(self.params)
        return ModelConfig(key=self.key, params=p)


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    default_params: Dict[str, float]


def available_models() -> Dict[str, str]:
    return {k: v.label for k, v in MODELS.items()}


class Envelope(nn.Module):
    """Dirichlet-ish envelope.

    psi(x) = envelope(x) * amplitude(x)

    envelope(x) = exp(-a x^2) * clamp(1 - (x/L)^2, 0, +inf)

    - exp(-a x^2) encourages localization
    - (1 - (x/L)^2) enforces psi(±L)=0
    """

    def __init__(self, L: float, a: float = 0.35):
        super().__init__()
        self.L = float(L)
        self.a = float(a)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(-self.a * x**2) * (1.0 - (x / self.L) ** 2).clamp(min=0.0)


class PlainMLPCore(nn.Module):
    def __init__(self, hidden: int = 64, depth: int = 3, act: str = "tanh"):
        super().__init__()
        acts = {"tanh": nn.Tanh(), "silu": nn.SiLU(), "gelu": nn.GELU()}
        activation = acts.get(act, nn.Tanh())

        layers = [nn.Linear(1, hidden), activation]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), activation]
        layers += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x1: torch.Tensor) -> torch.Tensor:
        return self.net(x1)


class FourierFeatures(nn.Module):
    """Fixed random Fourier feature map for 1D inputs."""

    def __init__(self, num_features: int = 16, scale: float = 3.0):
        super().__init__()
        B = torch.randn(int(num_features), 1) * float(scale)
        self.register_buffer("B", B)

    def forward(self, x1: torch.Tensor) -> torch.Tensor:
        proj = 2.0 * math.pi * (x1 @ self.B.T)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class FourierMLPCore(nn.Module):
    def __init__(self, hidden: int = 64, depth: int = 3, act: str = "tanh", num_fourier: int = 16, ff_scale: float = 3.0, include_x: bool = True):
        super().__init__()
        self.ff = FourierFeatures(num_features=num_fourier, scale=ff_scale)
        self.include_x = bool(include_x)
        in_dim = 2 * int(num_fourier) + (1 if include_x else 0)

        acts = {"tanh": nn.Tanh(), "silu": nn.SiLU(), "gelu": nn.GELU()}
        activation = acts.get(act, nn.Tanh())

        layers = [nn.Linear(in_dim, hidden), activation]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), activation]
        layers += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x1: torch.Tensor) -> torch.Tensor:
        feats = self.ff(x1)
        if self.include_x:
            feats = torch.cat([x1, feats], dim=-1)
        return self.net(feats)


class SineLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, is_first: bool, omega0: float = 30.0):
        super().__init__()
        self.omega0 = float(omega0)
        self.linear = nn.Linear(in_features, out_features)

        with torch.no_grad():
            if is_first:
                self.linear.weight.uniform_(-1 / in_features, 1 / in_features)
            else:
                bound = math.sqrt(6 / in_features) / self.omega0
                self.linear.weight.uniform_(-bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega0 * self.linear(x))


class SirenCore(nn.Module):
    def __init__(self, hidden: int = 64, depth: int = 3, omega0: float = 30.0):
        super().__init__()
        layers = [SineLayer(1, hidden, is_first=True, omega0=omega0)]
        for _ in range(depth - 1):
            layers.append(SineLayer(hidden, hidden, is_first=False, omega0=omega0))
        self.net = nn.Sequential(*layers)
        self.final = nn.Linear(hidden, 1)

    def forward(self, x1: torch.Tensor) -> torch.Tensor:
        return self.final(self.net(x1))


class RBFCore(nn.Module):
    """RBF mixture amplitude: Σ w_i exp(-0.5 ((x-mu_i)/sigma_i)^2)."""

    def __init__(self, M: int = 48, init_span: float = 3.0, init_sigma: float = 0.7):
        super().__init__()
        M = int(M)
        mu = torch.linspace(-float(init_span), float(init_span), M)
        self.mu = nn.Parameter(mu)
        self.log_sigma = nn.Parameter(torch.zeros(M) + math.log(float(init_sigma)))
        self.w = nn.Parameter(torch.randn(M) * 0.1)

    def forward(self, x1: torch.Tensor) -> torch.Tensor:
        x = x1.squeeze(-1)
        sigma = torch.exp(self.log_sigma) + 1e-8
        z = (x[:, None] - self.mu[None, :]) / sigma[None, :]
        phi = torch.exp(-0.5 * z**2)
        return (phi @ self.w[:, None])


class WaveAnsatz(nn.Module):
    """psi(x) = envelope(x) * amplitude(x).

    If positive=True, amplitude uses Softplus to bias toward node-less ground states.
    For excited states, set positive=False to allow sign changes.
    """

    def __init__(self, core: nn.Module, envelope: Envelope, positive: bool):
        super().__init__()
        self.core = core
        self.envelope = envelope
        self.positive = bool(positive)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = x[:, None] if x.ndim == 1 else x
        raw = self.core(x1).squeeze(-1)
        amp = (F.softplus(raw) + 1e-12) if self.positive else raw
        return self.envelope(x) * amp


MODELS: Dict[str, ModelSpec] = {
    "plain_mlp": ModelSpec(
        key="plain_mlp",
        label="MLP + envelope",
        default_params={"hidden": 64, "depth": 3, "act": "tanh", "a": 0.35},
    ),
    "fourier_mlp": ModelSpec(
        key="fourier_mlp",
        label="Fourier features + MLP + envelope",
        default_params={"hidden": 64, "depth": 3, "act": "tanh", "a": 0.35, "num_fourier": 24, "ff_scale": 2.5, "include_x": 1.0},
    ),
    "siren": ModelSpec(
        key="siren",
        label="SIREN (sin activations) + envelope",
        default_params={"hidden": 64, "depth": 3, "a": 0.35, "omega0": 30.0},
    ),
    "rbf": ModelSpec(
        key="rbf",
        label="RBF mixture + envelope",
        default_params={"M": 48, "a": 0.35, "init_span": 3.0, "init_sigma": 0.7},
    ),
}


def build_model(cfg: ModelConfig, L: float, positive: bool) -> nn.Module:
    """Instantiate a model from a ModelConfig."""
    cfg = cfg.with_defaults()
    p = dict(cfg.params) if cfg.params else {}

    env = Envelope(L=float(L), a=float(p.pop("a", 0.35)))

    if cfg.key == "plain_mlp":
        core = PlainMLPCore(hidden=int(p.get("hidden", 64)), depth=int(p.get("depth", 3)), act=str(p.get("act", "tanh")))
        return WaveAnsatz(core=core, envelope=env, positive=positive)

    if cfg.key == "fourier_mlp":
        core = FourierMLPCore(
            hidden=int(p.get("hidden", 64)),
            depth=int(p.get("depth", 3)),
            act=str(p.get("act", "tanh")),
            num_fourier=int(p.get("num_fourier", 24)),
            ff_scale=float(p.get("ff_scale", 2.5)),
            include_x=bool(int(p.get("include_x", 1))),
        )
        return WaveAnsatz(core=core, envelope=env, positive=positive)

    if cfg.key == "siren":
        core = SirenCore(hidden=int(p.get("hidden", 64)), depth=int(p.get("depth", 3)), omega0=float(p.get("omega0", 30.0)))
        return WaveAnsatz(core=core, envelope=env, positive=positive)

    if cfg.key == "rbf":
        init_span = float(p.get("init_span", min(3.0, 0.5 * float(L))))
        core = RBFCore(M=int(p.get("M", 48)), init_span=init_span, init_sigma=float(p.get("init_sigma", 0.7)))
        return WaveAnsatz(core=core, envelope=env, positive=positive)

    raise ValueError(f"Unknown model key: {cfg.key}")


def count_parameters(model: nn.Module) -> int:
    return sum(int(p.numel()) for p in model.parameters() if p.requires_grad)
