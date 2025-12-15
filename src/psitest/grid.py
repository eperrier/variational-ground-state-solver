from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch


@dataclass(frozen=True)
class GridConfig:
    """Uniform 1D grid on [-L, L]."""

    L: float = 8.0
    N: int = 4096


def make_grid(cfg: GridConfig, device: torch.device, dtype: torch.dtype) -> Tuple[np.ndarray, torch.Tensor, float]:
    """Create numpy and torch representations of the uniform grid."""
    x_np = np.linspace(-cfg.L, cfg.L, cfg.N)
    dx = float(x_np[1] - x_np[0])
    x = torch.tensor(x_np, device=device, dtype=dtype)
    return x_np, x, dx
