from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def save_fig(fig, out_dir: Path, filename: str, dpi: int = 180) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path


def plot_potential(x_np: np.ndarray, V_np: np.ndarray, title: str = "Potential"):
    fig = plt.figure()
    plt.plot(x_np, V_np)
    plt.xlabel("x")
    plt.ylabel("V(x)")
    plt.title(title)
    plt.grid(True)
    return fig


def plot_training(df: pd.DataFrame, title: str = "Training"):
    fig = plt.figure()
    plt.plot(df["step"], df["E"], label="E")
    plt.plot(df["step"], df["loss"], label="loss")
    plt.xlabel("step")
    plt.ylabel("value")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    return fig


def plot_boundary_mass(df: pd.DataFrame, title: str = "Boundary mass"):
    fig = plt.figure()
    plt.plot(df["step"], df["boundary_mass"])
    plt.xlabel("step")
    plt.ylabel("boundary probability mass")
    plt.title(title)
    plt.grid(True)
    return fig


def plot_wavefunctions(x_np: np.ndarray, psis: np.ndarray, labels: Optional[list] = None, title: str = "Wavefunctions"):
    fig = plt.figure()
    K = psis.shape[0]
    for i in range(K):
        lab = labels[i] if labels is not None else f"psi{i}"
        plt.plot(x_np, psis[i], label=lab)
    plt.xlabel("x")
    plt.ylabel("psi(x)")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    return fig


def plot_overlap_matrix(O: np.ndarray, title: str = "|Overlap| matrix"):
    fig = plt.figure()
    plt.imshow(O, aspect="auto")
    plt.colorbar()
    plt.xlabel("ref state j")
    plt.ylabel("nn state i")
    plt.title(title)
    return fig
