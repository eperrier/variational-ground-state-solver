from __future__ import annotations

from typing import Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def fd_eigensolve_k(x_np: np.ndarray, V_np: np.ndarray, dx: float, k: int = 6) -> Tuple[np.ndarray, np.ndarray]:
    """Sparse finite-difference eigensolver with Dirichlet boundaries.

    Discretizes:
      H = -1/2 d^2/dx^2 + V(x)

    On interior points (exclude endpoints) with psi(±L)=0.

    Returns:
      evals: shape (k,)
      psis:  shape (k, N_full) padded with boundary zeros, normalized so ∫|psi|^2 dx = 1.
    """

    assert x_np.ndim == 1
    assert V_np.shape == x_np.shape

    N_full = x_np.size
    V_int = V_np[1:-1]
    Nint = V_int.size

    # Kinetic operator: (-1/2) d^2/dx^2 using central differences.
    main = (1.0 / dx**2) * np.ones(Nint)
    off = (-1.0 / (2.0 * dx**2)) * np.ones(Nint - 1)
    T = sp.diags([off, main, off], offsets=[-1, 0, 1], format="csr")
    Vmat = sp.diags(V_int, offsets=0, format="csr")
    H = T + Vmat

    evals, evecs = spla.eigsh(H, k=int(k), which="SA", tol=1e-10, maxiter=20000)
    idx = np.argsort(evals)
    evals = np.array(evals[idx], dtype=np.float64)
    evecs = np.array(evecs[:, idx], dtype=np.float64)

    psis = np.zeros((int(k), N_full), dtype=np.float64)
    for i in range(int(k)):
        psi_int = evecs[:, i]
        psi_int = psi_int / np.sqrt(dx * np.sum(psi_int**2))
        psis[i, 1:-1] = psi_int

    return evals, psis
