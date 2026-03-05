# eager_map.py
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.sparse.linalg import LinearOperator, eigsh

from .knn import knn_jax_to_csr


def eager_dmap_sparse(
    R_iX: np.ndarray,
    *,
    α: float = 1.0,                    # Coifman–Lafon normalization
    t: int = 1,                        # diffusion time
    β: float | ArrayLike | None = None,# inverse temperature (scalar or (h,))
    L: np.ndarray | None = None,       # optional Mahalanobis factors (h, D, r)
    k_nn: int = 64,
    q_block: int = 1024,
    r_block: int = 8192,
    k_eigs: int | None = None,         # per-head output dim (includes trivial mode internally)
    ϵ: float = 1e-12,
    seed: int = 0,
) -> dict:
    """
    Sparse multi-head diffusion maps on a symmetric kNN graph.

    - Builds squared-distance kNN graph via knn_jax_to_csr (CSR.data = D^2).
    - ε (kernel bandwidth) = median(D^2.data) per head.
    - K_ij = exp(-β_h * D^2_ij / ε)
    - Coifman–Lafon: K_α = diag(q^{-α}) K diag(q^{-α}), q_i = Σ_j K_ij
    - Symmetric operator: A = D^{-1/2} K_α D^{-1/2}
    - eigsh(A) (default which="LA"), rotate back: ψ = D^{-1/2} u
    - diffusion coordinates at time t: W = ψ * λ^t

    Returns:
      {
        "β": (h,),
        "ε_kernel": (h,),
        "spectral": {"λ_x": (h,m), "ψ_ix": (h,N,m)},
        "W": (h,N,m),
        "q": (h,N),
      }
    where m = (k_eigs-1) after dropping the trivial mode.
    """
    rng = np.random.default_rng(seed)
    R = np.asarray(R_iX, dtype=np.float64)
    if R.ndim != 2:
        raise ValueError(f"R_iX must be (N,D), got {R.shape}.")
    N, D = R.shape

    # heads inferred from β (or from L if β is scalar/None)
    if β is None or np.isscalar(β):
        if L is None:
            h = 1
        else:
            L = np.asarray(L, dtype=np.float64)
            if L.ndim != 3 or L.shape[1] != D:
                raise ValueError(f"L must be (h,D,r), got {L.shape} for D={D}.")
            h = int(L.shape[0])
        β_heads = np.full((h,), 1.0 if β is None else float(β), dtype=np.float64)
    else:
        β_heads = np.asarray(β, dtype=np.float64).reshape(-1)
        h = int(β_heads.shape[0])
        if L is not None:
            L = np.asarray(L, dtype=np.float64)
            if L.ndim != 3 or L.shape[0] != h or L.shape[1] != D:
                raise ValueError(f"L must be (h={h},D={D},r), got {L.shape}.")

    # choose eig count (includes trivial mode)
    k_total = (min(N - 1, 2) if k_eigs is None else int(k_eigs))
    if k_total < 2:
        # need at least 2 to drop trivial mode
        k_total = min(N - 1, 2)
    if k_total >= N:
        k_total = N - 1

    m = max(0, k_total - 1)  # after dropping trivial eigenvector

    W = np.zeros((h, N, m), dtype=np.float64)
    λ_out = np.zeros((h, m), dtype=np.float64)
    ψ_out = np.zeros((h, N, m), dtype=np.float64)
    q_out = np.zeros((h, N), dtype=np.float64)
    ε_out = np.zeros((h,), dtype=np.float64)

    for hh in range(h):
        # metric transform per head (Mahalanobis via factor)
        if L is None:
            X = R
        else:
            X = R @ L[hh]  # (N, r)

        # squared-distance kNN graph (symmetric union)
        D2 = knn_jax_to_csr(
            np.asarray(X, dtype=np.float32),
            None,
            k=int(k_nn),
            q_block=int(q_block),
            r_block=int(r_block),
        ).astype(np.float64)

        if D2.nnz == 0:
            raise ValueError("kNN graph has nnz=0. Increase k_nn?")

        # ε = median of squared distances on edges
        ε = float(np.median(D2.data))
        if not np.isfinite(ε) or ε <= 0.0:
            pos = D2.data[D2.data > 0]
            ε = float(np.mean(pos)) if pos.size else 1.0
        ε_out[hh] = ε

        # K = exp(-β * D2 / ε)
        βh = float(β_heads[hh])
        K = D2.copy()
        K.data = np.exp(-(βh / ε) * K.data)

        # q = K 1
        q = np.asarray(K.sum(axis=1)).reshape(-1)
        q = np.maximum(q, ϵ)
        q_out[hh] = q

        qα = np.power(q, -float(α))  # q^{-α}

        # d = K_α 1 = q^{-α} * (K @ q^{-α})
        d = qα * K.dot(qα)
        d = np.maximum(d, ϵ)
        d_inv_sqrt = 1.0 / np.sqrt(d)

        # A x = D^{-1/2} K_α D^{-1/2} x
        def matvec(x: np.ndarray) -> np.ndarray:
            x = np.asarray(x, dtype=np.float64)
            z = d_inv_sqrt * x
            z = qα * z
            y = K.dot(z)
            y = qα * y
            y = d_inv_sqrt * y
            return y

        A = LinearOperator((N, N), matvec=matvec, rmatvec=matvec, dtype=np.float64)

        v0 = rng.standard_normal(N)
        w, U = eigsh(A, k=k_total, which="LA", v0=v0)

        # sort descending, drop trivial mode
        order = np.argsort(w)[::-1]
        w = w[order]
        U = U[:, order]

        w = w[1:k_total]
        U = U[:, 1:k_total]

        # rotate back: ψ = D^{-1/2} u
        ψ = d_inv_sqrt[:, None] * U

        # diffusion coordinates: W = ψ * λ^t
        λt = np.power(w, int(t), dtype=np.float64)
        Φ = ψ * λt[None, :]

        λ_out[hh] = w
        ψ_out[hh] = ψ
        W[hh] = Φ

    return {
        "β": β_heads.astype(np.float32),
        "ε_kernel": ε_out.astype(np.float32),
        "q": q_out.astype(np.float32),
        "spectral": {
            "λ_x": λ_out.astype(np.float32),
            "ψ_ix": ψ_out.astype(np.float32),
        },
        "W": W.astype(np.float32),
    }
