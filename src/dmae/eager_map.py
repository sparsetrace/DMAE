# eager_map.py
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import LinearOperator, eigsh

from .knn import knn_jax_to_csr


def _as_1d_beta(beta: float | ArrayLike | None, h: int) -> np.ndarray:
    if beta is None:
        return np.ones((h,), dtype=np.float64)
    if np.isscalar(beta):
        return np.full((h,), float(beta), dtype=np.float64)
    β = np.asarray(beta, dtype=np.float64).reshape(-1)
    if β.shape[0] != h:
        raise ValueError(f"beta must have length h={h}, got {β.shape[0]}.")
    return β


def _init_L(
    rng: np.random.Generator,
    *,
    h: int,
    D: int,
    metric_rank: int | None,
    metric_init: str,
    metric_mix: float,
) -> np.ndarray:
    r = D if metric_rank is None else int(metric_rank)
    if not (1 <= r <= D):
        raise ValueError(f"metric_rank must be in [1, D], got {r} for D={D}.")

    if metric_init == "euclidean":
        # Q_h is (D,r): take first r columns of I
        I = np.eye(D, dtype=np.float64)[:, :r]
        return np.repeat(I[None, :, :], h, axis=0)

    if metric_init == "wishart_mix":
        # Random low-rank factor mixed with identity-ish scale.
        # Q = (1-mix)*I + mix*G, then (optional) QR to stabilize.
        I = np.eye(D, dtype=np.float64)[:, :r]
        L = np.empty((h, D, r), dtype=np.float64)
        for hh in range(h):
            G = rng.standard_normal((D, r))
            Q = (1.0 - metric_mix) * I + metric_mix * G
            # Orthonormalize columns for numerical stability
            Q, _ = np.linalg.qr(Q)
            L[hh] = Q[:, :r]
        return L

    raise ValueError(f"Unknown metric_init={metric_init!r} (expected 'euclidean' or 'wishart_mix').")


def eager_dmap_sparse(
    R_iX: np.ndarray,
    *,
    head_dim: int,
    h: int = 1,
    alpha: float = 1.0,   # Coifman–Lafon α
    t: int = 1,           # diffusion time
    beta: float | ArrayLike | None = None,   # inverse temperature β (scalar or (h,))
    L: np.ndarray | None = None,             # optional Mahalanobis factors (h,D,r)
    mahalanobis: bool = False,
    metric_rank: int | None = None,
    metric_init: str = "euclidean",
    metric_mix: float = 0.1,
    # kNN / JAX blocking
    k_nn: int = 64,
    q_block: int = 1024,
    r_block: int = 8192,
    # eigsh
    k_eigs: int | None = None,   # defaults to head_dim + 1 (includes trivial mode)
    which: str = "LA",
    # numerics
    eps: float = 1e-12,
    seed: int = 0,
    return_graph: bool = False,
) -> dict:
    """
    Sparse multi-head diffusion maps via kNN graph + Lanczos (eigsh).

    For each head h:
      - build symmetric kNN distance graph (CSR) of squared distances D^2
      - set ε = median(D^2.data)
      - kernel: K_ij = exp( -β_h * D^2_ij / ε )
      - Coifman–Lafon: K_α = diag(q^{-α}) K diag(q^{-α}), q_i = sum_j K_ij
      - row-sums: d_i = sum_j (K_α)_ij
      - symmetric operator: A = D^{-1/2} K_α D^{-1/2}
      - eigsh on A, then rotate back:
            A u = λ u
            P ψ = λ ψ   with  P = D^{-1} K_α,  ψ = D^{-1/2} u

    Returns a dict shaped like your eager init conventions:
      {
        "params": {
          "SMD": {"R_iX": (N,D), "L": (h,D,r) optional},
          "cl_softmax": {"q": (h,N)},
          "W": (h,N,head_dim),
        },
        "beta_heads": (h,),
        "spectral": {"λ_x": (h,head_dim), "ψ_ix": (h,N,head_dim)},
        "ε_kernel": (h,),
        (optional) "graph": {"D2": [csr per head], "K": [csr per head]}
      }
    """
    rng = np.random.default_rng(seed)

    R = np.asarray(R_iX, dtype=np.float64)
    if R.ndim != 2:
        raise ValueError(f"R_iX must be (N,D), got {R.shape}.")
    N, D = R.shape

    if head_dim <= 0:
        raise ValueError("head_dim must be positive.")
    if h <= 0:
        raise ValueError("h must be positive.")
    α = float(alpha)
    if t < 0:
        raise ValueError("t must be >= 0.")

    β_heads = _as_1d_beta(beta, h)

    # Decide / validate L
    L_used: np.ndarray | None = None
    if L is not None:
        L_used = np.asarray(L, dtype=np.float64)
        if L_used.ndim != 3:
            raise ValueError(f"L must be (h,D,r), got {L_used.shape}.")
        if L_used.shape[0] != h or L_used.shape[1] != D:
            raise ValueError(f"L shape must be (h={h}, D={D}, r), got {L_used.shape}.")
        mahalanobis = True
    elif mahalanobis:
        L_used = _init_L(
            rng,
            h=h,
            D=D,
            metric_rank=metric_rank,
            metric_init=metric_init,
            metric_mix=float(metric_mix),
        )

    k_total = (head_dim + 1) if (k_eigs is None) else int(k_eigs)
    if k_total >= N:
        # eigsh needs k < N
        k_total = max(1, N - 1)

    W_out = np.zeros((h, N, head_dim), dtype=np.float64)
    λ_out = np.zeros((h, head_dim), dtype=np.float64)
    ψ_out = np.zeros((h, N, head_dim), dtype=np.float64)
    q_out = np.zeros((h, N), dtype=np.float64)
    ε_out = np.zeros((h,), dtype=np.float64)

    D2_graphs: list[csr_matrix] = []
    K_graphs: list[csr_matrix] = []

    for hh in range(h):
        # --- metric transform for this head ---
        if L_used is None:
            X = R
        else:
            # Mahalanobis via factor: M = L L^T, distance^2 = || (x_i - x_j) L ||^2
            X = R @ L_used[hh]  # (N,r)

        # --- kNN squared distances (symmetric union CSR) ---
        D2: csr_matrix = knn_jax_to_csr(
            np.asarray(X, dtype=np.float32),
            None,
            k=int(k_nn),
            q_block=int(q_block),
            r_block=int(r_block),
        ).astype(np.float64)

        if D2.nnz == 0:
            raise ValueError("kNN graph has no edges (nnz=0). Increase k_nn?")

        # ε = median(D^2) over edges
        ε_kernel = float(np.median(D2.data))
        if not np.isfinite(ε_kernel) or ε_kernel <= 0.0:
            # fallback: mean of positive values
            pos = D2.data[D2.data > 0]
            ε_kernel = float(np.mean(pos)) if pos.size else 1.0
        ε_out[hh] = ε_kernel

        # --- sparse kernel K_ij = exp(-β * D^2 / ε) ---
        β = float(β_heads[hh])
        K = D2.copy()
        K.data = np.exp(-(β / ε_kernel) * K.data).astype(np.float64)

        # --- Coifman–Lafon normalization ---
        # q_i = sum_j K_ij
        q = np.asarray(K.sum(axis=1)).reshape(-1)
        q = np.maximum(q, eps)
        q_out[hh] = q

        qα = np.power(q, -α)  # q^{-α}

        # d_i = sum_j (K_α)_ij where K_α = diag(q^{-α}) K diag(q^{-α})
        # K_α 1 = q^{-α} * (K @ q^{-α})
        tmp = K.dot(qα)
        d_vec = qα * tmp
        d_vec = np.maximum(d_vec, eps)
        d_inv_sqrt = 1.0 / np.sqrt(d_vec)

        # symmetric LinearOperator: A x = D^{-1/2} (K_α (D^{-1/2} x))
        def matvec(x: np.ndarray) -> np.ndarray:
            x = np.asarray(x, dtype=np.float64)
            z = d_inv_sqrt * x
            z = qα * z
            y = K.dot(z)
            y = qα * y
            y = d_inv_sqrt * y
            return y

        A = LinearOperator((N, N), matvec=matvec, rmatvec=matvec, dtype=np.float64)

        # --- eigsh (largest eigenvalues near 1) ---
        v0 = rng.standard_normal(N)
        w, U = eigsh(A, k=k_total, which=which, v0=v0)

        # sort descending by λ
        order = np.argsort(w)[::-1]
        w = w[order]
        U = U[:, order]

        # drop trivial mode (λ≈1) and keep head_dim
        keep = slice(1, 1 + head_dim)
        λ = w[keep].copy()
        u = U[:, keep].copy()

        # rotate back: ψ = D^{-1/2} u  (right eigenvectors of P)
        ψ = (d_inv_sqrt[:, None] * u)

        # diffusion coordinates at time t: ψ * λ^t
        λt = np.power(λ, int(t), dtype=np.float64)
        Φ = ψ * λt[None, :]

        λ_out[hh] = λ
        ψ_out[hh] = ψ
        W_out[hh] = Φ

        if return_graph:
            D2_graphs.append(D2)
            K_graphs.append(K)

    params: dict = {
        "SMD": {"R_iX": np.asarray(R, dtype=np.float32)},
        # store an unconstrained parameter for positivity if you later softmax/exp it
        "cl_softmax": {"q": np.log(q_out + eps).astype(np.float32)},
        "W": W_out.astype(np.float32),
    }
    if L_used is not None:
        params["SMD"]["L"] = L_used.astype(np.float32)

    out = {
        "params": params,
        "beta_heads": β_heads.astype(np.float32),
        "spectral": {
            "λ_x": λ_out.astype(np.float32),
            "ψ_ix": ψ_out.astype(np.float32),
        },
        "ε_kernel": ε_out.astype(np.float32),
    }
    if return_graph:
        out["graph"] = {"D2": D2_graphs, "K": K_graphs}
    return out
