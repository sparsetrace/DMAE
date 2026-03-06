import numpy as np
from scipy.sparse.linalg import eigsh

def eager_dmap(
    R_iX: np.ndarray,
    *,
    head_dim: int,
    h: int = 1,
    alpha: float = 1.0,
    t: int = 1,
    beta: float | None = None,
    beta_mode: str = "per_head",   # "shared" | "per_head"
    mahalanobis: bool = False,
    Q: np.ndarray | None = None,   # optional user-provided metric factors (h,D,r), M_h = Q_h Q_h^T
    metric_rank: int | None = None,
    metric_init: str = "euclidean",  # "euclidean" | "wishart_mix"
    metric_mix: float = 0.1,
    zero_diag: bool = True,
    k_eigs: int | None = None,     # defaults to head_dim+1 (includes trivial mode)
    which: str = "LA",
    eps: float = 1e-12,
    seed: int = 0,
    ) -> dict:
    """
    Eager O(h*N^2) kernel + Lanczos eigsh per head.

    Returns:
      {
        "params": {
          "SMD": {"R_iX": (N,D), "L": (h,D,r) optional},
          "cl_softmax": {"q": (h,N)},
          "W": (h,N,head_dim),
        },
        "beta_heads": (h,),
        "spectral": {
          "lambdas": (h,head_dim),
          "psi": (h,N,head_dim),
        }
      }

    Rotation back to row-stochastic:
      If A u = λ u with A = D^{-1/2} Kα D^{-1/2}, then
      P ψ = λ ψ with P = D^{-1} Kα and ψ = D^{-1/2} u.
    """
    rng = np.random.default_rng(seed)

    R = np.asarray(R_iX, dtype=np.float64)
    N, D = R.shape

    # ---- metric factors Q_h (called "L" in your Flax SMD) ----
    L = None
    if mahalanobis:
        if Q is not None:
            L = np.asarray(Q, dtype=np.float64)
            if L.ndim != 3 or L.shape[0] != h or L.shape[1] != D:
                raise ValueError(f"Q must have shape (h,D,r); got {L.shape}")
            r = L.shape[2]
        else:
            r = D if metric_rank is None else int(metric_rank)
            if r <= 0 or r > D:
                raise ValueError(f"metric_rank must be in [1,D], got r={r}, D={D}")

            I = np.eye(D, dtype=np.float64)[:, :r]  # (D,r)
            if metric_init == "euclidean":
                L = np.broadcast_to(I[None, :, :], (h, D, r)).copy()
            elif metric_init == "wishart_mix":
                A = rng.standard_normal(size=(h, D, r)) / np.sqrt(max(r, 1))
                m = float(metric_mix)
                L = (1.0 - m) * I[None, :, :] + m * A
            else:
                raise ValueError(f"Unknown metric_init={metric_init}")
    else:
        r = None

    # ---- helper: pairwise squared Euclidean distances for (N, dproj) ----
    def pairwise_dist2(X: np.ndarray) -> np.ndarray:
        x2 = np.sum(X * X, axis=1, keepdims=True)   # (N,1)
        d2 = x2 + x2.T - 2.0 * (X @ X.T)            # (N,N)
        return np.maximum(d2, 0.0)

    # ---- per-head projected anchors (for mahalanobis) ----
    R_proj = []
    for hh in range(h):
        if not mahalanobis:
            R_proj.append(R)                # (N,D)
        else:
            R_proj.append(R @ L[hh])        # (N,r)

    # ---- choose beta(s) ----
    if beta is not None:
        beta_heads = np.full((h,), float(beta), dtype=np.float64)
    else:
        beta_heads = np.zeros((h,), dtype=np.float64)
        if beta_mode == "shared":
            d2 = pairwise_dist2(R_proj[0])
            off = d2[~np.eye(N, dtype=bool)]
            med = np.median(off) if off.size else 1.0
            beta_heads[:] = 1.0 / (med + eps)
        elif beta_mode == "per_head":
            for hh in range(h):
                d2 = pairwise_dist2(R_proj[hh])
                off = d2[~np.eye(N, dtype=bool)]
                med = np.median(off) if off.size else 1.0
                beta_heads[hh] = 1.0 / (med + eps)
        else:
            raise ValueError(f"Unknown beta_mode={beta_mode}")

    # ---- eig count (include trivial) ----
    if k_eigs is None:
        k_eigs = head_dim + 1
    if k_eigs >= N:
        raise ValueError(f"k_eigs must be < N (Lanczos). Got k_eigs={k_eigs}, N={N}")

    # ---- outputs ----
    q_heads = np.zeros((h, N), dtype=np.float64)
    W_heads = np.zeros((h, N, head_dim), dtype=np.float64)
    psi_heads = np.zeros((h, N, head_dim), dtype=np.float64)
    lam_heads = np.zeros((h, head_dim), dtype=np.float64)

    # ---- per head: build kernel, normalize, eigsh, rotate back ----
    for hh in range(h):
        Rp = R_proj[hh]
        d2 = pairwise_dist2(Rp)                          # (N,N)

        K = np.exp(-beta_heads[hh] * d2)                 # (N,N)
        if zero_diag:
            np.fill_diagonal(K, 0.0)

        # pre-alpha degrees (this is what your CL-softmax "q" is trying to capture)
        q = np.sum(K, axis=1)
        q = np.maximum(q, eps)
        q_heads[hh] = q

        # Coifman–Lafon alpha normalization: Kα = K / (q^α q^α^T)
        if alpha != 0.0:
            qa = q ** alpha
            K_alpha = K / np.maximum(qa[:, None] * qa[None, :], eps)
        else:
            K_alpha = K

        # post-alpha degree for Markov normalization
        drow = np.sum(K_alpha, axis=1)
        drow = np.maximum(drow, eps)

        inv_sqrt_d = 1.0 / np.sqrt(drow)

        # symmetric conjugate A = D^{-1/2} Kα D^{-1/2}
        A = (inv_sqrt_d[:, None] * K_alpha) * inv_sqrt_d[None, :]

        # Lanczos: largest algebraic eigenvalues/eigenvectors
        evals, evecs = eigsh(A, k=k_eigs, which=which)
        # sort descending
        idx = np.argsort(evals)[::-1]
        evals, evecs = evals[idx], evecs[:, idx]

        # drop trivial first mode
        lambdas = evals[1:head_dim+1].copy()
        u = evecs[:, 1:head_dim+1].copy()

        # rotate back: right eigenvectors of P are psi = D^{-1/2} u
        psi = (inv_sqrt_d[:, None] * u)

        # W for Nyström-style apply: W = lambda^(t-1) * psi  (so t=1 => W=psi)
        # (Assumes integer t; if you use non-integer t and get negative lambdas, handle separately.)
        W = psi * (lambdas[None, :] ** (t - 1))

        lam_heads[hh] = lambdas
        psi_heads[hh] = psi
        W_heads[hh] = W

    params = {
        "SMD": {"R_iX": R.astype(np.float32)},
        "cl_softmax": {"q": q_heads.astype(np.float32)},      # per-head q for your Softmax
        "W": W_heads.astype(np.float32),                      # (h,N,head_dim)
    }
    if mahalanobis:
        params["SMD"]["L"] = L.astype(np.float32)             # your SMD expects "L"

    return {
        "params": params,
        "beta_heads": beta_heads.astype(np.float32),
        "spectral": {
            "lambdas": lam_heads.astype(np.float32),
            "psi": psi_heads.astype(np.float32),
        },
    }
