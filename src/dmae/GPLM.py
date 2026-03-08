# dmae/GPLM.py
from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import jax.numpy as jnp

from .hf_io import (
    write_json,
    read_json,
    save_npz,
    load_npz,
    save_params,
    load_params,
    write_model_card,
    hf_upload_folder,
    hf_download_folder,
)

BetaSpec = Union[float, Tuple[float, ...]]


def _resolve_beta_alias(beta_greek, beta_ascii, default=None):
    if beta_greek is not None and beta_ascii is not None:
        a = np.asarray(beta_greek if not np.isscalar(beta_greek) else [beta_greek], dtype=np.float64)
        b = np.asarray(beta_ascii if not np.isscalar(beta_ascii) else [beta_ascii], dtype=np.float64)
        if a.shape != b.shape or not np.allclose(a, b):
            raise ValueError("Got both β and beta with different values; provide only one.")
    if beta_greek is None and beta_ascii is None:
        return default
    return beta_ascii if beta_greek is None else beta_greek


def _pairwise_dist2(X: np.ndarray) -> np.ndarray:
    x2 = np.sum(X * X, axis=1, keepdims=True)
    d2 = x2 + x2.T - 2.0 * (X @ X.T)
    return np.maximum(d2, 0.0)


def _normalize_latents_for_solver(R_ix: np.ndarray, N_expected: int) -> np.ndarray:
    """
    Normalize external latent shapes to internal solver shape (h, N, d).

    Accepted:
      - (N, d)      -> only valid for h=1, becomes (1, N, d)
      - (N, h, d)   -> becomes (h, N, d)
    """
    R_ix = np.asarray(R_ix)

    if R_ix.ndim == 2:
        if R_ix.shape[0] != N_expected:
            raise ValueError(
                f"R_ix shape (N,d) must have N={N_expected} on axis 0, got {R_ix.shape}."
            )
        return R_ix[None, ...]

    if R_ix.ndim == 3:
        if R_ix.shape[0] != N_expected:
            raise ValueError(
                f"R_ix shape (N,h,d) must have N={N_expected} on axis 0, got {R_ix.shape}."
            )
        return np.transpose(R_ix, (1, 0, 2))

    raise ValueError(f"R_ix must have shape (N,d) or (N,h,d), got {R_ix.shape}.")


def _make_identity_metric(h: int, d: int, metric_rank: Optional[int]) -> Optional[np.ndarray]:
    if metric_rank is None:
        return None
    r = int(metric_rank)
    if r <= 0 or r > d:
        raise ValueError(f"metric_rank must be in [1,d], got r={r}, d={d}.")
    I = np.eye(d, dtype=np.float64)[:, :r]
    return np.broadcast_to(I[None, :, :], (h, d, r)).copy()


def _apply_metric(Z: np.ndarray, L: Optional[np.ndarray]) -> np.ndarray:
    if L is None:
        return Z
    return Z @ L


def _prepare_beta_heads(
    R_ix_hNd: np.ndarray,
    beta: float | Tuple[float, ...] | np.ndarray | None,
    eps: float,
) -> np.ndarray:
    h, N, _ = R_ix_hNd.shape
    if beta is None:
        beta_heads = np.zeros((h,), dtype=np.float64)
        for hh in range(h):
            d2 = _pairwise_dist2(R_ix_hNd[hh])
            off = d2[~np.eye(N, dtype=bool)]
            med = np.median(off) if off.size else 1.0
            beta_heads[hh] = 1.0 / (med + eps)
        return beta_heads

    if np.isscalar(beta):
        return np.full((h,), float(beta), dtype=np.float64)

    beta_heads = np.asarray(beta, dtype=np.float64).reshape(-1)
    if beta_heads.shape[0] != h:
        raise ValueError(f"β must be scalar or length h={h}, got shape {beta_heads.shape}.")
    return beta_heads


class _KernelLinearOperator:
    """
    Matrix-free operator for A = K + sigma2 * I, where K is the latent RBF kernel.

    This class never materializes K. A matrix-vector or matrix-matrix product is
    computed in row blocks using the identity

        K_ij = exp(-beta * ||z_i - z_j||^2).

    Complexity per call:
      - compute: O(N^2 * (r + m)) for an input with m right-hand sides and metric rank r
      - peak memory: O(block_size * N + N * m)

    The important point is that memory is sub-quadratic; we do not store the full N x N kernel.
    """

    def __init__(self, Z_train: np.ndarray, beta: float, sigma2: float, block_size: int = 1024):
        Z_train = np.asarray(Z_train, dtype=np.float64)
        if Z_train.ndim != 2:
            raise ValueError(f"Z_train must have shape (N,r), got {Z_train.shape}.")
        self.Z_train = Z_train
        self.beta = float(beta)
        self.sigma2 = float(sigma2)
        self.block_size = int(block_size)
        self.N = int(Z_train.shape[0])
        self.z2 = np.sum(Z_train * Z_train, axis=1, dtype=np.float64)

    def kernel_mm(self, V: np.ndarray) -> np.ndarray:
        """Return K @ V without materializing K."""
        V = np.asarray(V, dtype=np.float64)
        was_vector = (V.ndim == 1)
        if was_vector:
            V = V[:, None]
        if V.ndim != 2 or V.shape[0] != self.N:
            raise ValueError(f"V must have shape (N,) or (N,m), got {V.shape} with N={self.N}.")

        m = V.shape[1]
        out = np.empty((self.N, m), dtype=np.float64)
        Z = self.Z_train
        z2 = self.z2
        for start in range(0, self.N, self.block_size):
            stop = min(start + self.block_size, self.N)
            Zb = Z[start:stop]                    # (B,r)
            d2 = z2[start:stop, None] + z2[None, :] - 2.0 * (Zb @ Z.T)
            np.maximum(d2, 0.0, out=d2)
            K_block = np.exp(-self.beta * d2)
            out[start:stop] = K_block @ V
        return out[:, 0] if was_vector else out

    def mv(self, V: np.ndarray) -> np.ndarray:
        return self.kernel_mm(V) + self.sigma2 * np.asarray(V, dtype=np.float64)

    def cross_kernel_mm(self, Z_query: np.ndarray, V: np.ndarray) -> np.ndarray:
        """Return K(Z_query, Z_train) @ V without materializing the cross-kernel."""
        Z_query = np.asarray(Z_query, dtype=np.float64)
        V = np.asarray(V, dtype=np.float64)
        if Z_query.ndim != 2:
            raise ValueError(f"Z_query must have shape (A,r), got {Z_query.shape}.")
        if V.ndim != 2 or V.shape[0] != self.N:
            raise ValueError(f"V must have shape (N,m), got {V.shape} with N={self.N}.")

        A = Z_query.shape[0]
        m = V.shape[1]
        out = np.empty((A, m), dtype=np.float64)
        q2 = np.sum(Z_query * Z_query, axis=1, dtype=np.float64)
        Z = self.Z_train
        z2 = self.z2
        for start in range(0, A, self.block_size):
            stop = min(start + self.block_size, A)
            Zb = Z_query[start:stop]
            d2 = q2[start:stop, None] + z2[None, :] - 2.0 * (Zb @ Z.T)
            np.maximum(d2, 0.0, out=d2)
            K_block = np.exp(-self.beta * d2)
            out[start:stop] = K_block @ V
        return out


@dataclass
class _CGInfo:
    converged: bool
    iters: int
    rel_residual: float
    abs_residual: float


@dataclass
class GPLMInit:
    R_ix_hNd: np.ndarray              # (h,N,d)
    R_iX: np.ndarray                  # (N,D)
    beta_heads: np.ndarray            # (h,)
    W: np.ndarray                     # (h,N,D)
    L: Optional[np.ndarray] = None    # (h,d,r)
    W_O: Optional[np.ndarray] = None  # (h*D, D_out)
    b_O: Optional[np.ndarray] = None  # (D_out,)
    cg_info: Optional[Dict[str, Any]] = None


def _pcg_matrix(
    A_mv,
    B: np.ndarray,
    *,
    M_inv_mv=None,
    x0: Optional[np.ndarray] = None,
    tol: float = 1e-6,
    atol: float = 0.0,
    maxiter: Optional[int] = None,
    eps: float = 1e-30,
) -> tuple[np.ndarray, _CGInfo]:
    """
    Conjugate gradient on the vectorized system with multiple right-hand sides.

    We treat B in R^(N x D) as one long vector under the Frobenius inner product.
    That is mathematically equivalent to solving (I_D ⊗ A) vec(X) = vec(B) with CG.

    Notes:
      - A must be SPD.
      - This avoids explicit K^{-1}; it solves A X = B directly.
      - For fixed D, each iteration costs one matrix-free kernel matmul, i.e. O(N^2).
        More precisely the arithmetic is O(N^2 * D) for the kernel-times-matrix part.
    """
    B = np.asarray(B, dtype=np.float64)
    if B.ndim == 1:
        B = B[:, None]
    if B.ndim != 2:
        raise ValueError(f"B must have shape (N,) or (N,m), got {B.shape}.")

    X = np.zeros_like(B) if x0 is None else np.asarray(x0, dtype=np.float64).copy()
    if X.shape != B.shape:
        raise ValueError(f"x0 must match B shape {B.shape}, got {X.shape}.")

    if M_inv_mv is None:
        def M_inv_mv(R):
            return R

    R = B - A_mv(X)
    Z = M_inv_mv(R)
    P = Z.copy()

    b_norm = float(np.linalg.norm(B))
    rz_old = float(np.sum(R * Z))
    r_norm = float(np.linalg.norm(R))
    threshold = max(float(atol), float(tol) * max(b_norm, eps))

    if maxiter is None:
        maxiter = 5 * B.shape[0]

    if r_norm <= threshold:
        info = _CGInfo(converged=True, iters=0, rel_residual=0.0 if b_norm == 0 else r_norm / b_norm, abs_residual=r_norm)
        return X, info

    converged = False
    iters = 0

    for k in range(1, int(maxiter) + 1):
        AP = A_mv(P)
        denom = float(np.sum(P * AP))
        if abs(denom) <= eps:
            break

        alpha = rz_old / denom
        X = X + alpha * P
        R = R - alpha * AP
        r_norm = float(np.linalg.norm(R))
        iters = k

        if r_norm <= threshold:
            converged = True
            break

        Z = M_inv_mv(R)
        rz_new = float(np.sum(R * Z))
        beta = rz_new / max(rz_old, eps)
        P = Z + beta * P
        rz_old = rz_new

    rel = 0.0 if b_norm == 0.0 else (r_norm / b_norm)
    info = _CGInfo(converged=converged, iters=iters, rel_residual=rel, abs_residual=r_norm)
    return X, info


def _solve_gplm_cg(
    R_ix_hNd: np.ndarray,
    R_iX: np.ndarray,
    *,
    beta: float | Tuple[float, ...] | np.ndarray | None = None,
    metric_rank: int | None = None,
    sigma2: float = 1e-6,
    cg_tol: float = 1e-6,
    cg_atol: float = 0.0,
    cg_maxiter: Optional[int] = None,
    block_size: int = 1024,
    warm_start: Optional[np.ndarray] = None,
    eps: float = 1e-12,
) -> dict:
    """
    Solve for decoder weights W in

        (K_h + sigma2 I) W_h = R_iX,

    one head at a time, using matrix-free conjugate gradients.

    This is the CG replacement for the dense exact solve

        W_h = (K_h + sigma2 I)^{-1} R_iX,

    used in the original implementation.

    Complexity:
      - compute: O(h * T * N^2 * D_head) for T CG iterations and fixed latent rank
      - memory:  O(h * N * D_head + h * N * d + block_size * N)

    The memory stays sub-quadratic because K is never formed explicitly.
    """
    R_iX = np.asarray(R_iX, dtype=np.float64)
    if R_iX.ndim != 2:
        raise ValueError(f"R_iX must have shape (N, D), got {R_iX.shape}.")
    N, D = R_iX.shape

    R_ix_hNd = np.asarray(R_ix_hNd, dtype=np.float64)
    if R_ix_hNd.ndim != 3:
        raise ValueError(f"R_ix_hNd must have shape (h,N,d), got {R_ix_hNd.shape}.")

    h, N2, d = R_ix_hNd.shape
    if N2 != N:
        raise ValueError(f"R_ix and R_iX must have same N, got {N2} and {N}.")

    beta_heads = _prepare_beta_heads(R_ix_hNd=R_ix_hNd, beta=beta, eps=eps)
    L = _make_identity_metric(h=h, d=d, metric_rank=metric_rank)

    W_heads = np.zeros((h, N, D), dtype=np.float64)
    info_per_head = []

    if warm_start is not None:
        warm_start = np.asarray(warm_start, dtype=np.float64)
        if warm_start.shape != (h, N, D):
            raise ValueError(
                f"warm_start must have shape {(h, N, D)}, got {warm_start.shape}."
            )

    for hh in range(h):
        Z = _apply_metric(R_ix_hNd[hh], None if L is None else L[hh])
        op = _KernelLinearOperator(
            Z_train=Z,
            beta=float(beta_heads[hh]),
            sigma2=float(sigma2),
            block_size=int(block_size),
        )
        x0 = None if warm_start is None else warm_start[hh]
        W_h, cg_info = _pcg_matrix(
            op.mv,
            R_iX,
            M_inv_mv=None,   # identity preconditioner; no inducing points / landmarks
            x0=x0,
            tol=float(cg_tol),
            atol=float(cg_atol),
            maxiter=cg_maxiter,
        )
        W_heads[hh] = W_h
        info_per_head.append({
            "converged": bool(cg_info.converged),
            "iters": int(cg_info.iters),
            "rel_residual": float(cg_info.rel_residual),
            "abs_residual": float(cg_info.abs_residual),
        })

    params = {
        "R_ix": R_ix_hNd.astype(np.float32),
        "W": W_heads.astype(np.float32),
    }
    if L is not None:
        params["L"] = L.astype(np.float32)

    return {
        "params": params,
        "beta_heads": beta_heads.astype(np.float32),
        "cg_info": info_per_head,
    }


class GPLM:
    """
    Matrix-free RBF GPLM decoder solved with conjugate gradients.

    External latent convention:
      - single-head: R_ix may be (N,d)
      - multi-head:  R_ix must be (N,h,d)

    Inference accepts:
      - single-head: z may be (B,d)
      - multi-head:  z must be (B,h,d)

    Important complexity notes:
      - This implementation avoids explicit Cholesky and never materializes K.
      - Solve cost is O(T * N^2 * D_head) per head, where T is the CG iteration count.
      - Peak memory is sub-quadratic: O(N*d + N*D_head + block_size*N), not O(N^2).
      - Test-time decoding of A query points costs O(A * N * D_head) for the final cross-kernel
        times weights, plus the cost of evaluating the RBF distances.
    """

    def __init__(
        self,
        R_ix: np.ndarray,   # external: (N,d) for h=1 or (N,h,d)
        R_iX: np.ndarray,   # (N,D)
        *,
        β: float | Tuple[float, ...] | np.ndarray | None = None,
        beta: float | Tuple[float, ...] | np.ndarray | None = None,
        metric_rank: int | None = None,
        sigma2: float = 1e-6,
        use_W_O: bool = False,
        D_out: int | None = None,
        cg_tol: float = 1e-6,
        cg_atol: float = 0.0,
        cg_maxiter: Optional[int] = None,
        block_size: int = 1024,
        warm_start: Optional[np.ndarray] = None,
        eps: float = 1e-12,
        seed: int = 0,
        dtype: Any = jnp.float32,
        param_dtype: Any = jnp.float32,
    ):
        del seed  # kept for backward-compatible signature; solver is deterministic.
        β = _resolve_beta_alias(β, beta, None)

        R_iX = np.asarray(R_iX, dtype=np.float64)
        if R_iX.ndim != 2:
            raise ValueError(f"R_iX must have shape (N, D), got {R_iX.shape}.")
        N, D = R_iX.shape

        R_ix_hNd = _normalize_latents_for_solver(R_ix, N_expected=N).astype(np.float64)
        h, N2, d = R_ix_hNd.shape
        if N2 != N:
            raise ValueError(f"R_ix and R_iX must have same N, got {N2} and {N}.")

        solved = _solve_gplm_cg(
            R_ix_hNd=R_ix_hNd,
            R_iX=R_iX,
            beta=β,
            metric_rank=metric_rank,
            sigma2=float(sigma2),
            cg_tol=float(cg_tol),
            cg_atol=float(cg_atol),
            cg_maxiter=cg_maxiter,
            block_size=int(block_size),
            warm_start=warm_start,
            eps=float(eps),
        )

        params_solved = solved["params"]
        beta_heads = np.asarray(solved["beta_heads"], dtype=np.float32)
        R_ix_fit = np.asarray(params_solved["R_ix"], dtype=np.float32)
        W_fit = np.asarray(params_solved["W"], dtype=np.float32)
        cg_info = solved["cg_info"]

        L_fit = None
        metric_rank_fit = None
        if "L" in params_solved:
            L_fit = np.asarray(params_solved["L"], dtype=np.float32)
            metric_rank_fit = int(L_fit.shape[-1])

        W_O_init = None
        b_O_init = None
        if use_W_O:
            D_out_final = D if D_out is None else int(D_out)
            W_O_init = np.zeros((h * D, D_out_final), dtype=np.float32)
            if D_out_final == D:
                for hh in range(h):
                    W_O_init[hh * D:(hh + 1) * D, :] = np.eye(D, dtype=np.float32) / float(h)
            b_O_init = np.zeros((D_out_final,), dtype=np.float32)
        else:
            D_out_final = h * D

        self.init_data = GPLMInit(
            R_ix_hNd=R_ix_fit,
            R_iX=R_iX.astype(np.float32),
            beta_heads=beta_heads,
            W=W_fit,
            L=L_fit,
            W_O=W_O_init,
            b_O=b_O_init,
            cg_info={"per_head": cg_info},
        )

        beta_module: BetaSpec
        if h == 1:
            beta_module = float(beta_heads[0])
        else:
            beta_module = tuple(float(b) for b in beta_heads)

        self.dtype = dtype
        self.param_dtype = param_dtype
        self.variables = {
            "params": {
                "R_ix": np.asarray(self.init_data.R_ix_hNd, dtype=np.float32),
                "W": np.asarray(self.init_data.W, dtype=np.float32),
            }
        }
        if self.init_data.L is not None:
            self.variables["params"]["L"] = np.asarray(self.init_data.L, dtype=np.float32)
        if use_W_O:
            self.variables["params"]["W_O"] = np.asarray(self.init_data.W_O, dtype=np.float32)
            self.variables["params"]["b_O"] = np.asarray(self.init_data.b_O, dtype=np.float32)

        self.config = {
            "R_ix_external_shape": tuple(np.asarray(R_ix).shape),
            "R_iX_shape": tuple(R_iX.shape),
            "h": int(h),
            "d": int(d),
            "D_head": int(D),
            "D_out": int(D_out_final),
            "β": beta_module,
            "metric_rank": metric_rank_fit,
            "sigma2": float(sigma2),
            "use_W_O": bool(use_W_O),
            "cg_tol": float(cg_tol),
            "cg_atol": float(cg_atol),
            "cg_maxiter": None if cg_maxiter is None else int(cg_maxiter),
            "block_size": int(block_size),
            "eps": float(eps),
            "solver": "matrix_free_cg",
            "cg_info": cg_info,
        }

    # -------------------------
    # Inference helpers
    # -------------------------

    def _normalize_inference_latents(self, z: np.ndarray | jnp.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=np.float64)
        h = self.config["h"]

        if z.ndim == 2:
            if h != 1:
                raise ValueError(
                    f"Got latent input shape {z.shape}; for h={h}, expected (B,h,d)."
                )
            z = z[:, None, :]
            return z

        if z.ndim == 3:
            if z.shape[1] != h:
                raise ValueError(
                    f"Expected latent input shape (B,h,d) with h={h}, got {z.shape}."
                )
            return z

        raise ValueError(
            f"Latent input must have shape (B,d) (only for h=1) or (B,h,d), got {z.shape}."
        )

    def _predict_with_params(self, z_hBd: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        h = self.config["h"]
        D = self.config["D_head"]
        block_size = self.config["block_size"]

        R_ix_hNd = np.asarray(params["R_ix"], dtype=np.float64)
        W_hND = np.asarray(params["W"], dtype=np.float64)
        beta_cfg = self.config["β"]
        if np.isscalar(beta_cfg):
            beta_heads = np.full((h,), float(beta_cfg), dtype=np.float64)
        else:
            beta_heads = np.asarray(beta_cfg, dtype=np.float64)
        L = None if "L" not in params else np.asarray(params["L"], dtype=np.float64)

        head_outputs = []
        for hh in range(h):
            Z_train = _apply_metric(R_ix_hNd[hh], None if L is None else L[hh])
            Z_query = _apply_metric(z_hBd[:, hh, :], None if L is None else L[hh])
            op = _KernelLinearOperator(
                Z_train=Z_train,
                beta=float(beta_heads[hh]),
                sigma2=float(self.config["sigma2"]),
                block_size=int(block_size),
            )
            Y_h = op.cross_kernel_mm(Z_query=Z_query, V=W_hND[hh])
            head_outputs.append(Y_h)

        Y_cat = np.concatenate(head_outputs, axis=1)  # (B, h*D)
        if self.config["use_W_O"]:
            W_O = np.asarray(params["W_O"], dtype=np.float64)
            b_O = np.asarray(params["b_O"], dtype=np.float64)
            Y = Y_cat @ W_O + b_O
        else:
            Y = Y_cat if h > 1 else head_outputs[0]

        return Y

    # -------------------------
    # Inference
    # -------------------------

    def __call__(self, z: np.ndarray | jnp.ndarray) -> jnp.ndarray:
        z = self._normalize_inference_latents(z)
        y = self._predict_with_params(z, self.variables["params"])
        return jnp.asarray(y, dtype=self.dtype)

    def apply(self, z: np.ndarray | jnp.ndarray, variables: Optional[Dict[str, Any]] = None):
        z = self._normalize_inference_latents(z)
        vars_use = self.variables if variables is None else variables
        if "params" not in vars_use:
            raise ValueError("variables must be a dict containing a 'params' key.")
        y = self._predict_with_params(z, vars_use["params"])
        return jnp.asarray(y, dtype=self.dtype)

    @property
    def params(self):
        return self.variables["params"]

    # -------------------------
    # Refit / warm-start utility
    # -------------------------

    def refit(
        self,
        *,
        R_ix: Optional[np.ndarray] = None,
        R_iX: Optional[np.ndarray] = None,
        β: float | Tuple[float, ...] | np.ndarray | None = None,
        beta: float | Tuple[float, ...] | np.ndarray | None = None,
        sigma2: Optional[float] = None,
        cg_tol: Optional[float] = None,
        cg_atol: Optional[float] = None,
        cg_maxiter: Optional[int] = None,
        block_size: Optional[int] = None,
        warm_start: bool = True,
    ) -> None:
        """
        Re-solve the GPLM weights after latents or hyperparameters change.

        This is the operation you should call if you fine-tune R_ix, beta, or sigma2
        outside this class and want W to remain consistent with the kernel.
        """
        beta_merged = _resolve_beta_alias(β, beta, self.config["β"])
        R_iX_use = self.init_data.R_iX if R_iX is None else np.asarray(R_iX, dtype=np.float64)
        N = R_iX_use.shape[0]
        R_ix_use = np.transpose(self.init_data.R_ix_hNd, (1, 0, 2)) if R_ix is None else np.asarray(R_ix)
        R_ix_hNd = _normalize_latents_for_solver(R_ix_use, N_expected=N).astype(np.float64)

        solved = _solve_gplm_cg(
            R_ix_hNd=R_ix_hNd,
            R_iX=R_iX_use,
            beta=beta_merged,
            metric_rank=self.config["metric_rank"],
            sigma2=float(self.config["sigma2"] if sigma2 is None else sigma2),
            cg_tol=float(self.config["cg_tol"] if cg_tol is None else cg_tol),
            cg_atol=float(self.config["cg_atol"] if cg_atol is None else cg_atol),
            cg_maxiter=self.config["cg_maxiter"] if cg_maxiter is None else int(cg_maxiter),
            block_size=int(self.config["block_size"] if block_size is None else block_size),
            warm_start=self.init_data.W.astype(np.float64) if warm_start else None,
            eps=float(self.config["eps"]),
        )

        self.init_data.R_ix_hNd = solved["params"]["R_ix"]
        self.init_data.R_iX = np.asarray(R_iX_use, dtype=np.float32)
        self.init_data.beta_heads = np.asarray(solved["beta_heads"], dtype=np.float32)
        self.init_data.W = np.asarray(solved["params"]["W"], dtype=np.float32)
        if "L" in solved["params"]:
            self.init_data.L = np.asarray(solved["params"]["L"], dtype=np.float32)
        self.init_data.cg_info = {"per_head": solved["cg_info"]}

        self.variables["params"]["R_ix"] = np.asarray(self.init_data.R_ix_hNd, dtype=np.float32)
        self.variables["params"]["W"] = np.asarray(self.init_data.W, dtype=np.float32)
        if self.init_data.L is not None:
            self.variables["params"]["L"] = np.asarray(self.init_data.L, dtype=np.float32)

        h = self.config["h"]
        beta_heads = np.asarray(self.init_data.beta_heads, dtype=np.float32)
        self.config["β"] = float(beta_heads[0]) if h == 1 else tuple(float(b) for b in beta_heads)
        self.config["sigma2"] = float(self.config["sigma2"] if sigma2 is None else sigma2)
        self.config["cg_tol"] = float(self.config["cg_tol"] if cg_tol is None else cg_tol)
        self.config["cg_atol"] = float(self.config["cg_atol"] if cg_atol is None else cg_atol)
        self.config["cg_maxiter"] = self.config["cg_maxiter"] if cg_maxiter is None else int(cg_maxiter)
        self.config["block_size"] = int(self.config["block_size"] if block_size is None else block_size)
        self.config["cg_info"] = solved["cg_info"]

    # -------------------------
    # Low-rank factorization of solved weights
    # -------------------------

    def factorize(self, rank: int) -> Dict[str, np.ndarray]:
        """
        Compute truncated SVD per head:
            W_h ≈ U_h V_h
        where W_h = self.init_data.W[h] has shape (N, D).
        """
        S = np.asarray(self.init_data.W, dtype=np.float64)   # (h, N, D)
        h, N, D = S.shape

        max_rank = min(N, D)
        if rank <= 0 or rank > max_rank:
            raise ValueError(
                f"rank must be in [1, min(N,D)] = [1, {max_rank}], got rank={rank}."
            )

        U_factors = np.zeros((h, N, rank), dtype=np.float32)
        V_factors = np.zeros((h, rank, D), dtype=np.float32)
        S_hat = np.zeros((h, N, D), dtype=np.float32)
        rel_errs = np.zeros((h,), dtype=np.float32)

        for hh in range(h):
            Sh = S[hh]
            U0, s, Vt = np.linalg.svd(Sh, full_matrices=False)

            U0_r = U0[:, :rank]
            s_r = s[:rank]
            Vt_r = Vt[:rank, :]

            sqrt_s = np.sqrt(np.maximum(s_r, 0.0))
            Uh = U0_r * sqrt_s[None, :]
            Vh = sqrt_s[:, None] * Vt_r
            Sh_hat = Uh @ Vh

            denom = np.linalg.norm(Sh, ord="fro")
            err = np.linalg.norm(Sh - Sh_hat, ord="fro")
            rel = 0.0 if denom == 0.0 else (err / denom)

            U_factors[hh] = Uh.astype(np.float32)
            V_factors[hh] = Vh.astype(np.float32)
            S_hat[hh] = Sh_hat.astype(np.float32)
            rel_errs[hh] = np.float32(rel)

        result = {
            "U": U_factors,
            "V": V_factors,
            "S_hat": S_hat,
            "rel_frob_error_per_head": rel_errs,
            "rel_frob_error": float(np.mean(rel_errs)),
            "rank": int(rank),
        }
        self.factorization = result
        return result

    # -------------------------
    # Local save / load
    # -------------------------

    def save_pretrained(self, local_dir: str | Path) -> None:
        local_dir = Path(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)

        write_json(local_dir / "config.json", {
            "class_name": "GPLM",
            "config": self.config,
        })

        save_params(local_dir / "params.safetensors", dict(self.variables["params"]))

        arrays = {
            "R_ix_hNd": np.asarray(self.init_data.R_ix_hNd),
            "R_iX": np.asarray(self.init_data.R_iX),
            "beta_heads": np.asarray(self.init_data.beta_heads),
            "W": np.asarray(self.init_data.W),
        }
        if self.init_data.L is not None:
            arrays["L"] = np.asarray(self.init_data.L)
        if self.init_data.W_O is not None:
            arrays["W_O"] = np.asarray(self.init_data.W_O)
        if self.init_data.b_O is not None:
            arrays["b_O"] = np.asarray(self.init_data.b_O)

        save_npz(local_dir / "init_data.npz", arrays)
        write_model_card(local_dir / "README.md", class_name="GPLM")

    @classmethod
    def from_pretrained(cls, local_dir: str | Path) -> "GPLM":
        local_dir = Path(local_dir)

        meta = read_json(local_dir / "config.json")
        cfg = meta["config"]
        init_npz = load_npz(local_dir / "init_data.npz")
        params = load_params(local_dir / "params.safetensors")

        obj = cls.__new__(cls)
        obj.dtype = jnp.float32
        obj.param_dtype = jnp.float32
        obj.init_data = GPLMInit(
            R_ix_hNd=init_npz["R_ix_hNd"],
            R_iX=init_npz["R_iX"],
            beta_heads=init_npz["beta_heads"],
            W=init_npz["W"],
            L=init_npz["L"] if "L" in init_npz else None,
            W_O=init_npz["W_O"] if "W_O" in init_npz else None,
            b_O=init_npz["b_O"] if "b_O" in init_npz else None,
            cg_info={"per_head": cfg.get("cg_info", [])},
        )
        obj.variables = {"params": params}
        obj.config = cfg
        return obj

    # -------------------------
    # Hugging Face save / load
    # -------------------------

    def hf_save(self, hf_repo: str, hf_token: str | None = None) -> None:
        with tempfile.TemporaryDirectory() as td:
            self.save_pretrained(td)
            hf_upload_folder(td, hf_repo, hf_token, commit_message="Upload GPLM")

    @classmethod
    def hf_load(cls, hf_repo: str, hf_token: str | None = None) -> "GPLM":
        local_dir = hf_download_folder(hf_repo, hf_token)
        return cls.from_pretrained(local_dir)

    @classmethod
    def from_latents(cls, *args, **kwargs) -> "GPLM":
        return cls(*args, **kwargs)
