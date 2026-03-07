# GPLM.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import jax
import jax.numpy as jnp
from flax.core import freeze, unfreeze

from .blocks import gplm


BetaSpec = Union[float, Tuple[float, ...]]


def _resolve_scalar_alias(
    greek_name: str,
    greek_val,
    latin_name: str,
    latin_val,
    default: float,
) -> float:
    if greek_val is not None and latin_val is not None and float(greek_val) != float(latin_val):
        raise ValueError(
            f"Got both {greek_name}={greek_val} and {latin_name}={latin_val}; provide only one."
        )
    if greek_val is None and latin_val is None:
        return float(default)
    return float(latin_val) if greek_val is None else float(greek_val)


def _resolve_beta_alias(β, beta, default=None):
    if β is not None and beta is not None:
        a = np.asarray(β if not np.isscalar(β) else [β], dtype=np.float64)
        b = np.asarray(beta if not np.isscalar(beta) else [beta], dtype=np.float64)
        if a.shape != b.shape or not np.allclose(a, b):
            raise ValueError("Got both β and beta with different values; provide only one.")
    if β is None and beta is None:
        return default
    return beta if β is None else β


def _pairwise_dist2(X: np.ndarray) -> np.ndarray:
    x2 = np.sum(X * X, axis=1, keepdims=True)      # (N,1)
    d2 = x2 + x2.T - 2.0 * (X @ X.T)               # (N,N)
    return np.maximum(d2, 0.0)


def _normalize_latents_for_solver(R_ix: np.ndarray, N_expected: int) -> np.ndarray:
    """
    Normalize external latent shapes to internal solver shape (h, N, d).

    Accepted:
      - (N, d)      -> only valid for h=1, becomes (1, N, d)
      - (N, h, d)   -> becomes (h, N, d)

    Rejected:
      - any other shape
    """
    R_ix = np.asarray(R_ix, dtype=np.float32)

    if R_ix.ndim == 2:
        # (N, d) -> (1, N, d)
        if R_ix.shape[0] != N_expected:
            raise ValueError(
                f"R_ix shape (N,d) must have N={N_expected} on axis 0, got {R_ix.shape}."
            )
        return R_ix[None, ...]

    if R_ix.ndim == 3:
        # (N, h, d) -> (h, N, d)
        if R_ix.shape[0] != N_expected:
            raise ValueError(
                f"R_ix shape (N,h,d) must have N={N_expected} on axis 0, got {R_ix.shape}."
            )
        return np.transpose(R_ix, (1, 0, 2))

    raise ValueError(
        f"R_ix must have shape (N,d) or (N,h,d), got {R_ix.shape}."
    )


def _solve_gplm_exact(
    R_ix_hNd: np.ndarray,     # (h, N, d)
    R_iX: np.ndarray,         # (N, D)
    *,
    β: float | Tuple[float, ...] | np.ndarray | None = None,
    metric_rank: int | None = None,
    sigma2: float = 1e-6,
    eps: float = 1e-12,
    seed: int = 0,
) -> dict:
    """
    Exact dense GPLM solve per head.

    For each head h:
        K_h[i,j] = exp(-β_h * d_h^2(R_ix[i], R_ix[j]))
        S_h      = (K_h + sigma2 I)^(-1) R_iX

    Returns
    -------
    {
      "params": {
        "R_ix": (h,N,d),
        "W":    (h,N,D),
        "L":    (h,d,r) optional,
      },
      "beta_heads": (h,),
    }
    """
    _ = np.random.default_rng(seed)

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

    if β is None:
        beta_heads = np.zeros((h,), dtype=np.float64)
        for hh in range(h):
            d2 = _pairwise_dist2(R_ix_hNd[hh])
            off = d2[~np.eye(N, dtype=bool)]
            med = np.median(off) if off.size else 1.0
            beta_heads[hh] = 1.0 / (med + eps)
    elif np.isscalar(β):
        beta_heads = np.full((h,), float(β), dtype=np.float64)
    else:
        beta_heads = np.asarray(β, dtype=np.float64).reshape(-1)
        if beta_heads.shape[0] != h:
            raise ValueError(f"β must be scalar or length h={h}, got shape {beta_heads.shape}.")

    W_heads = np.zeros((h, N, D), dtype=np.float64)

    L = None
    if metric_rank is not None:
        r = int(metric_rank)
        if r <= 0 or r > d:
            raise ValueError(f"metric_rank must be in [1,d], got r={r}, d={d}.")
        I = np.eye(d, dtype=np.float64)[:, :r]
        L = np.broadcast_to(I[None, :, :], (h, d, r)).copy()

    for hh in range(h):
        Z = R_ix_hNd[hh]  # (N,d)
        if L is None:
            Zp = Z
        else:
            Zp = Z @ L[hh]  # (N,r)

        d2 = _pairwise_dist2(Zp)
        K = np.exp(-beta_heads[hh] * d2)
        A = K + float(sigma2) * np.eye(N, dtype=np.float64)

        # Exact dense solve
        S = np.linalg.solve(A, R_iX)  # (N,D)
        W_heads[hh] = S

    params = {
        "R_ix": R_ix_hNd.astype(np.float32),
        "W": W_heads.astype(np.float32),
    }
    if L is not None:
        params["L"] = L.astype(np.float32)

    return {
        "params": params,
        "beta_heads": beta_heads.astype(np.float32),
    }


@dataclass
class GPLMInit:
    R_ix_hNd: np.ndarray              # (h,N,d) internal
    R_iX: np.ndarray                  # (N,D)
    β: np.ndarray                     # (h,)
    W: np.ndarray                     # (h,N,D)
    L: Optional[np.ndarray] = None    # (h,d,r)
    W_O: Optional[np.ndarray] = None  # (h*D, D_out)
    b_O: Optional[np.ndarray] = None  # (D_out,)


class GPLM:
    """
    Exact dense GPLM decoder wrapper around the abstract Flax `gplm` block.

    External latent convention:
      - single-head: R_ix may be (N,d)
      - multi-head:  R_ix must be (N,h,d)

    Inference accepts:
      - single-head: z may be (B,d)
      - multi-head:  z must be (B,h,d)

    Internally, latents are converted to (h,N,d) for the exact solve and
    the Flax parameter tree.
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
        eps: float = 1e-12,
        seed: int = 0,
        dtype: Any = jnp.float32,
        param_dtype: Any = jnp.float32,
    ):
        β = _resolve_beta_alias(β, beta, None)

        R_iX = np.asarray(R_iX, dtype=np.float32)
        if R_iX.ndim != 2:
            raise ValueError(f"R_iX must have shape (N, D), got {R_iX.shape}.")
        N, D = R_iX.shape

        # Normalize external latent convention to internal (h,N,d)
        R_ix_hNd = _normalize_latents_for_solver(R_ix, N_expected=N)
        h, N2, d = R_ix_hNd.shape
        if N2 != N:
            raise ValueError(f"R_ix and R_iX must have same N, got {N2} and {N}.")

        eager = _solve_gplm_exact(
            R_ix_hNd=R_ix_hNd,
            R_iX=R_iX,
            β=β,
            metric_rank=metric_rank,
            sigma2=float(sigma2),
            eps=float(eps),
            seed=int(seed),
        )

        params_eager = eager["params"]
        beta_heads = np.asarray(eager["beta_heads"], dtype=np.float32)      # (h,)
        R_ix_fit = np.asarray(params_eager["R_ix"], dtype=np.float32)       # (h,N,d)
        W_fit = np.asarray(params_eager["W"], dtype=np.float32)             # (h,N,D)

        L_fit = None
        metric_rank_fit = None
        if "L" in params_eager:
            L_fit = np.asarray(params_eager["L"], dtype=np.float32)
            metric_rank_fit = int(L_fit.shape[-1])

        W_O_init = None
        b_O_init = None
        if use_W_O:
            D_out_final = D if D_out is None else int(D_out)
            W_O_init = np.zeros((h * D, D_out_final), dtype=np.float32)
            if D_out_final == D:
                for hh in range(h):
                    W_O_init[hh * D : (hh + 1) * D, :] = np.eye(D, dtype=np.float32) / float(h)
            b_O_init = np.zeros((D_out_final,), dtype=np.float32)
        else:
            D_out_final = h * D

        self.init_data = GPLMInit(
            R_ix_hNd=R_ix_fit,
            R_iX=R_iX,
            β=beta_heads,
            W=W_fit,
            L=L_fit,
            W_O=W_O_init,
            b_O=b_O_init,
        )

        beta_module: BetaSpec
        if h == 1:
            beta_module = float(beta_heads[0])
        else:
            beta_module = tuple(float(b) for b in beta_heads)

        self.module = gplm(
            D_head=int(D),
            N=int(N),
            h=int(h),
            β=beta_module,
            metric_rank=metric_rank_fit,
            use_W_O=bool(use_W_O),
            D_out=None if not use_W_O else int(D_out_final),
            eps=float(eps),
            dtype=dtype,
            param_dtype=param_dtype,
        )

        dummy_z = jnp.zeros((1, h, d), dtype=dtype)
        variables = self.module.init(jax.random.PRNGKey(seed), dummy_z)
        vars_mut = unfreeze(variables)
        params = vars_mut["params"]

        params["R_ix"] = jnp.asarray(self.init_data.R_ix_hNd, dtype=param_dtype)
        params["W"] = jnp.asarray(self.init_data.W, dtype=param_dtype)

        if self.init_data.L is not None:
            params["L"] = jnp.asarray(self.init_data.L, dtype=param_dtype)

        if use_W_O:
            params["W_O"] = jnp.asarray(self.init_data.W_O, dtype=param_dtype)
            params["b_O"] = jnp.asarray(self.init_data.b_O, dtype=param_dtype)

        self.variables = freeze(vars_mut)

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
            "eps": float(eps),
            "seed": int(seed),
        }

    def _normalize_inference_latents(self, z: np.ndarray | jnp.ndarray) -> jnp.ndarray:
        """
        External inference convention:
          - if h=1: accept (B,d) or (B,1,d)
          - if h>1: accept only (B,h,d)
        """
        z = jnp.asarray(z, dtype=self.module.dtype)
        h = self.config["h"]

        if z.ndim == 2:
            if h != 1:
                raise ValueError(
                    f"Got latent input shape {z.shape}; for h={h}, expected (B,h,d)."
                )
            z = z[:, None, :]  # (B,1,d)
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

    def __call__(self, z: np.ndarray | jnp.ndarray) -> jnp.ndarray:
        z = self._normalize_inference_latents(z)
        return self.module.apply(self.variables, z)

    def apply(self, z: np.ndarray | jnp.ndarray, variables: Optional[Dict[str, Any]] = None):
        z = self._normalize_inference_latents(z)
        vars_use = self.variables if variables is None else variables
        return self.module.apply(vars_use, z)

    @property
    def params(self):
        return self.variables["params"]

    @classmethod
    def from_latents(cls, *args, **kwargs) -> "GPLM":
        return cls(*args, **kwargs)

    def factorize(self, rank: int) -> Dict[str, np.ndarray]:
        """
        Compute an exact algebraic low-rank factorization of the solved GPLM weights.
    
        For each head h, factorize:
            S_h ≈ U_h V_h
        where:
            S_h = self.init_data.W[h]    # shape (N, D)
    
        using truncated SVD:
            S_h ≈ U0_h[:, :m] diag(s_h[:m]) V0_h[:m, :]
    
        We absorb singular values symmetrically:
            U_h = U0_h[:, :m] * sqrt(s_h[:m])
            V_h = sqrt(s_h[:m])[:, None] * V0_h[:m, :]
    
        so that:
            U_h @ V_h ≈ S_h
    
        Parameters
        ----------
        rank:
            Target factor rank m. Must satisfy 1 <= rank <= min(N, D).
    
        Returns
        -------
        dict with keys:
            "U":      (h, N, m)
            "V":      (h, m, D)
            "S_hat":  (h, N, D)
            "rel_frob_error_per_head": (h,)
            "rel_frob_error": scalar float
            "rank":   int
    
        Notes
        -----
        This does NOT modify the Flax model. It only factorizes the exact solved
        decoder weights stored in self.init_data.W. If you later want inference to
        use the low-rank form, build a separate low-rank GPLM module or a method
        that patches/rebuilds the decoder.
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
            Sh = S[hh]  # (N, D)
    
            U0, s, Vt = np.linalg.svd(Sh, full_matrices=False)
    
            U0_r = U0[:, :rank]                      # (N, rank)
            s_r = s[:rank]                           # (rank,)
            Vt_r = Vt[:rank, :]                      # (rank, D)
    
            sqrt_s = np.sqrt(np.maximum(s_r, 0.0))   # (rank,)
    
            Uh = U0_r * sqrt_s[None, :]              # (N, rank)
            Vh = sqrt_s[:, None] * Vt_r              # (rank, D)
    
            Sh_hat = Uh @ Vh
    
            denom = np.linalg.norm(Sh, ord="fro")
            err = np.linalg.norm(Sh - Sh_hat, ord="fro")
            rel = 0.0 if denom == 0.0 else (err / denom)
    
            U_factors[hh] = Uh.astype(np.float32)
            V_factors[hh] = Vh.astype(np.float32)
            S_hat[hh] = Sh_hat.astype(np.float32)
            rel_errs[hh] = np.float32(rel)
    
        result = {
            "U": U_factors,                                  # (h, N, rank)
            "V": V_factors,                                  # (h, rank, D)
            "S_hat": S_hat,                                  # (h, N, D)
            "rel_frob_error_per_head": rel_errs,             # (h,)
            "rel_frob_error": float(np.mean(rel_errs)),      # scalar
            "rank": int(rank),
        }
    
        # Optional convenience cache on the Python wrapper
        self.factorization = result
        return result
