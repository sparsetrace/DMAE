from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import jax
import jax.numpy as jnp
from flax.core import freeze, unfreeze

from .eager_map import eager_dmap_sparse
from .blocks import dmap


BetaSpec = Union[float, Tuple[float, ...]]


@dataclass
class DMAPInit:
    R_iX: np.ndarray
    β: np.ndarray
    q: np.ndarray
    λ_x: np.ndarray
    ψ_ix: np.ndarray
    R_ix: np.ndarray
    W_linear: np.ndarray
    ε_kernel: np.ndarray
    L: Optional[np.ndarray] = None


class DMAP:
    """
    Eager-initialized wrapper around the trainable Flax `dmap` encoder.

    Usage:
        encoder = DMAP(R_iX=train_X, d=16)
        z = encoder(test_X)
    """

    def __init__(
        self,
        R_iX: np.ndarray,
        d: int,
        *,
        h: int = 1,
        α: float | None = None,
        β: float | Tuple[float, ...] | np.ndarray | None = None,
        alpha: float | None = None,
        beta: float | Tuple[float, ...] | np.ndarray | None = None,
        L: np.ndarray | None = None,
        t: int = 1,
        k_nn: int = 64,
        q_block: int = 1024,
        r_block: int = 8192,
        ϵ: float = 1e-12,
        eps: float | None = None,
        seed: int = 0,
        dtype: Any = jnp.float32,
        param_dtype: Any = jnp.float32,
        ):
        # Resolve aliases safely
        if α is not None and alpha is not None and float(α) != float(alpha):
            raise ValueError(f"Got both α={α} and alpha={alpha}; please provide only one.")
        if β is not None and beta is not None:
            β_arr = np.asarray(β if not np.isscalar(β) else [β], dtype=np.float64)
            beta_arr = np.asarray(beta if not np.isscalar(beta) else [beta], dtype=np.float64)
            if β_arr.shape != beta_arr.shape or not np.allclose(β_arr, beta_arr):
                raise ValueError("Got both β and beta with different values; please provide only one.")
        if eps is not None and float(ϵ) != float(eps):
            raise ValueError(f"Got both ϵ={ϵ} and eps={eps}; please provide only one.")

        α = 1.0 if (α is None and alpha is None) else (float(alpha) if α is None else float(α))
        β = beta if β is None else β
        ϵ = float(eps) if eps is not None else float(ϵ)
        
        
        R_iX = np.asarray(R_iX, dtype=np.float32)
        if R_iX.ndim != 2:
            raise ValueError(f"R_iX must have shape (N, D), got {R_iX.shape}.")

        N, D = R_iX.shape

        if d <= 0:
            raise ValueError(f"`d` must be positive, got {d}.")
        if h <= 0:
            raise ValueError(f"`h` must be positive, got {h}.")

        # Normalize beta input for eager_dmap_sparse
        if β is None:
            beta_eager = None
        elif np.isscalar(β):
            beta_eager = float(β)
        else:
            beta_eager = np.asarray(β, dtype=np.float64).reshape(-1)
            if beta_eager.shape[0] != h:
                raise ValueError(f"β must be scalar or length h={h}, got shape {beta_eager.shape}.")

        metric_rank = None
        if L is not None:
            L = np.asarray(L, dtype=np.float32)
            if L.ndim != 3 or L.shape[0] != h or L.shape[1] != D:
                raise ValueError(f"L must have shape (h={h}, D={D}, r), got {L.shape}.")
            metric_rank = int(L.shape[-1])

        # eager solve
        eager = eager_dmap_sparse(
            R_iX=R_iX,
            α=float(α),
            t=int(t),
            β=beta_eager,
            L=None if L is None else np.asarray(L, dtype=np.float64),
            k_nn=int(k_nn),
            q_block=int(q_block),
            r_block=int(r_block),
            k_eigs=int(d + 1),   # eager drops trivial mode internally
            ϵ=float(ϵ),
            seed=int(seed),
        )

        β_fit = np.asarray(eager["β"], dtype=np.float32)                    # (h,)
        q_fit = np.asarray(eager["q"], dtype=np.float32)                    # (h, N)
        λ_fit = np.asarray(eager["spectral"]["λ_x"], dtype=np.float32)      # (h, d)
        ψ_fit = np.asarray(eager["spectral"]["ψ_ix"], dtype=np.float32)     # (h, N, d)
        R_ix = np.asarray(eager["W"], dtype=np.float32)                     # (h, N, d)
        ε_kernel = np.asarray(eager["ε_kernel"], dtype=np.float32)          # (h,)

        if λ_fit.shape != (h, d):
            raise ValueError(f"Expected λ_x shape {(h, d)}, got {λ_fit.shape}.")
        if ψ_fit.shape != (h, N, d):
            raise ValueError(f"Expected ψ_ix shape {(h, N, d)}, got {ψ_fit.shape}.")
        if R_ix.shape != (h, N, d):
            raise ValueError(f"Expected W shape {(h, N, d)}, got {R_ix.shape}.")
        if q_fit.shape != (h, N):
            raise ValueError(f"Expected q shape {(h, N)}, got {q_fit.shape}.")

        # abstract dmap uses final Linear = R_ix / λ_x
        λ_safe = np.maximum(λ_fit, float(ϵ))
        W_linear = R_ix / λ_safe[:, None, :]  # (h, N, d)

        self.init_data = DMAPInit(
            R_iX=R_iX,
            β=β_fit,
            q=q_fit,
            λ_x=λ_fit,
            ψ_ix=ψ_fit,
            R_ix=R_ix,
            W_linear=W_linear,
            ε_kernel=ε_kernel,
            L=L,
        )

        # build abstract flax module
        beta_module: BetaSpec
        if h == 1:
            beta_module = float(β_fit[0])
        else:
            beta_module = tuple(float(b) for b in β_fit)

        self.module = dmap(
            d=d,
            N=N,
            h=h,
            α=float(α),
            β=beta_module,
            metric_rank=metric_rank,
            eps=float(ϵ),
            dtype=dtype,
            param_dtype=param_dtype,
        )

        # initialize flax variables
        dummy_x = jnp.zeros((1, D), dtype=dtype)
        variables = self.module.init(jax.random.PRNGKey(seed), dummy_x)
        vars_mut = unfreeze(variables)
        params = vars_mut["params"]

        # patch eager solution into parameter tree
        params["rbf"]["R_iX"] = jnp.asarray(R_iX, dtype=param_dtype)
        params["norm"]["q"] = jnp.asarray(q_fit, dtype=param_dtype)
        params["W"] = jnp.asarray(W_linear, dtype=param_dtype)

        if L is not None:
            params["rbf"]["L"] = jnp.asarray(L, dtype=param_dtype)

        self.variables = freeze(vars_mut)

        self.config = {
            "R_iX_shape": tuple(R_iX.shape),
            "d": int(d),
            "h": int(h),
            "α": float(α),
            "β": beta_module,
            "metric_rank": metric_rank,
            "t": int(t),
            "k_nn": int(k_nn),
            "q_block": int(q_block),
            "r_block": int(r_block),
            "ϵ": float(ϵ),
            "seed": int(seed),
        }

    def __call__(self, x: np.ndarray | jnp.ndarray) -> jnp.ndarray:
        x = jnp.asarray(x, dtype=self.module.dtype)
        return self.module.apply(self.variables, x)

    @property
    def params(self):
        return self.variables["params"]

    def apply(self, x: np.ndarray | jnp.ndarray, variables: Optional[Dict[str, Any]] = None):
        x = jnp.asarray(x, dtype=self.module.dtype)
        vars_use = self.variables if variables is None else variables
        return self.module.apply(vars_use, x)
