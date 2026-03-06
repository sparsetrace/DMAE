# NWLM.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import jax
import jax.numpy as jnp
from flax.core import freeze, unfreeze

from .blocks import nwlm


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


@dataclass
class NWLMInit:
    R_ix: np.ndarray                 # (h, N, d)
    R_iX: np.ndarray                 # (N, D_out)
    β: np.ndarray                    # (h,)
    W: np.ndarray                    # (h, N, D_head)
    W_O: Optional[np.ndarray] = None # (h*D_head, D_out)
    b_O: Optional[np.ndarray] = None # (D_out,)


class NWLM:
    """
    Wrapper around the trainable Flax `nwlm` decoder.

    Parameters
    ----------
    R_ix:
        Training latent anchors. Shape `(N, d)` for one head, or `(h, N, d)` for multi-head.
    R_iX:
        Ambient targets / anchor values. Shape `(N, D_out)`.
    β / beta:
        Scalar or per-head tuple.
    α / alpha:
        CL exponent used by Norm.
    use_W_O:
        If False, each head outputs `D_head` and all heads are concatenated.
        If True, concatenated head outputs are projected to `D_out`.

    Notes
    -----
    The default initialization uses the ambient anchor values directly:
        W[h, i, :] = R_iX[i, :]
    for each head.
    """

    def __init__(
        self,
        R_ix: np.ndarray,
        R_iX: np.ndarray,
        *,
        α: float | None = None,
        β: float | Tuple[float, ...] | np.ndarray | None = None,
        alpha: float | None = None,
        beta: float | Tuple[float, ...] | np.ndarray | None = None,
        metric_rank: int | None = None,
        use_W_O: bool = False,
        D_out: int | None = None,
        eps: float = 1e-12,
        seed: int = 0,
        dtype: Any = jnp.float32,
        param_dtype: Any = jnp.float32,
    ):
        α = _resolve_scalar_alias("α", α, "alpha", alpha, 0.0)
        β = _resolve_beta_alias(β, beta, 1.0)

        R_iX = np.asarray(R_iX, dtype=np.float32)
        if R_iX.ndim != 2:
            raise ValueError(f"R_iX must have shape (N, D_out), got {R_iX.shape}.")
        N_x, D_ambient = R_iX.shape

        R_ix = np.asarray(R_ix, dtype=np.float32)
        if R_ix.ndim == 2:
            # one-head convenience: (N, d) -> (1, N, d)
            R_ix = R_ix[None, ...]
        if R_ix.ndim != 3:
            raise ValueError(f"R_ix must have shape (N, d) or (h, N, d), got {R_ix.shape}.")

        h, N_z, d = R_ix.shape
        if N_z != N_x:
            raise ValueError(
                f"R_ix and R_iX must have the same anchor count N, got {N_z} and {N_x}."
            )

        # Decoder per-head output dimension
        # The simplest useful default: each head predicts the full ambient dimension.
        D_head = D_ambient

        # Beta handling
        if β is None:
            beta_heads = np.ones((h,), dtype=np.float32)
        elif np.isscalar(β):
            beta_heads = np.full((h,), float(β), dtype=np.float32)
        else:
            beta_heads = np.asarray(β, dtype=np.float32).reshape(-1)
            if beta_heads.shape[0] != h:
                raise ValueError(f"β must be scalar or length h={h}, got shape {beta_heads.shape}.")

        # Initialize per-head value maps from ambient anchor values
        # Same ambient anchor values for each head by default.
        W_init = np.broadcast_to(R_iX[None, :, :], (h, N_x, D_head)).copy()

        W_O_init = None
        b_O_init = None
        if use_W_O:
            D_out_final = D_ambient if D_out is None else int(D_out)
            # Initialize W^O to an average over heads if dimensions match.
            W_O_init = np.zeros((h * D_head, D_out_final), dtype=np.float32)
            if D_out_final == D_head:
                for hh in range(h):
                    W_O_init[hh * D_head : (hh + 1) * D_head, :] = np.eye(
                        D_head, dtype=np.float32
                    ) / float(h)
            b_O_init = np.zeros((D_out_final,), dtype=np.float32)
        else:
            D_out_final = h * D_head

        self.init_data = NWLMInit(
            R_ix=R_ix,
            R_iX=R_iX,
            β=beta_heads,
            W=W_init,
            W_O=W_O_init,
            b_O=b_O_init,
        )

        beta_module: BetaSpec
        if h == 1:
            beta_module = float(beta_heads[0])
        else:
            beta_module = tuple(float(b) for b in beta_heads)

        self.module = nwlm(
            D_head=int(D_head),
            N=int(N_x),
            h=int(h),
            α=float(α),
            β=beta_module,
            metric_rank=metric_rank,
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

        # Patch initialization into the Flax tree
        params["R_ix"] = jnp.asarray(self.init_data.R_ix, dtype=param_dtype)
        params["W"] = jnp.asarray(self.init_data.W, dtype=param_dtype)

        if use_W_O:
            params["W_O"] = jnp.asarray(self.init_data.W_O, dtype=param_dtype)
            params["b_O"] = jnp.asarray(self.init_data.b_O, dtype=param_dtype)

        self.variables = freeze(vars_mut)

        self.config = {
            "R_ix_shape": tuple(R_ix.shape),
            "R_iX_shape": tuple(R_iX.shape),
            "h": int(h),
            "d": int(d),
            "D_head": int(D_head),
            "D_out": int(D_out_final),
            "α": float(α),
            "β": beta_module,
            "metric_rank": None if metric_rank is None else int(metric_rank),
            "use_W_O": bool(use_W_O),
            "eps": float(eps),
            "seed": int(seed),
        }

    def __call__(self, z: np.ndarray | jnp.ndarray) -> jnp.ndarray:
        z = jnp.asarray(z, dtype=self.module.dtype)
        if z.ndim == 2:
            # one-head convenience: (B, d) -> (B, 1, d)
            if self.config["h"] != 1:
                raise ValueError(
                    f"Got 2D latent input {z.shape}, but decoder has h={self.config['h']} heads."
                )
            z = z[:, None, :]
        return self.module.apply(self.variables, z)

    def apply(self, z: np.ndarray | jnp.ndarray, variables: Optional[Dict[str, Any]] = None):
        z = jnp.asarray(z, dtype=self.module.dtype)
        if z.ndim == 2:
            if self.config["h"] != 1:
                raise ValueError(
                    f"Got 2D latent input {z.shape}, but decoder has h={self.config['h']} heads."
                )
            z = z[:, None, :]
        vars_use = self.variables if variables is None else variables
        return self.module.apply(vars_use, z)

    @property
    def params(self):
        return self.variables["params"]

    @classmethod
    def from_latents(cls, *args, **kwargs) -> "NWLM":
        return cls(*args, **kwargs)
