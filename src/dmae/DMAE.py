# DMAE.py
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import jax.numpy as jnp

from .DMAP import DMAP
from .GPLM import GPLM


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


class DMAE:
    """
    Dense exact DMAE wrapper combining:
        DMAP encoder + GPLM decoder

    Shared latent geometry:
        - training ambient anchors R_iX are given once
        - encoder is built from R_iX
        - shared training latents R_ix are computed once from encoder(R_iX)
        - decoder is built from those same shared R_ix

    Public API:
        model = DMAE(R_iX, d=32)

        Q_ix = model.encode(R_iX)        # ambient -> latent
        Q_iX = model(Q_ix)               # latent  -> ambient   (__call__ = decode)
        Q_iX = model.decode(Q_ix)        # same as above
        Q_iX = model.reconstruct(R_iX)   # ambient -> latent -> ambient
    """

    def __init__(
        self,
        R_iX: np.ndarray,
        d: int = 32,
        *,
        h: int = 1,

        # shared user-friendly aliases
        α: float | None = None,
        β: float | Tuple[float, ...] | np.ndarray | None = None,
        alpha: float | None = None,
        beta: float | Tuple[float, ...] | np.ndarray | None = None,

        # encoder-specific
        t: int = 1,
        mahalanobis: bool = False,
        Q: np.ndarray | None = None,
        metric_rank: int | None = None,
        metric_init: str = "euclidean",
        metric_mix: float = 0.1,
        zero_diag: bool = True,
        k_eigs: int | None = None,
        which: str = "LA",

        # decoder-specific
        sigma2: float = 1e-6,
        use_W_O: bool = False,
        D_out: int | None = None,

        # numerics
        eps: float = 1e-12,
        seed: int = 0,
        dtype: Any = jnp.float32,
        param_dtype: Any = jnp.float32,
    ):
        R_iX = np.asarray(R_iX, dtype=np.float32)
        if R_iX.ndim != 2:
            raise ValueError(f"R_iX must have shape (N, D), got {R_iX.shape}.")

        α = _resolve_scalar_alias("α", α, "alpha", alpha, 1.0)
        β = _resolve_beta_alias(β, beta, None)

        self.R_iX = R_iX
        self.N, self.D = R_iX.shape

        # Build encoder
        self.encoder = DMAP(
            R_iX=R_iX,
            d=int(d),
            h=int(h),
            α=float(α),
            β=β,
            t=int(t),
            mahalanobis=bool(mahalanobis),
            Q=Q,
            metric_rank=metric_rank,
            metric_init=metric_init,
            metric_mix=float(metric_mix),
            zero_diag=bool(zero_diag),
            k_eigs=k_eigs,
            which=which,
            eps=float(eps),
            seed=int(seed),
            dtype=dtype,
            param_dtype=param_dtype,
        )

        # Shared latent anchors are exactly what the encoder produces on training anchors
        # External latent convention is (N, h, d), which is exactly what GPLM expects.
        self.R_ix = np.asarray(self.encoder(R_iX), dtype=np.float32)  # (N, h, d)

        # Build decoder from the SAME shared latent bank
        self.decoder = GPLM(
            R_ix=self.R_ix,
            R_iX=R_iX,
            β=β,
            metric_rank=metric_rank if mahalanobis else None,
            sigma2=float(sigma2),
            use_W_O=bool(use_W_O),
            D_out=D_out,
            eps=float(eps),
            seed=int(seed),
            dtype=dtype,
            param_dtype=param_dtype,
        )

        self.config: Dict[str, Any] = {
            "R_iX_shape": tuple(R_iX.shape),
            "d": int(d),
            "h": int(h),
            "α": float(α),
            "β": β,
            "t": int(t),
            "mahalanobis": bool(mahalanobis),
            "metric_rank": metric_rank,
            "metric_init": str(metric_init),
            "metric_mix": float(metric_mix),
            "zero_diag": bool(zero_diag),
            "k_eigs": None if k_eigs is None else int(k_eigs),
            "which": str(which),
            "sigma2": float(sigma2),
            "use_W_O": bool(use_W_O),
            "D_out": D_out,
            "eps": float(eps),
            "seed": int(seed),
        }

    def encode(self, X: np.ndarray | jnp.ndarray) -> jnp.ndarray:
        """
        Encode ambient points to latent points.

        Input:
            X: (B, D)

        Output:
            Z: (B, h, d)
        """
        return self.encoder(X)

    def decode(self, Z: np.ndarray | jnp.ndarray) -> jnp.ndarray:
        """
        Decode latent points to ambient points.

        Input:
            Z: (B, h, d), or (B, d) if h=1

        Output:
            X_hat: (B, D_out) where D_out = D by default
        """
        return self.decoder(Z)

    def reconstruct(self, X: np.ndarray | jnp.ndarray) -> jnp.ndarray:
        """
        Reconstruct ambient points through the full autoencoder:
            X -> encode(X) -> decode(Z)
        """
        Z = self.encode(X)
        return self.decode(Z)

    def __call__(self, Z: np.ndarray | jnp.ndarray) -> jnp.ndarray:
        """
        By design, __call__ performs decoding:
            DMAE(R_iX). __call__(R_ix) -> ambient reconstruction
        """
        return self.decode(Z)

    @property
    def params(self) -> Dict[str, Any]:
        return {
            "encoder": self.encoder.params,
            "decoder": self.decoder.params,
        }
