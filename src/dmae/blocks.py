# This file defines the core multi-head kernel blocks for a trainable DMAE-style
# model family. It provides five main components: `RBF`, `Norm`, `dmap`, `nwlm`,
# and `gplm`.
#
# `RBF` is the shared kernel layer. It learns anchor points and computes Gaussian
# radial-basis affinities between inputs and anchors. In Euclidean mode it uses
# ordinary squared distance; when `metric_rank` is an integer, it instead uses a
# per-head low-rank Mahalanobis factor `L`, so each head measures distance in its
# own learned metric. The output is a tensor of positive kernel weights with shape
# `(..., h, N)`.
#
# `Norm` performs Coifman–Lafon normalization followed by row normalization. It
# takes positive kernel weights, optionally applies a trainable per-head density
# correction `q`, scales by the exponent `α`, and then normalizes rows so the last
# axis sums to one. This is the normalization used by the DMAP-style encoder and
# the NWLM decoder.
#
# `dmap` is the multi-head encoder. For an ambient input `x`, it applies
# `RBF -> Norm -> Linear` independently across heads. The learned linear map for
# each head has shape `(N, d)`, so each head produces a latent feature block of
# size `d`. The encoder does not concatenate heads; it returns latent features in
# matrix form with shape `(..., h, d)`.
#
# `nwlm` is the normalized kernel decoder. It expects head-structured latent input
# `(..., h, d)`. For each head it computes kernel weights against learned latent
# anchors, applies `Norm`, and maps the normalized weights through a learned
# linear value map to produce a per-head output block. These head outputs are then
# concatenated. If `use_W_O=True`, a final learned output projection `W^O` maps
# the concatenated decoder output to the requested output dimension.
#
# `gplm` is the unnormalized kernel decoder. It has the same overall structure as
# `nwlm`, except that after the RBF computation it skips normalization and applies
# the linear value map directly. This makes it the simpler kernel-regression-style
# decoder. Like `nwlm`, it concatenates head outputs automatically and can
# optionally apply a final projection `W^O`.
#
# In short, the file implements a consistent multi-head architecture:
# - `dmap`: `RBF -> Norm -> Linear`
# - `nwlm`: `RBF -> Norm -> Linear`
# - `gplm`: `RBF -> Linear`
#
# The encoder keeps the head dimension explicit, while both decoders merge head
# outputs into one feature vector and can optionally project that vector into the
# final output space.


from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import jax.numpy as jnp
import flax.linen as nn


BetaSpec = Union[float, Tuple[float, ...]]


class RBF(nn.Module):
    """
    Multi-head RBF kernel layer with learned anchors and optional Mahalanobis factor.

    For each head h and anchor i:
        K_h(x, R_i) = exp( -β_h * d_h^2(x, R_i) )

    Euclidean:
        if metric_rank is None
        d_h^2(x, R_i) = ||x - R_i||^2

    Mahalanobis:
        if metric_rank = r is an int
        d_h^2(x, R_i) = ||(x - R_i) L_h||^2
        with L: (h, D, r)

    Shapes:
        x:   (..., D)
        out: (..., h, N)
    """
    N: int
    h: int = 1

    β: BetaSpec = 1.0
    metric_rank: Optional[int] = None

    eps: float = 1e-12
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = jnp.asarray(x, self.dtype)
        D = x.shape[-1]

        R = self.param(
            "R_iX",
            nn.initializers.lecun_normal(),
            (self.N, D),
            self.param_dtype,
        )  # (N, D)

        if isinstance(self.β, tuple):
            if len(self.β) != self.h:
                raise ValueError(f"β tuple must have length h={self.h}, got {len(self.β)}.")
            beta = jnp.asarray(self.β, dtype=self.dtype)
        else:
            beta = jnp.full((self.h,), float(self.β), dtype=self.dtype)

        if self.metric_rank is None:
            x2 = jnp.sum(x * x, axis=-1, keepdims=True)   # (..., 1)
            cross = -2.0 * (x @ R.T)                      # (..., N)
            R2 = jnp.sum(R * R, axis=-1)                  # (N,)
            dist2 = jnp.maximum(x2 + cross + R2, 0.0)     # (..., N)

            dist2_h = dist2[..., None, :]
            if self.h != 1:
                dist2_h = jnp.broadcast_to(
                    dist2_h,
                    dist2.shape[:-1] + (self.h, self.N),
                )
        else:
            r = int(self.metric_rank)
            if r <= 0 or r > D:
                raise ValueError(f"RBF.metric_rank must be in [1, D], got r={r}, D={D}.")

            L = self.param(
                "L",
                nn.initializers.lecun_normal(),
                (self.h, D, r),
                self.param_dtype,
            )  # (h, D, r)

            xh = jnp.einsum("...d,hdr->...hr", x, L)      # (..., h, r)
            Rh = jnp.einsum("nd,hdr->hnr", R, L)          # (h, N, r)

            xh2 = jnp.sum(xh * xh, axis=-1, keepdims=True)              # (..., h, 1)
            Rh2 = jnp.sum(Rh * Rh, axis=-1)                             # (h, N)
            cross_h = -2.0 * jnp.einsum("...hr,hnr->...hn", xh, Rh)     # (..., h, N)
            dist2_h = jnp.maximum(xh2 + cross_h + Rh2, 0.0)

        beta = beta.reshape((1,) * (dist2_h.ndim - 2) + (self.h, 1))
        K = jnp.exp(-beta * dist2_h)
        return K.astype(self.dtype)

    @property
    def config(self) -> Dict[str, Any]:
        beta_cfg: Any = [float(b) for b in self.β] if isinstance(self.β, tuple) else float(self.β)
        return {
            "N": int(self.N),
            "h": int(self.h),
            "β": beta_cfg,
            "metric_rank": None if self.metric_rank is None else int(self.metric_rank),
            "eps": float(self.eps),
        }

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "RBF":
        beta_cfg = cfg.get("β", cfg.get("beta", 1.0))
        beta_val: BetaSpec = tuple(float(b) for b in beta_cfg) if isinstance(beta_cfg, list) else float(beta_cfg)
        return cls(
            N=int(cfg["N"]),
            h=int(cfg.get("h", 1)),
            β=beta_val,
            metric_rank=cfg.get("metric_rank", None),
            eps=float(cfg.get("eps", 1e-12)),
        )


class Norm(nn.Module):
    """
    Coifman-Lafon normalization followed by row normalization.

    Given positive affinities K_{...hi}:
        Ksum_a = sum_i K_{ahi}
        Ktilde_{ahi} = Ksum_a^{-α} * K_{ahi} * q_{hi}^{-α}
        P_{ahi} = Ktilde_{ahi} / sum_i Ktilde_{ahi}

    Shapes:
        K:   (..., h, N) or (..., N)
        out: same shape as K
    """
    N: int
    α: float = 0.0
    h: int = 1
    per_head_q: bool = True

    eps: float = 1e-12
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, K: jnp.ndarray) -> jnp.ndarray:
        K = jnp.asarray(K, self.dtype)
        if K.shape[-1] != self.N:
            raise ValueError(f"Norm expected last dim {self.N}, got {K.shape[-1]}.")

        K = jnp.maximum(K, 0.0)
        has_head_axis = (K.ndim >= 2) and (K.shape[-2] == self.h)

        if self.α != 0.0:
            Ksum = jnp.sum(K, axis=-1, keepdims=True)
            Ksum = jnp.maximum(Ksum, self.eps)

            if self.per_head_q and has_head_axis:
                q_raw = self.param("q", nn.initializers.ones, (self.h, self.N), self.param_dtype)
                q = nn.relu(q_raw) + self.eps
                K = (Ksum ** (-self.α)) * K * (q ** (-self.α))
            else:
                q_raw = self.param("q", nn.initializers.ones, (self.N,), self.param_dtype)
                q = nn.relu(q_raw) + self.eps
                K = (Ksum ** (-self.α)) * K * (q ** (-self.α))

        row_sum = jnp.sum(K, axis=-1, keepdims=True)
        row_sum = jnp.maximum(row_sum, self.eps)
        return (K / row_sum).astype(self.dtype)

    @property
    def config(self) -> Dict[str, Any]:
        return {
            "N": int(self.N),
            "α": float(self.α),
            "h": int(self.h),
            "per_head_q": bool(self.per_head_q),
            "eps": float(self.eps),
        }

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "Norm":
        alpha = cfg.get("α", cfg.get("alpha", 0.0))
        return cls(
            N=int(cfg["N"]),
            α=float(alpha),
            h=int(cfg.get("h", 1)),
            per_head_q=bool(cfg.get("per_head_q", True)),
            eps=float(cfg.get("eps", 1e-12)),
        )


class dmap(nn.Module):
    """
    Multi-head DMAP encoder.

    Pipeline per head:
        x -> RBF -> Norm -> Linear

    Output is NOT concatenated:
        (..., h, d)

    Parameters:
        d: per-head latent dimension
        N: number of encoder anchors
        h: number of heads
        β: scalar or per-head tuple
        α: CL exponent
        metric_rank: None => Euclidean, int => Mahalanobis rank
    """
    d: int
    N: int
    h: int = 1

    α: float = 1.0
    β: BetaSpec = 1.0
    metric_rank: Optional[int] = None

    eps: float = 1e-12
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = jnp.asarray(x, self.dtype)

        K = RBF(
            N=self.N,
            h=self.h,
            β=self.β,
            metric_rank=self.metric_rank,
            eps=self.eps,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="rbf",
        )(x)  # (..., h, N)

        P = Norm(
            N=self.N,
            α=self.α,
            h=self.h,
            per_head_q=True,
            eps=self.eps,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="norm",
        )(K)  # (..., h, N)

        W = self.param(
            "W",
            nn.initializers.lecun_normal(),
            (self.h, self.N, self.d),
            self.param_dtype,
        )  # (h, N, d)

        y = jnp.einsum("...hn,hnd->...hd", P, W)  # (..., h, d)
        return y.astype(self.dtype)

    @property
    def config(self) -> Dict[str, Any]:
        beta_cfg: Any = [float(b) for b in self.β] if isinstance(self.β, tuple) else float(self.β)
        return {
            "d": int(self.d),
            "N": int(self.N),
            "h": int(self.h),
            "α": float(self.α),
            "β": beta_cfg,
            "metric_rank": None if self.metric_rank is None else int(self.metric_rank),
            "eps": float(self.eps),
        }

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "dmap":
        alpha = cfg.get("α", cfg.get("alpha", 1.0))
        beta_cfg = cfg.get("β", cfg.get("beta", 1.0))
        beta_val: BetaSpec = tuple(float(b) for b in beta_cfg) if isinstance(beta_cfg, list) else float(beta_cfg)
        return cls(
            d=int(cfg["d"]),
            N=int(cfg["N"]),
            h=int(cfg.get("h", 1)),
            α=float(alpha),
            β=beta_val,
            metric_rank=cfg.get("metric_rank", None),
            eps=float(cfg.get("eps", 1e-12)),
        )


class nwlm(nn.Module):
    """
    Multi-head NWLM decoder.

    Pipeline per head:
        z_h -> RBF -> Norm -> Linear

    Input:
        z: (..., h, d)

    Per-head output:
        (..., h, D_head)

    Final output:
        concat heads -> (..., h * D_head)
        optional W^O -> (..., D_out)
    """
    D_head: int
    N: int
    h: int = 1

    α: float = 0.0
    β: BetaSpec = 1.0
    metric_rank: Optional[int] = None

    use_W_O: bool = False
    D_out: Optional[int] = None

    eps: float = 1e-12
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, z: jnp.ndarray) -> jnp.ndarray:
        z = jnp.asarray(z, self.dtype)
        if z.ndim < 2 or z.shape[-2] != self.h:
            raise ValueError(f"nwlm expected input shape (..., h, d) with h={self.h}, got {z.shape}.")

        d = z.shape[-1]

        # Per-head anchors and optional Mahalanobis factors
        R = self.param(
            "R_ix",
            nn.initializers.lecun_normal(),
            (self.h, self.N, d),
            self.param_dtype,
        )  # (h, N, d)

        if isinstance(self.β, tuple):
            if len(self.β) != self.h:
                raise ValueError(f"β tuple must have length h={self.h}, got {len(self.β)}.")
            beta = jnp.asarray(self.β, dtype=self.dtype)
        else:
            beta = jnp.full((self.h,), float(self.β), dtype=self.dtype)

        if self.metric_rank is None:
            z2 = jnp.sum(z * z, axis=-1, keepdims=True)                        # (..., h, 1)
            R2 = jnp.sum(R * R, axis=-1)                                       # (h, N)
            cross = -2.0 * jnp.einsum("...hd,hnd->...hn", z, R)                # (..., h, N)
            dist2 = jnp.maximum(z2 + cross + R2, 0.0)                          # (..., h, N)
        else:
            r = int(self.metric_rank)
            if r <= 0 or r > d:
                raise ValueError(f"nwlm.metric_rank must be in [1, d], got r={r}, d={d}.")

            L = self.param(
                "L",
                nn.initializers.lecun_normal(),
                (self.h, d, r),
                self.param_dtype,
            )  # (h, d, r)

            zh = jnp.einsum("...hd,hdr->...hr", z, L)                          # (..., h, r)
            Rh = jnp.einsum("hnd,hdr->hnr", R, L)                              # (h, N, r)

            zh2 = jnp.sum(zh * zh, axis=-1, keepdims=True)                     # (..., h, 1)
            Rh2 = jnp.sum(Rh * Rh, axis=-1)                                    # (h, N)
            cross = -2.0 * jnp.einsum("...hr,hnr->...hn", zh, Rh)              # (..., h, N)
            dist2 = jnp.maximum(zh2 + cross + Rh2, 0.0)                        # (..., h, N)

        beta = beta.reshape((1,) * (dist2.ndim - 2) + (self.h, 1))
        K = jnp.exp(-beta * dist2)                                             # (..., h, N)

        P = Norm(
            N=self.N,
            α=self.α,
            h=self.h,
            per_head_q=True,
            eps=self.eps,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="norm",
        )(K)  # (..., h, N)

        W = self.param(
            "W",
            nn.initializers.lecun_normal(),
            (self.h, self.N, self.D_head),
            self.param_dtype,
        )  # (h, N, D_head)

        y_h = jnp.einsum("...hn,hnd->...hd", P, W)  # (..., h, D_head)
        y = y_h.reshape(*y_h.shape[:-2], self.h * self.D_head)  # (..., h*D_head)

        if not self.use_W_O:
            return y.astype(self.dtype)

        D_out = self.h * self.D_head if self.D_out is None else int(self.D_out)
        W_O = self.param(
            "W_O",
            nn.initializers.lecun_normal(),
            (self.h * self.D_head, D_out),
            self.param_dtype,
        )
        b_O = self.param(
            "b_O",
            nn.initializers.zeros,
            (D_out,),
            self.param_dtype,
        )
        return (y @ W_O + b_O).astype(self.dtype)

    @property
    def config(self) -> Dict[str, Any]:
        beta_cfg: Any = [float(b) for b in self.β] if isinstance(self.β, tuple) else float(self.β)
        return {
            "D_head": int(self.D_head),
            "N": int(self.N),
            "h": int(self.h),
            "α": float(self.α),
            "β": beta_cfg,
            "metric_rank": None if self.metric_rank is None else int(self.metric_rank),
            "use_W_O": bool(self.use_W_O),
            "D_out": None if self.D_out is None else int(self.D_out),
            "eps": float(self.eps),
        }

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "nwlm":
        alpha = cfg.get("α", cfg.get("alpha", 0.0))
        beta_cfg = cfg.get("β", cfg.get("beta", 1.0))
        beta_val: BetaSpec = tuple(float(b) for b in beta_cfg) if isinstance(beta_cfg, list) else float(beta_cfg)
        return cls(
            D_head=int(cfg["D_head"]),
            N=int(cfg["N"]),
            h=int(cfg.get("h", 1)),
            α=float(alpha),
            β=beta_val,
            metric_rank=cfg.get("metric_rank", None),
            use_W_O=bool(cfg.get("use_W_O", False)),
            D_out=cfg.get("D_out", None),
            eps=float(cfg.get("eps", 1e-12)),
        )


class gplm(nn.Module):
    """
    Multi-head GPLM decoder.

    Pipeline per head:
        z_h -> RBF -> Linear

    Input:
        z: (..., h, d)

    Per-head output:
        (..., h, D_head)

    Final output:
        concat heads -> (..., h * D_head)
        optional W^O -> (..., D_out)
    """
    D_head: int
    N: int
    h: int = 1

    β: BetaSpec = 1.0
    metric_rank: Optional[int] = None

    use_W_O: bool = False
    D_out: Optional[int] = None

    eps: float = 1e-12
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, z: jnp.ndarray) -> jnp.ndarray:
        z = jnp.asarray(z, self.dtype)
        if z.ndim < 2 or z.shape[-2] != self.h:
            raise ValueError(f"gplm expected input shape (..., h, d) with h={self.h}, got {z.shape}.")

        d = z.shape[-1]

        R = self.param(
            "R_ix",
            nn.initializers.lecun_normal(),
            (self.h, self.N, d),
            self.param_dtype,
        )  # (h, N, d)

        if isinstance(self.β, tuple):
            if len(self.β) != self.h:
                raise ValueError(f"β tuple must have length h={self.h}, got {len(self.β)}.")
            beta = jnp.asarray(self.β, dtype=self.dtype)
        else:
            beta = jnp.full((self.h,), float(self.β), dtype=self.dtype)

        if self.metric_rank is None:
            z2 = jnp.sum(z * z, axis=-1, keepdims=True)                        # (..., h, 1)
            R2 = jnp.sum(R * R, axis=-1)                                       # (h, N)
            cross = -2.0 * jnp.einsum("...hd,hnd->...hn", z, R)                # (..., h, N)
            dist2 = jnp.maximum(z2 + cross + R2, 0.0)
        else:
            r = int(self.metric_rank)
            if r <= 0 or r > d:
                raise ValueError(f"gplm.metric_rank must be in [1, d], got r={r}, d={d}.")

            L = self.param(
                "L",
                nn.initializers.lecun_normal(),
                (self.h, d, r),
                self.param_dtype,
            )  # (h, d, r)

            zh = jnp.einsum("...hd,hdr->...hr", z, L)                          # (..., h, r)
            Rh = jnp.einsum("hnd,hdr->hnr", R, L)                              # (h, N, r)

            zh2 = jnp.sum(zh * zh, axis=-1, keepdims=True)
            Rh2 = jnp.sum(Rh * Rh, axis=-1)
            cross = -2.0 * jnp.einsum("...hr,hnr->...hn", zh, Rh)
            dist2 = jnp.maximum(zh2 + cross + Rh2, 0.0)

        beta = beta.reshape((1,) * (dist2.ndim - 2) + (self.h, 1))
        K = jnp.exp(-beta * dist2)                                             # (..., h, N)

        W = self.param(
            "W",
            nn.initializers.lecun_normal(),
            (self.h, self.N, self.D_head),
            self.param_dtype,
        )  # (h, N, D_head)

        y_h = jnp.einsum("...hn,hnd->...hd", K, W)  # (..., h, D_head)
        y = y_h.reshape(*y_h.shape[:-2], self.h * self.D_head)  # (..., h*D_head)

        if not self.use_W_O:
            return y.astype(self.dtype)

        D_out = self.h * self.D_head if self.D_out is None else int(self.D_out)
        W_O = self.param(
            "W_O",
            nn.initializers.lecun_normal(),
            (self.h * self.D_head, D_out),
            self.param_dtype,
        )
        b_O = self.param(
            "b_O",
            nn.initializers.zeros,
            (D_out,),
            self.param_dtype,
        )
        return (y @ W_O + b_O).astype(self.dtype)

    @property
    def config(self) -> Dict[str, Any]:
        beta_cfg: Any = [float(b) for b in self.β] if isinstance(self.β, tuple) else float(self.β)
        return {
            "D_head": int(self.D_head),
            "N": int(self.N),
            "h": int(self.h),
            "β": beta_cfg,
            "metric_rank": None if self.metric_rank is None else int(self.metric_rank),
            "use_W_O": bool(self.use_W_O),
            "D_out": None if self.D_out is None else int(self.D_out),
            "eps": float(self.eps),
        }

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "gplm":
        beta_cfg = cfg.get("β", cfg.get("beta", 1.0))
        beta_val: BetaSpec = tuple(float(b) for b in beta_cfg) if isinstance(beta_cfg, list) else float(beta_cfg)
        return cls(
            D_head=int(cfg["D_head"]),
            N=int(cfg["N"]),
            h=int(cfg.get("h", 1)),
            β=beta_val,
            metric_rank=cfg.get("metric_rank", None),
            use_W_O=bool(cfg.get("use_W_O", False)),
            D_out=cfg.get("D_out", None),
            eps=float(cfg.get("eps", 1e-12)),
        )
