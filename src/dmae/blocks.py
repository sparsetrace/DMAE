# src/dmae/blocks.py
# Core DMAP encoder building blocks (to be wrapped later, e.g. in `DMAP.py`)
#
# Goals:
# - Keep things minimal and NN-friendly.
# - Multi-head works like MHSA structurally: per-head scores -> per-head weights -> per-head value map -> concat.
# - Geometric: distances are symmetric; metric is global (sample-location independent).
# - SMD can be Euclidean (no extra metric params) or Mahalanobis (global PSD metric factor L).
# - All classes include config/from_config for save/load.

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import jax.numpy as jnp
import flax.linen as nn


BetaSpec = Union[float, Tuple[float, ...]]


class SMD(nn.Module):
    """
    Squared distance to anchors, multi-head.

    Anchors are shared across heads (compression-friendly):
      R_iX: (N, D)

    Euclidean (mahalanobis=False):
      d_h^2(x, R_i) = ||x - R_i||^2   (same across heads; heads differ via β_h, q_h, W_h)

    Mahalanobis (mahalanobis=True), per-head global PSD metric M_h = L_h L_h^T:
      d_h^2(x, R_i) = || (x - R_i) L_h ||^2
      L: (h, D, r), where r defaults to D (full rank) if metric_rank is None

    Shapes:
      x: (..., D)
      out: (..., h, N)
    """
    N: int
    h: int = 1

    mahalanobis: bool = False
    metric_rank: Optional[int] = None  # if None and mahalanobis=True, uses r=D

    eps: float = 1e-12
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = jnp.asarray(x, self.dtype)
        D = x.shape[-1]

        # Shared anchors
        R = self.param(
            "R_iX",
            nn.initializers.lecun_normal(),
            (self.N, D),
            self.param_dtype,
        )  # (N, D)

        # Base Euclidean dist^2: (..., N)
        x2 = jnp.sum(x * x, axis=-1, keepdims=True)  # (..., 1)
        cross = -2.0 * (x @ R.T)                     # (..., N)
        R2 = jnp.sum(R * R, axis=-1)                 # (N,)
        dist2 = jnp.maximum(x2 + cross + R2, 0.0).astype(self.dtype)  # (..., N)

        if not self.mahalanobis:
            # Broadcast Euclidean dist to heads: (..., h, N)
            dist2_h = dist2[..., None, :]  # (..., 1, N)
            if self.h != 1:
                dist2_h = jnp.broadcast_to(dist2_h, dist2.shape[:-1] + (self.h, self.N))
            return dist2_h.astype(self.dtype)

        # Mahalanobis: per-head projection
        r = int(self.metric_rank) if (self.metric_rank is not None) else int(D)
        if r <= 0 or r > D:
            raise ValueError(f"SMD.metric_rank must be in [1, D], got r={r}, D={D}.")

        L = self.param(
            "L",
            nn.initializers.lecun_normal(),
            (self.h, D, r),
            self.param_dtype,
        )  # (h, D, r)

        # Project: x_h = x @ L_h, R_h = R @ L_h
        xh = jnp.einsum("...d,hdr->...hr", x, L)   # (..., h, r)
        Rh = jnp.einsum("nd,hdr->hnr", R, L)       # (h, N, r)

        # dist^2 in projected space: (..., h, N)
        xh2 = jnp.sum(xh * xh, axis=-1, keepdims=True)                 # (..., h, 1)
        Rh2 = jnp.sum(Rh * Rh, axis=-1)                                # (h, N)
        cross_h = -2.0 * jnp.einsum("...hr,hnr->...hn", xh, Rh)        # (..., h, N)
        dist2_h = jnp.maximum(xh2 + cross_h + Rh2, 0.0)                # (..., h, N)
        return dist2_h.astype(self.dtype)

    @property
    def config(self) -> Dict[str, Any]:
        return {
            "N": int(self.N),
            "h": int(self.h),
            "mahalanobis": bool(self.mahalanobis),
            "metric_rank": None if self.metric_rank is None else int(self.metric_rank),
            "eps": float(self.eps),
        }

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "SMD":
        return cls(
            N=int(cfg["N"]),
            h=int(cfg.get("h", 1)),
            mahalanobis=bool(cfg.get("mahalanobis", False)),
            metric_rank=cfg.get("metric_rank", None),
            eps=float(cfg.get("eps", 1e-12)),
        )

class Softmax(nn.Module):
    N: int
    α: float = 0.0
    h: int = 1
    per_head_q: bool = True

    eps: float = 1e-12
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, logits: jnp.ndarray) -> jnp.ndarray:
        logits = jnp.asarray(logits, self.dtype)
        if logits.shape[-1] != self.N:
            raise ValueError(f"Softmax expected last dim {self.N}, got {logits.shape[-1]}.")

        # stable exp
        z = logits - jnp.max(logits, axis=-1, keepdims=True)
        w = jnp.exp(z)

        # Detect whether logits has a head axis (..., h, N)
        has_head_axis = (logits.ndim >= 2) and (logits.shape[-2] == self.h)

        if self.α != 0.0:
            if self.per_head_q and has_head_axis:
                # q is per-head even when h==1 (shape (1,N))
                q_raw = self.param("q", nn.initializers.ones, (self.h, self.N), self.param_dtype)
                q = nn.relu(q_raw) + self.eps                       # (h, N)
                w = w * (q ** (-self.α))                            # broadcast to (..., h, N)
            else:
                q_raw = self.param("q", nn.initializers.ones, (self.N,), self.param_dtype)
                q = nn.relu(q_raw) + self.eps                       # (N,)
                w = w * (q ** (-self.α))                            # broadcast to (..., N)

        s = jnp.sum(w, axis=-1, keepdims=True)
        return (w / (s + self.eps)).astype(self.dtype)

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
    def from_config(cls, cfg: Dict[str, Any]) -> "Softmax":
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
    Multi-head DMAP-style encoder (compression / encoder use-case).

    Per-head:
      dist2_h = SMD(x)                             # (..., h, N)
      dist2_h = relu(β_h * dist2_h)                # positive safety, β is per-head
      p_h     = CLSoftmax(logits=-dist2_h, α)      # (..., h, N)
      y_h     = p_h @ W_h                          # (..., h, head_dim)
    Output:
      concat_h(y_h) -> (..., d)

    Params:
      d: total output dim (must be divisible by h)
      N: number of anchors
      h: number of heads
      β: scalar or tuple of length h (per-head distance scale)
      mahalanobis: if True, learns per-head metric factors L_h (global, location-independent)
      metric_rank: if None and mahalanobis=True, uses full rank r=D
    """
    d: int
    N: int
    h: int = 1

    α: float = 1.0
    β: BetaSpec = 1.0

    mahalanobis: bool = False
    metric_rank: Optional[int] = None

    eps: float = 1e-12

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = jnp.asarray(x, jnp.float32)

        if self.d % self.h != 0:
            raise ValueError(f"dmap requires d % h == 0, got d={self.d}, h={self.h}.")
        head_dim = self.d // self.h

        dist2_h = SMD(
            N=self.N,
            h=self.h,
            mahalanobis=self.mahalanobis,
            metric_rank=self.metric_rank,
            eps=self.eps,
            name="SMD",
        )(x)  # (..., h, N)

        # Build per-head beta vector
        if isinstance(self.β, tuple):
            if len(self.β) != self.h:
                raise ValueError(f"β tuple must have length h={self.h}, got {len(self.β)}.")
            beta = jnp.asarray(self.β, dtype=jnp.float32)  # (h,)
        else:
            beta = jnp.full((self.h,), float(self.β), dtype=jnp.float32)

        beta = beta.reshape((1,) * (dist2_h.ndim - 2) + (self.h, 1))  # broadcast to (..., h, 1)

        dist2_h = nn.relu(dist2_h * beta)  # (..., h, N)
        logits = -dist2_h

        p = Softmax(
            N=self.N,
            α=self.α,
            h=self.h,
            per_head_q=True,
            eps=self.eps,
            name="cl_softmax",
        )(logits)  # (..., h, N)

        # Per-head value map W_h : (h, N, d)  <-- changed
        W = self.param(
            "W",
            nn.initializers.lecun_normal(),
            (self.h, self.N, self.d),
            jnp.float32,
        )

        y = jnp.einsum("...hn,hnd->...hd", p, W)  # (..., h, d)  <-- already right
        return y.astype(jnp.float32)

    @property
    def config(self) -> Dict[str, Any]:
        beta_cfg: Any
        if isinstance(self.β, tuple):
            beta_cfg = [float(b) for b in self.β]  # JSON-friendly
        else:
            beta_cfg = float(self.β)

        return {
            "d": int(self.d),
            "N": int(self.N),
            "h": int(self.h),
            "α": float(self.α),
            "β": beta_cfg,
            "mahalanobis": bool(self.mahalanobis),
            "metric_rank": None if self.metric_rank is None else int(self.metric_rank),
            "eps": float(self.eps),
        }

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "dmap":
        alpha = cfg.get("α", cfg.get("alpha", 1.0))

        beta_cfg = cfg.get("β", cfg.get("beta", 1.0))
        if isinstance(beta_cfg, list):
            beta_val: BetaSpec = tuple(float(b) for b in beta_cfg)
        else:
            beta_val = float(beta_cfg)

        return cls(
            d=int(cfg["d"]),
            N=int(cfg["N"]),
            h=int(cfg.get("h", 1)),
            α=float(alpha),
            β=beta_val,
            mahalanobis=bool(cfg.get("mahalanobis", False)),
            metric_rank=cfg.get("metric_rank", None),
            eps=float(cfg.get("eps", 1e-12)),
        )

# src/dima/_dmap.py  (or rename to blocks.py)
from typing import Any, Dict, Optional, Tuple, Union
# assumes SMD and Softmax are defined above in this same file

BetaSpec = Union[float, Tuple[float, ...]]


class kgpr(nn.Module):
    """
    Kernel-GPR / Nyström-KRR decoder block (mean only), multihead-friendly.

    dist2_h = SMD(z)                  # (..., h, M)
    k_h     = exp(-β_h * dist2_h)     # (..., h, M)
    y_h     = k_h @ M_hX              # (..., h, D)
    output  = concat(y_h) or return per-head

    Wrapper fills:
      params["SMD"]["R_iX"] = inducing points Z  (M, d_lat)
      params["M_hX"]        = weights            (h, M, D)

    `β` is intended to already include any legacy scaling like (beta/eps).
    """
    D: int
    M: int
    h: int = 1
    β: BetaSpec = 1.0

    concat_heads: bool = True  # if True -> (..., h*D); else -> (..., h, D)

    eps: float = 1e-12
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, z: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        z = jnp.asarray(z, self.dtype)  # (..., d_lat)

        dist2_h = SMD(
            N=self.M,
            h=self.h,
            mahalanobis=False,
            eps=self.eps,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="SMD",
        )(z)  # (..., h, M)

        # β broadcast to (..., h, 1)
        if isinstance(self.β, tuple):
            if len(self.β) != self.h:
                raise ValueError(f"β tuple must have length h={self.h}, got {len(self.β)}.")
            beta = jnp.asarray(self.β, dtype=jnp.float32)  # (h,)
        else:
            beta = jnp.full((self.h,), float(self.β), dtype=jnp.float32)
        beta = beta.reshape((1,) * (dist2_h.ndim - 2) + (self.h, 1))

        k_h = jnp.exp(-(beta * dist2_h)).astype(self.dtype)  # (..., h, M)

        M_hX = self.param(
            "M_hX",
            nn.initializers.zeros,
            (self.h, self.M, self.D),
            self.param_dtype,
        )  # (h, M, D)

        y_h = jnp.einsum("...hm,hmd->...hd", k_h, M_hX).astype(self.dtype)  # (..., h, D)

        if self.concat_heads:
            return y_h.reshape(*y_h.shape[:-2], self.h * self.D)
        return y_h

    @property
    def config(self) -> Dict[str, Any]:
        beta_cfg: Any = [float(b) for b in self.β] if isinstance(self.β, tuple) else float(self.β)
        return {
            "D": int(self.D),
            "M": int(self.M),
            "h": int(self.h),
            "β": beta_cfg,
            "concat_heads": bool(self.concat_heads),
            "eps": float(self.eps),
        }

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "kgpr":
        beta_cfg = cfg.get("β", cfg.get("beta", 1.0))
        if isinstance(beta_cfg, list):
            beta_val: BetaSpec = tuple(float(b) for b in beta_cfg)
        else:
            beta_val = float(beta_cfg)

        return cls(
            D=int(cfg["D"]),
            M=int(cfg["M"]),
            h=int(cfg.get("h", 1)),
            β=beta_val,
            concat_heads=bool(cfg.get("concat_heads", True)),
            eps=float(cfg.get("eps", 1e-12)),
        )

# in _dmap.py (or blocks.py)
class nwlm(nn.Module):
    D: int
    M: int
    h: int = 1

    β: BetaSpec = 1.0
    α: float = 0.0

    # NEW:
    mahalanobis: bool = False
    metric_rank: Optional[int] = None

    concat_heads: bool = True
    eps: float = 1e-12
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, z: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        z = jnp.asarray(z, self.dtype)

        dist2_h = SMD(
            N=self.M,
            h=self.h,
            mahalanobis=self.mahalanobis,      # <-- now configurable
            metric_rank=self.metric_rank,      # <-- now configurable
            eps=self.eps,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="SMD",
        )(z)  # (..., h, M)

        # ... rest unchanged ...

    @property
    def config(self) -> Dict[str, Any]:
        beta_cfg: Any = [float(b) for b in self.β] if isinstance(self.β, tuple) else float(self.β)
        return {
            "D": int(self.D),
            "M": int(self.M),
            "h": int(self.h),
            "β": beta_cfg,
            "α": float(self.α),
            "mahalanobis": bool(self.mahalanobis),                 # NEW
            "metric_rank": None if self.metric_rank is None else int(self.metric_rank),  # NEW
            "concat_heads": bool(self.concat_heads),
            "eps": float(self.eps),
        }

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "nwlm":
        alpha = cfg.get("α", cfg.get("alpha", 0.0))
        beta_cfg = cfg.get("β", cfg.get("beta", 1.0))
        beta_val: BetaSpec = tuple(float(b) for b in beta_cfg) if isinstance(beta_cfg, list) else float(beta_cfg)

        return cls(
            D=int(cfg["D"]),
            M=int(cfg["M"]),
            h=int(cfg.get("h", 1)),
            β=beta_val,
            α=float(alpha),
            mahalanobis=bool(cfg.get("mahalanobis", False)),       # NEW
            metric_rank=cfg.get("metric_rank", None),              # NEW
            concat_heads=bool(cfg.get("concat_heads", True)),
            eps=float(cfg.get("eps", 1e-12)),
        )
