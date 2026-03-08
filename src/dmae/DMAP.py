# DMAP.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import jax
import jax.numpy as jnp
from flax.core import freeze, unfreeze

from .eager_map import eager_dmap
from .blocks import dmap
from .hf_io import (
    write_json, read_json, save_npz, load_npz,
    save_params, load_params, write_model_card,
    hf_upload_folder, hf_download_folder,
)


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


def _resolve_beta_alias(
    β,
    beta,
    default=None,
):
    if β is not None and beta is not None:
        a = np.asarray(β if not np.isscalar(β) else [β], dtype=np.float64)
        b = np.asarray(beta if not np.isscalar(beta) else [beta], dtype=np.float64)
        if a.shape != b.shape or not np.allclose(a, b):
            raise ValueError("Got both β and beta with different values; provide only one.")
    if β is None and beta is None:
        return default
    return beta if β is None else β


@dataclass
class DMAPInit:
    R_iX: np.ndarray                  # (N, D)
    β: np.ndarray                     # (h,)
    q: np.ndarray                     # (h, N)
    W: np.ndarray                     # (h, N, d)
    λ_x: np.ndarray                   # (h, d)
    ψ_ix: np.ndarray                  # (h, N, d)
    L: Optional[np.ndarray] = None    # (h, D, r)


class DMAP:
    """
    Dense eager-initialized wrapper around the trainable Flax `dmap` encoder.

    Example
    -------
    encoder = DMAP(R_iX, d=2)
    Z = encoder(R_iX)   # shape (N, h, d)
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
        t: int = 1,
        mahalanobis: bool = False,
        Q: np.ndarray | None = None,
        metric_rank: int | None = None,
        metric_init: str = "euclidean",
        metric_mix: float = 0.1,
        zero_diag: bool = True,
        k_eigs: int | None = None,
        which: str = "LA",
        eps: float = 1e-12,
        seed: int = 0,
        dtype: Any = jnp.float32,
        param_dtype: Any = jnp.float32,
    ):
        R_iX = np.asarray(R_iX, dtype=np.float32)
        if R_iX.ndim != 2:
            raise ValueError(f"R_iX must have shape (N, D), got {R_iX.shape}.")

        N, D = R_iX.shape
        if d <= 0:
            raise ValueError(f"`d` must be positive, got {d}.")
        if h <= 0:
            raise ValueError(f"`h` must be positive, got {h}.")

        α = _resolve_scalar_alias("α", α, "alpha", alpha, 1.0)
        β = _resolve_beta_alias(β, beta, None)

        # Dense eager solve
        eager = eager_dmap(
            R_iX=R_iX,
            head_dim=int(d),
            h=int(h),
            alpha=float(α),
            t=int(t),
            beta=β,
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
        )

        params_eager = eager["params"]

        β_fit = np.asarray(eager["beta_heads"], dtype=np.float32)               # (h,)
        q_fit = np.asarray(params_eager["cl_softmax"]["q"], dtype=np.float32)   # (h, N)
        W_fit = np.asarray(params_eager["W"], dtype=np.float32)                 # (h, N, d)
        λ_fit = np.asarray(eager["spectral"]["lambdas"], dtype=np.float32)      # (h, d)
        ψ_fit = np.asarray(eager["spectral"]["psi"], dtype=np.float32)          # (h, N, d)

        if q_fit.shape != (h, N):
            raise ValueError(f"Expected q shape {(h, N)}, got {q_fit.shape}.")
        if W_fit.shape != (h, N, d):
            raise ValueError(f"Expected W shape {(h, N, d)}, got {W_fit.shape}.")
        if λ_fit.shape != (h, d):
            raise ValueError(f"Expected lambdas shape {(h, d)}, got {λ_fit.shape}.")
        if ψ_fit.shape != (h, N, d):
            raise ValueError(f"Expected psi shape {(h, N, d)}, got {ψ_fit.shape}.")

        L_fit = None
        metric_rank_fit = None
        if "L" in params_eager["SMD"]:
            L_fit = np.asarray(params_eager["SMD"]["L"], dtype=np.float32)
            metric_rank_fit = int(L_fit.shape[-1])

        self.init_data = DMAPInit(
            R_iX=np.asarray(params_eager["SMD"]["R_iX"], dtype=np.float32),
            β=β_fit,
            q=q_fit,
            W=W_fit,
            λ_x=λ_fit,
            ψ_ix=ψ_fit,
            L=L_fit,
        )

        beta_module: BetaSpec
        if h == 1:
            beta_module = float(β_fit[0])
        else:
            beta_module = tuple(float(b) for b in β_fit)

        self.module = dmap(
            d=int(d),
            N=int(N),
            h=int(h),
            α=float(α),
            β=beta_module,
            metric_rank=metric_rank_fit,
            eps=float(eps),
            dtype=dtype,
            param_dtype=param_dtype,
        )

        dummy_x = jnp.zeros((1, D), dtype=dtype)
        variables = self.module.init(jax.random.PRNGKey(seed), dummy_x)
        vars_mut = unfreeze(variables)
        params = vars_mut["params"]

        # Patch dense eager solution into abstract Flax encoder
        params["rbf"]["R_iX"] = jnp.asarray(self.init_data.R_iX, dtype=param_dtype)
        params["norm"]["q"] = jnp.asarray(self.init_data.q, dtype=param_dtype)
        params["W"] = jnp.asarray(self.init_data.W, dtype=param_dtype)

        if self.init_data.L is not None:
            params["rbf"]["L"] = jnp.asarray(self.init_data.L, dtype=param_dtype)

        self.variables = freeze(vars_mut)

        self.config = {
            "R_iX_shape": tuple(R_iX.shape),
            "d": int(d),
            "h": int(h),
            "α": float(α),
            "β": beta_module,
            "t": int(t),
            "mahalanobis": bool(mahalanobis),
            "metric_rank": metric_rank_fit,
            "metric_init": str(metric_init),
            "metric_mix": float(metric_mix),
            "zero_diag": bool(zero_diag),
            "k_eigs": None if k_eigs is None else int(k_eigs),
            "which": str(which),
            "eps": float(eps),
            "seed": int(seed),
        }

    def __call__(self, x: np.ndarray | jnp.ndarray) -> jnp.ndarray:
        x = jnp.asarray(x, dtype=self.module.dtype)
        return self.module.apply(self.variables, x)

    def apply(self, x: np.ndarray | jnp.ndarray, variables: Optional[Dict[str, Any]] = None):
        x = jnp.asarray(x, dtype=self.module.dtype)
        vars_use = self.variables if variables is None else variables
        return self.module.apply(vars_use, x)

    @property
    def params(self):
        return self.variables["params"]

    @property
    def latent_train(self) -> np.ndarray:
        """
        Dense eager training coordinates R_ix = λ^t ψ are not stored explicitly here,
        but for the current eager_dmap implementation:
            W = ψ * λ^(t-1)
        so when t=1, W = ψ.
        If you need exact R_ix for general t, store it directly in eager_dmap.
        """
        return np.asarray(self.init_data.W)

    @classmethod
    def from_eager(cls, *args, **kwargs) -> "DMAP":
        return cls(*args, **kwargs)



    ############## SAVE
    def save_pretrained(self, local_dir: str) -> None:
        local_dir = Path(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)
    
        write_json(local_dir / "config.json", {
            "class_name": "DMAP",
            "config": self.config,
        })
    
        save_params(local_dir / "params.safetensors", dict(self.variables["params"]))
    
        arrays = {
            "R_iX": np.asarray(self.init_data.R_iX),
            "β": np.asarray(self.init_data.β),
            "q": np.asarray(self.init_data.q),
            "W": np.asarray(self.init_data.W),
            "λ_x": np.asarray(self.init_data.λ_x),
            "ψ_ix": np.asarray(self.init_data.ψ_ix),
        }
        if self.init_data.L is not None:
            arrays["L"] = np.asarray(self.init_data.L)
    
        save_npz(local_dir / "init_data.npz", arrays)
        write_model_card(local_dir / "README.md", class_name="DMAP")
    
    
    @classmethod
    def from_pretrained(cls, local_dir: str) -> "DMAP":
        local_dir = Path(local_dir)
    
        meta = read_json(local_dir / "config.json")
        cfg = meta["config"]
        init_npz = load_npz(local_dir / "init_data.npz")
        params = load_params(local_dir / "params.safetensors")
    
        beta_module = cfg["β"]
        if isinstance(beta_module, list):
            beta_module = tuple(float(b) for b in beta_module)
    
        module = dmap(
            d=int(cfg["d"]),
            N=int(cfg["R_iX_shape"][0]),
            h=int(cfg["h"]),
            α=float(cfg["α"]),
            β=beta_module,
            metric_rank=cfg.get("metric_rank", None),
            eps=float(cfg["eps"]),
            dtype=jnp.float32,
            param_dtype=jnp.float32,
        )
    
        D = int(cfg["R_iX_shape"][1])
        dummy_x = jnp.zeros((1, D), dtype=jnp.float32)
        variables = module.init(jax.random.PRNGKey(int(cfg["seed"])), dummy_x)
        vars_mut = unfreeze(variables)
        vars_mut["params"] = params
        variables = freeze(vars_mut)
    
        init_data = DMAPInit(
            R_iX=init_npz["R_iX"],
            β=init_npz["β"],
            q=init_npz["q"],
            W=init_npz["W"],
            λ_x=init_npz["λ_x"],
            ψ_ix=init_npz["ψ_ix"],
            L=init_npz["L"] if "L" in init_npz else None,
        )
    
        obj = cls.__new__(cls)
        obj.module = module
        obj.variables = variables
        obj.init_data = init_data
        obj.config = cfg
        return obj
    
    
    def hf_save(self, hf_repo: str, hf_token: str | None = None) -> None:
        with tempfile.TemporaryDirectory() as td:
            self.save_pretrained(td)
            hf_upload_folder(td, hf_repo, hf_token, commit_message="Upload DMAP")
    
    
    @classmethod
    def hf_load(cls, hf_repo: str, hf_token: str | None = None) -> "DMAP":
        local_dir = hf_download_folder(hf_repo, hf_token)
        return cls.from_pretrained(local_dir)
