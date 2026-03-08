# dmae/DMAE.py
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import jax
import jax.numpy as jnp
from flax.core import freeze, unfreeze

from .DMAP import DMAP
from .GPLM import GPLM
from .DMAP import DMAPInit
from .GPLM import GPLMInit
from .blocks import dmap, gplm
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

    Shared state:
        - R_iX : ambient training anchors
        - R_ix : latent training anchors (computed once from encoder(R_iX))

    Public API:
        Model = DMAE(R_iX, d=32)

        Q_ix = Model.encode(R_iX)        # ambient -> latent
        Q_iX = Model.decode(Q_ix)        # latent  -> ambient
        Q_iX = Model.reconstruct(R_iX)   # ambient -> latent -> ambient

    Dispatch API:
        Model(X)
          - if last dim == ambient D : encode
          - else if last dim == latent d : decode
          - if both happen to match, prefer encode
    """

    def __init__(
        self,
        R_iX: np.ndarray,
        d: int = 32,
        *,
        h: int = 1,

        α: float | None = None,
        β: float | Tuple[float, ...] | np.ndarray | None = None,
        alpha: float | None = None,
        beta: float | Tuple[float, ...] | np.ndarray | None = None,

        # encoder
        t: int = 1,
        mahalanobis: bool = False,
        Q: np.ndarray | None = None,
        metric_rank: int | None = None,
        metric_init: str = "euclidean",
        metric_mix: float = 0.1,
        zero_diag: bool = True,
        k_eigs: int | None = None,
        which: str = "LA",

        # decoder
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

        # Shared latent bank in external convention (N, h, d)
        self.R_ix = np.asarray(self.encoder(R_iX), dtype=np.float32)

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

    # -------------------------
    # Core API
    # -------------------------

    def encode(self, X: np.ndarray | jnp.ndarray) -> jnp.ndarray:
        return self.encoder(X)

    def decode(self, Z: np.ndarray | jnp.ndarray) -> jnp.ndarray:
        return self.decoder(Z)

    def reconstruct(self, X: np.ndarray | jnp.ndarray) -> jnp.ndarray:
        return self.decode(self.encode(X))

    def __call__(self, X_or_Z: np.ndarray | jnp.ndarray) -> jnp.ndarray:
        arr = jnp.asarray(X_or_Z)
        if arr.ndim < 2:
            raise ValueError(f"DMAE input must have at least 2 dims, got shape {arr.shape}.")

        last_dim = int(arr.shape[-1])

        # Prefer encode on ambiguity
        if last_dim == self.D:
            return self.encode(arr)

        if last_dim == self.config["d"]:
            return self.decode(arr)

        raise ValueError(
            f"Could not infer whether input is ambient or latent from shape {arr.shape}. "
            f"Expected last dim {self.D} (ambient) or {self.config['d']} (latent)."
        )

    @property
    def params(self) -> Dict[str, Any]:
        return {
            "encoder": self.encoder.params,
            "decoder": self.decoder.params,
        }

    # -------------------------
    # Local save / load
    # -------------------------

    def save_pretrained(self, local_dir: str | Path) -> None:
        """
        Save DMAE without duplicating shared R_iX / R_ix.
        """
        local_dir = Path(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)

        write_json(local_dir / "config.json", {
            "class_name": "DMAE",
            "config": self.config,
        })

        # Shared geometry/state saved ONCE
        save_npz(local_dir / "shared.npz", {
            "R_iX": np.asarray(self.R_iX),
            "R_ix": np.asarray(self.R_ix),  # external convention (N,h,d)
        })

        # Encoder-only
        write_json(local_dir / "encoder_config.json", self.encoder.config)
        save_params(local_dir / "encoder_params.safetensors", dict(self.encoder.variables["params"]))

        # Save only encoder-specific init arrays not shared
        encoder_arrays = {
            "β": np.asarray(self.encoder.init_data.β),
            "q": np.asarray(self.encoder.init_data.q),
            "W": np.asarray(self.encoder.init_data.W),
            "λ_x": np.asarray(self.encoder.init_data.λ_x),
            "ψ_ix": np.asarray(self.encoder.init_data.ψ_ix),
        }
        if self.encoder.init_data.L is not None:
            encoder_arrays["L"] = np.asarray(self.encoder.init_data.L)
        save_npz(local_dir / "encoder_init.npz", encoder_arrays)

        # Decoder-only
        write_json(local_dir / "decoder_config.json", self.decoder.config)
        save_params(local_dir / "decoder_params.safetensors", dict(self.decoder.variables["params"]))

        decoder_arrays = {
            "β": np.asarray(self.decoder.init_data.β),
            "W": np.asarray(self.decoder.init_data.W),
        }
        if self.decoder.init_data.L is not None:
            decoder_arrays["L"] = np.asarray(self.decoder.init_data.L)
        if self.decoder.init_data.W_O is not None:
            decoder_arrays["W_O"] = np.asarray(self.decoder.init_data.W_O)
        if self.decoder.init_data.b_O is not None:
            decoder_arrays["b_O"] = np.asarray(self.decoder.init_data.b_O)
        save_npz(local_dir / "decoder_init.npz", decoder_arrays)

        write_model_card(local_dir / "README.md", class_name="DMAE")

    @classmethod
    def from_pretrained(cls, local_dir: str | Path) -> "DMAE":
        """
        Load DMAE from local directory, reconstructing encoder and decoder while
        reusing shared R_iX / R_ix stored once at the top level.
        """
        local_dir = Path(local_dir)

        meta = read_json(local_dir / "config.json")
        shared = load_npz(local_dir / "shared.npz")
        enc_cfg = read_json(local_dir / "encoder_config.json")
        dec_cfg = read_json(local_dir / "decoder_config.json")
        enc_params = load_params(local_dir / "encoder_params.safetensors")
        dec_params = load_params(local_dir / "decoder_params.safetensors")
        enc_init = load_npz(local_dir / "encoder_init.npz")
        dec_init = load_npz(local_dir / "decoder_init.npz")

        R_iX = shared["R_iX"].astype(np.float32)
        R_ix = shared["R_ix"].astype(np.float32)   # external convention (N,h,d)

        # -------------------------
        # Rebuild encoder
        # -------------------------
        beta_enc = enc_cfg["β"]
        if isinstance(beta_enc, list):
            beta_enc = tuple(float(b) for b in beta_enc)

        enc_module = dmap(
            d=int(enc_cfg["d"]),
            N=int(enc_cfg["R_iX_shape"][0]),
            h=int(enc_cfg["h"]),
            α=float(enc_cfg["α"]),
            β=beta_enc,
            metric_rank=enc_cfg.get("metric_rank", None),
            eps=float(enc_cfg["eps"]),
            dtype=jnp.float32,
            param_dtype=jnp.float32,
        )

        D = int(enc_cfg["R_iX_shape"][1])
        dummy_x = jnp.zeros((1, D), dtype=jnp.float32)
        enc_variables = enc_module.init(jax.random.PRNGKey(int(enc_cfg["seed"])), dummy_x)
        enc_vars_mut = unfreeze(enc_variables)
        enc_vars_mut["params"] = enc_params
        enc_variables = freeze(enc_vars_mut)

        enc_init_data = DMAPInit(
            R_iX=R_iX,
            β=enc_init["β"],
            q=enc_init["q"],
            W=enc_init["W"],
            λ_x=enc_init["λ_x"],
            ψ_ix=enc_init["ψ_ix"],
            L=enc_init["L"] if "L" in enc_init else None,
        )

        encoder = DMAP.__new__(DMAP)
        encoder.module = enc_module
        encoder.variables = enc_variables
        encoder.init_data = enc_init_data
        encoder.config = enc_cfg

        # -------------------------
        # Rebuild decoder
        # -------------------------
        beta_dec = dec_cfg["β"]
        if isinstance(beta_dec, list):
            beta_dec = tuple(float(b) for b in beta_dec)

        dec_module = gplm(
            D_head=int(dec_cfg["D_head"]),
            N=int(dec_cfg["R_iX_shape"][0]),
            h=int(dec_cfg["h"]),
            β=beta_dec,
            metric_rank=dec_cfg.get("metric_rank", None),
            use_W_O=bool(dec_cfg["use_W_O"]),
            D_out=None if not dec_cfg["use_W_O"] else int(dec_cfg["D_out"]),
            eps=float(dec_cfg["eps"]),
            dtype=jnp.float32,
            param_dtype=jnp.float32,
        )

        h = int(dec_cfg["h"])
        d = int(dec_cfg["d"])
        dummy_z = jnp.zeros((1, h, d), dtype=jnp.float32)
        dec_variables = dec_module.init(jax.random.PRNGKey(int(dec_cfg["seed"])), dummy_z)
        dec_vars_mut = unfreeze(dec_variables)
        dec_vars_mut["params"] = dec_params
        dec_variables = freeze(dec_vars_mut)

        # GPLM internal init stores latent anchors as (h,N,d)
        R_ix_hNd = np.transpose(R_ix, (1, 0, 2))

        dec_init_data = GPLMInit(
            R_ix_hNd=R_ix_hNd,
            R_iX=R_iX,
            β=dec_init["β"],
            W=dec_init["W"],
            L=dec_init["L"] if "L" in dec_init else None,
            W_O=dec_init["W_O"] if "W_O" in dec_init else None,
            b_O=dec_init["b_O"] if "b_O" in dec_init else None,
        )

        decoder = GPLM.__new__(GPLM)
        decoder.module = dec_module
        decoder.variables = dec_variables
        decoder.init_data = dec_init_data
        decoder.config = dec_cfg

        # -------------------------
        # Rebuild wrapper
        # -------------------------
        obj = cls.__new__(cls)
        obj.encoder = encoder
        obj.decoder = decoder
        obj.R_iX = R_iX
        obj.R_ix = R_ix
        obj.N, obj.D = R_iX.shape
        obj.config = meta["config"]
        return obj

    # -------------------------
    # Hugging Face save / load
    # -------------------------

    def hf_save(self, hf_repo: str, hf_token: str | None = None) -> None:
        with tempfile.TemporaryDirectory() as td:
            self.save_pretrained(td)
            hf_upload_folder(td, hf_repo, hf_token, commit_message="Upload DMAE")

    @classmethod
    def hf_load(cls, hf_repo: str, hf_token: str | None = None) -> "DMAE":
        local_dir = hf_download_folder(hf_repo, hf_token)
        return cls.from_pretrained(local_dir)
