# src/dmae/DMAP.py
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
from flax.core import freeze, unfreeze
from flax.serialization import msgpack_serialize, msgpack_restore

from .blocks import dmap as DMAPFlax
from .eager_map import eager_dmap_sparse  # <- your simplified sparse eager init


BetaSpec = Union[float, Tuple[float, ...]]


@dataclass
class DMAPConfig:
    # architecture
    d: int = 16
    h: int = 1
    α: float = 1.0
    eps: float = 1e-12

    # metric
    mahalanobis: bool = False
    metric_rank: Optional[int] = None  # None => full-rank (D)

    # eager init (sparse)
    t: int = 1
    k_nn: int = 64
    q_block: int = 1024
    r_block: int = 8192
    k_eigs: Optional[int] = None  # None => head_dim + 1 (includes trivial)


class DMAP:
    """
    Wrapper around the Flax DMAP encoder in blocks.py.

    If params are not provided, this runs sparse eager diffusion-map init:
      - anchors R_iX are set to the provided R_iX
      - q (Coifman–Lafon) is set from the kernel row-sums
      - W is filled from diffusion coordinates of the anchors
        (packed into per-head slices of size head_dim inside the last dim d)
    """

    CONFIG_NAME = "config.json"
    WEIGHTS_NAME = "flax_model.msgpack"

    def __init__(
        self,
        R_iX: Any,
        *,
        config: DMAPConfig | None = None,
        β: float | Sequence[float] | None = None,
        params: Optional[Dict[str, Any]] = None,
        seed: int = 0,
    ):
        self.cfg = config or DMAPConfig()

        R_iX = jnp.asarray(R_iX, dtype=jnp.float32)
        if R_iX.ndim != 2:
            raise ValueError(f"R_iX must be (N,D). Got shape {R_iX.shape}.")
        self.N, self.D = int(R_iX.shape[0]), int(R_iX.shape[1])

        if self.cfg.d % self.cfg.h != 0:
            raise ValueError(f"Need d % h == 0, got d={self.cfg.d}, h={self.cfg.h}.")
        self.head_dim = self.cfg.d // self.cfg.h

        β_heads = self._normalize_beta(β, self.cfg.h)  # (h,)

        # Flax module (β is a module field, not a param)
        self.model = DMAPFlax(
            d=int(self.cfg.d),
            N=int(self.N),
            h=int(self.cfg.h),
            α=float(self.cfg.α),
            β=tuple(float(b) for b in β_heads.tolist()),
            mahalanobis=bool(self.cfg.mahalanobis),
            metric_rank=self.cfg.metric_rank,
            eps=float(self.cfg.eps),
        )

        if params is None:
            # --- initialize metric factor L if needed and not provided ---
            L0 = None
            if self.cfg.mahalanobis:
                r = self.D if self.cfg.metric_rank is None else int(self.cfg.metric_rank)
                if r <= 0 or r > self.D:
                    raise ValueError(f"metric_rank must be in [1,D], got {r} for D={self.D}.")
                I = jnp.eye(self.D, dtype=jnp.float32)[:, :r]              # (D,r)
                L0 = jnp.broadcast_to(I[None, :, :], (self.cfg.h, self.D, r))  # (h,D,r)

            # choose eig count including trivial mode
            k_total = self.cfg.k_eigs
            if k_total is None:
                k_total = self.head_dim + 1

            # force eager to use exactly h heads: pass β as length-h vector
            init = eager_dmap_sparse(
                R_iX=jax.device_get(R_iX),
                α=float(self.cfg.α),
                t=int(self.cfg.t),
                β=jax.device_get(β_heads).tolist(),
                L=None if L0 is None else jax.device_get(L0),
                k_nn=int(self.cfg.k_nn),
                q_block=int(self.cfg.q_block),
                r_block=int(self.cfg.r_block),
                k_eigs=int(k_total),
                ϵ=float(self.cfg.eps),
                seed=int(seed),
            )

            # init outputs: q (h,N), W_spec (h,N,head_dim)
            q = init["q"]                 # (h,N)
            W_spec = init["W"]            # (h,N,head_dim) because k_eigs=head_dim+1

            # pack W_spec into (h,N,d) by placing each head in its slice
            W_full = np.zeros((self.cfg.h, self.N, self.cfg.d), dtype=np.float32)
            for hh in range(self.cfg.h):
                s = hh * self.head_dim
                e = (hh + 1) * self.head_dim
                W_full[hh, :, s:e] = W_spec[hh]

            params = {
                "SMD": {"R_iX": jax.device_get(R_iX).astype("float32")},
                "W": W_full,
            }

            if self.cfg.mahalanobis:
                params["SMD"]["L"] = jax.device_get(L0).astype("float32")

            # Only include q if α != 0 (Softmax won’t create/use it otherwise)
            if float(self.cfg.α) != 0.0:
                params["cl_softmax"] = {"q": np.asarray(q, dtype=np.float32)}

        self.params = freeze(self._to_jax_tree(params))
        self._apply_jit = jax.jit(lambda p, x: self.model.apply({"params": p}, x))

    def __call__(self, R_aX: Any) -> jnp.ndarray:
        x = jnp.asarray(R_aX, dtype=jnp.float32)
        return self._apply_jit(self.params, x)

    # ---------------------------
    # Save / Load (local)
    # ---------------------------

    def save_pretrained(self, save_directory: str | os.PathLike) -> None:
        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)

        payload = {
            "wrapper_config": asdict(self.cfg),
            "N": self.N,
            "D": self.D,
            "flax_module_config": self.model.config,
        }
        (save_dir / self.CONFIG_NAME).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False)
        )

        params_py = unfreeze(self.params)
        (save_dir / self.WEIGHTS_NAME).write_bytes(msgpack_serialize(params_py))

    @classmethod
    def from_pretrained(
        cls,
        path_or_repo_id: str,
        *,
        token: Optional[str] = None,
        revision: Optional[str] = None,
        R_iX: Any = None,
        seed: int = 0,
        β: float | Sequence[float] | None = None,
    ) -> "DMAP":
        path = Path(path_or_repo_id)
        if path.exists() and path.is_dir():
            local_dir = path
        else:
            local_dir = Path(_hf_snapshot_download(path_or_repo_id, token=token, revision=revision))

        cfg_path = local_dir / cls.CONFIG_NAME
        if not cfg_path.exists():
            raise FileNotFoundError(f"Missing {cls.CONFIG_NAME} in {local_dir}")

        payload = json.loads(cfg_path.read_text())
        wrapper_cfg = DMAPConfig(**payload.get("wrapper_config", {}))
        N = int(payload["N"])
        D = int(payload["D"])

        weights_path = local_dir / cls.WEIGHTS_NAME
        if weights_path.exists():
            params = msgpack_restore(weights_path.read_bytes())
            if R_iX is None:
                R_iX = jnp.zeros((N, D), dtype=jnp.float32)
            return cls(R_iX, config=wrapper_cfg, params=params, seed=seed, β=β)

        if R_iX is None:
            raise FileNotFoundError(
                f"Missing {cls.WEIGHTS_NAME} in {local_dir}. Provide R_iX to recompute eagerly."
            )
        return cls(R_iX, config=wrapper_cfg, params=None, seed=seed, β=β)

    # ---------------------------
    # HF Hub helpers
    # ---------------------------

    def push_to_hub(
        self,
        hf_repo_id: str,
        *,
        token: Optional[str] = None,
        private: bool = False,
        commit_message: str = "Add DMAP Flax model",
    ) -> None:
        token = token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")

        with tempfile.TemporaryDirectory() as tmp:
            self.save_pretrained(tmp)
            _hf_upload_folder(
                repo_id=hf_repo_id,
                folder_path=tmp,
                token=token,
                private=private,
                commit_message=commit_message,
            )

    # ---------------------------
    # internal utils
    # ---------------------------

    @staticmethod
    def _normalize_beta(beta: float | Sequence[float] | None, h: int) -> jnp.ndarray:
        if beta is None:
            return jnp.ones((h,), dtype=jnp.float32)
        if isinstance(beta, (list, tuple)):
            if len(beta) != h:
                raise ValueError(f"β list/tuple must have length h={h}, got {len(beta)}.")
            return jnp.asarray([float(b) for b in beta], dtype=jnp.float32)
        return jnp.full((h,), float(beta), dtype=jnp.float32)

    @staticmethod
    def _to_jax_tree(tree: Any) -> Any:
        if isinstance(tree, dict):
            return {k: DMAP._to_jax_tree(v) for k, v in tree.items()}
        if isinstance(tree, (list, tuple)):
            return type(tree)(DMAP._to_jax_tree(v) for v in tree)
        try:
            import numpy as np
            if isinstance(tree, np.ndarray):
                return jnp.asarray(tree)
        except Exception:
            pass
        return tree


def _hf_snapshot_download(repo_id: str, *, token: Optional[str], revision: Optional[str]) -> str:
    from huggingface_hub import snapshot_download
    return snapshot_download(repo_id=repo_id, token=token, revision=revision)


def _hf_upload_folder(
    *,
    repo_id: str,
    folder_path: str,
    token: Optional[str],
    private: bool,
    commit_message: str,
) -> None:
    from huggingface_hub import create_repo, upload_folder
    create_repo(repo_id, token=token, private=private, exist_ok=True)
    upload_folder(
        repo_id=repo_id,
        folder_path=folder_path,
        token=token,
        commit_message=commit_message,
    )
