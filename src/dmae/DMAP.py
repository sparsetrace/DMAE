# DMAP.py
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

from ._dmap import dmap as DMAPFlax
from .eager_dmap import eager_dmap


BetaSpec = Union[float, Tuple[float, ...]]


@dataclass
class DMAPConfig:
    # architecture
    d: int = 16
    h: int = 1
    alpha: float = 1.0
    eps: float = 1e-12

    # metric
    mahalanobis: bool = False
    metric_rank: Optional[int] = None  # None => full-rank (D) inside SMD

    # eager init / spectral
    t: int = 1
    beta_mode: str = "per_head"    # "shared" | "per_head"
    zero_diag: bool = True
    k_eigs: Optional[int] = None   # None => head_dim + 1
    which: str = "LA"              # eigsh mode

    # optional metric init (only used if mahalanobis and no weights)
    metric_init: str = "euclidean"     # "euclidean" | "wishart_mix"
    metric_mix: float = 0.1


class DMAP:
    """
    Wrapper around the Flax DMAP encoder.

    - "Training" in __init__ means: eager spectral init (O(h*N^2) kernel + Lanczos).
      You can add SGD fine-tuning later, but this gives a strong geometric starting point.
    - Inference in __call__ uses jitted Flax apply.
    - save/load like a normal Flax model: config.json + flax_model.msgpack,
      with optional push_to_hub / from_pretrained.
    """

    CONFIG_NAME = "config.json"
    WEIGHTS_NAME = "flax_model.msgpack"

    def __init__(
        self,
        R_iX: Any,
        *,
        config: DMAPConfig | None = None,
        beta: float | Sequence[float] | None = None,
        params: Optional[Dict[str, Any]] = None,
        seed: int = 0,
    ):
        """
        If `params` is provided, uses it directly.
        Otherwise computes eager initialization from `R_iX` using eager_dmap.py.
        """
        self.cfg = config or DMAPConfig()

        R_iX = jnp.asarray(R_iX, dtype=jnp.float32)
        if R_iX.ndim != 2:
            raise ValueError(f"R_iX must be (N,D). Got shape {R_iX.shape}.")
        self.N, self.D = int(R_iX.shape[0]), int(R_iX.shape[1])

        if self.cfg.d % self.cfg.h != 0:
            raise ValueError(f"Need d % h == 0, got d={self.cfg.d}, h={self.cfg.h}.")
        self.head_dim = self.cfg.d // self.cfg.h

        # Decide betas (per head) for the Flax module field β
        beta_heads = self._normalize_beta(beta, self.cfg.h)

        # Flax module (β is stored in the module fields, not params)
        self.model = DMAPFlax(
            d=self.cfg.d,
            N=self.N,
            h=self.cfg.h,
            α=float(self.cfg.alpha),
            β=tuple(float(b) for b in beta_heads),
            mahalanobis=bool(self.cfg.mahalanobis),
            metric_rank=self.cfg.metric_rank,
            eps=float(self.cfg.eps),
        )

        if params is None:
            # Eager init: returns params compatible with your Flax module param tree
            init = eager_dmap(
                R_iX=jax.device_get(R_iX),     # eager code uses numpy/scipy
                head_dim=self.head_dim,
                h=self.cfg.h,
                alpha=self.cfg.alpha,
                t=self.cfg.t,
                beta=None if beta is None else float(beta_heads[0]),
                beta_mode=self.cfg.beta_mode,
                mahalanobis=self.cfg.mahalanobis,
                metric_rank=self.cfg.metric_rank,
                metric_init=self.cfg.metric_init,
                metric_mix=self.cfg.metric_mix,
                zero_diag=self.cfg.zero_diag,
                k_eigs=self.cfg.k_eigs,
                which=self.cfg.which,
                eps=self.cfg.eps,
                seed=seed,
            )

            # Use the eager betas if user didn't specify beta
            if beta is None:
                beta_heads = jax.device_get(init["beta_heads"]).tolist()
                # rebuild model with these betas
                self.model = DMAPFlax(
                    d=self.cfg.d,
                    N=self.N,
                    h=self.cfg.h,
                    α=float(self.cfg.alpha),
                    β=tuple(float(b) for b in beta_heads),
                    mahalanobis=bool(self.cfg.mahalanobis),
                    metric_rank=self.cfg.metric_rank,
                    eps=float(self.cfg.eps),
                )

            params = init["params"]

        # Store params as FrozenDict (Flax-friendly)
        self.params = freeze(self._to_jax_tree(params))

        # JIT apply
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

        # config.json includes wrapper config + shape info + module config (for safety)
        payload = {
            "wrapper_config": asdict(self.cfg),
            "N": self.N,
            "D": self.D,
            "flax_module_config": self.model.config,  # relies on your _dmap.py dmap.config
        }
        (save_dir / self.CONFIG_NAME).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False)
        )

        # flax_model.msgpack: serialize params pytree
        params_py = unfreeze(self.params)
        weights_bytes = msgpack_serialize(params_py)  # :contentReference[oaicite:2]{index=2}
        (save_dir / self.WEIGHTS_NAME).write_bytes(weights_bytes)

    @classmethod
    def from_pretrained(
        cls,
        path_or_repo_id: str,
        *,
        token: Optional[str] = None,
        revision: Optional[str] = None,
        R_iX: Any = None,
        seed: int = 0,
    ) -> "DMAP":
        """
        Load from local folder or HF repo id.

        If weights are missing but config exists, and you provide R_iX,
        it will run eager init to reconstruct weights.
        """
        path = Path(path_or_repo_id)
        if path.exists() and path.is_dir():
            local_dir = path
        else:
            # HF download
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
            params = msgpack_restore(weights_path.read_bytes())  # :contentReference[oaicite:3]{index=3}
            # If R_iX not provided, anchors come from params; good.
            if R_iX is None:
                # fabricate R_iX just to satisfy constructor shape checks (won't be used if params provided)
                R_iX = jnp.zeros((N, D), dtype=jnp.float32)
            return cls(R_iX, config=wrapper_cfg, params=params, seed=seed)

        # No weights: recompute eagerly if R_iX provided
        if R_iX is None:
            raise FileNotFoundError(
                f"Missing {cls.WEIGHTS_NAME} in {local_dir}. Provide R_iX to recompute eagerly."
            )
        return cls(R_iX, config=wrapper_cfg, params=None, seed=seed)

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
        """
        Save to a temp folder then upload to Hugging Face Hub.
        Uses huggingface_hub upload_folder / create_repo. :contentReference[oaicite:4]{index=4}
        """
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
            # placeholder; eager init will likely override if beta_mode uses median heuristic
            return jnp.ones((h,), dtype=jnp.float32)
        if isinstance(beta, (list, tuple)):
            if len(beta) != h:
                raise ValueError(f"beta list/tuple must have length h={h}, got {len(beta)}.")
            return jnp.asarray([float(b) for b in beta], dtype=jnp.float32)
        return jnp.full((h,), float(beta), dtype=jnp.float32)

    @staticmethod
    def _to_jax_tree(tree: Any) -> Any:
        # convert numpy leaves to jax arrays recursively
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
    from huggingface_hub import snapshot_download  # :contentReference[oaicite:5]{index=5}
    return snapshot_download(repo_id=repo_id, token=token, revision=revision)


def _hf_upload_folder(
    *,
    repo_id: str,
    folder_path: str,
    token: Optional[str],
    private: bool,
    commit_message: str,
) -> None:
    from huggingface_hub import create_repo, upload_folder  # :contentReference[oaicite:6]{index=6}
    create_repo(repo_id, token=token, private=private, exist_ok=True)
    upload_folder(
        repo_id=repo_id,
        folder_path=folder_path,
        token=token,
        commit_message=commit_message,
    )
