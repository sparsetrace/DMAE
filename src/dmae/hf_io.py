# dmae/hf_io.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import jax.numpy as jnp
from flax.traverse_util import flatten_dict, unflatten_dict
from safetensors.flax import save_file as flax_save_file, load_file as flax_load_file
from huggingface_hub import HfApi, snapshot_download


def split_hf_repo(hf_repo: str) -> Tuple[str, str]:
    """
    Split:
      'username/repo' -> ('username/repo', '')
      'username/repo/subfolder/a' -> ('username/repo', 'subfolder/a')
    """
    parts = hf_repo.strip("/").split("/")
    if len(parts) < 2:
        raise ValueError(
            f"hf_repo must be 'username/repo' or 'username/repo/subfolder', got {hf_repo!r}."
        )
    repo_id = "/".join(parts[:2])
    subpath = "/".join(parts[2:]) if len(parts) > 2 else ""
    return repo_id, subpath


def write_json(path: str | Path, obj: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def read_json(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def save_npz(path: str | Path, arrays: Dict[str, np.ndarray]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def load_npz(path: str | Path) -> Dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=False)
    return {k: data[k] for k in data.files}


def flatten_params_for_safetensors(params: Dict[str, Any]) -> Dict[str, Any]:
    flat = flatten_dict(params, sep="::")
    return {k: jnp.asarray(v) for k, v in flat.items()}


def unflatten_params_from_safetensors(flat_tensors: Dict[str, Any]) -> Dict[str, Any]:
    return unflatten_dict({tuple(k.split("::")): v for k, v in flat_tensors.items()})


def save_params(path: str | Path, params: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    flax_save_file(flatten_params_for_safetensors(params), str(path))


def load_params(path: str | Path) -> Dict[str, Any]:
    flat = flax_load_file(str(path))
    return unflatten_params_from_safetensors(flat)


def write_model_card(path: str | Path, class_name: str, extra: str = "") -> None:
    text = f"""---
library_name: flax
tags:
- dmae
- flax
- jax
---

# {class_name}

Custom DMAE-family model.

{extra}
"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def hf_upload_folder(local_dir: str | Path, hf_repo: str, hf_token: str | None = None, commit_message: str = "Upload model") -> None:
    repo_id, subpath = split_hf_repo(hf_repo)
    api = HfApi(token=hf_token)
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(local_dir),
        path_in_repo=subpath or None,
        commit_message=commit_message,
    )


def hf_download_folder(hf_repo: str, hf_token: str | None = None) -> str:
    repo_id, subpath = split_hf_repo(hf_repo)
    local_repo = snapshot_download(repo_id=repo_id, repo_type="model", token=hf_token)
    return str(Path(local_repo) / subpath) if subpath else str(local_repo)
