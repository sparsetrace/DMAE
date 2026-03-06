# cloud.py
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional


def load_token(name: str, *, required: bool = True, set_env: bool = True) -> Optional[str]:
    """
    Load a secret token from:
      1) os.environ[name]
      2) Colab Secrets (google.colab.userdata.get)

    If required=False, returns None if not found.
    If set_env=True, stores it back into os.environ[name] when found via Colab.
    """
    v = os.environ.get(name)
    if v:
        return v

    try:
        from google.colab import userdata  # type: ignore
        v = userdata.get(name)
        if v:
            if set_env:
                os.environ[name] = v
            return v
    except Exception:
        pass

    if required:
        raise RuntimeError(f"Missing token: set {name} in env or Colab Secrets.")
    return None


# ----------------------------
# Hugging Face
# ----------------------------

def ensure_hf_token(name: str = "HF_TOKEN", *, required: bool = True) -> Optional[str]:
    """Load HF token into env (if present) and return it."""
    return load_token(name, required=required)


# ----------------------------
# Modal
# ----------------------------

def ensure_modal_tokens(
    *,
    token_id_name: str = "MODAL_TOKEN_ID",
    token_secret_name: str = "MODAL_TOKEN_SECRET",
    modal_bin: str = "modal",
    required: bool = False,
    quiet: bool = True,
) -> bool:
    """
    Ensure Modal CLI is configured using tokens from env or Colab Secrets.

    Looks up:
      MODAL_TOKEN_ID, MODAL_TOKEN_SECRET (customizable)

    Runs:
      modal token set --token-id ... --token-secret ...

    Returns True if configured, False if skipped (when required=False and tokens missing).
    """
    token_id = load_token(token_id_name, required=required)
    token_secret = load_token(token_secret_name, required=required)

    if not token_id or not token_secret:
        return False

    if not shutil.which(modal_bin):
        raise RuntimeError(
            "Modal CLI not found on PATH. Install with `pip install modal` and ensure `modal` is available."
        )

    cmd = [modal_bin, "token", "set", "--token-id", token_id, "--token-secret", token_secret]
    subprocess.run(
        cmd,
        check=True,
        stdout=subprocess.DEVNULL if quiet else None,
        stderr=subprocess.DEVNULL if quiet else None,
    )
    if not quiet:
        print("✅ Modal tokens configured.")
    return True


# ----------------------------
# GitHub
# ----------------------------

def ensure_github_netrc(
    *,
    token_name: str = "GITHUB_TOKEN",
    host: str = "github.com",
    login: str = "x-access-token",
    required: bool = False,
    overwrite: bool = True,
) -> bool:
    """
    Ensure GitHub HTTPS auth works by writing ~/.netrc using a token from env or Colab Secrets.

    netrc entry:
      machine github.com
      login x-access-token
      password <TOKEN>

    Returns True if written, False if skipped (when required=False and token missing).
    """
    tok = load_token(token_name, required=required)
    if not tok:
        return False

    netrc = Path.home() / ".netrc"
    if netrc.exists() and not overwrite:
        return True

    netrc.write_text(
        f"machine {host}\n"
        f"login {login}\n"
        f"password {tok}\n"
    )
    try:
        netrc.chmod(0o600)
    except Exception:
        # chmod may fail on some platforms; best-effort only
        pass
    return True


def github_pip_url(owner: str, repo: str, ref: str = "main") -> str:
    """
    Build a pip VCS URL that works once ~/.netrc is configured.
    """
    return f"git+https://{host}/{owner}/{repo}.git@{ref}".replace("{host}", "github.com")
