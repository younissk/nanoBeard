"""Locate and load the project's .env.

The env file lives in `envs/`, which is NOT the directory `python-dotenv`
searches by default — a bare `load_dotenv()` walks up from the *cwd* and would
silently find nothing, leaving HF_TOKEN / WANDB_API_KEY unset at runtime with
no error. Always go through `load_env()`.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


def repo_root() -> Path:
    for d in Path(__file__).resolve().parents:
        if (d / "pyproject.toml").is_file():
            return d
    return Path.cwd()


def env_path() -> Path:
    return repo_root() / "envs" / ".env"


def load_env() -> None:
    """Load envs/.env, then any cwd-local .env as an override."""
    p = env_path()
    if p.is_file():
        load_dotenv(p)
    load_dotenv(override=False)  # legacy ./.env, if someone still has one


def require(name: str) -> str:
    """Fetch a required secret, with a message that says where to put it."""
    load_env()
    val = os.getenv(name)
    if not val:
        raise SystemExit(
            f"{name} is not set.\nAdd it to {env_path()} (see envs/example.env)."
        )
    return val
