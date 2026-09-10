"""Human-readable model identity. One place to change display format."""

import torch.nn as nn

from nanobeard.config import Config


def format_params(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n / 1e9:.1f}B"
    if n >= 1_000_000:
        return f"{n / 1e6:.1f}M"
    if n >= 1_000:
        return f"{n / 1e3:.1f}K"
    return str(n)


def display_name(cfg: Config, model: nn.Module | None = None) -> str:
    """E.g. 'nanoBeard Sloop (15.2M params)'."""
    from . import build_model, spec_for

    spec = spec_for(cfg)
    m = model if model is not None else build_model(cfg)
    return f"nanoBeard {spec.codename} ({format_params(m.num_parameters())} params)"  # type: ignore[operator]
