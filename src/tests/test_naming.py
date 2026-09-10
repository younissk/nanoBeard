from __future__ import annotations

from nanobeard.config import Config
from nanobeard.models import build_model
from nanobeard.models.naming import display_name, format_params


def test_format_params_small():
    assert format_params(500) == "500"


def test_format_params_thousands():
    assert format_params(1_500) == "1.5K"


def test_format_params_millions():
    assert format_params(13_800_000) == "13.8M"


def test_format_params_billions():
    assert format_params(7_000_000_000) == "7.0B"


def test_display_name_format(tiny_cfg: Config):
    m = build_model(tiny_cfg)
    name = display_name(tiny_cfg, m)
    assert name.startswith("nanoBeard Sloop (")
    assert name.endswith(" params)")
    assert "M" in name or "K" in name or "params)" in name


def test_display_name_builds_model_if_not_provided(tiny_cfg: Config):
    """display_name(cfg) without a model should still work — it builds one."""
    name = display_name(tiny_cfg)
    assert "nanoBeard Sloop" in name
