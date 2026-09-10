"""Model registry — spec_for, build_model, ARCH_FIELDS hygiene."""

from __future__ import annotations

import torch.nn as nn

from nanobeard.config import Config
from nanobeard.models import MODEL_REGISTRY, build_model, spec_for


def test_registry_non_empty():
    assert len(MODEL_REGISTRY) >= 1


def test_sloop_is_registered():
    assert "sloop" in MODEL_REGISTRY
    spec = MODEL_REGISTRY["sloop"]
    assert spec.codename == "Sloop"
    assert spec.hf_repo == "younissk/nanoBeard"
    assert issubclass(spec.cls, nn.Module)


def test_dispatch_key_matches_registry_key():
    for key, spec in MODEL_REGISTRY.items():
        assert key == spec.dispatch_key, (
            f"Registry key {key!r} must match spec.dispatch_key {spec.dispatch_key!r}"
        )


def test_spec_for_returns_correct_spec():
    cfg = Config(model_name="sloop")
    assert spec_for(cfg) is MODEL_REGISTRY["sloop"]


def test_build_model_returns_registered_class(tiny_cfg: Config):
    m = build_model(tiny_cfg)
    expected_cls = MODEL_REGISTRY[tiny_cfg.model_name].cls
    assert isinstance(m, expected_cls)


def test_arch_fields_present_on_config():
    """Every field a spec lists must actually exist on Config."""
    cfg = Config()
    for spec in MODEL_REGISTRY.values():
        for field in spec.arch_fields:
            assert hasattr(cfg, field), (
                f"{spec.codename} declares arch_field {field!r} not on Config"
            )
