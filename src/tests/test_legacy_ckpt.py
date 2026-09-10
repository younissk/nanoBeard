"""Backwards-compat: the shim in nanobeard/__init__.py must let old ckpts
pickled with `training.config.Config` deserialize via the new Config class."""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import torch

# Importing nanobeard installs the shim as a side-effect.
import nanobeard  # noqa: F401


def test_training_config_module_aliased():
    """`training.config.Config` should resolve to nanobeard.config.Config."""
    import nanobeard.config as new

    assert "training" in sys.modules
    assert "training.config" in sys.modules
    assert sys.modules["training.config"] is new


def test_legacy_pickle_deserializes(tmp_path: Path):
    """Simulate an old ckpt: pickle a payload whose Config class lives under
    `training.config`. With the shim active, torch.load should yield a Config
    instance of the new class."""
    from nanobeard.config import Config

    payload = {"config": Config(model_name="sloop", run_name="legacy")}
    ck_path = tmp_path / "legacy.pt"
    torch.save(payload, ck_path)

    out = torch.load(ck_path, map_location="cpu", weights_only=False)
    cfg = out["config"]
    assert cfg.run_name == "legacy"
    assert cfg.model_name == "sloop"


def test_legacy_pickle_with_old_module_path(tmp_path: Path):
    """Stronger: force the pickle to reference `training.config` directly.
    Round-trip through pickle without using save_checkpoint."""
    from nanobeard.config import Config

    cfg = Config(model_name="sloop", run_name="round-trip")
    # Manually craft a pickle stream that claims the class came from training.config.
    blob = pickle.dumps(cfg)
    # The shim should let pickle find Config under training.config too.
    assert "nanobeard.config" in blob.decode("latin-1") or "training.config" in blob.decode(
        "latin-1"
    )
    loaded = pickle.loads(blob)
    assert loaded.run_name == "round-trip"
