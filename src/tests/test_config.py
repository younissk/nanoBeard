"""Config loading + derived paths + variant dispatch."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from nanobeard.config import Config, load_config


def test_default_config_has_required_fields():
    cfg = Config()
    assert cfg.model_name == "sloop"
    assert cfg.data_dir == "data/datasets/tiny_pirate_stories"
    assert cfg.run_dir == "runs/sloop"
    assert cfg.hf_model_repo == "younissk/nanoBeard"


def test_derived_paths_use_data_dir(tmp_path: Path):
    cfg = Config(data_dir=str(tmp_path / "d"), run_dir=str(tmp_path / "r"))
    assert cfg.train_bin == os.path.join(cfg.data_dir, "train.bin")
    assert cfg.val_bin == os.path.join(cfg.data_dir, "val.bin")
    assert cfg.tokenizer_path == os.path.join(cfg.data_dir, "pirate_bpe.json")
    assert cfg.ckpt_path == os.path.join(cfg.run_dir, "ckpt.pt")
    assert cfg.sft_ckpt_path == os.path.join(cfg.run_dir, "sft_ckpt.pt")


def test_out_dir_alias_returns_run_dir():
    cfg = Config(run_dir="custom/runs")
    assert cfg.out_dir == "custom/runs"


def test_load_config_sloop_smoke(monkeypatch):
    monkeypatch.setenv("CONFIG_VARIANT", "smoke")
    cfg = load_config("configs/sloop.py")
    assert cfg.model_name == "sloop"
    assert cfg.run_name == "sloop-m1-smoke"
    assert cfg.max_iters == 50


def test_load_config_sloop_gpu(monkeypatch):
    monkeypatch.setenv("CONFIG_VARIANT", "gpu")
    cfg = load_config("configs/sloop.py")
    assert cfg.device == "cuda"
    assert cfg.compile is True
    assert cfg.dtype == "bfloat16"


def test_load_config_sloop_sft(monkeypatch):
    monkeypatch.setenv("CONFIG_VARIANT", "sft")
    cfg = load_config("configs/sloop.py")
    assert cfg.dropout == 0.0
    assert cfg.weight_decay == 0.0
    assert cfg.resume is False


def test_load_config_missing_file():
    with pytest.raises((ImportError, FileNotFoundError)):
        load_config("configs/does_not_exist.py")


def test_load_config_module_missing_factory(tmp_path: Path):
    bad = tmp_path / "bad.py"
    bad.write_text("x = 1\n")
    with pytest.raises(AttributeError):
        load_config(str(bad))


def test_load_config_accepts_module_level_config(tmp_path: Path):
    """Modules with `config = Config(...)` also work."""
    mod = tmp_path / "viaattr.py"
    mod.write_text(
        "from nanobeard.config import Config\n"
        "config = Config(run_name='from-attr', model_name='sloop')\n"
    )
    cfg = load_config(str(mod))
    assert cfg.run_name == "from-attr"
