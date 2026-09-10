"""Publish flow — stages the right files, writes config.json with required keys.

HF upload is mocked so this test runs offline."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch

from nanobeard.config import Config
from nanobeard.models import build_model
from nanobeard.train import save_checkpoint


def _make_ckpt(cfg: Config) -> str:
    model = build_model(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    os.makedirs(cfg.run_dir, exist_ok=True)
    save_checkpoint(model, optimizer, cfg, iter_num=0, val_loss=1.0, best_val_loss=1.0)
    return cfg.ckpt_path


def test_publish_stages_required_files(tokenized_cfg: Config, tmp_path: Path, monkeypatch):
    cfg = tokenized_cfg
    _make_ckpt(cfg)

    config_module = tmp_path / "fake_config.py"
    config_module.write_text(
        "from nanobeard.config import Config\n"
        f"def make_config(): return Config(**{cfg.__dict__!r})\n"
    )

    stage_dir = tmp_path / "stage"
    monkeypatch.setenv("HF_TOKEN", "test-token")

    fake_api = MagicMock()
    with patch("nanobeard.publish.HfApi", return_value=fake_api):
        # Invoke publish via CLI surface.
        argv = [
            "nanobeard.publish",
            "--config",
            str(config_module),
            "--ckpt",
            cfg.ckpt_path,
            "--stage-dir",
            str(stage_dir),
            "--readme",
            str(tmp_path / "missing-readme.md"),
            "--banner",
            str(tmp_path / "missing-banner.png"),
        ]
        with patch.object(sys, "argv", argv):
            from nanobeard import publish

            publish.main()

    assert (stage_dir / "model.safetensors").exists()
    assert (stage_dir / "config.json").exists()
    assert (stage_dir / "pirate_bpe.json").exists()
    assert (stage_dir / "training_metadata.json").exists()


def test_publish_config_json_contains_required_keys(
    tokenized_cfg: Config, tmp_path: Path, monkeypatch
):
    cfg = tokenized_cfg
    _make_ckpt(cfg)

    config_module = tmp_path / "fake_config.py"
    config_module.write_text(
        "from nanobeard.config import Config\n"
        f"def make_config(): return Config(**{cfg.__dict__!r})\n"
    )

    stage_dir = tmp_path / "stage"
    monkeypatch.setenv("HF_TOKEN", "test-token")

    with patch("nanobeard.publish.HfApi", return_value=MagicMock()):
        argv = [
            "nanobeard.publish",
            "--config",
            str(config_module),
            "--ckpt",
            cfg.ckpt_path,
            "--stage-dir",
            str(stage_dir),
            "--readme",
            str(tmp_path / "no.md"),
            "--banner",
            str(tmp_path / "no.png"),
        ]
        with patch.object(sys, "argv", argv):
            from nanobeard import publish

            publish.main()

    cj = json.loads((stage_dir / "config.json").read_text())
    # Must include arch fields for Space to reconstruct the model.
    for field in ("vocab_size", "block_size", "n_layer", "n_head", "n_embd"):
        assert field in cj
    # Multi-version metadata that the Space depends on:
    assert cj["model_name"] == "sloop"
    assert cj["codename"] == "Sloop"
    assert cj["model_type"] == "nanobeard-sloop"
    assert "display_name" in cj
    assert "nanoBeard Sloop" in cj["display_name"]
    assert cj["num_parameters"] > 0


def test_publish_uploads_to_spec_repo(tokenized_cfg: Config, tmp_path: Path, monkeypatch):
    """publish must call HfApi.upload_folder with the registry's hf_repo, not a stale literal."""
    cfg = tokenized_cfg
    _make_ckpt(cfg)

    config_module = tmp_path / "fake_config.py"
    config_module.write_text(
        "from nanobeard.config import Config\n"
        f"def make_config(): return Config(**{cfg.__dict__!r})\n"
    )

    stage_dir = tmp_path / "stage"
    monkeypatch.setenv("HF_TOKEN", "test-token")

    fake_api = MagicMock()
    with patch("nanobeard.publish.HfApi", return_value=fake_api):
        argv = [
            "nanobeard.publish",
            "--config",
            str(config_module),
            "--ckpt",
            cfg.ckpt_path,
            "--stage-dir",
            str(stage_dir),
            "--readme",
            str(tmp_path / "no.md"),
            "--banner",
            str(tmp_path / "no.png"),
        ]
        with patch.object(sys, "argv", argv):
            from nanobeard import publish

            publish.main()

    # Should have called create_repo and upload_folder with the Sloop HF repo.
    fake_api.create_repo.assert_called_once()
    assert fake_api.create_repo.call_args.kwargs["repo_id"] == "younissk/nanoBeard"
    fake_api.upload_folder.assert_called_once()
    assert fake_api.upload_folder.call_args.kwargs["repo_id"] == "younissk/nanoBeard"
