"""Checkpoint save/load roundtrip + resume."""

from __future__ import annotations

import os

import torch

from nanobeard.config import Config
from nanobeard.models import build_model
from nanobeard.train import save_checkpoint, try_resume


def test_save_and_reload_state_dict_match(tiny_cfg: Config):
    model = build_model(tiny_cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    os.makedirs(tiny_cfg.run_dir, exist_ok=True)

    save_checkpoint(model, optimizer, tiny_cfg, iter_num=0, val_loss=1.0, best_val_loss=1.0)

    ck = torch.load(tiny_cfg.ckpt_path, map_location="cpu", weights_only=False)
    model2 = build_model(tiny_cfg)
    model2.load_state_dict(ck["model"])

    # Forward should now produce identical logits (eval mode, no dropout).
    model.eval()
    model2.eval()
    torch.manual_seed(0)
    idx = torch.randint(0, tiny_cfg.vocab_size, (1, tiny_cfg.block_size))
    with torch.no_grad():
        out1, _ = model(idx)
        out2, _ = model2(idx)
    assert torch.allclose(out1, out2)


def test_ckpt_carries_required_metadata(tiny_cfg: Config):
    model = build_model(tiny_cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    os.makedirs(tiny_cfg.run_dir, exist_ok=True)
    save_checkpoint(model, optimizer, tiny_cfg, iter_num=7, val_loss=2.5, best_val_loss=2.5)

    ck = torch.load(tiny_cfg.ckpt_path, map_location="cpu", weights_only=False)
    assert ck["iter_num"] == 7
    assert ck["val_loss"] == 2.5
    assert ck["model_name"] == "sloop"
    assert isinstance(ck["config"], Config)
    assert ck["config"].model_name == "sloop"


def test_try_resume_loads_local_ckpt(tiny_cfg: Config):
    model = build_model(tiny_cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    os.makedirs(tiny_cfg.run_dir, exist_ok=True)
    save_checkpoint(model, optimizer, tiny_cfg, iter_num=3, val_loss=1.0, best_val_loss=1.0)

    ck = try_resume(tiny_cfg)
    assert ck is not None
    assert ck["iter_num"] == 3


def test_try_resume_returns_none_when_no_local_and_no_repo(tiny_cfg: Config):
    """No local ckpt + no hf_ckpt_repo configured -> None, no exception."""
    tiny_cfg.hf_ckpt_repo = None
    tiny_cfg.resume = True
    assert try_resume(tiny_cfg) is None
