"""Optimizer selection: AdamW (default) and the hybrid Muon."""

from __future__ import annotations

from typing import Any, cast

import pytest
import torch

from nanobeard.config import Config
from nanobeard.models.frigate import GPT
from nanobeard.optim import Muon, build_optimizer, zeropower_via_newtonschulz5


def _c_attn_weight(model: GPT) -> torch.Tensor:
    """First block's fused QKV weight.

    Walked as Any because nn.Module.__getattr__ is typed `Tensor | Module`, so
    every submodule hop otherwise needs its own cast.
    """
    return cast(Any, model).blocks[0].attn.c_attn.weight


def frigate_cfg(**overrides) -> Config:
    base = dict(
        model_name="frigate",
        vocab_size=128,
        block_size=16,
        n_layer=2,
        n_head=2,
        n_embd=16,
        dropout=0.0,
        use_rope=True,
        use_swiglu=True,
        use_rmsnorm=True,
        use_qk_norm=True,
    )
    base.update(overrides)
    return Config(**base)


def test_default_is_adamw():
    cfg = frigate_cfg()
    opt = build_optimizer(GPT(cfg), cfg)
    assert isinstance(opt, torch.optim.AdamW)


def test_muon_selected_by_config():
    cfg = frigate_cfg(optimizer="muon")
    opt = build_optimizer(GPT(cfg), cfg)
    assert isinstance(opt, Muon)


def test_unknown_optimizer_raises():
    cfg = frigate_cfg(optimizer="lion")
    with pytest.raises(ValueError, match="Unknown optimizer"):
        build_optimizer(GPT(cfg), cfg)


def test_muon_excludes_embeddings_and_head():
    """Embeddings + tied LM head must land in an AdamW group, never Muon."""
    cfg = frigate_cfg(optimizer="muon")
    model = GPT(cfg)
    opt = build_optimizer(model, cfg)
    muon_ids = {
        id(p) for g in opt.param_groups if g["use_muon"] for p in g["params"]
    }
    # Tied embedding/head tensor must not be optimized by Muon.
    assert id(model.lm_head.weight) not in muon_ids
    # The 2D attention projection must be.
    assert id(_c_attn_weight(model)) in muon_ids


def test_muon_groups_have_lr_ratio():
    cfg = frigate_cfg(optimizer="muon", learning_rate=3e-4, muon_lr=0.02)
    opt = build_optimizer(GPT(cfg), cfg)
    ratios = {g["use_muon"]: g["lr_ratio"] for g in opt.param_groups}
    assert ratios[False] == 1.0
    assert ratios[True] == pytest.approx(0.02 / 3e-4)


def _one_step(cfg: Config):
    torch.manual_seed(0)
    model = GPT(cfg)
    opt = build_optimizer(model, cfg)
    before = _c_attn_weight(model).detach().clone()
    idx = torch.randint(0, cfg.vocab_size, (2, cfg.block_size))
    tgt = torch.randint(0, cfg.vocab_size, (2, cfg.block_size))
    _, loss = model(idx, tgt)
    loss.backward()
    opt.step()
    after = _c_attn_weight(model).detach()
    return before, after


def test_muon_step_updates_2d_weights():
    before, after = _one_step(frigate_cfg(optimizer="muon", muon_lr=0.02))
    assert not torch.allclose(before, after)
    assert torch.isfinite(after).all()


def test_muon_step_runs_for_adamw_too():
    before, after = _one_step(frigate_cfg(optimizer="adamw"))
    assert not torch.allclose(before, after)


def test_muon_state_dict_roundtrip():
    """Resume path: save + load optimizer state survives."""
    cfg = frigate_cfg(optimizer="muon")
    model = GPT(cfg)
    opt = build_optimizer(model, cfg)
    idx = torch.randint(0, cfg.vocab_size, (2, cfg.block_size))
    tgt = torch.randint(0, cfg.vocab_size, (2, cfg.block_size))
    _, loss = model(idx, tgt)
    loss.backward()
    opt.step()
    sd = opt.state_dict()

    opt2 = build_optimizer(model, cfg)
    opt2.load_state_dict(sd)
    assert len(opt2.state) == len(opt.state)


def test_newton_schulz_orthogonalizes_square():
    """Output of NS on a square matrix is near-orthogonal (singular values ~1)."""
    torch.manual_seed(0)
    g = torch.randn(32, 32)
    o = zeropower_via_newtonschulz5(g, steps=5).float()
    s = torch.linalg.svdvals(o)
    assert s.max() < 1.3 and s.min() > 0.5
