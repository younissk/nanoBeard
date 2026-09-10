"""Frigate-specific tests: the v2 arch flags (RoPE, SwiGLU, RMSNorm, QK-norm).

The parametrized model contract (test_model_contract.py) builds every model
from tiny_cfg, where all arch flags default False — so it exercises Frigate's
Sloop-compatible fallback path. These tests flip the flags ON to cover the
actual Frigate architecture.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch

from nanobeard.config import Config
from nanobeard.models.frigate import GPT, SwiGLU


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


def test_builds_with_all_flags_on():
    cfg = frigate_cfg()
    m = GPT(cfg)
    assert m.wpe is None  # RoPE replaces the learned position table
    assert m.num_parameters() > 0


def test_forward_and_loss_finite():
    cfg = frigate_cfg()
    m = GPT(cfg)
    idx = torch.randint(0, cfg.vocab_size, (2, cfg.block_size))
    targets = torch.randint(0, cfg.vocab_size, (2, cfg.block_size))
    logits, loss = m(idx, targets)
    assert logits.shape == (2, cfg.block_size, cfg.vocab_size)
    assert loss.ndim == 0 and torch.isfinite(loss)


def test_backward_grads_finite():
    cfg = frigate_cfg()
    m = GPT(cfg)
    idx = torch.randint(0, cfg.vocab_size, (2, cfg.block_size))
    targets = torch.randint(0, cfg.vocab_size, (2, cfg.block_size))
    _, loss = m(idx, targets)
    loss.backward()
    for name, p in m.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"non-finite grad in {name}"


def test_causal_no_future_leakage_with_rope():
    cfg = frigate_cfg()
    m = GPT(cfg)
    m.eval()
    idx_a = torch.randint(0, cfg.vocab_size, (1, cfg.block_size))
    idx_b = idx_a.clone()
    idx_b[0, -1] = (idx_a[0, -1] + 1) % cfg.vocab_size
    with torch.no_grad():
        logits_a, _ = m(idx_a)
        logits_b, _ = m(idx_b)
    assert torch.allclose(logits_a[:, :-1], logits_b[:, :-1], atol=1e-5)


def test_overlong_context_raises():
    cfg = frigate_cfg()
    m = GPT(cfg)
    idx = torch.randint(0, cfg.vocab_size, (1, cfg.block_size + 1))
    with pytest.raises(AssertionError):
        m(idx)


def test_weight_tying():
    cfg = frigate_cfg()
    m = GPT(cfg)
    assert m.wte.weight.data_ptr() == m.lm_head.weight.data_ptr()


def test_swiglu_hidden_is_multiple_of_64():
    cfg = frigate_cfg(n_embd=384)
    ffn = SwiGLU(cfg)
    hidden = ffn.w_gate.out_features
    assert hidden % 64 == 0
    assert hidden == 1024  # 8/3 * 384 = 1024


def test_flags_off_reduces_to_sloop_shape():
    """All flags False -> learned wpe, GELU MLP, LayerNorm (Sloop-compatible)."""
    cfg = frigate_cfg(
        use_rope=False, use_swiglu=False, use_rmsnorm=False, use_qk_norm=False
    )
    m = GPT(cfg)
    assert m.wpe is not None
    assert isinstance(m.ln_f, torch.nn.LayerNorm)


def test_qk_norm_changes_output():
    """Sanity: toggling qk_norm actually changes the forward pass."""
    cfg_on = frigate_cfg(use_qk_norm=True)
    cfg_off = dataclasses.replace(cfg_on, use_qk_norm=False)
    torch.manual_seed(0)
    m_on = GPT(cfg_on)
    torch.manual_seed(0)
    m_off = GPT(cfg_off)
    idx = torch.randint(0, cfg_on.vocab_size, (1, cfg_on.block_size))
    with torch.no_grad():
        out_on, _ = m_on(idx)
        out_off, _ = m_off(idx)
    assert not torch.allclose(out_on, out_off)
