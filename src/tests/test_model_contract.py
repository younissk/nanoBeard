"""Contract that every model in MODEL_REGISTRY must satisfy.

Parametrized over the registry — when v2 (Brig, etc.) lands, it's automatically
tested for free. If a model adds a quirk (different forward signature, no weight
tying, etc.), update the contract here so the deviation is documented.
"""

from __future__ import annotations

import pytest
import torch

from nanobeard.config import Config
from nanobeard.models import MODEL_REGISTRY


@pytest.fixture(params=list(MODEL_REGISTRY.keys()))
def model_key(request) -> str:
    return request.param


@pytest.fixture
def model_and_cfg(model_key: str, tiny_cfg: Config):
    cfg = Config(**{**tiny_cfg.__dict__, "model_name": model_key})
    spec = MODEL_REGISTRY[model_key]
    model = spec.cls(cfg)
    return model, cfg


def test_model_builds_from_config(model_and_cfg):
    model, cfg = model_and_cfg
    assert hasattr(model, "config")
    assert model.config.block_size == cfg.block_size


def test_num_parameters_is_positive(model_and_cfg):
    model, _ = model_and_cfg
    assert model.num_parameters() > 0


def test_forward_no_targets_returns_logits_only(model_and_cfg):
    model, cfg = model_and_cfg
    idx = torch.randint(0, cfg.vocab_size, (2, cfg.block_size))
    logits, loss = model(idx)
    assert logits.shape == (2, cfg.block_size, cfg.vocab_size)
    assert loss is None


def test_forward_with_targets_returns_scalar_loss(model_and_cfg):
    model, cfg = model_and_cfg
    idx = torch.randint(0, cfg.vocab_size, (2, cfg.block_size))
    targets = torch.randint(0, cfg.vocab_size, (2, cfg.block_size))
    _, loss = model(idx, targets)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_backward_produces_finite_grads(model_and_cfg):
    model, cfg = model_and_cfg
    idx = torch.randint(0, cfg.vocab_size, (2, cfg.block_size))
    targets = torch.randint(0, cfg.vocab_size, (2, cfg.block_size))
    _, loss = model(idx, targets)
    loss.backward()
    for name, p in model.named_parameters():
        if p.requires_grad and p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"non-finite grad in {name}"


def test_short_context_works(model_and_cfg):
    """T < block_size is a valid input — common when sampling early in a sequence."""
    model, cfg = model_and_cfg
    idx = torch.randint(0, cfg.vocab_size, (1, max(1, cfg.block_size // 2)))
    logits, _ = model(idx)
    assert logits.shape[1] == idx.shape[1]


def test_full_block_works(model_and_cfg):
    model, cfg = model_and_cfg
    idx = torch.randint(0, cfg.vocab_size, (1, cfg.block_size))
    logits, _ = model(idx)
    assert logits.shape[1] == cfg.block_size


def test_overlong_context_raises(model_and_cfg):
    """T > block_size must be rejected — position embedding has fixed shape."""
    model, cfg = model_and_cfg
    idx = torch.randint(0, cfg.vocab_size, (1, cfg.block_size + 1))
    with pytest.raises(AssertionError):
        model(idx)


def test_causal_mask_no_future_leakage(model_and_cfg):
    """Changing a token at position k must not affect logits at positions < k."""
    model, cfg = model_and_cfg
    model.eval()
    torch.manual_seed(0)
    idx_a = torch.randint(0, cfg.vocab_size, (1, cfg.block_size))
    idx_b = idx_a.clone()
    # Mutate the LAST token only.
    idx_b[0, -1] = (idx_a[0, -1] + 1) % cfg.vocab_size
    with torch.no_grad():
        logits_a, _ = model(idx_a)
        logits_b, _ = model(idx_b)
    # Logits at all positions BEFORE the last must be identical.
    assert torch.allclose(logits_a[:, :-1], logits_b[:, :-1], atol=1e-5)


def test_weight_tying_sloop():
    """Sloop ties wte with lm_head. Document this as a Sloop-specific contract.
    If a future model breaks tying, fork this test rather than relaxing it."""
    from nanobeard.models.sloop import GPT

    cfg = Config(vocab_size=64, block_size=8, n_layer=1, n_head=2, n_embd=16)
    m = GPT(cfg)
    assert m.wte.weight.data_ptr() == m.lm_head.weight.data_ptr()
