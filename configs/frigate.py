"""nanoBeard Frigate — v2 architecture preset, trained on pirate_enhanced.

Frigate is a deeper, thinner Galleon with four modern upgrades (all implemented
in nanobeard/models/frigate.py, gated by Config flags):

  use_rope     RoPE rotary position embedding, replacing the learned wpe table.
  use_swiglu   SwiGLU FFN (8/3 expansion) instead of the 4x GELU MLP.
  use_rmsnorm  RMSNorm instead of LayerNorm.
  use_qk_norm  per-head RMSNorm on Q and K — stabilizes the deep stack.

Sizing (frigate-gpu): n_layer=16, n_head=12, n_embd=768, block_size=512
  -> ~126M params (16 * (4*768^2 + 3*768*2048) transformer + 16384*768 tied
  embeddings). n_head=12 keeps head_dim = 768/12 = 64 — even (required by RoPE)
  and on the flash-attention kernel path.

vocab_size is NOT set here: train.resolve_vocab_size() overrides it from the
dataset's tokenizer (16384 for pirate_enhanced).

Token budget: tokens/iter = batch_size * block_size * grad_accum = 32*512*3 = 49152.
  1 epoch (1.55B) ~= 31.5k iters. Training length is set by `epochs` (1.6 here
  ~= 50k iters); max_iters is just a safety ceiling. resolve_max_iters() derives
  the real horizon from the actual train.bin size at train start.

Variants: CONFIG_VARIANT=smoke|gpu|sft (default: smoke).
"""

import os

from nanobeard.config import Config

DATA_DIR = "data/datasets/pirate_enhanced"

# Frigate architecture flags — shared by every variant.
ARCH = dict(
    block_size=512,
    n_layer=16,
    n_head=12,
    n_embd=768,
    use_rope=True,
    use_swiglu=True,
    use_rmsnorm=True,
    use_qk_norm=True,
)


def make_config_smoke() -> Config:
    """Tiny M1 Mac sanity-check. Real arch but few iters — NOT a good model."""
    return Config(
        run_name="frigate-m1-smoke",
        model_name="frigate",
        data_dir=DATA_DIR,
        run_dir="runs/frigate-smoke",
        hf_model_repo="younissk/nanoBeard-frigate-126M",
        dropout=0.05,
        optimizer="muon",
        device="mps",
        dtype="float32",
        compile=False,
        batch_size=2,
        gradient_accumulation_steps=1,
        max_iters=50,
        eval_interval=25,
        eval_iters=5,
        warmup_iters=10,
        lr_decay_iters=50,
        **ARCH,
    )


def make_config_gpu() -> Config:
    """Full Frigate run for a single RTX 4090 (24GB)."""
    return Config(
        run_name="frigate-gpu",
        model_name="frigate",
        data_dir=DATA_DIR,
        run_dir="runs/frigate",
        hf_model_repo="younissk/nanoBeard-frigate-126M",
        # Own ckpt repo — Frigate's weights are incompatible with Galleon's.
        hf_ckpt_repo="younissk/frigate-ckpts",
        dropout=0.05,
        # Muon for the 2D hidden matrices; embeddings/head/norms stay on AdamW.
        optimizer="muon",
        muon_lr=0.02,
        muon_momentum=0.95,
        device="cuda",
        dtype="bfloat16",
        compile=True,
        # Chunked lm_head+CE: frees the [B*T, vocab] logits tensor, which is
        # what caps the micro-batch on a 24GB card.
        fused_loss=True,
        batch_size=32,
        gradient_accumulation_steps=3,  # effective batch 96
        # epochs is the real knob: 1.6 passes over pirate_enhanced (~1.55B tokens)
        # ≈ 50k iters at 49152 tokens/iter. max_iters is just the safety ceiling.
        epochs=1.6,
        max_iters=60000,
        warmup_iters=400,
        lr_decay_iters=60000,
        eval_interval=500,
        eval_iters=100,
        wandb_project="pirate-llm",
        **ARCH,
    )


def make_config_sft() -> Config:
    """Chat SFT on top of the pretrained Frigate (RTX 4090).

    block_size MUST equal the pretraining block_size (512) — load_pretrained
    hard-fails otherwise.
    """
    return Config(
        run_name="frigate-sft",
        model_name="frigate",
        data_dir=DATA_DIR,
        run_dir="runs/frigate-sft",
        hf_model_repo="younissk/nanoBeard-frigate-126M",
        hf_ckpt_repo="younissk/frigate-sft-ckpts",
        # SFT loads the pretraining ckpt.pt from here (not the model repo).
        pretrained_ckpt_repo="younissk/frigate-ckpts",
        dropout=0.0,
        device="cuda",
        dtype="bfloat16",
        compile=False,  # short run; compile warmup not worth it
        # SFT optimizer: low LR, no weight decay.
        learning_rate=2e-5,
        min_lr=2e-6,
        weight_decay=0.0,
        warmup_iters=100,
        lr_decay_iters=3000,
        max_iters=3000,
        batch_size=16,
        gradient_accumulation_steps=2,  # effective batch 32
        eval_interval=200,
        eval_iters=50,
        log_interval=20,
        resume=False,
        wandb_project="pirate-llm",
        **ARCH,
    )


def make_config() -> Config:
    """Variant dispatcher. CONFIG_VARIANT=smoke|gpu|sft (default: smoke)."""
    variant = os.environ.get("CONFIG_VARIANT", "smoke")
    return {
        "smoke": make_config_smoke,
        "gpu": make_config_gpu,
        "sft": make_config_sft,
    }[variant]()
