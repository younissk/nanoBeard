"""Frigate 126M — 1 epoch on pirate_enhanced_full (~4.75B tokens). RTX 4090 / A100.

16 layers / 12 heads / 768 embd -> 125,856,512 params. head_dim = 768/12 = 64
(RoPE-even + flash path). Same arch as the original frigate-gpu preset, retrained
for a single pass over the larger full corpus.

Variants: CONFIG_VARIANT=smoke|gpu (default: smoke).
"""

import os

from nanobeard.config import Config

DATA_DIR = "data/datasets/pirate_enhanced_full"

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
        run_name="frigate-125m-full-smoke",
        model_name="frigate",
        data_dir=DATA_DIR,
        run_dir="runs/frigate-125m-full-smoke",
        hf_model_repo="younissk/nanoBeard-frigate-125M-full",
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
    """Full 1-epoch Frigate run for a single RTX 4090 (24GB) or A100."""
    return Config(
        run_name="frigate-125m-full",
        model_name="frigate",
        data_dir=DATA_DIR,
        run_dir="runs/frigate-125m-full",
        hf_model_repo="younissk/nanoBeard-frigate-125M-full",
        # Own ckpt repo — keeps the full-corpus weights separate from frigate-ckpts.
        hf_ckpt_repo="younissk/frigate-125M-full-ckpts",
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
        gradient_accumulation_steps=3,  # effective batch 96, 49,152 tokens/iter
        # epochs is the real knob: 1 full pass over pirate_enhanced_full
        # (~4.75B tokens) ≈ 96.6k iters at 49,152 tokens/iter. max_iters is just
        # the safety ceiling; resolve_max_iters() derives the real horizon from
        # the actual train.bin size at train start.
        epochs=1.0,
        max_iters=100000,
        warmup_iters=1000,
        lr_decay_iters=100000,
        eval_interval=1000,
        eval_iters=100,
        wandb_project="pirate-llm",
        **ARCH,
    )


def make_config_sft() -> Config:
    """Chat SFT on top of the pretrained 125M-full Frigate (RTX 4090 / A100).

    Loads the pretraining ckpt.pt from pretrained_ckpt_repo; block_size MUST
    equal the pretraining block_size (512) — load_pretrained hard-fails otherwise.
    SFT data is built on the fly (dolly-pirate + empathetic_dialogues); only the
    tokenizer (pirate_bpe.json from pirate_enhanced_full) is needed locally, and
    its sha256 must match the pretrained ckpt.
    """
    return Config(
        run_name="frigate-125m-full-sft",
        model_name="frigate",
        data_dir=DATA_DIR,  # for pirate_bpe.json (tokenizer + sha verify)
        run_dir="runs/frigate-125m-full-sft",
        hf_model_repo="younissk/nanoBeard-frigate-125M-full",
        # hf_ckpt_repo intentionally None: save_sft_checkpoint pushes a ~1.5GB
        # ckpt synchronously inside the training loop on every val improvement,
        # which on a slow/flaky uplink stalls the whole loop. SFT is short, so
        # checkpoint locally only and upload the final sft_ckpt.pt once, out of
        # band, to younissk/frigate-125M-full-sft-ckpts (public).
        hf_ckpt_repo=None,
        # SFT loads the pretraining ckpt.pt from here (not the model repo).
        pretrained_ckpt_repo="younissk/frigate-125M-full-ckpts",
        hf_private=False,
        dropout=0.0,
        device="cuda",
        dtype="bfloat16",
        compile=False,  # short run; compile warmup not worth it
        # SFT optimizer: low LR, no weight decay (AdamW, the Config default).
        learning_rate=2e-5,
        min_lr=2e-6,
        weight_decay=0.0,
        warmup_iters=100,
        lr_decay_iters=3000,
        max_iters=3000,
        batch_size=16,
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
