"""Frigate 358M — 1.6 epochs on pirate_enhanced_full (~4.75B tokens). A100 / RTX 4090.

27 layers / 16 heads / 1024 embd -> 358,345,088 params. head_dim = 1024/16 = 64
(RoPE-even + flash path). A scaled-up Frigate: ~2.85x the 126M preset's depth*width.

Memory: batch sized down for a 24GB 4090; comfortable on an A100 (40/80GB).
If OOM on the 4090, halve batch_size and double gradient_accumulation_steps
(keeps the 49,152 tokens/iter constant so the horizon math is unchanged).

Variants: CONFIG_VARIANT=smoke|gpu (default: smoke).
"""

import os

from nanobeard.config import Config

DATA_DIR = "data/datasets/pirate_enhanced_full"

ARCH = dict(
    block_size=512,
    n_layer=27,
    n_head=16,
    n_embd=1024,
    use_rope=True,
    use_swiglu=True,
    use_rmsnorm=True,
    use_qk_norm=True,
)


def make_config_smoke() -> Config:
    """Tiny M1 Mac sanity-check. Real arch but few iters — NOT a good model."""
    return Config(
        run_name="frigate-360m-full-smoke",
        model_name="frigate",
        data_dir=DATA_DIR,
        run_dir="runs/frigate-360m-full-smoke",
        hf_model_repo="younissk/nanoBeard-frigate-360M-full",
        dropout=0.05,
        optimizer="muon",
        device="mps",
        dtype="float32",
        compile=False,
        batch_size=1,
        gradient_accumulation_steps=1,
        max_iters=50,
        eval_interval=25,
        eval_iters=5,
        warmup_iters=10,
        lr_decay_iters=50,
        **ARCH,
    )


def make_config_gpu() -> Config:
    """Full 1.6-epoch Frigate-358M run. A100 recommended (40/80GB); fits a 4090."""
    return Config(
        run_name="frigate-360m-full",
        model_name="frigate",
        data_dir=DATA_DIR,
        run_dir="runs/frigate-360m-full",
        hf_model_repo="younissk/nanoBeard-frigate-360M-full",
        # Own ckpt repo — keeps the 358M full-corpus weights separate.
        hf_ckpt_repo="younissk/frigate-360M-full-ckpts",
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
        batch_size=12,
        gradient_accumulation_steps=8,  # effective batch 96, 49,152 tokens/iter
        # epochs is the real knob: 1.6 passes over pirate_enhanced_full
        # (~4.75B tokens) ≈ 154.6k iters at 49,152 tokens/iter. max_iters is just
        # the safety ceiling; resolve_max_iters() derives the real horizon from
        # the actual train.bin size at train start.
        epochs=1.6,
        max_iters=160000,
        warmup_iters=2000,
        lr_decay_iters=160000,
        eval_interval=1000,
        eval_iters=100,
        wandb_project="pirate-llm",
        **ARCH,
    )


def make_config_sft() -> Config:
    """Chat SFT on top of the pretrained 360M-full Frigate (A100 / RTX 4090).

    Loads the pretraining ckpt.pt from pretrained_ckpt_repo; block_size MUST
    equal the pretraining block_size (512). SFT data is built on the fly
    (dolly-pirate + empathetic_dialogues); only the tokenizer (pirate_bpe.json)
    is needed locally, and its sha256 must match the pretrained ckpt.
    """
    return Config(
        run_name="frigate-360m-full-sft",
        model_name="frigate",
        data_dir=DATA_DIR,  # for pirate_bpe.json (tokenizer + sha verify)
        run_dir="runs/frigate-360m-full-sft",
        hf_model_repo="younissk/nanoBeard-frigate-360M-full",
        # hf_ckpt_repo intentionally None: save_sft_checkpoint pushes a multi-GB
        # ckpt synchronously inside the loop on every val improvement, which on a
        # flaky uplink stalls the whole run. SFT is short — checkpoint locally and
        # upload the final sft_ckpt.pt once, out of band, to
        # younissk/frigate-360M-full-sft-ckpts (public).
        hf_ckpt_repo=None,
        # SFT loads the pretraining ckpt.pt from here (not the model repo).
        pretrained_ckpt_repo="younissk/frigate-360M-full-ckpts",
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
        # 360M is ~2.85x the 125M; batch 8 keeps it comfortably within a 24GB
        # 4090 and well within an A100 40GB.
        batch_size=8,
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
