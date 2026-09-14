import argparse
import math
import os
import time
from contextlib import nullcontext
from typing import cast

import torch
import torch.nn as nn
from dotenv import load_dotenv
from huggingface_hub import HfApi, hf_hub_download
from torch.amp import GradScaler, autocast

from nanobeard.config import Config, load_config
from nanobeard.models import build_model
from nanobeard.models.naming import display_name
from nanobeard.optim import build_optimizer
from nanobeard.sft_data import build_sft_dataset, get_sft_batch
from nanobeard.tokenizer_hash import hash_file, verify_match

load_dotenv()


def setup_training(config: Config):
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    os.makedirs(config.run_dir, exist_ok=True)

    ptdtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[config.dtype]

    ctx: nullcontext | autocast
    if config.device == "cuda" and config.dtype != "float32":
        ctx = autocast(device_type="cuda", dtype=ptdtype)
    else:
        ctx = nullcontext()

    scaler = GradScaler(enabled=(config.dtype == "float16"))
    return ctx, scaler


def load_pretrained(config: Config, pretrained_repo: str) -> nn.Module:
    if os.path.exists(pretrained_repo):
        path = pretrained_repo
    else:
        path = hf_hub_download(
            repo_id=pretrained_repo,
            filename="ckpt.pt",
            token=os.environ.get("HF_TOKEN"),
            local_dir=config.run_dir,
        )
    ckpt = torch.load(path, map_location=config.device, weights_only=False)
    arch_cfg: Config = ckpt["config"]

    # Architecture is fixed by the checkpoint — block_size, n_layer, n_embd, vocab_size.
    if config.block_size != arch_cfg.block_size:
        raise ValueError(
            f"SFT block_size ({config.block_size}) must equal pretraining "
            f"block_size ({arch_cfg.block_size}) — the position embedding is fixed."
        )

    if config.model_name != arch_cfg.model_name:
        raise ValueError(
            f"SFT model_name ({config.model_name!r}) must match pretraining "
            f"({arch_cfg.model_name!r}). Use the same architecture."
        )

    # Hard-fail on tokenizer mismatch: SFT against the wrong tokenizer
    # silently corrupts the model.
    expected = ckpt.get("tokenizer_sha256")
    if expected is not None and os.path.exists(config.tokenizer_path):
        verify_match(config.tokenizer_path, expected)

    arch_cfg.dropout = config.dropout

    # The architecture (incl. vocab_size) is fixed by the pretrained ckpt.
    # Sync it onto the SFT config so save_sft_checkpoint stores a config that
    # matches the weights — otherwise sampling rebuilds the model at the
    # Config default vocab (8192) and load_state_dict fails on a size mismatch.
    config.vocab_size = arch_cfg.vocab_size

    model = build_model(arch_cfg).to(config.device)
    model.load_state_dict(ckpt["model"])
    print(
        f"Loaded pretrained {display_name(arch_cfg, model)} from {pretrained_repo} "
        f"(iter {ckpt['iter_num']}, val {ckpt['val_loss']:.4f})"
    )
    return model


def get_lr(it: int, config: Config) -> float:
    if it < config.warmup_iters:
        return config.learning_rate * (it + 1) / (config.warmup_iters + 1)
    if it > config.lr_decay_iters:
        return config.min_lr
    decay_ratio = (it - config.warmup_iters) / (config.lr_decay_iters - config.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)


@torch.no_grad()
def estimate_val_loss(model, val_examples, config: Config, ctx) -> float:
    model.eval()
    losses = torch.zeros(config.eval_iters)
    for k in range(config.eval_iters):
        x, y = get_sft_batch(val_examples, config)
        with ctx:
            _, loss = model(x, y)
        losses[k] = loss.item()
    model.train()
    return losses.mean().item()


def save_sft_checkpoint(
    model, optimizer, config: Config, iter_num, val_loss, best_val_loss, tag="latest", push=False
):
    raw = model._orig_mod if hasattr(model, "_orig_mod") else model
    tokenizer_sha256 = None
    if os.path.exists(config.tokenizer_path):
        tokenizer_sha256 = hash_file(config.tokenizer_path)
    ckpt = {
        "model": raw.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": config,
        "model_name": config.model_name,
        "iter_num": iter_num,
        "val_loss": val_loss,
        "best_val_loss": best_val_loss,
        "stage": "sft",
        "tokenizer_sha256": tokenizer_sha256,
    }
    path = config.sft_ckpt_path
    torch.save(ckpt, path)
    print(f"  -> saved SFT ckpt to {path} ({tag}, val {val_loss:.4f})")

    # Only push improvements to the Hub — uploading every eval churns ~165MB
    # (model+optimizer) onto the same filename for no gain.
    if config.hf_ckpt_repo and push:
        try:
            HfApi().create_repo(
                repo_id=config.hf_ckpt_repo,
                private=config.hf_private,
                exist_ok=True,
                token=os.environ.get("HF_TOKEN"),
            )
            HfApi().upload_file(
                path_or_fileobj=path,
                path_in_repo="sft_ckpt.pt",
                repo_id=config.hf_ckpt_repo,
                token=os.environ.get("HF_TOKEN"),
                commit_message=f"sft iter {iter_num} | val {val_loss:.4f} ({tag})",
            )
            print(f"  -> pushed to {config.hf_ckpt_repo}")
        except Exception as e:
            print(f"  ! Hub upload failed ({type(e).__name__}: {e}) — continuing")


def sft_train(config: Config, pretrained_repo: str):
    print(f"\n=== SFT run: {config.run_name} ===")
    print(f"Device: {config.device}, dtype: {config.dtype}, compile: {config.compile}")

    ctx, scaler = setup_training(config)
    train_examples, val_examples, _ = build_sft_dataset(config)

    model = load_pretrained(config, pretrained_repo)
    if config.compile:
        print("Compiling model...")
        model = cast(nn.Module, torch.compile(model))

    optimizer = build_optimizer(model, config)

    iter_num = 0
    best_val_loss = float("inf")
    t0 = time.time()

    while iter_num < config.max_iters:
        lr = get_lr(iter_num, config)
        for pg in optimizer.param_groups:
            pg["lr"] = lr * pg.get("lr_ratio", 1.0)

        if iter_num % config.eval_interval == 0:
            val_loss = estimate_val_loss(model, val_examples, config, ctx)
            elapsed = time.time() - t0
            print(f"step {iter_num:>6d} | val {val_loss:.4f} | lr {lr:.2e} | {elapsed:.1f}s")
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
            save_sft_checkpoint(
                model,
                optimizer,
                config,
                iter_num,
                val_loss,
                best_val_loss,
                tag="best" if is_best else "latest",
                push=is_best,
            )

        optimizer.zero_grad(set_to_none=True)
        micro_steps = max(1, config.gradient_accumulation_steps)
        loss_sum = 0.0
        for _ in range(micro_steps):
            x, y = get_sft_batch(train_examples, config)
            with ctx:
                _, loss = model(x, y)
                loss = loss / micro_steps
            scaler.scale(loss).backward()
            loss_sum += loss.item()

        if config.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        if iter_num % config.log_interval == 0 and iter_num > 0:
            print(f"  iter {iter_num} | loss {loss_sum:.4f} | lr {lr:.2e}")

        iter_num += 1

    # Final checkpoint: the eval-interval cadence otherwise discards the last
    # (max_iters % eval_interval) steps. Evaluate + save the finished model;
    # push if it's the best we've seen.
    val_loss = estimate_val_loss(model, val_examples, config, ctx)
    is_best = val_loss < best_val_loss
    if is_best:
        best_val_loss = val_loss
    save_sft_checkpoint(
        model,
        optimizer,
        config,
        iter_num,
        val_loss,
        best_val_loss,
        tag="best" if is_best else "final",
        push=is_best,
    )

    print(f"\nSFT done. Best val: {best_val_loss:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config .py file")
    parser.add_argument(
        "--pretrained-repo",
        default=None,
        help="HF repo to load pretrained ckpt from. Defaults to config.hf_model_repo.",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    # The pretraining ckpt.pt lives in pretrained_ckpt_repo, NOT the model repo
    # (which holds safetensors). Fall back to hf_model_repo only for legacy use.
    pretrained = args.pretrained_repo or config.pretrained_ckpt_repo or config.hf_model_repo
    sft_train(config, pretrained_repo=pretrained)


if __name__ == "__main__":
    main()
