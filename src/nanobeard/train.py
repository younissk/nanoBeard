import argparse
import contextlib
import math
import os
import shutil
import threading
import time
from contextlib import nullcontext
from typing import cast

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast

from nanobeard.config import Config, load_config
from nanobeard.data import get_batch
from nanobeard.env import load_env
from nanobeard.models import build_model
from nanobeard.models.naming import display_name
from nanobeard.optim import build_optimizer
from nanobeard.tokenizer_hash import hash_file

load_env()


def resolve_vocab_size(config: Config) -> Config:
    """The dataset's tokenizer is the source of truth for vocab_size.

    The model embedding (config.vocab_size) MUST equal the tokenizer vocab, or
    token ids index out of range. If the built tokenizer exists, override
    config.vocab_size to match it — so the recipe's vocab_size is the single
    knob and configs need not track it. No-op if the tokenizer isn't built yet.
    """
    if not os.path.exists(config.tokenizer_path):
        return config

    from tokenizers import Tokenizer

    tok_vocab = Tokenizer.from_file(config.tokenizer_path).get_vocab_size()
    if config.vocab_size != tok_vocab:
        print(
            f"vocab_size: overriding config value {config.vocab_size} -> "
            f"{tok_vocab} (from tokenizer {config.tokenizer_path})"
        )
        config.vocab_size = tok_vocab
    return config


def resolve_max_iters(config: Config) -> Config:
    """Translate config.epochs into a training horizon (max_iters + lr_decay_iters).

    1 epoch = one full pass over train.bin:
        iters/epoch = train_tokens / (batch_size * block_size * grad_accum)
    The configured max_iters stays a HARD CEILING — it keeps smoke runs short
    (their tiny max_iters wins over a full epoch on a big corpus) and stops any
    run from overshooting. lr_decay_iters is synced to the resolved horizon so
    cosine decay spans the whole run. No-op if train.bin isn't built yet.
    """
    if not os.path.exists(config.train_bin):
        return config

    train_tokens = os.path.getsize(config.train_bin) // 2  # uint16 = 2 bytes/token
    # max(1, ...) mirrors the training loop's guard — the two must agree on
    # tokens/iter or the horizon is wrong by exactly the accumulation factor.
    tokens_per_iter = (
        config.batch_size * config.block_size * max(1, config.gradient_accumulation_steps)
    )
    epoch_iters = max(1, round(config.epochs * train_tokens / tokens_per_iter))
    horizon = min(epoch_iters, config.max_iters)

    print(
        f"epochs={config.epochs}: {epoch_iters} iters for a full pass over "
        f"{train_tokens:,} tokens; training {horizon} iters "
        f"(max_iters ceiling {config.max_iters})."
    )
    print(
        f"  tokens/iter = {config.batch_size} x {config.block_size} x "
        f"{config.gradient_accumulation_steps} = {tokens_per_iter:,}; "
        f"total = {horizon * tokens_per_iter:,} tokens "
        f"({horizon * tokens_per_iter / train_tokens:.2f} passes)."
    )
    config.max_iters = horizon
    config.lr_decay_iters = horizon
    return config


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


def get_lr(it: int, config: Config) -> float:
    if it < config.warmup_iters:
        return config.learning_rate * (it + 1) / (config.warmup_iters + 1)
    if it > config.lr_decay_iters:
        return config.min_lr
    decay_ratio = (it - config.warmup_iters) / (config.lr_decay_iters - config.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)


@torch.no_grad()
def estimate_loss(model: nn.Module, config: Config, ctx) -> dict[str, float]:
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(config.eval_iters)
        for k in range(config.eval_iters):
            x, y = get_batch(split, config)
            with ctx:
                _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def try_resume(config: Config) -> dict | None:
    local_path = config.ckpt_path

    if os.path.exists(local_path):
        print(f"Resuming from local checkpoint: {local_path}")
        return torch.load(local_path, map_location=config.device, weights_only=False)

    if not (config.resume and config.hf_ckpt_repo):
        return None

    try:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            repo_id=config.hf_ckpt_repo,
            filename="ckpt.pt",
            token=os.environ.get("HF_TOKEN"),
            local_dir=config.run_dir,
        )
        print(f"Resuming from Hub checkpoint: {config.hf_ckpt_repo}")
        return torch.load(path, map_location=config.device, weights_only=False)
    except Exception as e:
        print(f"No Hub checkpoint to resume from ({type(e).__name__}: {e})")
        return None


def ensure_hub_repo(config: Config):
    if not config.hf_ckpt_repo:
        return
    from huggingface_hub import HfApi

    HfApi().create_repo(
        repo_id=config.hf_ckpt_repo,
        private=config.hf_private,
        exist_ok=True,
        token=os.environ.get("HF_TOKEN"),
    )


def maybe_init_wandb(config: Config):
    if not config.wandb_project:
        return None
    import wandb

    return wandb.init(
        project=config.wandb_project,
        entity=config.wandb_entity,
        name=config.run_name,
        config=vars(config),
    )


# One upload at a time. A second push starting while the first is in flight
# would race on the staged file and waste the uplink the training box is paying
# for; skipping is always the right call because the next push is minutes away.
_upload_lock = threading.Lock()


def _push_to_hub(staged: str, repo_id: str, iter_num: int, val_loss: float, tag: str) -> None:
    """Upload a staged checkpoint copy. Runs on a background thread."""
    if not _upload_lock.acquire(blocking=False):
        print("  → Hub push already in flight, skipping this one")
        return
    try:
        from huggingface_hub import HfApi

        HfApi().upload_file(
            path_or_fileobj=staged,
            path_in_repo="ckpt.pt",
            repo_id=repo_id,
            token=os.environ.get("HF_TOKEN"),
            commit_message=f"iter {iter_num} | val {val_loss:.4f} ({tag})",
        )
        print(f"  → pushed to {repo_id} (iter {iter_num})")
    except Exception as e:
        print(f"  ! Hub upload failed ({type(e).__name__}: {e}) — continuing")
    finally:
        with contextlib.suppress(OSError):
            os.remove(staged)
        _upload_lock.release()


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: Config,
    iter_num: int,
    val_loss: float,
    best_val_loss: float,
    tag: str = "latest",
    push: bool = True,
) -> threading.Thread | None:
    """Write the checkpoint locally; optionally start a background Hub push.

    Returns the upload thread so a caller that is about to exit can join it —
    the thread is a daemon, so nothing else keeps it alive.
    """
    raw_model: nn.Module = getattr(model, "_orig_mod", model)

    tokenizer_sha256 = None
    if os.path.exists(config.tokenizer_path):
        tokenizer_sha256 = hash_file(config.tokenizer_path)

    checkpoint = {
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": config,
        "model_name": config.model_name,
        "iter_num": iter_num,
        "val_loss": val_loss,
        "best_val_loss": best_val_loss,
        "tokenizer_sha256": tokenizer_sha256,
    }
    path = config.ckpt_path
    torch.save(checkpoint, path)
    print(f"  → saved checkpoint to {path} ({tag}, val {val_loss:.4f})")

    if push and config.hf_ckpt_repo:
        # Upload a copy: the next torch.save would otherwise rewrite the file
        # mid-transfer and push a truncated checkpoint over a good one.
        staged = f"{path}.upload"
        try:
            shutil.copyfile(path, staged)
        except OSError as e:
            print(f"  ! could not stage checkpoint for upload ({e}) — continuing")
            return None
        thread = threading.Thread(
            target=_push_to_hub,
            args=(staged, config.hf_ckpt_repo, iter_num, val_loss, tag),
            daemon=True,
        )
        thread.start()
        return thread

    return None


def train(config: Config):
    print(f"\n=== Training run: {config.run_name} ===")
    print(f"Device: {config.device}, dtype: {config.dtype}, compile: {config.compile}")

    config = resolve_vocab_size(config)
    config = resolve_max_iters(config)
    ctx, scaler = setup_training(config)
    ensure_hub_repo(config)
    wandb_run = maybe_init_wandb(config)

    model = build_model(config).to(config.device)
    print(display_name(config, model))

    if config.compile:
        print("Compiling model with torch.compile...")
        # torch.compile returns OptimizedModule (callable wrapper). Same interface
        # as nn.Module for our purposes; cast so type-checkers see it that way.
        model = cast(nn.Module, torch.compile(model))

    optimizer = build_optimizer(model, config)

    iter_num = 0
    best_val_loss = float("inf")

    ckpt = try_resume(config)
    if ckpt is not None:
        raw_model = cast(nn.Module, getattr(model, "_orig_mod", model))
        raw_model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        iter_num = ckpt["iter_num"] + 1
        best_val_loss = ckpt.get("best_val_loss", ckpt.get("val_loss", float("inf")))
        print(f"Resumed at iter {iter_num}, best_val_loss={best_val_loss:.4f}")

    t0 = time.time()
    last_ckpt_t = t0
    last_push_t = t0
    last_val_loss = best_val_loss

    while iter_num < config.max_iters:
        lr = get_lr(iter_num, config)
        for pg in optimizer.param_groups:
            # lr_ratio lets Muon's group run at a higher peak LR than AdamW's
            # while sharing one cosine schedule (ratio 1.0 for AdamW groups).
            pg["lr"] = lr * pg.get("lr_ratio", 1.0)

        if iter_num % config.eval_interval == 0:
            losses = estimate_loss(model, config, ctx)
            elapsed = time.time() - t0
            print(
                f"step {iter_num:>6d} | "
                f"train {losses['train']:.4f} | val {losses['val']:.4f} | "
                f"lr {lr:.2e} | {elapsed:.1f}s"
            )
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "train/loss": losses["train"],
                        "val/loss": losses["val"],
                        "lr": lr,
                        "elapsed_s": elapsed,
                    },
                    step=iter_num,
                )

            last_val_loss = losses["val"]
            is_best = losses["val"] < best_val_loss
            if is_best:
                best_val_loss = losses["val"]
            # A new best is worth the uplink immediately; routine saves ride the
            # wall-clock cadence below instead of pushing on every eval.
            save_checkpoint(
                model,
                optimizer,
                config,
                iter_num,
                losses["val"],
                best_val_loss,
                tag="best" if is_best else "latest",
                push=is_best,
            )
            last_ckpt_t = time.time()
            if is_best:
                last_push_t = last_ckpt_t

        # Wall-clock checkpointing. On an interruptible host eviction arrives
        # with no warning, so the thing that must be bounded is minutes of lost
        # work, not iterations — and iteration time is not constant.
        now = time.time()
        due_ckpt = (
            config.ckpt_interval_min > 0
            and now - last_ckpt_t >= config.ckpt_interval_min * 60
        )
        due_push = (
            config.hf_ckpt_repo
            and config.hub_push_interval_min > 0
            and now - last_push_t >= config.hub_push_interval_min * 60
        )
        if due_ckpt or due_push:
            save_checkpoint(
                model,
                optimizer,
                config,
                iter_num,
                last_val_loss,
                best_val_loss,
                tag="periodic",
                push=bool(due_push),
            )
            last_ckpt_t = now
            if due_push:
                last_push_t = now

        # One optimizer step = gradient_accumulation_steps micro-batches. The
        # loss is divided by the count so the accumulated gradient is the mean
        # over the effective batch, not the sum — otherwise the effective LR
        # scales with the accumulation factor.
        optimizer.zero_grad(set_to_none=True)
        micro_steps = max(1, config.gradient_accumulation_steps)
        loss_sum = 0.0
        for _ in range(micro_steps):
            x, y = get_batch("train", config)
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
            print(f"  iter {iter_num} | minibatch loss {loss_sum:.4f} | lr {lr:.2e}")
            if wandb_run is not None:
                wandb_run.log(
                    {"train/minibatch_loss": loss_sum, "lr": lr},
                    step=iter_num,
                )

        iter_num += 1

    # Final checkpoint: the cadences above otherwise discard whatever happened
    # since the last one, which on a short run can be the whole tail.
    final_push = save_checkpoint(
        model, optimizer, config, iter_num, last_val_loss, best_val_loss,
        tag="final", push=True,
    )
    # Daemon thread: nothing else keeps the process alive for it, and on a spot
    # box the machine may be gone shortly after. Wait for the final upload.
    if final_push is not None:
        final_push.join(timeout=600)
        if final_push.is_alive():
            print("  ! final Hub push still running after 10 min — abandoning it")

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    if wandb_run is not None:
        wandb_run.finish()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        required=True,
        help="Path to config .py file, e.g. configs/sloop.py",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    train(config)


if __name__ == "__main__":
    main()
