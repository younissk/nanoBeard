"""Find the micro-batch that gives the best tokens/sec on *this* GPU.

    uv run python -m nanobeard.autobatch --config configs/frigate_360m_full.py

Micro-batch size is the last throughput lever left once bf16, FlashAttention and
torch.compile are in — and it is the one that cannot be chosen from a laptop,
because it depends on the card in front of you. Guessing it low wastes rented
GPU-hours; guessing it high crashes an hour into a run.

This measures rather than reasons: for each candidate size it runs real
forward/backward/step iterations and reports tokens/sec, then names the winner
and the `gradient_accumulation_steps` that keeps your configured effective batch
unchanged.

Note that bigger is not automatically better. Past the point where the GPU is
saturated, extra micro-batch buys nothing and extra *accumulation* actively
costs — which is why this reports a throughput curve and not just the largest
size that fits.
"""

from __future__ import annotations

import argparse
import os
import time

import torch

from nanobeard.config import Config, load_config
from nanobeard.models import build_model
from nanobeard.optim import build_optimizer

# Steps thrown away before timing: the first touches allocate caches and, with
# compile=True, pay for the graph capture.
WARMUP_STEPS = 3
TIMED_STEPS = 8


def candidate_sizes(start: int, limit: int) -> list[int]:
    """Powers of two from `start`, plus the configured size if it is not one."""
    out = []
    n = start
    while n <= limit:
        out.append(n)
        n *= 2
    return out


def _oom(exc: BaseException) -> bool:
    return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()


def measure(config: Config, batch_size: int) -> float | None:
    """Tokens/sec at this micro-batch, or None if it does not fit.

    Builds a fresh model per size: reusing one leaves the previous size's
    activations cached and the OOM boundary lands in the wrong place.
    """
    cfg = Config(**{**config.__dict__, "batch_size": batch_size})
    device = cfg.device
    try:
        model = build_model(cfg).to(device)
        optimizer = build_optimizer(model, cfg)
        x = torch.randint(0, cfg.vocab_size, (batch_size, cfg.block_size), device=device)
        y = torch.randint(0, cfg.vocab_size, (batch_size, cfg.block_size), device=device)

        for i in range(WARMUP_STEPS + TIMED_STEPS):
            if i == WARMUP_STEPS:
                if device == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            _, loss = model(x, y)
            loss.backward()
            optimizer.step()

        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        return (TIMED_STEPS * batch_size * cfg.block_size) / elapsed
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if _oom(e):
            return None
        raise
    finally:
        if device == "cuda":
            torch.cuda.empty_cache()


def run(config: Config, start: int, limit: int) -> None:
    target_effective = config.batch_size * max(1, config.gradient_accumulation_steps)
    print(
        f"device={config.device} dtype={config.dtype} block_size={config.block_size} "
        f"fused_loss={config.fused_loss}\n"
        f"configured: batch_size={config.batch_size} x "
        f"grad_accum={config.gradient_accumulation_steps} = effective {target_effective}\n"
    )
    print(f"{'micro-batch':>12} {'tokens/sec':>12} {'vs configured':>14}")
    print("-" * 40)

    results: dict[int, float] = {}
    baseline = None
    for size in candidate_sizes(start, limit):
        tps = measure(config, size)
        if tps is None:
            print(f"{size:>12} {'OOM':>12}")
            break
        results[size] = tps
        if size == config.batch_size:
            baseline = tps
        rel = f"{tps / baseline:.2f}x" if baseline else "—"
        print(f"{size:>12} {tps:>12,.0f} {rel:>14}")

    if not results:
        print("\nNothing fit — lower --start or the model does not fit at all.")
        return

    best = max(results, key=lambda k: results[k])
    accum = max(1, round(target_effective / best))
    print(
        f"\nbest micro-batch: {best} at {results[best]:,.0f} tok/s\n"
        f"to keep effective batch {target_effective}: "
        f"batch_size={best}, gradient_accumulation_steps={accum} "
        f"(effective {best * accum})"
    )
    if accum > 1:
        print(
            "  note: accumulation steps cost throughput. If you can accept a "
            "smaller effective batch, batch_size={best}, grad_accum=1 is faster."
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--variant", default="gpu",
                    help="Variant to probe; sets CONFIG_VARIANT, which the config dispatches on")
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--limit", type=int, default=512)
    args = ap.parse_args()

    # Configs dispatch on the env var, not an argument.
    os.environ["CONFIG_VARIANT"] = args.variant
    config = load_config(args.config)
    run(config, args.start, args.limit)


if __name__ == "__main__":
    main()
