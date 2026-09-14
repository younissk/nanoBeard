"""Sampling / generation from a trained nanoBeard checkpoint.

Run:
  uv run python -m nanobeard.sample --config configs/sloop.py --prompt "Once upon a time"
"""

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer

from nanobeard.config import Config, load_config
from nanobeard.models import build_model
from nanobeard.models.naming import display_name
from nanobeard.tokenizer_hash import TokenizerMismatch, verify_match


@torch.no_grad()
def generate(
    model: nn.Module,
    idx: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 0.8,
    top_k: int | None = 40,
    eos_id: int | None = None,
) -> torch.Tensor:
    block_size: int = model.config.block_size  # type: ignore[union-attr,assignment]

    for _ in range(max_new_tokens):
        idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature

        if top_k is not None and top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("inf")

        probs = F.softmax(logits, dim=-1)
        next_idx = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, next_idx), dim=1)
        # Stop a turn as soon as the model emits end-of-text (used by --chat).
        if eos_id is not None and next_idx.item() == eos_id:
            break

    return idx


def chat_repl(model, tokenizer, args) -> None:
    """Interactive pirate chat with a rolling memory window.

    Renders the transcript in the SFT format (User:/Pirate:), appends a
    "Pirate:" cue, generates a reply until <|endoftext|>, and keeps the most
    recent turns that fit in block_size.
    """
    from nanobeard.sft_data import Turn, build_chat_prompt_ids

    block_size: int = model.config.block_size
    eos_id = tokenizer.token_to_id("<|endoftext|>")

    print("\n⚓ Pirate chat — type your message, Ctrl-C or 'quit' to leave.\n")
    history: list[Turn] = []
    while True:
        try:
            user_msg = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nFair winds!")
            return
        if user_msg.lower() in {"quit", "exit"}:
            print("Fair winds!")
            return
        if not user_msg:
            continue

        history.append(Turn("user", user_msg))
        prompt_ids = build_chat_prompt_ids(
            history, tokenizer, eos_id, block_size, reserve=args.max_tokens
        )
        idx = torch.tensor(prompt_ids, dtype=torch.long, device=args.device).unsqueeze(0)
        out_idx = generate(
            model,
            idx,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            eos_id=eos_id,
        )
        reply_ids = out_idx[0, len(prompt_ids):].cpu().tolist()
        if reply_ids and reply_ids[-1] == eos_id:
            reply_ids = reply_ids[:-1]
        reply = tokenizer.decode(reply_ids).strip()
        history.append(Turn("bot", reply))
        print(f"Pirate: {reply}\n")


def load_checkpoint(ckpt_path: str, device: str, tokenizer_path: str | None = None) -> nn.Module:
    print(f"Loading checkpoint from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    config: Config = checkpoint["config"]
    config.device = device

    if tokenizer_path is not None:
        expected = checkpoint.get("tokenizer_sha256")
        try:
            verify_match(tokenizer_path, expected)
        except TokenizerMismatch as e:
            # Warn, don't hard-fail — user may want to inspect outputs anyway.
            print(f"WARNING: {e}")

    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    print(
        f"{display_name(config, model)} | "
        f"trained for {checkpoint['iter_num']} iters, "
        f"val loss {checkpoint['val_loss']:.4f}"
    )
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default=None, help="Path to config .py file (provides ckpt + tokenizer paths)"
    )
    parser.add_argument("--ckpt", default=None, help="Explicit checkpoint path (overrides config)")
    parser.add_argument(
        "--tokenizer", default=None, help="Explicit tokenizer path (overrides config)"
    )
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--device", default=None, help="cuda / mps / cpu (auto-detect if None)")
    parser.add_argument(
        "--chat",
        action="store_true",
        help="Interactive multi-turn pirate chat (for SFT checkpoints).",
    )
    args = parser.parse_args()

    if args.config:
        cfg = load_config(args.config)
        ckpt_path = args.ckpt or cfg.ckpt_path
        tokenizer_path = args.tokenizer or cfg.tokenizer_path
        if args.device is None:
            args.device = cfg.device
    else:
        if not (args.ckpt and args.tokenizer):
            parser.error("Provide --config OR both --ckpt and --tokenizer")
        ckpt_path = args.ckpt
        tokenizer_path = args.tokenizer

    if args.device is None:
        if torch.cuda.is_available():
            args.device = "cuda"
        elif torch.backends.mps.is_available():
            args.device = "mps"
        else:
            args.device = "cpu"
    print(f"Device: {args.device}")

    tokenizer = Tokenizer.from_file(tokenizer_path)
    model = load_checkpoint(ckpt_path, args.device, tokenizer_path=tokenizer_path)

    if args.chat:
        chat_repl(model, tokenizer, args)
        return

    prompt_ids = tokenizer.encode(args.prompt).ids
    idx = torch.tensor(prompt_ids, dtype=torch.long, device=args.device).unsqueeze(0)
    print(f"\nPrompt: {args.prompt!r} ({len(prompt_ids)} tokens)")
    print(f"Sampling: temperature={args.temperature}, top_k={args.top_k}")
    print("=" * 60)

    for i in range(args.num_samples):
        out_idx = generate(
            model,
            idx,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )
        out_text = tokenizer.decode(out_idx[0].cpu().tolist())
        print(f"\n--- Sample {i + 1} ---")
        print(out_text)
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
