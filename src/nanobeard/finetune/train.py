"""LoRA fine-tune of Qwen3-0.6B into the pirate assistant.

    uv run --group finetune python -m nanobeard.finetune.train \
        --data runs/distill/train.jsonl --out runs/lora/pirate-v1

LoRA rather than a full fine-tune, and not for the compute — at this size both
are minutes. The measured risk is forgetting: a heavy pirate system prompt alone
took this model's tool calling from 8/15 to 0/15 and cost 13 points of GSM8K
(see `runs/evals/`). Training the full weights toward that same style is the
efficient way to make the damage permanent; a low-rank adapter leaves the base
behaviour mostly intact and can be thrown away if the gates say it hurt.

Nothing here decides whether the result is good. That is `nanobeard.evals.run`,
before and after, and the ship rule is voice up with gsm8k / tool choice /
restraint holding.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

BASE_MODEL = "Qwen/Qwen3-0.6B"
# Attention and MLP projections. Embeddings stay frozen: they carry the
# 151k-token vocabulary and adapting them is how a small style SFT starts
# damaging unrelated knowledge.
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default="runs/distill/train.jsonl")
    ap.add_argument("--out", default="runs/lora/pirate-v1")
    ap.add_argument("--base", default=BASE_MODEL)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--device", default=None, help="cuda | mps | cpu (default: best available)")
    ap.add_argument("--limit", type=int, default=None, help="Use only N examples (smoke runs)")
    ap.add_argument("--push-to-hub", default=None, metavar="REPO",
                    help="Upload the adapter here when done, e.g. younissk/nanoBeard-pirate-lora")
    args = ap.parse_args()

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    from nanobeard.finetune.data import Collator, build_dataset, describe, load_rows, split

    device = args.device or (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"base={args.base} device={device}")

    tok = AutoTokenizer.from_pretrained(args.base)
    rows = load_rows(args.data)
    if args.limit:
        rows = rows[: args.limit]
    examples = build_dataset(tok, rows, max_len=args.max_len)
    print(f"\n{len(examples)} usable of {len(rows)} rows\n{describe(examples)}")
    if not examples:
        raise SystemExit("no usable examples — check --data and --max-len")

    train_ex, val_ex = split(examples, val_frac=args.val_frac, seed=args.seed)
    print(f"\ntrain={len(train_ex)} val={len(val_ex)}")

    model = AutoModelForCausalLM.from_pretrained(
        args.base, dtype=torch.float32 if device == "cpu" else torch.bfloat16
    )
    model.config.use_cache = False  # incompatible with gradient checkpointing / training
    lora = LoraConfig(
        r=args.rank, lora_alpha=args.alpha, lora_dropout=args.dropout,
        target_modules=TARGET_MODULES, task_type="CAUSAL_LM", bias="none",
    )
    # PeftModel wraps the base rather than subclassing its type, so rebind
    # instead of assigning over the annotated variable.
    peft_model = get_peft_model(model, lora)
    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in peft_model.parameters())
    print(f"LoRA r={args.rank} alpha={args.alpha}: "
          f"{trainable:,} trainable of {total:,} ({trainable / total:.2%})")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    steps_per_epoch = max(1, len(train_ex) // (args.batch_size * args.grad_accum))
    total_steps = max(1, int(steps_per_epoch * args.epochs))
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))
    print(f"~{total_steps} optimizer steps, {warmup_steps} warmup")

    targs = TrainingArguments(
        output_dir=str(out),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        # transformers 5.x dropped warmup_ratio; only warmup_steps survives, so
        # the ratio is resolved here against the real step count.
        warmup_steps=warmup_steps,
        logging_steps=10,
        eval_strategy="epoch" if val_ex else "no",
        save_strategy="epoch",
        save_total_limit=1,
        bf16=(device == "cuda"),
        # Without this the Trainer picks its own device and --device cpu is a
        # lie: on an 8GB Mac it silently lands on MPS and OOMs mid-step.
        use_cpu=(device == "cpu"),
        report_to=[],
        seed=args.seed,
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=peft_model, args=targs,
        train_dataset=train_ex, eval_dataset=val_ex or None,
        data_collator=Collator(tok.pad_token_id or 0),
    )
    trainer.train()

    peft_model.save_pretrained(str(out))
    tok.save_pretrained(str(out))
    (out / "training_args.json").write_text(json.dumps(vars(args), indent=2))
    print(f"\nadapter saved to {out}")

    # The Hub is the retrieval channel for a rented box, exactly as the
    # pretraining loop uses hf_ckpt_repo. Learned the hard way: an adapter that
    # exists only on a Vast instance is an adapter you may never see, because
    # `vastai execute` refuses on running instances, `vastai copy` to local can
    # be down for maintenance, and the SSH proxy can simply fail to forward.
    if args.push_to_hub:
        import os

        from huggingface_hub import HfApi

        from nanobeard.env import load_env

        load_env()
        token = os.getenv("HF_TOKEN")
        if not token or token == "none":
            print("  ! --push-to-hub given but HF_TOKEN is unset — adapter stays local")
        else:
            try:
                api = HfApi()
                api.create_repo(args.push_to_hub, token=token, exist_ok=True, private=True)
                api.upload_folder(
                    folder_path=str(out), repo_id=args.push_to_hub, token=token,
                    commit_message=f"LoRA r={args.rank} on {len(train_ex)} examples",
                )
                print(f"  -> pushed adapter to https://huggingface.co/{args.push_to_hub}")
            except Exception as e:
                print(f"  ! Hub upload failed ({type(e).__name__}: {e}) — adapter is still at {out}")
    print("next: uv run --group finetune python -m nanobeard.finetune.merge "
          f"--adapter {out} --out {out}-merged")


if __name__ == "__main__":
    main()
