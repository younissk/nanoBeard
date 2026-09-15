"""GRPO on the search task, written to be read.

    uv run --group finetune python -m nanobeard.rl.grpo --steps 50

The loop, in words:

1. Take a batch of questions.
2. For each, play GROUP_SIZE episodes at temperature > 0, so the rollouts differ.
3. Score each episode by whether its final answer matches the dataset's.
4. Within each group, subtract the group's mean reward and divide by its spread.
   That is the advantage: "better or worse than my other attempts at *this*
   question". It is why GRPO needs no value network — the group is the baseline.
5. Push up the probability of tokens from better-than-average episodes, down for
   worse, and hold the whole thing near a frozen copy of the starting model with
   a KL penalty so it cannot drift into gibberish that happens to score well.

Two details that are easy to get wrong and fatal if you do:

* Only tokens the *model* generated carry gradient. The transcript also holds
  <results> blocks the environment pasted in; training on those teaches the
  policy to invent search results rather than read them.
* If every episode in a group scores the same, the advantage is zero for all of
  them and the batch contributes nothing. That is expected early on, and it is
  why `frac_degenerate` is logged — if it stays near 1.0, the task is too hard
  or the reward too coarse, and no amount of training will help.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

BASE_MODEL = "Qwen/Qwen3-0.6B"


@dataclass
class Rollout:
    input_ids: list[int]
    action_mask: list[bool]   # True where the model generated the token
    reward: float
    advantage: float = 0.0
    answer_em: float = 0.0
    n_searches: int = 0
    retrieval_recall: float = 0.0


def group_advantages(rewards: list[float], eps: float = 1e-4) -> list[float]:
    """Centre and scale within the group. This is the whole of GRPO's baseline."""
    n = len(rewards)
    mean = sum(rewards) / n
    var = sum((r - mean) ** 2 for r in rewards) / n
    std = var ** 0.5
    if std < eps:
        # Every attempt scored alike: no information about which was better.
        return [0.0] * n
    return [(r - mean) / std for r in rewards]


def build_rollout(tok, episode, reward, system: str) -> Rollout | None:
    """Tokenize a transcript and mark which tokens the model is answerable for."""
    prompt = tok.apply_chat_template(
        [{"role": "system", "content": system},
         {"role": "user", "content": episode.transcript[: episode.generated_spans[0][0]]}],
        tokenize=False, add_generation_prompt=True,
    )
    body = episode.transcript[episode.generated_spans[0][0]:]
    offset = len(prompt)
    full = prompt + body

    enc = tok(full, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = enc["input_ids"], enc["offset_mapping"]
    if not ids:
        return None

    # Shift the episode's spans into the combined string's coordinates.
    base = episode.generated_spans[0][0]
    spans = [(a - base + offset, b - base + offset) for a, b in episode.generated_spans]
    mask = [any(a >= lo and b <= hi for lo, hi in spans) and b > a for (a, b) in offsets]
    if not any(mask):
        return None

    return Rollout(
        input_ids=ids,
        action_mask=mask,
        reward=reward.total,
        answer_em=reward.answer_em,
        n_searches=episode.n_searches,
        retrieval_recall=reward.breakdown["retrieval_recall"],
    )


def sequence_logprobs(model, input_ids, attention_mask, action_mask):
    """Per-token log-probs of the tokens actually produced, masked to the model's own.

    Shifted by one because position i predicts token i+1.
    """
    import torch

    out = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = out.logits[:, :-1, :]
    targets = input_ids[:, 1:]
    logp = torch.log_softmax(logits.float(), dim=-1)
    token_logp = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    mask = action_mask[:, 1:].float()
    entropy = -(logp.exp() * logp).sum(-1)
    return token_logp, mask, entropy


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", default=BASE_MODEL)
    ap.add_argument("--adapter", default=None, help="Start from a LoRA instead of the base")
    ap.add_argument("--index", default="data/search/hotpot_bm25.pkl")
    ap.add_argument("--out", default="runs/rl/search-v1")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--questions-per-step", type=int, default=4)
    ap.add_argument("--group-size", type=int, default=6, help="Rollouts per question")
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--kl-beta", type=float, default=0.02)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-new-tokens", type=int, default=100)
    ap.add_argument("--max-searches", type=int, default=2)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from nanobeard.rl import rewards as R
    from nanobeard.rl.corpus import load as load_corpus
    from nanobeard.rl.env import SYSTEM, run_episode

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    index, questions = load_corpus(Path(args.index))
    print(f"corpus {len(index):,} paragraphs | {len(questions):,} questions | device={device}")

    tok = AutoTokenizer.from_pretrained(args.base)
    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    # cast to Any: from_pretrained is typed as a union, and peft wraps rather
    # than subclasses, so the checker cannot follow either handoff.
    loaded: Any = AutoModelForCausalLM.from_pretrained(args.base, dtype=dtype)
    base_model: Any = loaded.to(device)
    model: Any
    if args.adapter:
        model = PeftModel.from_pretrained(base_model, args.adapter, is_trainable=True)
    else:
        model = get_peft_model(base_model, LoraConfig(
            r=args.rank, lora_alpha=args.rank * 2, lora_dropout=0.0, bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
        ))
    model.config.use_cache = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable {trainable:,}")

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    def generate(body: str) -> str:
        """One model turn. Stops at a closing tag so the model never writes the
        environment's half of the transcript itself."""
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": body}]
        # enable_thinking=False is not optional. Qwen3's template defaults it on,
        # and the model then spends the entire token budget inside <think> —
        # every rollout parses as invalid, every reward is 0.0, and every
        # advantage is 0.0, so the run looks like "the task is too hard" rather
        # than "the prompt was wrong".
        text = tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        enc = tok(text, return_tensors="pt", add_special_tokens=False).to(device)
        with torch.no_grad():
            gen = model.generate(
                **enc, max_new_tokens=args.max_new_tokens, do_sample=True,
                temperature=args.temperature, top_p=0.95,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )
        decoded = tok.decode(gen[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        chunk = decoded if isinstance(decoded, str) else str(decoded)
        for tag in ("</search>", "</answer>"):
            i = chunk.find(tag)
            if i != -1:
                return chunk[: i + len(tag)]
        return chunk

    rng = torch.Generator().manual_seed(args.seed)
    history = []
    for step in range(args.steps):
        t0 = time.time()
        idx = torch.randint(0, len(questions), (args.questions_per_step,), generator=rng)
        batch: list[Rollout] = []
        degenerate = 0

        model.eval()
        for qi in idx.tolist():
            q = questions[qi]
            group = []
            for _ in range(args.group_size):
                ep = run_episode(q, index, generate, max_searches=args.max_searches)
                rw = R.compute(ep.answer, q.answer, did_search=ep.did_search,
                               retrieved_titles=ep.retrieved_titles, gold_titles=q.gold_titles)
                r = build_rollout(tok, ep, rw, SYSTEM)
                if r is not None:
                    group.append(r)
            if not group:
                continue
            advs = group_advantages([g.reward for g in group])
            if all(a == 0.0 for a in advs):
                degenerate += 1
            for g, a in zip(group, advs, strict=True):
                g.advantage = a
            batch.extend(group)

        if not batch:
            print(f"step {step}: no usable rollouts")
            continue

        # ---- policy gradient over the batch, one rollout at a time ----
        model.train()
        opt.zero_grad(set_to_none=True)
        tot_loss = tot_kl = tot_ent = 0.0
        used = 0
        for r in batch:
            if r.advantage == 0.0:
                continue  # no signal; skip the forward pass entirely
            ids = torch.tensor([r.input_ids], device=device)
            attn = torch.ones_like(ids)
            act = torch.tensor([r.action_mask], device=device)

            logp, mask, entropy = sequence_logprobs(model, ids, attn, act)
            n = mask.sum().clamp(min=1)
            pg = -(logp * mask).sum() / n * r.advantage

            # The reference policy is this same model with the adapter switched
            # off — no second copy in memory, and exactly the weights training
            # started from.
            with torch.no_grad(), model.disable_adapter():
                ref_logp, _, _ = sequence_logprobs(model, ids, attn, act)
            # k3 estimator: non-negative and lower variance than (logp - ref).
            diff = ref_logp - logp
            kl = ((diff.exp() - diff - 1) * mask).sum() / n

            (pg + args.kl_beta * kl).backward()
            tot_loss += pg.item()
            tot_kl += kl.item()
            tot_ent += ((entropy * mask).sum() / n).item()
            used += 1

        if used:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()

        m = {
            "step": step,
            "reward": sum(r.reward for r in batch) / len(batch),
            "exact_match": sum(r.answer_em for r in batch) / len(batch),
            "retrieval_recall": sum(r.retrieval_recall for r in batch) / len(batch),
            "searches": sum(r.n_searches for r in batch) / len(batch),
            "kl": tot_kl / max(1, used),
            "entropy": tot_ent / max(1, used),
            "frac_degenerate": degenerate / max(1, args.questions_per_step),
            "rollouts": len(batch),
            "trained_on": used,
            "seconds": round(time.time() - t0, 1),
        }
        history.append(m)
        print(f"step {m['step']:>3} reward {m['reward']:.3f} em {m['exact_match']:.2f} "
              f"recall {m['retrieval_recall']:.2f} kl {m['kl']:.4f} ent {m['entropy']:.2f} "
              f"degen {m['frac_degenerate']:.0%} ({m['seconds']}s)")
        (out / "metrics.jsonl").open("a").write(json.dumps(m) + "\n")

    model.save_pretrained(str(out))
    tok.save_pretrained(str(out))
    (out / "args.json").write_text(json.dumps(vars(args), indent=2))
    if history:
        first, last = history[0], history[-1]
        print(f"\nreward {first['reward']:.3f} -> {last['reward']:.3f} | "
              f"em {first['exact_match']:.2f} -> {last['exact_match']:.2f}")
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
