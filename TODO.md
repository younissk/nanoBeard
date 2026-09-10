# TODO

Project state as of 2026-09-10 (verified against the working tree, not
remembered). Captured so future-you (and Claude in follow-up sessions) can pick
this up without re-deriving context.

## Gates

All green as of 2026-09-10: `make lint`, `make typecheck` and `make test`.
`typecheck` runs mypy *then* pyright, and pyright had never reached completion
before — mypy always exited first and masked it.

- [ ] Make CI a required check on `main` once `.github/workflows/test.yml` has
      run green on a real runner. It never has. The `UV_TORCH_BACKEND=cpu` env
      (keeps uv off multi-GB CUDA wheels a CPU runner cannot use) is the part
      most likely to need adjusting.
- [ ] Dependabot or Renovate for `uv.lock` upkeep.
- [ ] pyright still reports ~177 **warnings** — the config downgrades
      `reportArgumentType`, `reportOptionalMemberAccess` and
      `reportGeneralTypeIssues` from error to warning. Not a gate today. To make
      them one, promote a single rule at a time; most of the noise is
      `Config(**overrides)` in test helpers.

## Decide before the next GPU run

- [ ] **Check `make vast-offers` before every run — the ranking is not stable.**
      Measured 2026-09-10 within one session: the RTX_5090 bid floor moved
      $0.202 -> $0.333 while the 4090 held at $0.200, so the cheapest card
      swapped twice in ten minutes. `GPU` is a candidate *list* now and the
      launcher picks the cheapest; pin a single name only when you need
      specific VRAM.

- [ ] **The 360M token budget changed meaning on 2026-09-10.** Gradient
      accumulation was declared, budgeted for, and never executed — the loop
      took one micro-batch per optimizer step. `frigate_360m_full` gpu
      (`batch_size=12`, `grad_accum=8`) therefore ran at 6,144 tokens/iter while
      `resolve_max_iters` sized the horizon for 49,152, so a "1.6 epoch" run saw
      **950M tokens, 12.5% of the 7.6B intended**.
      The loop now accumulates, which means the same config is honest *and*
      roughly 8x the compute. Nothing was changed to hide that: pick a real
      `epochs` for the next run rather than inheriting 1.6 by accident.
      `resolve_max_iters` now prints tokens/iter and the total, so the two can
      never drift apart silently again.
- [ ] **Run `make autobatch` on the rented box before the next long run.**
      Micro-batch is the last throughput lever after bf16 + flash + compile, and
      it depends on the card. `batch_size=12` predates `fused_loss`, which frees
      the `[B*T, vocab]` logits tensor — the allocation that set that number.
      12 is almost certainly now too low.

## Half-built — scaffolded, needs flesh

- [ ] **Self-play** (`src/nanobeard/rejection/selfplay.py`). Has a Makefile
      target now, but it is the second place the 512-token ceiling bites: its
      docstring notes a 15-turn transcript does not fit, so it trims from the
      oldest turn each round. Reasoning-roadmap step 1 fixes this too.
- [ ] **Chat playground** (`src/nanobeard/chat/`) has no automated coverage of
      the llama-server subprocess lifecycle — that needs llama.cpp on PATH, so
      it is a `slow`-marked test at best. Discovery, history trimming and prompt
      assembly are unit-tested. Start / switch / SIGTERM cleanup were checked by
      hand on 2026-09-10: switching leaves exactly one process, SIGTERM leaves
      none (it did leak before — plain SIGTERM skipped the cleanup path).

## Reasoning roadmap — teach Frigate to reason

Ordered. Each step is a hard blocker for the next. Plan derived 2026-09-10 from
`src/nanobeard/config.py` + the `pirate_enhanced` data recipe.

- [x] **0. Tokenizer fertility check — done 2026-09-10.** `make fertility`
      (`src/nanobeard/fertility.py`, corpus ships beside it, no network).
      `REFERENCE=Qwen/Qwen3-0.6B make fertility` adds a control.

      **The hypothesis was wrong in the useful direction: the tokenizer is not
      the problem.** Tokens per character on `pirate_enhanced` (16k), each
      domain as a multiple of plain English:

      | domain | pirate_bpe 16k | GPT-2 50k | Qwen3 151k |
      |---|---|---|---|
      | english_prose | 1.00x | 1.00x | 1.00x |
      | gsm8k_problem | 1.17x | 1.11x | 1.19x |
      | gsm8k_cot | **2.25x** | 1.92x | 2.42x |
      | arithmetic | **2.74x** | 2.26x | 4.18x |
      | algebra | 2.23x | 2.22x | 2.55x |
      | python_code | 2.51x | 2.27x | 1.43x |

      CoT costs ~2x prose in *every* tokenizer — that is the domain, not us. On
      arithmetic `pirate_bpe` beats Qwen3 outright, because Qwen3 splits every
      digit on purpose (`1024` -> `1 0 2 4`) while ours merges pairs
      (`1024` -> `10 2 4`). Fewer tokens, but note that per-digit splitting is
      the choice math-capable models make deliberately, so this is cheaper
      context, not better arithmetic.

      **The number step 1 turns on:** one complete GSM8K episode
      (problem + CoT + `#### answer`) in the SFT chat format is **147 tokens**
      of 512. So:

      - zero-shot, one episode — 147/512, fits
      - 1-shot + generation — 294/512, fits
      - 3-shot + generation — 588/512, **overflows**
      - 5-shot + generation — 882/512, **overflows**

      So the roadmap's original claim ("a reasoning episode does not fit today")
      is too strong: an *average* episode fits zero-shot. What does not fit is
      few-shot prompting, harder problems with 6-8 CoT steps, or any headroom at
      all. Still a blocker for step 1, for a more precise reason.

      Side finding: the 8k `tiny_pirate_stories` tokenizer (Sloop) is much worse
      — 2.77x on CoT, 4.10x on arithmetic, 99% single-character tokens. Do not
      reuse it for anything numeric.

      **Conclusion: do not retrain the tokenizer for math.** Spend the effort on
      `block_size` (step 1) and the data mix (step 2). Revisit only if step 4
      shows arithmetic errors clustering on multi-digit intermediates, which
      would argue for per-digit splitting like Qwen3 — at a further ~1.5x
      context cost.
- [ ] **1. Extend context past `block_size=512`.** Hard blocker, quantified by
      step 0: one GSM8K episode is 147 tokens, so 512 holds a zero-shot episode
      and nothing else — no few-shot prefix, no long CoT, no headroom. 2048
      buys a 5-shot prompt with room to spare. Three places already work around
      this ceiling — `sample.chat_repl`, `chat/server.trim_turns` and
      `rejection/selfplay` all drop the oldest turns to fit. RoPE is already in, so: bump
      `block_size` to 2048-4096, optionally rebase `rope_theta` to ~10k-50k if
      going long, and continue pretraining at the new length.
- [ ] **2. Math/code mid-training (~1-2B tokens).** Current mix is TinyStories +
      Cosmopedia wikihow/stories + 5 Gutenberg books — narrative only, and even
      the Cosmopedia slices are piratized. RL *elicits* reasoning, it does not
      instil it; the model must see math first. Mix: FineWeb-Edu slice + a math
      corpus (FineMath / OpenWebMath) + some code.
      **Do not piratize the math data** — keep the pirate persona for chat only;
      "x² + 3x = 4" in dialect burns tokens and confuses the model.
- [ ] **3. CoT distillation SFT.** The realistic path at 358M: SFT on existing
      traces from big reasoning models, not RL-from-scratch. A few-thousand-step
      run on a filtered subset (MetaMathQA, OpenR1-Math-220k, OpenThoughts),
      restricted to problems the model can plausibly do — single/double-step
      arithmetic word problems. Teaches the *format*: think → then answer.
- [ ] **4. GRPO with verifiable reward.** Only after step 3. Write the loop in
      nanobeard — it is small: sample n completions per prompt, score by exact
      match against the GSM8K answer (no reward model needed — this is why
      verifiable-reward RL is the budget choice), normalize rewards within the
      group for advantages, policy-gradient step KL-anchored to a frozen
      reference copy. `sample.py` is the head start; missing piece is the
      sampling-and-scoring harness.

## Removed 2026-09-10 (recoverable from git)

- **`src/nanobeard/eval/`** + `src/tests/test_eval.py` + `make eval` /
  `make eval-quick` — perplexity and sample-gallery harness. Removed on request:
  it never produced a committed result, so there was nothing to regress against.
  If eval returns it should aim at the reasoning benchmarks above (exact-match
  on GSM8K-style answers), not perplexity on `val.bin`.

## Nice-to-have, when scale demands

- [ ] Ckpt migration script (drop the `training.config` shim long-term).
- [ ] Sample regression test — store golden samples per release tag.
- [ ] Cheapest-provider sweep across RunPod too. `make vast-offers` ranks bid
      floors across GPU types on vast; RunPod spot is the reliability hedge and
      is not covered.
- [ ] DVC or hash-based data versioning for the bins.
- [ ] HF Hub model-card auto-gen from `training_metadata.json`. `hf/model_card.md`
      now exists but is hand-written and nothing in `hf/*.py` reads it or
      `training_metadata.json` — so the numbers in it drift silently per release.
