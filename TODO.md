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

- [ ] **0. Tokenizer fertility check (free, 10 min).** Run math/code text through
      `pirate_bpe.json` and measure tokens-per-char vs plain English. 16k vocab
      trained on story text may tokenize numbers/symbols badly, which silently
      shrinks effective context. Do this before committing to a context length.
- [ ] **1. Extend context past `block_size=512`.** Hard blocker: a CoT trace for
      even an easy GSM8K problem is 300-800 tokens and the prompt eats the rest.
      A reasoning episode does not fit today. Three places already work around
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
- [ ] Docker image pinned for Vast.ai (CUDA + torch versions).
- [ ] DVC or hash-based data versioning for the bins.
- [ ] HF Hub model-card auto-gen from `training_metadata.json`. `hf/model_card.md`
      now exists but is hand-written and nothing in `hf/*.py` reads it or
      `training_metadata.json` — so the numbers in it drift silently per release.
