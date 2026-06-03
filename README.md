![nanoBeard Banner](./banner.png)

# nanoBeard ☠️

Tiny pirate-themed language models, trained from scratch on piratized corpora and
SFT-tuned to chat — then exported to GGUF so they run **on-device** in the NanoBeard
mobile app. Closer to nanoGPT than to a production LM; built as a learning project.

One codebase, **multiple ship-class versions** (each a config + an entry in the model
registry). Two architectures: a classic GPT-2 (Sloop, Galleon) and a modern v2 stack
(Frigate — RoPE + SwiGLU + RMSNorm + per-head QK-norm, trained with Muon).

| Class | Params | Arch | Trained on | Mode | Val loss | Weights | GGUF (on-device) |
|---|---|---|---|---|---|---|---|
| **Sloop** | 14M | GPT-2 | tiny_pirate_stories | completion | 4.28 | [`nanoBeard-sloop-14M`](https://huggingface.co/younissk/nanoBeard-sloop-14M) | [`…-sloop-14M-GGUF`](https://huggingface.co/younissk/nanoBeard-sloop-14M-GGUF) |
| **Galleon** | 34M | GPT-2 (scaled) | pirate_enhanced (~1.55B tok) | chat SFT | 3.01 | [`nanoBeard-galleon-34M`](https://huggingface.co/younissk/nanoBeard-galleon-34M) | [`…-galleon-34M-GGUF`](https://huggingface.co/younissk/nanoBeard-galleon-34M-GGUF) |
| **Frigate 125M** | 125.9M | v2 | pirate_enhanced_full (~4.75B tok) | chat SFT | 2.88 | [`…-125M-full-ckpts`](https://huggingface.co/younissk/frigate-125M-full-ckpts) | [`…-frigate-125M-GGUF`](https://huggingface.co/younissk/nanoBeard-frigate-125M-GGUF) |
| **Frigate 360M** | 358.3M | v2 | pirate_enhanced_full (~4.75B tok) | chat SFT | 3.04 | [`…-360M-full-ckpts`](https://huggingface.co/younissk/frigate-360M-full-ckpts) | [`…-frigate-360M-GGUF`](https://huggingface.co/younissk/nanoBeard-frigate-360M-GGUF) |

- **Source code:** https://github.com/younissk/pirate_llm
- **Docs:** see `docs/` or run `make docs-serve` (architecture, eval, vast.ai, adding a model)

## Architectures

Two model families share the config-driven GPT in `nanobeard/models/`, dispatched by
`model_name` into `MODEL_REGISTRY`:

**Sloop / Galleon (`sloop` arch)** — classic GPT-2: learned position embeddings,
4× GELU MLP, LayerNorm, `nn.MultiheadAttention`, tied embeddings, no biases. Galleon is
the same architecture scaled up (8L / 8H / 512d) to exploit a bigger corpus.

**Frigate (`frigate` arch, v2)** — a deeper, thinner stack with four modern upgrades,
each gated by a `Config` flag in `nanobeard/models/frigate.py`:

| Flag | Upgrade |
|---|---|
| `use_rope` | RoPE rotary position embedding (replaces the learned `wpe` table) |
| `use_swiglu` | SwiGLU FFN (8/3 expansion) instead of the 4× GELU MLP |
| `use_rmsnorm` | RMSNorm instead of LayerNorm |
| `use_qk_norm` | per-head RMSNorm on Q and K — stabilizes the deep stack |

This combination (no biases, tied embeddings, head_dim 64) is **byte-for-byte a Qwen3
model**, which is what makes the GGUF export trivial (see below). Frigate is trained
with **Muon** (`nanobeard/optim.py`): the 2D hidden matmul weights get an orthogonalized
momentum step via Newton-Schulz; embeddings, the LM head, and norms fall back to AdamW —
all inside one optimizer so resume stays a single `state_dict`.

## On-device export (GGUF)

Every shipped model also exists as GGUF quants for [llama.cpp](https://github.com/ggml-org/llama.cpp)
and the **NanoBeard mobile app** (which downloads files straight from the HF `resolve/main`
URLs). Two quants per model: `Q4_K_M` (default — smallest/fastest for phones) and `Q8_0`
(near-lossless fallback).

`scripts/export_gguf.py` re-emits checkpoint weights as a HF folder, then leans on the
upstream `convert_hf_to_gguf.py` — Frigate → `Qwen3ForCausalLM` (`qwen3` arch), Sloop/Galleon
→ `GPT2LMHeadModel` (`gpt2` arch). No bespoke runtime needed. `scripts/publish_gguf.py`
uploads the quants + a generated model card.

```bash
# Frigate checkpoint -> GGUF (Q4_K_M + Q8_0)
uv run python -m scripts.export_gguf \
  --ckpt runs/frigate-125m-full-sft/sft_ckpt.pt \
  --tokenizer data/datasets/pirate_enhanced_full/pirate_bpe.json \
  --name frigate-125M --out-dir export/gguf/frigate-125m

# publish to a public HF GGUF repo (dry-run without --push)
uv run python -m scripts.publish_gguf \
  --gguf-dir export/gguf/frigate-125m \
  --repo younissk/nanoBeard-frigate-125M-GGUF \
  --title "nanoBeard Frigate 125M (pirate chat)" \
  --params 125.9M --val-loss 2.882 \
  --base-model younissk/nanoBeard-frigate-125M-full --push
```

## Layout

```
nanobeard/             # package: training, SFT, sampling, eval, publish, Muon optimizer
  models/              # one file per architecture (sloop, frigate), MODEL_REGISTRY
  eval/                # perplexity + sample-gallery harness
  dataset_pipeline/    # sources (piratized corpora) + recipe-driven dataset builds
configs/               # one .py per ship-class; CONFIG_VARIANT=smoke|gpu|sft
scripts/               # GGUF/ONNX export, GGUF/space publish, vast.ai provisioning
data/sources/<name>/   # reusable corpora (cached arrow + source.json)
data/datasets/<name>/  # composed datasets: recipe.json + tokenizer + bins + metadata
export/gguf/<model>/   # GGUF quants + generated model card
runs/<version>/        # per-version checkpoints
evals/                 # prompt set + per-eval reports
space/                 # Gradio playground (multi-version dropdown)
tests/                 # pytest suite (parametrized over MODEL_REGISTRY)
docs/                  # mkdocs-material site
```

## Data

Datasets are composed from reusable **sources** via a `recipe.json` (equal-weight
concatenate, or weighted interleave); the tokenizer + `train.bin`/`val.bin` + metadata
are built from the recipe. To skip the ~30-min piratize on a remote GPU box, push/pull
the built dataset to/from HF (`make push-data`).

| Dataset | Vocab | Train tokens | Sources |
|---|---|---|---|
| `tiny_pirate_stories` | 8192 | — | piratized [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) |
| `pirate_enhanced` | 16384 | ~1.55B | TinyStories + cosmopedia (wikihow, stories) piratized + Gutenberg books (plain) |
| `pirate_enhanced_full` | 16384 | ~4.75B | full cosmopedia (stories + stanford + wikihow) + TinyStories, all piratized; + Wikipedia pirate articles + Gutenberg pirate books (both **not** piratized, for authentic plain-English exposure) |

Piratization is done with [`arrr`](https://pypi.org/project/arrr/) in
`nanobeard/dataset_pipeline/piratize.py`.

**SFT (chat)** uses a plaintext `User:/Pirate:` transcript with loss masked on everything
but the bot replies:
- `TeeZee/dolly-15k-pirate-speech` — single-turn instruction following (responses
  re-piratized through `arrr` for a consistent voice).
- `Estwld/empathetic_dialogues_llm` — multi-turn chit-chat; teaches turn-taking and
  short-range memory.

## Quick start

```bash
make install                       # uv sync
uv sync --dev                      # dev tooling (pytest, ruff, mypy, mkdocs)
make env                           # .env from example
pre-commit install                 # format/lint on commit

make dataset DATASET=pirate_enhanced_full      # build -> data/datasets/<name>/
make train   CONFIG=frigate_125m_full          # local smoke (CONFIG_VARIANT=smoke)
make train   CONFIG=frigate_125m_full CONFIG_VARIANT=gpu   # full GPU run
make sft     CONFIG=frigate_125m_full          # chat SFT on the pretrained ckpt
make sample  CONFIG=frigate_125m_full PROMPT='Ahoy matey'
make eval    CONFIG=frigate_125m_full          # perplexity + gallery
make publish CONFIG=frigate_125m_full          # push weights to its HF repo
```

Configs: `sloop`, `galleon`, `frigate`, `frigate_125m_full`, `frigate_360m_full`. Each
exposes `CONFIG_VARIANT=smoke|gpu|sft`. `epochs` (not `max_iters`) sets the real training
horizon — `resolve_max_iters()` derives the iteration count from the actual `train.bin`
size at run start.

### Training on cheap GPUs (vast.ai)

`scripts/vast_launch.sh` provisions a spot GPU (requires CUDA host ≥ 12.9, pins Python
3.12, installs build tooling for `torch.compile`), pulls the prebuilt dataset from HF,
and dispatches to pretraining or SFT. See `docs/vast-ai.md`.

```bash
make vast-launch CONFIG=frigate_360m_full
make vast-logs
make vast-destroy
```

## Tests

```bash
make test            # fast
make test-all        # include slow integration
```

- `tests/test_model_contract.py` — **parametrized over `MODEL_REGISTRY`**, so every
  architecture is checked for the same invariants (causal mask, weight tying, shape
  contracts, finite grads) automatically.
- `tests/test_publish.py` — HF Hub mocked; verifies `config.json` carries the model
  identity + arch fields.
- `tests/test_tokenizer_hash.py` — tokenizer fingerprint gated into every checkpoint to
  catch silent tokenizer drift.

## Loading a published model

nanoBeard models are **not** `transformers` models — load via the `nanobeard` package
(weights repos), or just grab the GGUF for llama.cpp / the mobile app.

```python
import json, torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_model
from tokenizers import Tokenizer

from nanobeard.config import Config
from nanobeard.models import build_model

repo = "younissk/nanoBeard-sloop-14M"
cfg_dict = json.load(open(hf_hub_download(repo, "config.json")))
cfg = Config(**{k: v for k, v in cfg_dict.items() if k in Config.__dataclass_fields__})
model = build_model(cfg).eval()
load_model(model, hf_hub_download(repo, "model.safetensors"))

tok = Tokenizer.from_file(hf_hub_download(repo, "pirate_bpe.json"))
ids = torch.tensor([tok.encode("Once upon a time").ids])
with torch.no_grad():
    for _ in range(80):
        logits, _ = model(ids[:, -cfg.block_size:])
        next_id = torch.multinomial(torch.softmax(logits[:, -1] / 0.8, -1), 1)
        ids = torch.cat([ids, next_id], dim=1)
print(tok.decode(ids[0].tolist()))
```

GGUF (chat models use plain `User:/Pirate:` turns; stop at `<|endoftext|>`):

```bash
llama-completion -m frigate-125M.Q4_K_M.gguf \
  -p $'User: Tell me about the sea.\nPirate:' -n 80 --temp 0.8 --top-k 40
```

## Adding a new model version

See `docs/adding-a-model.md`. TL;DR:

1. Write `nanobeard/models/<key>.py` (new arch, frozen contract) — or reuse an existing one.
2. Register a `ModelSpec` in `nanobeard/models/__init__.py`.
3. Drop a config in `configs/<key>.py`.
4. `make dataset DATASET=<name>` / `make train CONFIG=<key>`.
5. `make test` — the contract suite parametrizes over the registry automatically.
6. `make publish CONFIG=<key>`, then `scripts/export_gguf.py` + `scripts/publish_gguf.py`.

## Limitations

- Trained on (mostly) synthetic pirate corpora. Vocabulary, grammar, and world knowledge
  are narrow; the chat models are coherent-ish but small.
- Short context (256 tokens for Sloop; 512 for Galleon/Frigate).
- No safety tuning. Pirate-flavored fun, not a useful assistant.
- Educational artifact.

## Data licenses

- [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) — CDLA-Sharing-1.0
- [HuggingFaceTB/cosmopedia](https://huggingface.co/datasets/HuggingFaceTB/cosmopedia)
- Project Gutenberg books (public domain) and English Wikipedia (CC BY-SA) for plain-language exposure
- SFT: `TeeZee/dolly-15k-pirate-speech`, `Estwld/empathetic_dialogues_llm`
