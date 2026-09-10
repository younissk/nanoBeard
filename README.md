# nanoBeard ☠️

A tiny pirate-themed GPT trained from scratch on a piratized version of
TinyStories, then SFT-tuned. Built as a learning project — closer to nanoGPT
than to a production LM.

The repo is structured for **multiple ship-class versions** under one codebase:
add a model file, register a spec, drop in a config. No caller changes.

| Codename | Params | Status |
|---|---|---|
| **Sloop** (v1) | ~13.8M | shipped |
| **Galleon** | ~33.8M | shipped |
| **Frigate** (v2) | ~126M / ~358M | shipped (GGUF for on-device) |

- **Site:** https://youniss.dev/nanoBeard/
- **Models:** [huggingface.co/younissk](https://huggingface.co/younissk)

## Layout

```
src/
  nanobeard/           # the package — training, sampling, SFT, publish
    models/            # one file per architecture, registered via MODEL_REGISTRY
    dataset_pipeline/  # sources (piratized corpora) + recipe-driven builds
    chat/              # local browser playground over the exported GGUFs
    rejection/         # rejection sampling, LLM judge, self-play, diagnostics
  tests/               # pytest suite (sibling of the package, not shipped)
configs/               # one .py per model version

web/                   # the GitHub Pages site (index.html, legal pages, assets)
hf/                    # everything that ships to Hugging Face
  model_card.md        #   the model-repo README (NOT this file)
  banner.png
  space/               #   Gradio playground
  export_gguf.py       #   ckpt -> GGUF for llama.cpp / the mobile app
  publish_gguf.py
  publish_space.py
scripts/vast/          # vast.ai launch / bootstrap / destroy

data/sources/<name>/   # reusable piratized corpora (cached arrow + source.json)
data/datasets/<name>/  # recipe.json + tokenizer + bins + metadata
runs/<version>/        # checkpoints
export/                # local GGUF builds (gitignored)
```

Two READMEs on purpose: **this file** is for GitHub, `hf/model_card.md` is what
`make publish` uploads to the model repo. They used to be the same file, which
meant every GitHub-facing edit also rewrote the model card.

## Quick start

```bash
make install                       # uv sync
uv sync --dev                      # dev tooling (pytest, ruff, mypy, pyright)
make env                           # .env from example
pre-commit install                 # format/lint on commit

make dataset DATASET=tiny_pirate_stories   # build -> data/datasets/tiny_pirate_stories/
make train   CONFIG=sloop          # local smoke
make train   CONFIG=sloop CONFIG_VARIANT=gpu   # GPU run
make sample  CONFIG=sloop PROMPT='Ahoy matey'
make publish CONFIG=sloop          # push to HF model repo
```

`make help` lists everything.

## Talking to a model

```bash
make chat                          # browser UI over every GGUF in export/gguf/
```

Discovers `export/gguf/**/*.gguf`, starts `llama-server` on the newest one and
opens a chat page at http://127.0.0.1:8800 with a dropdown to switch between
builds (switching restarts the server — llama.cpp serves one model per process).
Needs llama.cpp on PATH (`brew install llama.cpp`) and at least one
`make export-gguf` behind it.

- `ATTACH=http://127.0.0.1:8899 make chat` reuses a llama-server you already
  have running (e.g. from `make serve`) instead of starting its own.
- It renders the SFT transcript exactly as `nanobeard.rejection.generate` does.
  That matters: the prompt must end at `"Pirate: "` *including the trailing
  space*, or a working model looks broken.

## Tests

```bash
make test            # fast (~2s)
make test-all        # include slow integration
```

Highlights:
- `src/tests/test_model_contract.py` — **parametrized over `MODEL_REGISTRY`**, so any
  new architecture is automatically checked for the same invariants (causal mask,
  weight tying, shape contracts, finite grads).
- `src/tests/test_publish.py` — HF Hub mocked; verifies `config.json` carries
  `model_name`, `codename`, `display_name`, `num_parameters`, and all arch fields.
- `src/tests/test_tokenizer_hash.py` — tokenizer fingerprint in every ckpt.

`make lint` is green. **`make typecheck` is not** — see [TODO.md](TODO.md).

## Adding a new model version

1. Write `src/nanobeard/models/<key>.py` (new arch, frozen contract).
2. Register a `ModelSpec` in `src/nanobeard/models/__init__.py`.
3. Drop a config in `configs/<key>.py`.
4. `make dataset DATASET=<name>` / `make train CONFIG=<key>`.
5. `make test` — the contract suite parametrizes automatically.
6. `make publish CONFIG=<key>` to its own HF model repo.

## The site

`web/` is published to GitHub Pages by `.github/workflows/pages.yml` on every
push to `main`. Edit `web/index.html` directly — there's no site generator.
