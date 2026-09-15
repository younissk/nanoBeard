.PHONY: help install env source prepare dataset train train-gpu sft sample chat fertility autobatch evals evals-diff distill lora lora-merge rl-corpus rl-train \
        publish publish-space export-gguf publish-gguf push-data serve rejection view judge judge-all analyze selfplay \
        vast-offers vast-launch vast-watch vast-ssh vast-logs vast-destroy \
        test test-all test-fast test-slow lint format typecheck clean clean-data clean-ckpt

UV ?= uv
# Longer HF download timeout — the default is short and trips on slow shards /
# brief network stalls during the multi-hour data build (esp. Wikipedia).
HF_HUB_DOWNLOAD_TIMEOUT ?= 60
export HF_HUB_DOWNLOAD_TIMEOUT
CONFIG ?= sloop
CONFIG_FILE = configs/$(CONFIG).py
DATASET ?= tiny_pirate_stories
DATA_DIR = data/datasets/$(DATASET)
SOURCE ?= tiny_stories_pirate
PROMPT ?= Once upon a time

help:
	@echo "nanoBeard make targets — pass CONFIG=<name> (default: sloop)"
	@echo ""
	@echo "  make install                Install Python deps via uv"
	@echo "  make env                    Copy envs/example.env -> envs/.env"
	@echo ""
	@echo "  make source SOURCE=<name>   Build+cache one source -> data/sources/<name>/"
	@echo "  make prepare DATASET=<name> Download+cache all of a recipe's sources (no tokenizer/bins)"
	@echo "  make dataset DATASET=<name> Build a dataset from its recipe.json:"
	@echo "                              combine sources -> tokenizer + train.bin/val.bin + metadata"
	@echo "                              -> $(DATA_DIR)/"
	@echo ""
	@echo "  make train CONFIG=$(CONFIG) CONFIG_VARIANT=smoke|gpu"
	@echo "                              Train model (default: smoke variant)"
	@echo "  make sft                    SFT a pretrained ckpt"
	@echo "  make sample PROMPT='Ahoy'   Generate from runs/$(CONFIG)/ckpt.pt"
	@echo "  make chat                   Browser chat UI over every exported GGUF"
	@echo "  make fertility              Tokens/char by domain (REFERENCE=Qwen/Qwen3-0.6B)"
	@echo "  make autobatch              Measure tok/s vs micro-batch on this GPU"
	@echo "  make distill DISTILL_N=40   Generate pirate SFT data with the Kimi teacher"
	@echo "  make rl-corpus              Build the HotpotQA search index"
	@echo "  make rl-train               GRPO on search (answer exact-match reward)"
	@echo "  make lora                   LoRA fine-tune Qwen3-0.6B on that data"
	@echo "  make lora-merge GGUF=1      Merge adapter -> HF -> GGUF"
	@echo "  make vast-watch             Wait for the run, fetch results, destroy the box"
	@echo "  make evals [PERSONA=1]      gsm8k + tool-calling + pirate-voice gates"
	@echo "  make evals-diff A=.. B=..   Compare two eval reports"
	@echo ""
	@echo "  make publish                Push CONFIG ckpt to its HF model repo"
	@echo "  make publish-space          Push playground Space"
	@echo "  make export-gguf            Frigate ckpt -> GGUF (needs local llama.cpp)"
	@echo "  make publish-gguf PUSH=1    Push a GGUF dir to HF (dry-run without PUSH)"
	@echo ""
	@echo "  make serve                  Serve GGUF_MODEL via llama-server (blocks)"
	@echo "  make rejection RS_N=30      N candidates per prompt -> $(RS_OUT)"
	@echo "  make view                   Build + open the HTML review UI"
	@echo "  make judge                  Local judge endpoint for the ask-qwen button"
	@echo "  make judge-all JUDGE_VOTES=3  Batch-judge all prompts (shuffled order)"
	@echo "  make analyze                Diversity + judge-bias diagnostics"
	@echo "  make selfplay               Qwen-driven conversations -> $(SP_OUT)"
	@echo ""
	@echo "  make test                   Fast tests (excludes slow marker)"
	@echo "  make test-slow              Slow integration tests only"
	@echo "  make test-all               Everything"
	@echo "  make lint / format / typecheck"
	@echo ""
	@echo "  make clean                  Remove caches + wandb dir"
	@echo "  make clean-ckpt             Remove runs/$(CONFIG)/"
	@echo "  make clean-data             Remove $(DATA_DIR)/ contents"

install:
	$(UV) sync

env:
	@if [ -f envs/.env ]; then \
		echo "envs/.env already exists — leaving alone"; \
	else \
		cp envs/example.env envs/.env && echo "Created envs/.env from envs/example.env — fill in your tokens"; \
	fi

# ----- Data pipeline -----
# Sources are reusable corpora; datasets compose them via data/datasets/<name>/recipe.json.

source:
	$(UV) run python -m nanobeard.dataset_pipeline.sources --source $(SOURCE) $(if $(FORCE),--force,)

prepare:
	$(UV) run python -m nanobeard.dataset_pipeline.build --dataset $(DATASET) --sources-only $(if $(FORCE),--force,)

dataset:
	$(UV) run python -m nanobeard.dataset_pipeline.build --dataset $(DATASET)

# ----- Training -----

train:
	$(UV) run python -m nanobeard.train --config $(CONFIG_FILE)

train-gpu:
	CONFIG_VARIANT=gpu $(UV) run python -m nanobeard.train --config $(CONFIG_FILE)

sft:
	CONFIG_VARIANT=sft $(UV) run python -m nanobeard.sft --config $(CONFIG_FILE)

# ----- Sampling -----

sample:
	$(UV) run python -m nanobeard.sample --config $(CONFIG_FILE) --prompt "$(PROMPT)"

# ----- Publishing -----

publish:
	$(UV) run python -m nanobeard.publish --config $(CONFIG_FILE)

publish-space:
	$(UV) run python hf/publish_space.py

# ----- Dataset to/from HF Hub (skips 30-min piratize on remote machines) -----

push-data:
	$(UV) run python -m nanobeard.dataset_pipeline.push_to_hf --data-dir $(DATA_DIR)

# ----- Vast.ai -----

# Ranked board of the cheapest bid offers across candidate GPUs. Read-only —
# creates nothing. Prices move by the minute, so check before a long run.
VAST_GPUS ?= RTX_4090,RTX_5090,RTX_3090
VAST_MAX_DPH ?= 0.40

vast-offers:
	$(UV) run python -m nanobeard.vast_offers --gpus $(VAST_GPUS) --max-dph $(VAST_MAX_DPH) --board

vast-launch:
	CONFIG=$(CONFIG) ./scripts/vast/vast_launch.sh

# Watch an instance, fetch results, then DESTROY it. Always destroys — on
# success, on timeout, and on Ctrl-C — because a stopped instance still bills.
VAST_TIMEOUT_MIN ?= 60

# setsid: the watchdog must outlive the shell that starts it. Started as a plain
# background job it stays in the caller's process group, so killing that shell
# signals the watchdog too — whose EXIT trap then destroys a perfectly healthy
# instance mid-bootstrap. Observed exactly that.
vast-watch:
	setsid nohup ./scripts/vast/vast_watch.sh $$(cat .vast_instance) \
		--timeout-min $(VAST_TIMEOUT_MIN) \
		--fetch /root/pirate_llm/runs/lora runs/ > runs/vast_watch.log 2>&1 &
	@echo "watchdog detached; tail runs/vast_watch.log"

vast-ssh:
	@INSTANCE=$$(cat .vast_instance 2>/dev/null) && vastai ssh-url $$INSTANCE

vast-logs:
	@INSTANCE=$$(cat .vast_instance 2>/dev/null) && vastai logs $$INSTANCE

vast-destroy:
	./scripts/vast/vast_destroy.sh

# ----- Eval -----

# ----- GGUF export (on-device llama.cpp builds) -----
# Requires a local llama.cpp clone; override with LLAMA_CPP=<path>.

GGUF_CKPT ?= runs/$(CONFIG)/sft_ckpt.pt
GGUF_TOKENIZER ?= $(DATA_DIR)/pirate_bpe.json
GGUF_NAME ?= $(CONFIG)
GGUF_OUT ?= export/gguf/$(CONFIG)
GGUF_QUANTS ?= Q4_K_M Q8_0

export-gguf:
	$(UV) run python hf/export_gguf.py \
		--ckpt $(GGUF_CKPT) --tokenizer $(GGUF_TOKENIZER) \
		--name $(GGUF_NAME) --out-dir $(GGUF_OUT) --quants $(GGUF_QUANTS) \
		$(if $(LLAMA_CPP),--llama-cpp $(LLAMA_CPP),) \
		$(if $(CONVERTER_PYTHON),--converter-python $(CONVERTER_PYTHON),)

# Dry-run by default. Add PUSH=1 to actually upload.
publish-gguf:
	$(UV) run python hf/publish_gguf.py \
		--gguf-dir $(GGUF_OUT) --repo $(GGUF_REPO) --title $(GGUF_TITLE) \
		--params $(GGUF_PARAMS) --val-loss $(GGUF_VAL_LOSS) \
		--base-model $(GGUF_BASE_MODEL) $(if $(PUSH),--push,)

# ----- Teacher data generation -----
# Kimi writes the pirate SFT data. Thinking is OFF by default: measured 5.2x
# fewer output tokens, better yield (40/40 vs 38/40) and no loss of voice.
# Always run a small --n first and read it before spending on the full set.

DISTILL_N ?= 40
DISTILL_OUT ?= runs/distill/sample.jsonl

distill:
	$(UV) run python -m nanobeard.distill.generate --out $(DISTILL_OUT) --n $(DISTILL_N) \
		$(if $(THINKING),--thinking,)

# ----- RL with verifiable rewards (search) -----
# The reward is answer exact-match, which cannot be faked. Retrieval precision
# and recall are logged as diagnostics only — a query of "the" retrieves
# everything and would score perfect recall.

RL_INDEX ?= data/search/hotpot_bm25.pkl
RL_OUT ?= runs/rl/search-v1
RL_STEPS ?= 50
RL_GROUP ?= 8

# One-off: build the searchable corpus and print the retrieval baseline to beat.
rl-corpus:
	$(UV) run python -m nanobeard.rl.corpus --max-questions $(or $(RL_QUESTIONS),2000)

rl-train:
	$(UV) run --group finetune python -m nanobeard.rl.grpo \
		--index $(RL_INDEX) --out $(RL_OUT) --steps $(RL_STEPS) --group-size $(RL_GROUP) \
		$(if $(RL_ADAPTER),--adapter $(RL_ADAPTER),) $(if $(DEVICE),--device $(DEVICE),)

# ----- LoRA fine-tune -----
# Needs the finetune dependency group: `uv sync --group finetune`.
# Run `make evals` before and after. Ship only if voice went up and gsm8k,
# tool choice and restraint all held.

LORA_DATA ?= runs/distill/train.jsonl
LORA_OUT ?= runs/lora/pirate-v1
LORA_RANK ?= 16
LORA_EPOCHS ?= 2

lora:
	$(UV) run --group finetune python -m nanobeard.finetune.train \
		--data $(LORA_DATA) --out $(LORA_OUT) --rank $(LORA_RANK) --epochs $(LORA_EPOCHS) \
		$(if $(DEVICE),--device $(DEVICE),) $(if $(LIMIT),--limit $(LIMIT),)

# Fold the adapter into the base weights; llama.cpp cannot load PEFT adapters.
lora-merge:
	$(UV) run --group finetune python -m nanobeard.finetune.merge \
		--adapter $(LORA_OUT) $(if $(GGUF),--gguf,)

# ----- Capability gates -----
# Run before AND after every fine-tune. Voice going up is not a result; voice
# going up while gsm8k and tool-calling hold is.

EVAL_MODEL ?= export/gguf/qwen3-0.6b/Qwen3-0.6B-Q4_K_M.gguf
EVAL_N ?= 200
EVAL_WORKERS ?= 6

evals:
	$(UV) run python -m nanobeard.evals.run --model $(EVAL_MODEL) \
		--n-gsm8k $(EVAL_N) --workers $(EVAL_WORKERS) \
		$(if $(PERSONA),--persona,) $(if $(LABEL),--label $(LABEL),)

# make evals-diff A=runs/evals/stock.json B=runs/evals/lora.json
evals-diff:
	$(UV) run python -m nanobeard.evals.run --compare $(A) $(B)

# ----- Throughput tuning -----
# Micro-batch is the last throughput lever after bf16 + flash + compile, and it
# is the one that can only be measured on the card you rented. Run this once on
# a fresh box before starting a long run.

AUTOBATCH_VARIANT ?= gpu
AUTOBATCH_LIMIT ?= 512

autobatch:
	$(UV) run python -m nanobeard.autobatch \
		--config $(CONFIG_FILE) --variant $(AUTOBATCH_VARIANT) --limit $(AUTOBATCH_LIMIT)

# ----- Tokenizer diagnostics -----
# Tokens-per-character by domain. REFERENCE=<hf-repo> adds a control tokenizer
# (downloads once) so you can tell "our tokenizer is bad at math" apart from
# "math is denser than prose".

REFERENCE ?=

fertility:
	$(UV) run python -m nanobeard.fertility $(if $(REFERENCE),--reference $(REFERENCE),)

# ----- Chat playground -----
# Browser UI over the exported GGUFs. Spawns its own llama-server on
# CHAT_LLAMA_PORT (not RS_PORT — so this never collides with `make serve`).
# ATTACH=<url> reuses an already-running one instead.

CHAT_PORT ?= 8800
CHAT_LLAMA_PORT ?= 8901
CHAT_GGUF_ROOT ?= export/gguf

# API=completion for the frigate line (raw SFT transcript); the default `chat`
# uses /v1/chat/completions so llama-server applies the GGUF's own template,
# which is what carries Qwen3's tool-call and think blocks.
CHAT_API ?= chat

chat:
	$(UV) run python -m nanobeard.chat.server \
		--port $(CHAT_PORT) --llama-port $(CHAT_LLAMA_PORT) \
		--gguf-root $(CHAT_GGUF_ROOT) --api $(CHAT_API) \
		$(if $(GGUF_MODEL_PICK),--model $(GGUF_MODEL_PICK),) \
		$(if $(ATTACH),--attach $(ATTACH),)

# ----- Rejection sampling -----
# Two steps: serve the GGUF, then generate N candidates per prompt.
# Needs llama.cpp on PATH (brew install llama.cpp) and a local GGUF build.

GGUF_MODEL ?= export/gguf/frigate-360m/frigate-360M.Q8_0.gguf
RS_PORT ?= 8899
RS_SLOTS ?= 8
RS_N ?= 30
RS_OUT ?= runs/rejection/$(notdir $(basename $(GGUF_MODEL))).jsonl

# --cache-ram is the one that matters: it defaults to 8192 MiB, which on a
# small machine lets the prompt cache grow until the OS kills the server
# mid-run. Cap it well under available RAM.
RS_CACHE_RAM ?= 512

serve:
	llama-server -m $(GGUF_MODEL) --host 127.0.0.1 --port $(RS_PORT) \
		-c 4096 -np $(RS_SLOTS) --cache-ram $(RS_CACHE_RAM) --no-webui

# RESUME=1 keeps existing rows and fills only the gaps.
rejection:
	$(UV) run python -m nanobeard.rejection.generate \
		--out $(RS_OUT) --n $(RS_N) --workers $(RS_SLOTS) \
		--server http://127.0.0.1:$(RS_PORT) $(if $(RESUME),--resume,)

# Local judge endpoint for the viewer's "ask qwen" button. Holds
# OPENROUTER_API_KEY server-side — it never enters the HTML.
JUDGE_MODEL ?= qwen/qwen3-30b-a3b-instruct-2507
# Candidate order is shuffled every call (fixes a measured primacy bias).
# JUDGE_VOTES>1 re-judges with a fresh order each round and takes the majority.
JUDGE_VOTES ?= 1

judge:
	$(UV) run python -m nanobeard.rejection.judge --serve --model $(JUDGE_MODEL) --votes $(JUDGE_VOTES)

# Batch-judge every prompt and bake the verdicts into the next `make view`.
judge-all:
	$(UV) run python -m nanobeard.rejection.judge --all --in $(RS_OUT) --model $(JUDGE_MODEL) --votes $(JUDGE_VOTES)

# Diversity + judge-bias diagnostics. Run this before trusting any picks.
analyze:
	$(UV) run python -m nanobeard.rejection.analyze --in $(RS_OUT)

# Qwen drives the conversation, nanoBeard answers by rejection sampling.
# Needs `make serve` running (it talks to the same llama-server).
SP_OUT ?= runs/selfplay/$(notdir $(basename $(GGUF_MODEL))).jsonl
SP_CONVERSATIONS ?= 100

selfplay:
	$(UV) run python -m nanobeard.rejection.selfplay \
		--out $(SP_OUT) --conversations $(SP_CONVERSATIONS) --n $(RS_N)

view:
	$(UV) run python -m nanobeard.rejection.viewer --in $(RS_OUT)
	@open $(basename $(RS_OUT)).html 2>/dev/null || echo "open $(basename $(RS_OUT)).html"

# ----- Tests -----

test: test-fast

test-fast:
	$(UV) run pytest

test-slow:
	$(UV) run pytest -m slow

test-all:
	$(UV) run pytest -m "slow or not slow"

# ----- Lint / type-check -----

lint:
	$(UV) run ruff check .

format:
	$(UV) run ruff format .

# NOTE: currently RED — 8 pre-existing mypy errors in optim.py / sample.py.
# See TODO.md. Scope is deliberately nanobeard-only until that's green.
typecheck:
	$(UV) run mypy src/nanobeard
	$(UV) run pyright

# ----- Cleanup -----

clean:
	rm -rf wandb/ .pytest_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

clean-ckpt:
	rm -rf runs/$(CONFIG)/

clean-data:
	rm -rf $(DATA_DIR)/train.bin $(DATA_DIR)/val.bin $(DATA_DIR)/pirate_bpe.json $(DATA_DIR)/metadata.json
