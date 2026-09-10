#!/usr/bin/env bash
# Run this once on a fresh Vast.ai instance to prepare it for training.
#
# Idempotent: safe to re-run. Reads CONFIG (default: sloop) and VARIANT
# (default: gpu) from env. Pulls dataset from HF instead of re-piratizing
# locally — saves ~30 minutes per cold start.
#
# Usage on the instance:
#   curl -fsSL https://raw.githubusercontent.com/younissk/pirate_llm/main/scripts/vast_bootstrap.sh | bash
#   # or, after cloning:
#   ./scripts/vast_bootstrap.sh

set -euo pipefail

CONFIG="${CONFIG:-sloop}"
VARIANT="${VARIANT:-gpu}"
DATASET="${DATASET:-tiny_pirate_stories}"
REPO_URL="${REPO_URL:-https://github.com/younissk/pirate_llm}"
REPO_DIR="${REPO_DIR:-$HOME/pirate_llm}"
DATA_HF_REPO="${DATA_HF_REPO:-younissk/nanobeard-data-${DATASET}}"

log() { echo -e "\033[1;34m[bootstrap]\033[0m $*"; }

# 1. System deps. Most CUDA images already have python + git.
# build-essential (gcc) is required: torch.compile's inductor/triton backend
# JIT-compiles CUDA kernels through a C compiler, which the pytorch *-runtime
# images do not ship. Without it, compile=True crashes at the first step.
log "Installing system tools"
if ! command -v git >/dev/null;  then apt-get update -y && apt-get install -y git curl; fi
if ! command -v tmux >/dev/null; then apt-get install -y tmux; fi
if ! command -v gcc  >/dev/null; then apt-get update -y && apt-get install -y build-essential; fi

# 2. uv (fast Python install).
if ! command -v uv >/dev/null; then
    log "Installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# 3. Clone repo.
if [ ! -d "$REPO_DIR/.git" ]; then
    log "Cloning $REPO_URL -> $REPO_DIR"
    git clone "$REPO_URL" "$REPO_DIR"
fi
cd "$REPO_DIR"
git pull --ff-only || true

# 4. Sync deps.
# Pin Python 3.12 explicitly: .python-version is gitignored, so a fresh box
# would otherwise let uv grab the newest interpreter (3.14), which drags in a
# different torch build than the locally-tested 3.12 environment.
log "uv sync (python 3.12)"
uv python install 3.12
uv sync --no-dev --python 3.12

# 5. Pull dataset from HF Hub (instead of running the full pipeline).
DATA_DIR="data/datasets/$DATASET"
log "Pulling dataset $DATA_HF_REPO -> $DATA_DIR/"
mkdir -p "$DATA_DIR"
if [ -n "${HF_TOKEN:-}" ]; then
    HF_TOKEN_FLAG="--token $HF_TOKEN"
else
    HF_TOKEN_FLAG=""
fi
# shellcheck disable=SC2086
uv run hf download "$DATA_HF_REPO" \
    --repo-type dataset \
    --local-dir "$DATA_DIR" \
    $HF_TOKEN_FLAG || {
    log "Dataset $DATA_HF_REPO not on Hub yet — fall back to local build from recipe"
    uv run python -m nanobeard.dataset_pipeline.build --dataset "$DATASET"
}

# 6. Resume training in a detachable tmux session.
# VARIANT=sft runs the supervised-finetuning entrypoint (loads the pretrained
# ckpt named by config.pretrained_ckpt_repo); anything else is pretraining.
SESSION="nanobeard-$CONFIG"
if [ "$VARIANT" = "sft" ]; then ENTRY="nanobeard.sft"; else ENTRY="nanobeard.train"; fi
log "Starting $ENTRY in tmux session: $SESSION"
log "  Reattach with:  tmux attach -t $SESSION"
log "  Detach with:    Ctrl-b d"

mkdir -p "runs/$CONFIG"
tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" \
    "cd $REPO_DIR && CONFIG_VARIANT=$VARIANT uv run python -m $ENTRY --config configs/$CONFIG.py 2>&1 | tee runs/$CONFIG/train.log"

log "Done. Training is running in tmux ($SESSION)."
log "Checkpoints will roll to HF if hf_ckpt_repo is set in $CONFIG.py."
