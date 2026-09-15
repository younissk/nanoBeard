#!/usr/bin/env bash
# Run this once on a fresh Vast.ai instance to prepare it for training.
#
# Idempotent: safe to re-run. Reads CONFIG (default: sloop) and VARIANT
# (default: gpu) from env. Pulls dataset from HF instead of re-piratizing
# locally — saves ~30 minutes per cold start.
#
# Usage on the instance:
#   curl -fsSL https://raw.githubusercontent.com/younissk/nanoBeard/main/scripts/vast/vast_bootstrap.sh | bash
#   # or, after cloning:
#   ./scripts/vast/vast_bootstrap.sh

set -euo pipefail

CONFIG="${CONFIG:-sloop}"
VARIANT="${VARIANT:-gpu}"
DATASET="${DATASET:-tiny_pirate_stories}"
REPO_URL="${REPO_URL:-https://github.com/younissk/nanoBeard}"
# Branch/tag to train from. Without this the clone silently takes the default
# branch, which is how a box ends up running main's code against a branch's
# plan — the failure surfaces much later as a missing dependency group.
REPO_REF="${REPO_REF:-main}"
REPO_DIR="${REPO_DIR:-$HOME/pirate_llm}"
DATA_HF_REPO="${DATA_HF_REPO:-younissk/nanobeard-data-${DATASET}}"
# VARIANT=lora only.
LORA_DATA="${LORA_DATA:-runs/distill/train.jsonl}"
LORA_OUT="${LORA_OUT:-runs/lora/pirate-v1}"
LORA_EPOCHS="${LORA_EPOCHS:-2}"
LORA_RANK="${LORA_RANK:-16}"
# Push the finished adapter to the Hub. On a rented box this is the only
# reliable way to get it back — SSH through Vast's proxy is not dependable.
LORA_PUSH_REPO="${LORA_PUSH_REPO:-}"
# e.g. "--cap chat=400 --cap math=300 --cap tool_none=100". Balances the
# supervised-token budget, which example counts misrepresent.
LORA_CAPS="${LORA_CAPS:-}"
# VARIANT=rl only.
RL_OUT="${RL_OUT:-runs/rl/search-v1}"
RL_STEPS="${RL_STEPS:-150}"
RL_GROUP="${RL_GROUP:-8}"
RL_QPS="${RL_QPS:-16}"
RL_MAX_TOKENS="${RL_MAX_TOKENS:-160}"
RL_QUESTIONS="${RL_QUESTIONS:-4000}"
RL_ADAPTER="${RL_ADAPTER:-}"
RL_PUSH_REPO="${RL_PUSH_REPO:-}"
# Watchdog contract: this file appears exactly once, containing the exit status.
DONE_MARKER="${DONE_MARKER:-$REPO_DIR/.vast_done}"

log() { echo -e "\033[1;34m[bootstrap]\033[0m $*"; }

# Repair SSH before anything else. The Vast PyTorch images ship
# /root/.ssh/authorized_keys with permissions sshd refuses ("bad ownership or
# modes"), so every key is rejected — including Vast's own proxy. Without this
# there is no way to reach the box at all: `vastai execute` only works on
# stopped instances, and `vastai copy` to local was down for maintenance when
# this bit. Costs nothing and makes the instance reachable minutes earlier.
if [ -d /root/.ssh ]; then
    chown -R root:root /root/.ssh || true
    chmod 700 /root/.ssh || true
    chmod 600 /root/.ssh/authorized_keys 2>/dev/null || true
    log "repaired /root/.ssh permissions"
fi

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
    log "Cloning $REPO_URL ($REPO_REF) -> $REPO_DIR"
    git clone --branch "$REPO_REF" "$REPO_URL" "$REPO_DIR"
fi
cd "$REPO_DIR"
git fetch origin "$REPO_REF" --depth=50 || true
git checkout "$REPO_REF" 2>/dev/null || git checkout -B "$REPO_REF" "origin/$REPO_REF"
git reset --hard "origin/$REPO_REF" || true
log "on $(git rev-parse --abbrev-ref HEAD) @ $(git rev-parse --short HEAD)"

# 4. Sync deps.
# Pin Python 3.12 explicitly: .python-version is gitignored, so a fresh box
# would otherwise let uv grab the newest interpreter (3.14), which drags in a
# different torch build than the locally-tested 3.12 environment.
log "uv sync (python 3.12)"
uv python install 3.12
if [ "$VARIANT" = "lora" ] || [ "$VARIANT" = "rl" ]; then
    # Both fine-tuning paths need transformers/peft, deliberately absent from
    # the default install (see pyproject).
    uv sync --no-dev --group finetune --python 3.12
else
    uv sync --no-dev --python 3.12
fi

# 5. Pull dataset from HF Hub (instead of running the full pipeline).
# The LoRA trains on teacher-generated JSONL, which is committed to the repo and
# therefore already on disk after the clone — nothing to download.
if [ "$VARIANT" = "rl" ]; then
    # 19k paragraphs of pickle, rebuilt from HotpotQA in under a minute, so it
    # is gitignored and built here rather than shipped.
    log "VARIANT=rl: building the search index ($RL_QUESTIONS questions)"
    uv run python -m nanobeard.rl.corpus --max-questions "$RL_QUESTIONS"
elif [ "$VARIANT" = "lora" ]; then
    log "VARIANT=lora: training data is in the repo at $LORA_DATA"
    # LORA_DATA may name several files; test each rather than the whole string.
    for f in $LORA_DATA; do
        [ -f "$f" ] || { log "MISSING $f — is REPO_REF=$REPO_REF the right branch?"; exit 1; }
        log "  $f: $(wc -l < "$f") examples"
    done
else

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

fi

# 6. Resume training in a detachable tmux session.
# VARIANT=sft runs the supervised-finetuning entrypoint (loads the pretrained
# ckpt named by config.pretrained_ckpt_repo); anything else is pretraining.
SESSION="nanobeard-$CONFIG"
case "$VARIANT" in
    sft)  ENTRY="nanobeard.sft" ;;
    lora) ENTRY="nanobeard.finetune.train" ;;
    rl)   ENTRY="nanobeard.rl.grpo" ;;
    *)    ENTRY="nanobeard.train" ;;
esac
log "Starting $ENTRY in tmux session: $SESSION"
log "  Reattach with:  tmux attach -t $SESSION"
log "  Detach with:    Ctrl-b d"

mkdir -p "runs/$CONFIG"
tmux kill-session -t "$SESSION" 2>/dev/null || true
# Write a runner to disk rather than nesting a command string through
# launch -> onstart -> bootstrap -> tmux. Four layers of quoting is how these
# break, and the failure shows up as a box that bills while doing nothing.
RUNNER="$REPO_DIR/.vast_run.sh"
mkdir -p "$(dirname "$LORA_OUT")" "$(dirname "$RL_OUT")" "runs/$CONFIG"
{
    echo '#!/usr/bin/env bash'
    echo "cd $REPO_DIR"
    echo 'set -o pipefail'
    if [ "$VARIANT" = "rl" ]; then
        echo "uv run --group finetune python -m $ENTRY \\"
        echo "    --out $RL_OUT --steps $RL_STEPS --group-size $RL_GROUP \\"
        echo "    --questions-per-step $RL_QPS --max-new-tokens $RL_MAX_TOKENS \\"
        echo "    ${RL_ADAPTER:+--adapter $RL_ADAPTER} \\"
        echo "    ${RL_PUSH_REPO:+--push-to-hub $RL_PUSH_REPO} 2>&1 | tee $RL_OUT.log"
    elif [ "$VARIANT" = "lora" ]; then
        echo "uv run --group finetune python -m $ENTRY \\"
        echo "    --data $LORA_DATA --out $LORA_OUT \\"
        echo "    ${LORA_CAPS} \\"
        echo "    --epochs $LORA_EPOCHS --rank $LORA_RANK \\"
        echo "    ${LORA_PUSH_REPO:+--push-to-hub $LORA_PUSH_REPO} 2>&1 | tee $LORA_OUT.log"
    else
        echo "CONFIG_VARIANT=$VARIANT uv run python -m $ENTRY \\"
        echo "    --config configs/$CONFIG.py 2>&1 | tee runs/$CONFIG/train.log"
    fi
    # The status file is the watchdog's signal to fetch results and destroy the
    # box. Written whatever happens, so a crash shuts down as promptly as a
    # success — an instance that fails silently still bills by the second.
    echo 'STATUS=$?'
    echo "echo \$STATUS > $DONE_MARKER"
    echo 'echo "=== RUN FINISHED status=$STATUS ==="'
} > "$RUNNER"
chmod +x "$RUNNER"

tmux new-session -d -s "$SESSION" "$RUNNER"

log "Done. Training is running in tmux ($SESSION)."
log "Checkpoints will roll to HF if hf_ckpt_repo is set in $CONFIG.py."
