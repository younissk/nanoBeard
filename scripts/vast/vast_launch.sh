#!/usr/bin/env bash
# Provision a Vast.ai instance and bootstrap it for nanoBeard training.
#
# Prereqs (local):
#   pip install vastai
#   vastai set api-key <your-key>             # one-time
#   set -a; source .env; set +a               # loads HF_TOKEN + WANDB_API_KEY
#   # or export them by hand:
#   export HF_TOKEN=<your-hf-token>
#   export WANDB_API_KEY=<your-wandb-key>     # optional; omit to skip wandb
#
# Usage:
#   ./scripts/vast/vast_launch.sh                  # default: interruptible RTX 4090
#   GPU=RTX_4090 CONFIG=sloop VARIANT=gpu ./scripts/vast/vast_launch.sh

set -euo pipefail

CONFIG="${CONFIG:-sloop}"
VARIANT="${VARIANT:-gpu}"
DATASET="${DATASET:-tiny_pirate_stories}"
GPU="${GPU:-RTX_4090}"
# -devel, not -runtime: it ships gcc/nvcc, which torch.compile's inductor
# backend needs to JIT kernels. CUDA 12.8 is also the floor for sm_120 (RTX
# 5090). The image torch is irrelevant — `uv sync` installs the pinned one.
IMAGE="${IMAGE:-pytorch/pytorch:2.11.0-cuda12.8-cudnn9-devel}"
DISK_GB="${DISK_GB:-50}"
# Interruptible (bid) instances run 35-70% under on-demand and can be evicted
# with no warning. That is a fair trade here because training checkpoints on a
# wall-clock cadence and resumes from hf_ckpt_repo, so an eviction costs
# minutes. INTERRUPTIBLE=0 falls back to on-demand for a run you cannot babysit.
INTERRUPTIBLE="${INTERRUPTIBLE:-1}"
# Price cap, and the bid when interruptible. Measured 4090 floors sit around
# $0.27-0.39/hr on-demand; bids clear well under that. Aggregator prices move
# fast (4090 floors shifted ~59% in a month) — re-check before a long run.
MAX_DPH="${MAX_DPH:-0.40}"
BID="${BID:-$MAX_DPH}"
# Datacenter hosts, not residential: bid instances get evicted either way, but
# DC hosts have the uplink to reload a checkpoint quickly afterwards.
DATACENTER="${DATACENTER:-1}"
INET_DOWN="${INET_DOWN:-200}"        # min Mbps
# Minimum host CUDA driver. The pinned torch (>=2.12) ships a cu12.9+ wheel that
# refuses to init on older drivers ("NVIDIA driver too old"), so reject hosts
# whose driver predates 12.9 up front instead of crashing after the dataset pull.
CUDA_VERS="${CUDA_VERS:-12.9}"
REPO_URL="${REPO_URL:-https://github.com/younissk/pirate_llm}"
# Branch/tag the instance pulls the bootstrap from. Override when testing a
# branch, or the box will run main's bootstrap against your branch's code.
REPO_REF="${REPO_REF:-main}"

log() { echo -e "\033[1;32m[vast]\033[0m $*"; }

command -v vastai >/dev/null || { echo "Install vast-cli: pip install vastai"; exit 1; }
[ -n "${HF_TOKEN:-}" ] || { echo "Set HF_TOKEN in env"; exit 1; }
[ -n "${WANDB_API_KEY:-}" ] || log "WANDB_API_KEY unset — training runs without wandb logging"

# 1. Find a cheap matching offer.
QUERY="gpu_name=$GPU num_gpus=1 dph_total<=$MAX_DPH inet_down>=$INET_DOWN reliability>=0.95 cuda_vers>=$CUDA_VERS"
[ "$INTERRUPTIBLE" = "1" ] && QUERY="$QUERY type=bid"
[ "$DATACENTER" = "1" ]    && QUERY="$QUERY datacenter=true"

log "Searching offers: $QUERY"
# `|| true`: a no-match must not abort under `set -e` (the python prints an
# empty id), so the friendly hint below can fire instead of dying silently.
OFFER=$(vastai search offers "$QUERY" \
    -o 'dph+' \
    --raw 2>/dev/null | python -c "import json,sys; offers=json.load(sys.stdin); print(offers[0]['id'] if offers else '')") || true

[ -n "$OFFER" ] || {
    echo "No offers matched: $QUERY"
    echo "Relax constraints, e.g.:"
    echo "  MAX_DPH=0.60 $0        # raise the cap"
    echo "  DATACENTER=0 $0        # allow residential hosts"
    echo "  INTERRUPTIBLE=0 $0     # on-demand instead of bid"
    echo "  GPU=RTX_3090 $0        # cheaper card"
    exit 1
}
log "Picked offer $OFFER"

# 2. Create the instance with --onstart so it bootstraps itself.
ONSTART=$(cat <<EOF
#!/bin/bash
set -e
export HF_TOKEN='$HF_TOKEN'
export WANDB_API_KEY='${WANDB_API_KEY:-}'
export CONFIG='$CONFIG'
export VARIANT='$VARIANT'
export DATASET='$DATASET'
curl -fsSL $REPO_URL/raw/$REPO_REF/scripts/vast/vast_bootstrap.sh | bash
EOF
)

PRICE_ARGS=()
if [ "$INTERRUPTIBLE" = "1" ]; then
    PRICE_ARGS=(--price "$BID")
    log "Creating INTERRUPTIBLE instance, bid \$$BID/hr"
else
    log "Creating on-demand instance"
fi

INSTANCE=$(vastai create instance "$OFFER" \
    --image "$IMAGE" \
    --disk "$DISK_GB" \
    --ssh \
    "${PRICE_ARGS[@]}" \
    --onstart-cmd "$ONSTART" \
    --raw 2>/dev/null | python -c "import json,sys; print(json.load(sys.stdin)['new_contract'])")

log "Created instance $INSTANCE"

# Verify the bid actually took. A silently-on-demand instance costs 2-4x what
# you planned and nothing else in this script would notice.
if [ "$INTERRUPTIBLE" = "1" ]; then
    ACTUAL=$(vastai show instance "$INSTANCE" --raw 2>/dev/null \
        | python -c "import json,sys; d=json.load(sys.stdin); print(f\"{d.get('is_bid')}|{d.get('dph_total')}\")" 2>/dev/null || echo "?|?")
    case "$ACTUAL" in
        True*) log "Confirmed interruptible at \$${ACTUAL#*|}/hr" ;;
        *)     log "WARNING: asked for interruptible, instance reports is_bid=${ACTUAL%%|*}, dph=${ACTUAL#*|}." ;;
    esac
fi
log "Check status:  vastai show instance $INSTANCE"
log "SSH:           vastai ssh-url $INSTANCE"
log "Logs (once up): vastai logs $INSTANCE"
log "Destroy:       ./scripts/vast/vast_destroy.sh $INSTANCE"
echo "$INSTANCE" > .vast_instance
log "Saved instance id to .vast_instance"
