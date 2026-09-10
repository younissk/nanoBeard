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
#   ./scripts/vast_launch.sh                       # default: RTX 4090, sloop config, gpu variant
#   GPU=RTX_4090 CONFIG=sloop VARIANT=gpu ./scripts/vast_launch.sh

set -euo pipefail

CONFIG="${CONFIG:-sloop}"
VARIANT="${VARIANT:-gpu}"
DATASET="${DATASET:-tiny_pirate_stories}"
GPU="${GPU:-RTX_4090}"
IMAGE="${IMAGE:-pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime}"
DISK_GB="${DISK_GB:-50}"
MAX_DPH="${MAX_DPH:-1.10}"           # dollars per hour cap (RTX_4090 floor is ~$1.07)
INET_DOWN="${INET_DOWN:-200}"        # min Mbps
# Minimum host CUDA driver. The pinned torch (>=2.12) ships a cu12.9+ wheel that
# refuses to init on older drivers ("NVIDIA driver too old"), so reject hosts
# whose driver predates 12.9 up front instead of crashing after the dataset pull.
CUDA_VERS="${CUDA_VERS:-12.9}"
REPO_URL="${REPO_URL:-https://github.com/younissk/pirate_llm}"

log() { echo -e "\033[1;32m[vast]\033[0m $*"; }

command -v vastai >/dev/null || { echo "Install vast-cli: pip install vastai"; exit 1; }
[ -n "${HF_TOKEN:-}" ] || { echo "Set HF_TOKEN in env"; exit 1; }
[ -n "${WANDB_API_KEY:-}" ] || log "WANDB_API_KEY unset — training runs without wandb logging"

# 1. Find a cheap matching offer.
log "Searching for offers: gpu=$GPU, dph<=$MAX_DPH, inet_down>=$INET_DOWN, cuda_vers>=$CUDA_VERS"
# `|| true`: a no-match must not abort under `set -e` (the python prints an
# empty id), so the friendly hint below can fire instead of dying silently.
OFFER=$(vastai search offers \
    "gpu_name=$GPU num_gpus=1 dph_total<=$MAX_DPH inet_down>=$INET_DOWN reliability>=0.95 cuda_vers>=$CUDA_VERS" \
    -o 'dph+' \
    --raw 2>/dev/null | python -c "import json,sys; offers=json.load(sys.stdin); print(offers[0]['id'] if offers else '')") || true

[ -n "$OFFER" ] || {
    echo "No offers matched gpu=$GPU at dph<=$MAX_DPH, inet_down>=$INET_DOWN, reliability>=0.95, cuda_vers>=$CUDA_VERS."
    echo "Relax constraints, e.g.:  MAX_DPH=1.20 $0   or   GPU=RTX_3090 $0   or   CUDA_VERS=12.4 $0"
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
curl -fsSL $REPO_URL/raw/main/scripts/vast_bootstrap.sh | bash
EOF
)

log "Creating instance"
INSTANCE=$(vastai create instance "$OFFER" \
    --image "$IMAGE" \
    --disk "$DISK_GB" \
    --ssh \
    --onstart-cmd "$ONSTART" \
    --raw 2>/dev/null | python -c "import json,sys; print(json.load(sys.stdin)['new_contract'])")

log "Created instance $INSTANCE"
log "Check status:  vastai show instance $INSTANCE"
log "SSH:           vastai ssh-url $INSTANCE"
log "Logs (once up): vastai logs $INSTANCE"
log "Destroy:       ./scripts/vast_destroy.sh $INSTANCE"
echo "$INSTANCE" > .vast_instance
log "Saved instance id to .vast_instance"
