#!/usr/bin/env bash
# Provision a Vast.ai instance and bootstrap it for nanoBeard training.
#
# Prereqs (local):
#   uv tool install vastai
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
# A LIST, cheapest wins. Measured 2026-09-10: the 5090 bid floor moved
# $0.202 -> $0.333 in ten minutes while the 4090 sat at $0.200, so the "best
# card" swapped twice inside one session. Pin a single name here to override,
# e.g. GPU=RTX_5090 if you need the 32GB.
GPU="${GPU:-RTX_4090,RTX_5090,RTX_3090}"
# -devel, not -runtime, for two reasons. It ships gcc/nvcc, which torch.compile's
# inductor backend needs to JIT kernels. And the -runtime images have no openssh
# at all: Vast's own /.launch dies with "ssh: command not found" and the instance
# is unreachable for its whole life. CUDA 12.8 is also the floor for sm_120
# (RTX 5090). The image torch is irrelevant — `uv sync` installs the pinned one.
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
# Bid derived from the chosen offer's own min_bid, NOT from MAX_DPH: vast bills
# your bid, so bidding the cap when the floor is half that simply donates the
# difference. The multiplier is headroom against being outbid immediately.
BID_MULTIPLIER="${BID_MULTIPLIER:-1.15}"
BID="${BID:-}"   # set explicitly to override the derived bid
# Off by default, against the usual advice. Measured on vast 2026-09-10 for a
# 4090: datacenter=true cut the board from 15 offers to 2 and raised the floor
# from $0.168 to $0.391 — 2.3x the price for the privilege. Combined with
# reliability>=0.95 it returned nothing at all. The non-DC hosts that survive
# the reliability filter sit at 0.997+, and checkpoint-resume already covers
# eviction, so paying the DC premium buys very little here.
DATACENTER="${DATACENTER:-0}"
INET_DOWN="${INET_DOWN:-200}"        # min Mbps
# Minimum host CUDA driver. The pinned torch (>=2.12) ships a cu12.9+ wheel that
# refuses to init on older drivers ("NVIDIA driver too old"), so reject hosts
# whose driver predates 12.9 up front instead of crashing after the dataset pull.
CUDA_VERS="${CUDA_VERS:-12.9}"
REPO_URL="${REPO_URL:-https://github.com/younissk/nanoBeard}"
# Branch/tag the instance pulls the bootstrap from. Override when testing a
# branch, or the box will run main's bootstrap against your branch's code.
REPO_REF="${REPO_REF:-main}"
START_TIMEOUT_S="${START_TIMEOUT_S:-600}"   # image pull on a slow host is minutes

log() { echo -e "\033[1;32m[vast]\033[0m $*"; }

command -v vastai >/dev/null || { echo "Install vast-cli: uv tool install vastai"; exit 1; }
[ -n "${HF_TOKEN:-}" ] || { echo "Set HF_TOKEN in env"; exit 1; }
[ -n "${WANDB_API_KEY:-}" ] || log "WANDB_API_KEY unset — training runs without wandb logging"

# 1. Find the cheapest usable offer across the candidate GPUs.
#
# The selection lives in nanobeard.vast_offers, not here: it has to survive
# vast's "Unrecognized field" warning (which otherwise returns on-demand offers
# with only a warning), derive the bid from the offer's own min_bid, and rank
# across GPU types. That is testable Python, not shell.
OFFER_ARGS=(--gpus "$GPU" --max-dph "$MAX_DPH" --bid-multiplier "$BID_MULTIPLIER"
            --inet-down "$INET_DOWN" --cuda-vers "$CUDA_VERS")
[ -n "${EXCLUDE_MACHINES:-}" ] && OFFER_ARGS+=(--exclude-machines "$EXCLUDE_MACHINES")
[ "$DATACENTER" = "1" ]    && OFFER_ARGS+=(--datacenter)
[ "$INTERRUPTIBLE" = "1" ] || OFFER_ARGS+=(--on-demand)

log "Searching offers across: $GPU (cap \$$MAX_DPH/hr)"
uv run python -m nanobeard.vast_offers "${OFFER_ARGS[@]}" --board || exit 1

PICK=$(uv run python -m nanobeard.vast_offers "${OFFER_ARGS[@]}" --pick) || exit 1
OFFER=$(echo "$PICK"    | awk '{print $1}')
MIN_BID=$(echo "$PICK"  | awk '{print $2}')
DERIVED=$(echo "$PICK"  | awk '{print $3}')
OFFER_DPH=$(echo "$PICK" | awk '{print $4}')
PICKED_GPU=$(echo "$PICK" | awk '{print $5}')
PICKED_MACHINE=$(echo "$PICK" | awk '{print $6}')

log "Picked $PICKED_GPU offer $OFFER on machine $PICKED_MACHINE (dph=$OFFER_DPH, min_bid=$MIN_BID)"
[ -n "$BID" ] || BID="$DERIVED"

# 2. Create the instance with --onstart so it bootstraps itself.
ONSTART=$(cat <<EOF
#!/bin/bash
set -e
export HF_TOKEN='$HF_TOKEN'
export WANDB_API_KEY='${WANDB_API_KEY:-}'
export CONFIG='$CONFIG'
export VARIANT='$VARIANT'
export DATASET='$DATASET'
export REPO_URL='$REPO_URL'
export REPO_REF='$REPO_REF'
export LORA_DATA='${LORA_DATA:-runs/distill/train.jsonl}'
export LORA_OUT='${LORA_OUT:-runs/lora/pirate-v1}'
export LORA_EPOCHS='${LORA_EPOCHS:-2}'
export LORA_RANK='${LORA_RANK:-16}'
export LORA_PUSH_REPO='${LORA_PUSH_REPO:-}'
export DONE_MARKER='/root/pirate_llm/.vast_done'
curl -fsSL $REPO_URL/raw/$REPO_REF/scripts/vast/vast_bootstrap.sh | bash
EOF
)

# macOS ships bash 3.2, where `set -u` treats "${arr[@]}" on an EMPTY array as
# an unbound variable and aborts. ${arr[@]+"${arr[@]}"} is the portable form.
# Getting this wrong made every on-demand launch die silently.
PRICE_ARGS=()
if [ "$INTERRUPTIBLE" = "1" ]; then
    # The flag is --bid_price. --price is a different thing and passing it
    # gets you an on-demand instance at full rate.
    PRICE_ARGS=(--bid_price "$BID")
    log "Creating INTERRUPTIBLE instance, bid \$$BID/hr (min_bid \$$MIN_BID)"
else
    log "Creating on-demand instance"
fi

INSTANCE=$(vastai create instance "$OFFER" \
    --image "$IMAGE" \
    --disk "$DISK_GB" \
    --ssh \
    ${PRICE_ARGS[@]+"${PRICE_ARGS[@]}"} \
    --onstart-cmd "$ONSTART" \
    --raw 2>&1 | tee /tmp/vast_create.json | python3 -c "
import json,sys
raw = sys.stdin.read()
try:
    print(json.loads(raw)['new_contract'])
except Exception:
    sys.stderr.write('create failed: ' + raw[:400])
")

# Swallowing this is what hid a bash-3.2 array bug for six launches.
[ -n "$INSTANCE" ] || { log "instance creation failed (see above)"; exit 1; }

log "Created instance $INSTANCE"
echo "$INSTANCE" > .vast_instance

# Wait for the box to actually start. Vast happily creates an instance on a
# machine whose GPU is broken; it sits in "created" with
# status_msg="Error: GPU error, unable to start instance." and never runs.
# Measured: one in three hosts tried. Fail loudly here instead of letting a
# watchdog time out 40 minutes later.
log "waiting for it to start (up to ${START_TIMEOUT_S}s)"
DEADLINE=$(( $(date +%s) + START_TIMEOUT_S ))
while [ "$(date +%s)" -lt "$DEADLINE" ]; do
    READ=$(vastai show instance "$INSTANCE" --raw 2>/dev/null | python3 -c "
import json,sys
d=json.load(sys.stdin)
print(f\"{d.get('actual_status')}|{d.get('intended_status')}|{(d.get('status_msg') or '')[:70]}\")" 2>/dev/null || echo "?|?|")
    ST=${READ%%|*}; REST=${READ#*|}; INTENT=${REST%%|*}; MSG=${REST#*|}
    case "$ST" in
        running) log "running"; break ;;
        exited)  log "FAILED: instance exited during startup — $MSG"; DEAD=1; break ;;
    esac
    # Only an actual error counts. status_msg carries docker pull progress
    # ("f81de80fb4b1: Verifying Checksum") during a normal startup, and treating
    # any non-empty message as a refusal killed healthy instances mid-pull.
    case "$MSG" in
        *Error*|*error*|*failed*|*Failed*)
            log "FAILED: host refused to start — $MSG"; DEAD=1; break ;;
    esac
    sleep 10
done
if [ "${DEAD:-0}" = "1" ] || [ "$(date +%s)" -ge "$DEADLINE" ]; then
    log "destroying the bad instance"
    log "retry excluding it:  EXCLUDE_MACHINES=${EXCLUDE_MACHINES:+$EXCLUDE_MACHINES,}$PICKED_MACHINE $0"
    vastai destroy instance "$INSTANCE" -y >/dev/null 2>&1
    rm -f .vast_instance
    exit 1
fi

# Verify the bid actually took. A silently-on-demand instance costs 2-4x what
# you planned and nothing else in this script would notice.
if [ "$INTERRUPTIBLE" = "1" ]; then
    ACTUAL=$(vastai show instance "$INSTANCE" --raw 2>/dev/null \
        | python3 -c "import json,sys; d=json.load(sys.stdin); print(f\"{d.get('is_bid')}|{d.get('dph_total')}\")" 2>/dev/null || echo "?|?")
    case "$ACTUAL" in
        True*) log "Confirmed interruptible at \$${ACTUAL#*|}/hr" ;;
        *)     log "WARNING: asked for interruptible, instance reports is_bid=${ACTUAL%%|*}, dph=${ACTUAL#*|}." ;;
    esac
fi
log "Check status:  vastai show instance $INSTANCE"
log "SSH:           vastai ssh-url $INSTANCE"
log "Logs (once up): vastai logs $INSTANCE"
log "Destroy:       ./scripts/vast/vast_destroy.sh $INSTANCE"
log "Saved instance id to .vast_instance"
