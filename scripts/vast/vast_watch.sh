#!/usr/bin/env bash
# Watch a Vast instance, fetch its results, then DESTROY it.
#
#   ./scripts/vast/vast_watch.sh <instance-id> [--timeout-min 60] \
#       [--fetch <remote-path> <local-dir>]
#
# Why a local watchdog rather than a self-destruct on the box: self-destruct
# means putting a Vast API key on a machine somebody else owns, and that key can
# create and destroy instances on the whole account. This process holds the key
# instead and the rented box holds nothing.
#
# Three ways it ends, and all of them destroy:
#   * the run writes its status marker  -> fetch, then destroy
#   * the timeout expires               -> fetch what exists, then destroy
#   * this script is killed             -> trap fires, destroy
#
# `destroy`, never `stop`: a stopped Vast instance still bills for its disk.
#
# To hand an instance over to a different watchdog, `kill -9` this one: a plain
# kill runs the EXIT trap and destroys the box, which is correct for Ctrl-C and
# wrong when you only meant to change the timeout.

set -uo pipefail

INSTANCE="${1:?usage: $0 <instance-id> [--timeout-min N] [--fetch REMOTE LOCAL]}"
shift

TIMEOUT_MIN=60
POLL_SEC=20
FETCH_REMOTE=""
FETCH_LOCAL=""
DONE_MARKER="${DONE_MARKER:-/root/pirate_llm/.vast_done}"
# Consecutive failed state lookups before giving up on the instance. One is
# routine API flakiness; a run of them means it really is gone.
MAX_UNKNOWN="${MAX_UNKNOWN:-10}"
UNKNOWN=0
SSH_KEY="${SSH_KEY:-$HOME/.ssh/pirate_llm_gpu}"

while [ $# -gt 0 ]; do
    case "$1" in
        --timeout-min) TIMEOUT_MIN="$2"; shift 2 ;;
        --poll-sec)    POLL_SEC="$2"; shift 2 ;;
        --fetch)       FETCH_REMOTE="$2"; FETCH_LOCAL="$3"; shift 3 ;;
        --marker)      DONE_MARKER="$2"; shift 2 ;;
        *) echo "unknown arg: $1"; exit 2 ;;
    esac
done

log() { echo "[watch $(date +%H:%M:%S)] $*"; }

DESTROYED=0
destroy_once() {
    [ "$DESTROYED" = "1" ] && return 0
    DESTROYED=1
    log "destroying instance $INSTANCE"
    # -y is load-bearing: without it the CLI prompts "[y/N]", reads EOF from a
    # detached watchdog, prints "Aborted." and leaves the instance billing. The
    # whole point of this script is that it cannot do that.
    vastai destroy instance "$INSTANCE" -y 2>&1 | sed 's/^/  /'
    rm -f .vast_instance 2>/dev/null || true
    sleep 5
    if vastai show instance "$INSTANCE" --raw 2>/dev/null \
        | python3 -c "import json,sys; sys.exit(0 if json.load(sys.stdin).get('id') else 1)" 2>/dev/null; then
        log "WARNING: instance $INSTANCE still exists after destroy — check manually"
    else
        log "confirmed destroyed"
    fi
}
# Covers Ctrl-C, SIGTERM, and any unexpected exit path.
trap destroy_once EXIT INT TERM

SSH_HOST=""
SSH_PORT=""
SSH_OPTS=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR)

# Resolve host/port once per poll. Kept as separate variables rather than one
# pre-baked argument string because ssh wants -p and scp wants -P, and splitting
# a string back apart for that is how the fetch silently targets the wrong host.
resolve_ssh() {
    local url
    url=$(vastai ssh-url "$INSTANCE" 2>/dev/null) || return 1
    [ -n "$url" ] || return 1
    SSH_HOST=$(echo "$url" | sed -E 's#ssh://[^@]+@([^:]+):.*#\1#')
    SSH_PORT=$(echo "$url" | sed -E 's#.*:([0-9]+)$#\1#')
    [ -n "$SSH_HOST" ] && [ -n "$SSH_PORT" ]
}

DEADLINE=$(( $(date +%s) + TIMEOUT_MIN * 60 ))
if pgrep -f "vast_watch.sh $INSTANCE" | grep -qv "^$$\$"; then
    others=$(pgrep -f "vast_watch.sh $INSTANCE" | grep -v "^$$\$" | tr '\n' ' ')
    log "another watchdog is already on $INSTANCE (pid $others) — refusing to double up"
    trap - EXIT INT TERM
    exit 1
fi
log "watching $INSTANCE (timeout ${TIMEOUT_MIN}m, marker $DONE_MARKER)"

STATUS="timeout"
while [ "$(date +%s)" -lt "$DEADLINE" ]; do
    STATE=$(vastai show instance "$INSTANCE" --raw 2>/dev/null \
        | python3 -c "import json,sys; print(json.load(sys.stdin).get('actual_status','?'))" 2>/dev/null || echo "?")

    # "?" means the API call failed or returned something unparseable — NOT that
    # the instance is gone. Treating the two alike destroyed a healthy run 55
    # minutes in, on a single transient lookup, minutes before it would have
    # pushed its results. Only a repeated unknown counts.
    if [ "$STATE" = "exited" ]; then
        log "instance exited"
        STATUS="gone"
        break
    fi
    if [ "$STATE" = "?" ]; then
        UNKNOWN=$(( ${UNKNOWN:-0} + 1 ))
        log "state lookup failed (${UNKNOWN}/${MAX_UNKNOWN})"
        if [ "$UNKNOWN" -ge "$MAX_UNKNOWN" ]; then
            log "state unknown ${MAX_UNKNOWN} times running — treating as gone"
            STATUS="gone"
            break
        fi
        sleep "$POLL_SEC"
        continue
    fi
    UNKNOWN=0

    if [ "$STATE" = "running" ] && resolve_ssh; then
        # shellcheck disable=SC2029  # $DONE_MARKER is ours and expands locally on purpose
        if RC=$(ssh -i "$SSH_KEY" -p "$SSH_PORT" "${SSH_OPTS[@]}" \
                    "root@$SSH_HOST" "cat $DONE_MARKER" 2>/dev/null); then
            log "run finished with exit status ${RC:-?}"
            STATUS="done:${RC:-?}"; break
        fi
    fi
    sleep "$POLL_SEC"
done

log "result: $STATUS"

if [ -n "$FETCH_REMOTE" ] && [ "$STATUS" != "gone" ]; then
    if resolve_ssh; then
        mkdir -p "$FETCH_LOCAL"
        log "fetching $FETCH_REMOTE -> $FETCH_LOCAL"
        if scp -r -i "$SSH_KEY" -P "$SSH_PORT" "${SSH_OPTS[@]}" \
                "root@$SSH_HOST:$FETCH_REMOTE" "$FETCH_LOCAL" 2>&1 | sed 's/^/  /'; then
            log "fetched"
        else
            # Destroy regardless: the instance bills by the second, the adapter
            # does not, and it can be regenerated for pennies.
            log "fetch FAILED — destroying anyway"
        fi
    else
        log "could not resolve ssh — destroying without fetching"
    fi
fi

destroy_once
log "done"
