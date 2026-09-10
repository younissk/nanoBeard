#!/usr/bin/env bash
# Destroy a Vast.ai instance. Pass the id explicitly or rely on .vast_instance.
set -euo pipefail

INSTANCE="${1:-$(cat .vast_instance 2>/dev/null || true)}"
[ -n "$INSTANCE" ] || { echo "usage: $0 <instance-id> (or save one to .vast_instance)"; exit 1; }

vastai destroy instance "$INSTANCE"
rm -f .vast_instance
echo "Destroyed $INSTANCE"
