#!/usr/bin/env python3
"""Run a command in its own session, detached from this process group.

`setsid` does not exist on macOS, and a plain `cmd &` leaves the child in the
caller's process group — so killing the caller (a timed-out shell, say) signals
the child too. For the Vast watchdog that means a healthy instance gets
destroyed mid-bootstrap by its own safety trap.
"""
import os
import sys

if len(sys.argv) < 2:
    sys.exit("usage: detach.py <cmd> [args...]")
if os.fork() != 0:
    sys.exit(0)          # parent returns immediately
os.setsid()              # child becomes a session leader: no shared signals
sys.stdout.flush()
os.execvp(sys.argv[1], sys.argv[1:])
