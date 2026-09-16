#!/usr/bin/env python3
"""Run a command in its own session, detached from this process group.

`setsid` does not exist on macOS, and a plain `cmd &` leaves the child in the
caller's process group — so killing the caller (a timed-out shell, say) signals
the child too. For the Vast watchdog that means a healthy instance gets
destroyed mid-bootstrap by its own safety trap.
"""
import os
import shutil
import stat
import sys
import tempfile

if len(sys.argv) < 2:
    sys.exit("usage: detach.py <cmd> [args...]")

argv = list(sys.argv[1:])

# Run shell scripts from a private copy. bash reads a script incrementally from
# disk, so editing the file while it runs shifts the byte offsets under the live
# process: it resumes parsing mid-statement and dies with a syntax error. A
# syntax error also skips the EXIT trap, so a watchdog killed this way leaves the
# instance running unguarded. Cost a full GPU run to learn; a two-line copy makes
# it impossible.
if argv[0].endswith(".sh") and os.path.isfile(argv[0]):
    fd, snapshot = tempfile.mkstemp(prefix="detached-", suffix=".sh")
    os.close(fd)
    shutil.copyfile(argv[0], snapshot)
    os.chmod(snapshot, os.stat(snapshot).st_mode | stat.S_IEXEC)
    argv[0] = snapshot

if os.fork() != 0:
    sys.exit(0)          # parent returns immediately
os.setsid()              # child becomes a session leader: no shared signals
sys.stdout.flush()
os.execvp(argv[0], argv)
