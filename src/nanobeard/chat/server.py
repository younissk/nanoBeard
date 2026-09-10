"""Local chat UI for the exported GGUFs — pick a model, talk to it.

    make chat                      # discovers export/gguf/**/*.gguf, opens a browser
    make chat GGUF_MODEL=<path>    # start on a specific one
    make chat ATTACH=http://127.0.0.1:8899   # reuse a running `make serve`

Why this exists: trying a checkout meant remembering the llama-server flags,
then curling `/completion` with the SFT transcript rendered by hand. Getting the
render wrong (see `render_prompt`) makes a working model look broken, so it is
worth having exactly one place that gets it right.

Two moving parts:

  * ``LlamaServer`` owns a llama-server subprocess. Switching models in the
    dropdown kills it and starts another — llama-server serves one model per
    process, so there is no cheaper way.
  * The HTTP handler serves ``index.html`` and proxies ``/api/chat`` to
    llama-server's ``/completion`` with ``stream: true``, forwarding tokens as
    server-sent events so replies appear as they generate.

Prompt format and stop strings are imported from `nanobeard.rejection.generate`
rather than re-derived: the trailing space in "Pirate: " is load-bearing, and
two copies of that rule would eventually disagree.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from nanobeard.rejection.generate import STOPS, render_prompt

HERE = Path(__file__).parent
DEFAULT_GGUF_ROOT = Path("export/gguf")
DEFAULT_UI_PORT = 8800
# Deliberately not 8899: that is `make serve`'s port, and clashing with a
# rejection-sampling run mid-flight is a bad surprise. Use --attach for that.
DEFAULT_LLAMA_PORT = 8901
DEFAULT_CTX = 4096
# Leave room for the reply when trimming history to fit the context window.
REPLY_HEADROOM = 512


# --------------------------------------------------------------------------
# model discovery
# --------------------------------------------------------------------------
def discover_models(root: Path = DEFAULT_GGUF_ROOT) -> list[dict[str, Any]]:
    """Every GGUF under `root`, newest first.

    The f16 intermediates that `export_gguf.py --keep-f16` leaves behind are
    skipped: they are 4x the size of the Q8_0 next to them and no better to
    talk to.
    """
    if not root.exists():
        return []
    out: list[dict[str, Any]] = []
    for p in sorted(root.rglob("*.gguf")):
        if p.name.endswith("-f16.gguf"):
            continue
        out.append(
            {
                "path": str(p),
                "label": str(p.relative_to(root)),
                "size_mb": round(p.stat().st_size / 1e6, 1),
                "mtime": p.stat().st_mtime,
            }
        )
    out.sort(key=lambda m: -m["mtime"])
    return out


def _port_is_free(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket() as s:
        s.settimeout(0.2)
        return s.connect_ex((host, port)) != 0


# --------------------------------------------------------------------------
# llama-server lifecycle
# --------------------------------------------------------------------------
class LlamaServer:
    """One llama-server subprocess, restartable against a different GGUF."""

    def __init__(self, port: int = DEFAULT_LLAMA_PORT, ctx: int = DEFAULT_CTX):
        self.port = port
        self.ctx = ctx
        self.proc: subprocess.Popen | None = None
        self.model: str | None = None
        self._lock = threading.Lock()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self, model: str, timeout: float = 120.0) -> None:
        with self._lock:
            self._stop_locked()
            binary = shutil.which("llama-server")
            if not binary:
                raise RuntimeError(
                    "llama-server not found on PATH — install llama.cpp "
                    "(`brew install llama.cpp`) or pass --attach <url>"
                )
            self.proc = subprocess.Popen(
                [binary, "-m", model, "--host", "127.0.0.1", "--port", str(self.port),
                 "-c", str(self.ctx), "-np", "1", "--no-webui"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self.model = model
            self._await_health(timeout)

    def _await_health(self, timeout: float) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.proc is not None and self.proc.poll() is not None:
                raise RuntimeError(
                    f"llama-server exited with code {self.proc.returncode} loading "
                    f"{self.model} — run it by hand to see why"
                )
            try:
                with urllib.request.urlopen(f"{self.url}/health", timeout=1) as r:
                    if r.status == 200:
                        return
            except (urllib.error.URLError, TimeoutError, OSError):
                time.sleep(0.3)
        raise RuntimeError(f"llama-server did not become healthy within {timeout:.0f}s")

    def _stop_locked(self) -> None:
        if self.proc is None:
            return
        self.proc.terminate()
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait(timeout=5)
        self.proc = None
        self.model = None

    def stop(self) -> None:
        with self._lock:
            self._stop_locked()


# --------------------------------------------------------------------------
# prompt assembly
# --------------------------------------------------------------------------
def _token_count(server_url: str, text: str) -> int:
    req = urllib.request.Request(
        f"{server_url}/tokenize", data=json.dumps({"content": text}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return len(json.load(r).get("tokens", []))


def trim_turns(turns: list[dict], budget: int, count) -> list[dict]:
    """Drop the oldest turns until the rendered prompt fits `budget` tokens.

    The models are trained at block_size=512 (see TODO.md), so a long chat does
    not fit and something has to go. Oldest-first keeps the exchange the user is
    actually in. The newest turn is never dropped — if even that overflows the
    model will truncate, which is more useful than an empty prompt.

    `count` is injected so this stays testable without a live server.
    """
    kept = list(turns)
    while len(kept) > 1 and count(render_prompt(kept)) > budget:
        kept = kept[1:]
    return kept


def build_payload(turns: list[dict], opts: dict) -> dict:
    """llama-server /completion payload for a chat turn."""
    return {
        "prompt": render_prompt(turns),
        "n_predict": int(opts.get("max_tokens", 200)),
        "temperature": float(opts.get("temperature", 0.8)),
        "top_p": float(opts.get("top_p", 0.95)),
        "top_k": int(opts.get("top_k", 40)),
        "stop": STOPS,
        "cache_prompt": True,
        "stream": True,
    }


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------
class ChatHandler(BaseHTTPRequestHandler):
    server_version = "nanobeard-chat"
    # Set by serve().
    llama: LlamaServer | None = None
    attached: str | None = None
    gguf_root: Path = DEFAULT_GGUF_ROOT

    def log_message(self, fmt, *args):  # noqa: A002 - stdlib signature
        pass  # The UI is the log; per-request noise buries the real errors.

    # ---- helpers ----
    @property
    def backend(self) -> str:
        if self.attached:
            return self.attached
        assert self.llama is not None
        return self.llama.url

    def _json(self, obj, status: int = 200) -> None:
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        n = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(n) or b"{}")

    # ---- routes ----
    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            body = (HERE / "index.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/api/models":
            self._json({
                "models": discover_models(self.gguf_root),
                "current": None if self.attached else (self.llama.model if self.llama else None),
                "attached": self.attached,
            })
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self) -> None:
        if self.path == "/api/model":
            self._switch_model()
        elif self.path == "/api/chat":
            self._chat()
        else:
            self._json({"error": "not found"}, 404)

    def _switch_model(self) -> None:
        if self.attached:
            self._json({"error": "attached to an external llama-server; cannot switch"}, 409)
            return
        assert self.llama is not None
        path = self._read_json().get("path")
        if not path or not Path(path).exists():
            self._json({"error": f"no such model: {path!r}"}, 400)
            return
        try:
            self.llama.start(path)
        except RuntimeError as e:
            self._json({"error": str(e)}, 500)
            return
        self._json({"current": path})

    def _chat(self) -> None:
        req = self._read_json()
        turns = req.get("turns") or []
        if not turns:
            self._json({"error": "no turns"}, 400)
            return
        if not self.attached and (self.llama is None or self.llama.model is None):
            self._json({"error": "no model loaded — pick one first"}, 409)
            return

        backend = self.backend
        budget = DEFAULT_CTX - REPLY_HEADROOM - int(req.get("max_tokens", 200))
        # Tokenizer endpoint unreachable -> skip trimming, let the server truncate.
        with contextlib.suppress(urllib.error.URLError, TimeoutError, OSError):
            turns = trim_turns(turns, budget, lambda t: _token_count(backend, t))

        payload = build_payload(turns, req)
        upstream = urllib.request.Request(
            f"{backend}/completion", data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        try:
            with urllib.request.urlopen(upstream, timeout=300) as r:
                for raw in r:
                    line = raw.decode(errors="replace").strip()
                    if not line.startswith("data:"):
                        continue
                    chunk = json.loads(line[5:])
                    self._send_event({"content": chunk.get("content", "")})
                    if chunk.get("stop"):
                        self._send_event({"done": True, "reason": chunk.get("stop_type")})
                        return
        except BrokenPipeError:
            return  # Browser navigated away mid-stream; nothing to report to.
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            # BrokenPipeError is an OSError, so it has to be caught above this.
            self._send_event({"error": f"{type(e).__name__}: {e}"})

    def _send_event(self, obj: dict) -> None:
        self.wfile.write(f"data: {json.dumps(obj)}\n\n".encode())
        self.wfile.flush()


def serve(
    *,
    ui_port: int = DEFAULT_UI_PORT,
    llama_port: int = DEFAULT_LLAMA_PORT,
    gguf_root: Path = DEFAULT_GGUF_ROOT,
    model: str | None = None,
    attach: str | None = None,
    ctx: int = DEFAULT_CTX,
    open_browser: bool = True,
) -> None:
    llama = None if attach else LlamaServer(port=llama_port, ctx=ctx)

    if llama is not None:
        if not _port_is_free(llama_port):
            sys.exit(
                f"port {llama_port} is busy — something is already listening there.\n"
                f"If it is a llama-server you want to use: --attach http://127.0.0.1:{llama_port}"
            )
        models = discover_models(gguf_root)
        chosen = model or (models[0]["path"] if models else None)
        if not chosen:
            sys.exit(
                f"no GGUFs under {gguf_root}/ — run `make export-gguf` first, "
                f"or point --gguf-root somewhere else"
            )
        print(f"loading {chosen} …")
        llama.start(chosen)

    ChatHandler.llama = llama
    ChatHandler.attached = attach
    ChatHandler.gguf_root = gguf_root

    httpd = ThreadingHTTPServer(("127.0.0.1", ui_port), ChatHandler)

    # llama-server is a child process, not a thread: if this process dies
    # without running the `finally` below it is orphaned and keeps the port and
    # a few hundred MB of RAM. Ctrl-C already unwinds; SIGTERM (what `kill` and
    # most supervisors send) does not, so turn it into the same unwind.
    def _on_sigterm(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _on_sigterm)
    url = f"http://127.0.0.1:{ui_port}"
    print(f"⚓ nanoBeard chat on {url}  (backend: {attach or llama.url})")  # type: ignore[union-attr]
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nFair winds!")
    finally:
        httpd.server_close()
        if llama is not None:
            llama.stop()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", type=int, default=DEFAULT_UI_PORT, help="UI port")
    ap.add_argument("--llama-port", type=int, default=DEFAULT_LLAMA_PORT)
    ap.add_argument("--gguf-root", default=str(DEFAULT_GGUF_ROOT))
    ap.add_argument("--model", default=None, help="GGUF to load first (default: newest found)")
    ap.add_argument("--attach", default=None, help="Use an already-running llama-server URL")
    ap.add_argument("--ctx", type=int, default=DEFAULT_CTX)
    ap.add_argument("--no-open", action="store_true", help="Do not open a browser")
    args = ap.parse_args()
    serve(
        ui_port=args.port,
        llama_port=args.llama_port,
        gguf_root=Path(args.gguf_root),
        model=args.model,
        attach=args.attach,
        ctx=args.ctx,
        open_browser=not args.no_open,
    )


if __name__ == "__main__":
    main()
