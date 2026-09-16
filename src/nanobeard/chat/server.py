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

Two prompt paths, because the repo now has two kinds of model:

  * ``completion`` — the frigate line. Raw ``/completion`` with the SFT
    transcript from `nanobeard.rejection.generate`; the trailing space in
    "Pirate: " is load-bearing, which is why it is imported rather than
    re-derived.
  * ``chat`` — the Qwen3 line (the pirate LoRA). ``/v1/chat/completions``, so
    llama-server applies the template baked into the GGUF. That template carries
    the tool-call and think blocks, so hand-rendering would quietly lose them.

Default is ``chat``: everything trained from here on is Qwen3-shaped. Pass
``--api completion`` for a frigate GGUF.
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
from typing import Any, cast

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

    def __init__(
        self, port: int = DEFAULT_LLAMA_PORT, ctx: int = DEFAULT_CTX, slots: int = 1
    ):
        self.port = port
        # llama-server splits -c across slots, so an N-slot server needs N times
        # the context to give each request the window the caller asked for.
        self.ctx = ctx * slots
        self.slots = slots
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
                 "-c", str(self.ctx), "-np", str(self.slots), "--no-webui"],
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
    """llama-server /completion payload for a chat turn (frigate line)."""
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


def to_messages(turns: list[dict], system: str = "") -> list[dict]:
    """UI turns -> OpenAI messages.

    Tool turns matter: a model that called a tool needs to see its own call and
    the result, or the follow-up answer is generated blind and the whole point of
    testing the loop is lost.
    """
    messages: list[dict] = []
    if system.strip():
        messages.append({"role": "system", "content": system.strip()})
    for t in turns:
        role = t.get("role")
        if role == "tool":
            messages.append({
                "role": "tool",
                "name": t.get("name") or "tool",
                "content": t.get("text") or "",
            })
        elif role == "bot" and t.get("tool_calls"):
            messages.append({
                "role": "assistant",
                "content": t.get("text") or "",
                "tool_calls": t["tool_calls"],
            })
        else:
            messages.append({
                "role": "user" if role == "user" else "assistant",
                "content": t.get("text") or "",
            })
    return messages


def build_chat_payload(turns: list[dict], opts: dict) -> dict:
    """/v1/chat/completions payload, so the GGUF's own template is applied."""
    payload_tools = opts.get("tools")
    out = {
        "messages": to_messages(turns, opts.get("system") or ""),
        "max_tokens": int(opts.get("max_tokens", 200)),
        "temperature": float(opts.get("temperature", 0.8)),
        "top_p": float(opts.get("top_p", 0.95)),
        "top_k": int(opts.get("top_k", 40)),
        # Qwen3 thinks by default; on a 0.6B that is mostly latency the user
        # never sees, so it is off unless asked for.
        "chat_template_kwargs": {"enable_thinking": bool(opts.get("thinking"))},
        "stream": True,
    }
    if payload_tools:
        out["tools"] = payload_tools
    return out


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------
class ChatHandler(BaseHTTPRequestHandler):
    server_version = "nanobeard-chat"
    # Set by serve().
    llama: LlamaServer | None = None
    attached: str | None = None
    gguf_root: Path = DEFAULT_GGUF_ROOT
    api: str = "chat"
    default_system: str = ""
    search_index: object | None = None   # BM25 over Wikipedia paragraphs, or None

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
                "api": self.api,
                "system": self.default_system,
                "search": self.search_index is not None,
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

    def _chat_search(self, req: dict) -> None:
        """Play a full search episode, streaming each stage as it happens.

        The model only writes queries and answers; this loop runs the searches
        and pastes the results back, exactly as the RL environment does — same
        index, same protocol, same stop strings — so what you see here is what
        the policy was trained against.
        """
        from nanobeard.rl.corpus import BM25
        from nanobeard.rl.env import STOP_STRINGS, format_results, parse_action
        from nanobeard.rl.env import SYSTEM as SEARCH_SYSTEM

        turns = req.get("turns") or []
        question = (turns[-1].get("text") or "").strip()
        body = f"Question: {question}\n"
        max_searches = int(req.get("max_searches", 3))
        backend = self.backend

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        try:
            for _ in range(max_searches + 1):
                payload = {
                    "messages": [{"role": "system", "content": SEARCH_SYSTEM},
                                 {"role": "user", "content": body}],
                    "max_tokens": int(req.get("max_tokens", 160)),
                    "temperature": float(req.get("temperature", 0.7)),
                    "stop": list(STOP_STRINGS),
                    "chat_template_kwargs": {"enable_thinking": bool(req.get("thinking"))},
                }
                r = urllib.request.Request(
                    f"{backend}/v1/chat/completions", data=json.dumps(payload).encode(),
                    headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(r, timeout=180) as resp:
                    d = json.load(resp)
                chunk = d["choices"][0]["message"].get("content") or ""
                # llama-server eats the stop string; restore it so the same
                # parser the environment uses still sees a closed tag.
                if d["choices"][0].get("finish_reason") == "stop":
                    for st in STOP_STRINGS:
                        tag = st[2:-1]
                        if f"<{tag}>" in chunk and st not in chunk:
                            chunk += st
                body += chunk
                step = parse_action(chunk)

                if step.kind == "search":
                    # Typed `object` on the handler so the chat server does not
                    # import the RL package unless the tool is switched on.
                    rendered, titles = format_results(
                        cast(BM25, self.search_index), step.content, 3)
                    body += "\n" + rendered + "\n"
                    self._send_event({"search": step.content, "results": titles})
                    continue
                if step.kind == "answer":
                    self._send_event({"final_answer": step.content})
                    self._send_event({"done": True, "reason": "answer"})
                    return
                # Prose instead of an action ends the episode — that failure is
                # the thing worth seeing.
                self._send_event({"invalid": chunk.strip()[:400]})
                self._send_event({"done": True, "reason": "no-action"})
                return
            self._send_event({"done": True, "reason": "out of searches"})
        except BrokenPipeError:
            return
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            self._send_event({"error": f"{type(e).__name__}: {e}"})

    def _chat(self) -> None:
        req = self._read_json()
        turns = req.get("turns") or []
        if not turns:
            self._json({"error": "no turns"}, 400)
            return
        if req.get("tools") and self.api != "chat":
            self._json({"error": "tools need --api chat"}, 400)
            return
        if not self.attached and (self.llama is None or self.llama.model is None):
            self._json({"error": "no model loaded — pick one first"}, 409)
            return

        if self.search_index is not None and req.get("search"):
            self._chat_search(req)
            return

        backend = self.backend
        budget = DEFAULT_CTX - REPLY_HEADROOM - int(req.get("max_tokens", 200))
        # Trimming renders the frigate transcript, which only makes sense there.
        # In chat mode llama-server owns the template and its own context window.
        if self.api != "chat":
            with contextlib.suppress(urllib.error.URLError, TimeoutError, OSError):
                turns = trim_turns(turns, budget, lambda t: _token_count(backend, t))

        if self.api == "chat":
            payload = build_chat_payload(turns, req)
            endpoint = f"{backend}/v1/chat/completions"
        else:
            payload = build_payload(turns, req)
            endpoint = f"{backend}/completion"
        upstream = urllib.request.Request(
            endpoint, data=json.dumps(payload).encode(),
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
                    body_txt = line[5:].strip()
                    if body_txt == "[DONE]":
                        self._send_event({"done": True, "reason": "stop"})
                        return
                    chunk = json.loads(body_txt)
                    if self.api == "chat":
                        choice = (chunk.get("choices") or [{}])[0]
                        delta = choice.get("delta") or {}
                        # Some builds stream the chain of thought in its own
                        # field; others leave it inline in <think> tags for the
                        # client to split out.
                        if delta.get("reasoning_content"):
                            self._send_event({"thinking": delta["reasoning_content"]})
                        if delta.get("content"):
                            self._send_event({"content": delta["content"]})
                        for tc in delta.get("tool_calls") or []:
                            fn = tc.get("function") or {}
                            self._send_event({"tool_call": {
                                "index": tc.get("index", 0),
                                "name": fn.get("name") or "",
                                "arguments": fn.get("arguments") or "",
                            }})
                        if choice.get("finish_reason"):
                            self._send_event({"done": True, "reason": choice["finish_reason"]})
                            return
                    else:
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


PIRATE_SYSTEM = (
    "You are Nano Beard, a warm, salty pirate assistant. Speak in a natural, "
    "readable pirate voice. Be concise. Never mention being an AI."
)


def serve(
    *,
    ui_port: int = DEFAULT_UI_PORT,
    llama_port: int = DEFAULT_LLAMA_PORT,
    gguf_root: Path = DEFAULT_GGUF_ROOT,
    model: str | None = None,
    attach: str | None = None,
    ctx: int = DEFAULT_CTX,
    open_browser: bool = True,
    api: str = "chat",
    system: str = PIRATE_SYSTEM,
    search_index: object | None = None,
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
    ChatHandler.api = api
    ChatHandler.default_system = system
    ChatHandler.search_index = search_index

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
    ap.add_argument("--api", choices=("chat", "completion"), default="chat",
                    help="chat = /v1/chat/completions with the GGUF's own template "
                         "(Qwen3 line); completion = raw SFT transcript (frigate line)")
    ap.add_argument("--system", default=PIRATE_SYSTEM,
                    help="Default system prompt, editable in the UI")
    ap.add_argument("--search-index", default=None, metavar="PKL",
                    help="Enable the Wikipedia search tool, e.g. "
                         "data/search/hotpot_bm25.pkl (build with make rl-corpus)")
    args = ap.parse_args()

    search_index = None
    if args.search_index:
        from nanobeard.rl.corpus import load as load_corpus

        search_index, _q = load_corpus(Path(args.search_index))
        print(f"search tool: {len(search_index):,} Wikipedia paragraphs")

    serve(
        ui_port=args.port,
        llama_port=args.llama_port,
        gguf_root=Path(args.gguf_root),
        model=args.model,
        attach=args.attach,
        ctx=args.ctx,
        open_browser=not args.no_open,
        api=args.api,
        system=args.system,
        search_index=search_index,
    )


if __name__ == "__main__":
    main()
