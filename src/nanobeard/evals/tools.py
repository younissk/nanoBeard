"""Tool-calling validity — the capability gate that a style SFT is most likely to break.

Scored in four layers, because "it called a tool" is not the same as "it called
the right tool correctly":

  well_formed   emitted a tool call whose arguments parse as JSON
  right_tool    picked the tool the scenario expects
  args_ok       every required parameter present, and every value we pinned matches
  restraint     did NOT call a tool when no tool applies

`restraint` is scored separately and is not a footnote. Over-calling is the
classic small-model failure — a model that fires get_weather at "I'm feeling a
bit down today" is worse than useless in an app, and a pirate-voice SFT tends to
push exactly that way by making every reply feel like an action.
"""

from __future__ import annotations

import json
from pathlib import Path

from nanobeard.evals.client import ChatClient

SCENARIOS = Path(__file__).parent / "tool_scenarios.jsonl"


def load_scenarios(path: Path = SCENARIOS) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def parse_call(tool_calls: list[dict]) -> tuple[str | None, dict | None, bool]:
    """(name, arguments, well_formed). Arguments arrive as a JSON *string*."""
    if not tool_calls:
        return None, None, True  # no call is well-formed; correctness is judged elsewhere
    fn = tool_calls[0].get("function") or {}
    name = fn.get("name")
    raw = fn.get("arguments")
    if isinstance(raw, dict):
        return name, raw, True
    try:
        return name, json.loads(raw or "{}"), True
    except (json.JSONDecodeError, TypeError):
        return name, None, False


def required_params(scenario: dict, tool_name: str) -> list[str]:
    for t in scenario["tools"]:
        if t["function"]["name"] == tool_name:
            return t["function"]["parameters"].get("required", [])
    return []


def grade(scenario: dict, tool_calls: list[dict]) -> dict:
    name, args, well_formed = parse_call(tool_calls)
    expected = scenario["expect"]

    if expected is None:
        # Negative case: the only right answer is no call at all.
        return {
            "id": scenario["id"], "negative": True, "called": name,
            "well_formed": well_formed, "restraint": name is None,
            "right_tool": None, "args_ok": None,
        }

    right_tool = name == expected
    args_ok = False
    if right_tool and args is not None:
        missing = [k for k in required_params(scenario, expected) if k not in args]
        # Pinned values are checked loosely: a model may answer 300 or "300".
        wrong = [
            k for k, v in (scenario.get("args") or {}).items()
            if v is not None and str(args.get(k, "")).strip().lower() != str(v).strip().lower()
        ]
        args_ok = not missing and not wrong
    return {
        "id": scenario["id"], "negative": False, "called": name,
        "well_formed": well_formed, "restraint": None,
        "right_tool": right_tool, "args_ok": args_ok,
    }


def run(client: ChatClient, scenarios: list[dict] | None = None, workers: int = 4) -> dict:
    scenarios = scenarios or load_scenarios()

    def one(s: dict) -> dict:
        reply = client.chat(s["user"], tools=s["tools"])
        row = grade(s, reply.tool_calls)
        row["error"] = reply.error
        return row

    rows = client.map(scenarios, one, workers=workers)
    pos = [r for r in rows if not r["negative"]]
    neg = [r for r in rows if r["negative"]]
    frac = lambda xs, k: (sum(bool(x[k]) for x in xs) / len(xs)) if xs else 0.0  # noqa: E731
    return {
        "name": "tools",
        "n": len(rows),
        "well_formed": frac(rows, "well_formed"),
        "right_tool": frac(pos, "right_tool"),
        "args_ok": frac(pos, "args_ok"),
        "restraint": frac(neg, "restraint"),
        "n_positive": len(pos),
        "n_negative": len(neg),
        "rows": rows,
    }
