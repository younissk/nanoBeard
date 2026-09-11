"""Prompt sources for teacher generation.

Prompts are free; teacher replies are not. So every prompt here comes from a
local list or a public dataset, and the API budget goes entirely on answers.

**Disjointness matters.** `nanobeard.evals.voice` scores against
`rejection/prompts.jsonl`, and `evals.tools` against `evals/tool_scenarios.jsonl`.
Training on either would turn those gates into a measurement of the training set.
Chat prompts therefore come from dolly, and tool scenarios from a separate pool
built here.
"""

from __future__ import annotations

import random

# Tool schemas deliberately disjoint from evals/tool_scenarios.jsonl.
TOOL_POOL: list[dict] = [
    {"type": "function", "function": {
        "name": "get_tide", "description": "Get tide times for a harbour",
        "parameters": {"type": "object", "properties": {
            "harbour": {"type": "string"}, "date": {"type": "string"}},
            "required": ["harbour"]}}},
    {"type": "function", "function": {
        "name": "play_music", "description": "Play a song or playlist",
        "parameters": {"type": "object", "properties": {
            "track": {"type": "string"}, "shuffle": {"type": "boolean"}},
            "required": ["track"]}}},
    {"type": "function", "function": {
        "name": "add_to_list", "description": "Add an item to a named list",
        "parameters": {"type": "object", "properties": {
            "list_name": {"type": "string"}, "item": {"type": "string"}},
            "required": ["list_name", "item"]}}},
    {"type": "function", "function": {
        "name": "get_distance", "description": "Distance between two places",
        "parameters": {"type": "object", "properties": {
            "origin": {"type": "string"}, "destination": {"type": "string"},
            "mode": {"type": "string", "enum": ["sail", "walk", "drive"]}},
            "required": ["origin", "destination"]}}},
    {"type": "function", "function": {
        "name": "translate_text", "description": "Translate text to a language",
        "parameters": {"type": "object", "properties": {
            "text": {"type": "string"}, "target_language": {"type": "string"}},
            "required": ["text", "target_language"]}}},
    {"type": "function", "function": {
        "name": "create_reminder", "description": "Create a reminder at a time",
        "parameters": {"type": "object", "properties": {
            "what": {"type": "string"}, "when": {"type": "string"}},
            "required": ["what", "when"]}}},
    {"type": "function", "function": {
        "name": "roll_dice", "description": "Roll dice and return the total",
        "parameters": {"type": "object", "properties": {
            "sides": {"type": "integer"}, "count": {"type": "integer"}},
            "required": ["sides"]}}},
    {"type": "function", "function": {
        "name": "get_stock_price", "description": "Look up a share price",
        "parameters": {"type": "object", "properties": {
            "ticker": {"type": "string"}}, "required": ["ticker"]}}},
]

TOOL_REQUESTS: list[tuple[str, str]] = [
    ("get_tide", "When's high tide at Bristol harbour tomorrow?"),
    ("get_tide", "Tide times for Galway, please."),
    ("play_music", "Put on some sea shanties."),
    ("play_music", "Play Wellerman, shuffled."),
    ("add_to_list", "Add rum to the provisions list."),
    ("add_to_list", "Put 'mend the mainsail' on my chores list."),
    ("get_distance", "How far is it from Lisbon to Madeira by sail?"),
    ("get_distance", "Distance from Dublin to Cork, driving?"),
    ("translate_text", "How do I say 'good morning' in Portuguese?"),
    ("translate_text", "Translate 'where is the harbour' into Spanish."),
    ("create_reminder", "Remind me to check the rigging at six tomorrow."),
    ("create_reminder", "Set a reminder to call Ana on Friday."),
    ("roll_dice", "Roll me two six-sided dice."),
    ("roll_dice", "Roll a twenty-sided die."),
    ("get_stock_price", "What's the share price of MAERSK-B?"),
    ("get_stock_price", "Look up AAPL for me."),
]

# Prompts where no tool fits. Without these the model learns that a tool call is
# always the answer, which is the failure the eval's restraint axis catches.
NO_TOOL_REQUESTS: list[str] = [
    "I'm feeling a bit lost lately.",
    "What's your favourite thing about the sea?",
    "Tell me a short joke.",
    "Do you ever get lonely out there?",
    "What is 14 plus 29?",
    "Explain what a knot is, as a unit of speed.",
]


def tool_examples(rng: random.Random, n: int, distractors: int = 2) -> list[dict]:
    """(tools, user, expected) triples, mixing positives and no-tool cases."""
    by_name = {t["function"]["name"]: t for t in TOOL_POOL}
    out: list[dict] = []
    for i in range(n):
        # Roughly one in four has no applicable tool.
        if i % 4 == 3:
            user = rng.choice(NO_TOOL_REQUESTS)
            tools = rng.sample(TOOL_POOL, k=min(distractors + 1, len(TOOL_POOL)))
            out.append({"tools": tools, "user": user, "expected": None})
            continue
        name, user = rng.choice(TOOL_REQUESTS)
        others = [t for t in TOOL_POOL if t["function"]["name"] != name]
        tools = [by_name[name]] + rng.sample(others, k=min(distractors, len(others)))
        rng.shuffle(tools)
        out.append({"tools": tools, "user": user, "expected": name})
    return out


def chat_prompts(n: int, seed: int = 7) -> list[str]:
    """General instructions from dolly, plus the local situation seeds.

    Dolly rather than `rejection/prompts.jsonl`: the latter is what the voice
    gate scores against.
    """
    from datasets import load_dataset

    # NOT personas.SITUATIONS: those are second-person briefs for a *user
    # simulator* ("You had an exhausting day and want to vent"), not messages a
    # user sends. Fed in directly they produce examples where the assistant
    # answers a stage direction, which is exactly the wrong thing to imitate.
    rng = random.Random(seed)
    ds = load_dataset("TeeZee/dolly-15k-pirate-speech", split="train").shuffle(seed=seed)
    out: list[str] = []
    i = 0
    while len(out) < n and i < len(ds):
        row = ds[i]
        i += 1
        instr = (row["instruction"] or "").strip()
        ctx = (row["context"] or "").strip()
        # Skip rows that only make sense with a long pasted passage.
        if not instr or len(ctx) > 400:
            continue
        out.append(f"{instr}\n\n{ctx}".strip() if ctx else instr)
    rng.shuffle(out)
    return out[:n]


def math_prompts(n: int, seed: int = 7) -> list[dict]:
    """GSM8K TRAIN only. Test is held out for the eval, permanently."""
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="train").shuffle(seed=seed)
    n = min(n, len(ds))
    return [{"question": ds[i]["question"], "reference": ds[i]["answer"]} for i in range(n)]
