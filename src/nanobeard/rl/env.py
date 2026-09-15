"""The search episode: the model queries, reads results, and answers.

Protocol is plain text rather than JSON tool calls:

    <search>Scott Derrickson nationality</search>
    -> the environment appends <results>...</results>
    <answer>American</answer>

Text rather than the tool-call API for three reasons. It is the format
Search-R1-style work uses, so results are comparable. A 0.6B model emits it far
more reliably than well-formed JSON, and a rollout that fails to parse is a
wasted sample. And it works with any checkpoint, including stock Qwen3, so the
RL run is not entangled with whether the pirate LoRA's tool training survived.

The episode is deliberately short — a couple of searches then an answer. Every
extra turn is more tokens to generate and more context for a small model to lose
the thread in.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from nanobeard.rl.corpus import BM25, Question

SEARCH_RE = re.compile(r"<search>(.*?)</search>", re.S)
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.S)
# Generation stops the moment a tag closes, so the model never writes the
# environment's half of the conversation for itself.
STOP_STRINGS = ("</search>", "</answer>")

# A worked example, not just rules. Measured on stock Qwen3: with rules alone,
# 4 of 6 rollouts ended by writing the answer as prose with no <answer> tag —
# the model knew "WINNER" and still scored zero. One demonstration is the
# cheapest way to stop burning rollouts on formatting, and RL then has headroom
# to spend on the actual searching.
SYSTEM = """You answer questions by searching an encyclopedia.

Rules:
- To search, write exactly: <search>your query</search>
- To answer, write exactly: <answer>your answer</answer>
- You MUST wrap the final answer in <answer></answer> tags or it does not count.
- Keep the answer short: a name, a date, a place, or yes/no.

Example:

Question: What river runs through the city where Mozart was born?

<search>Mozart birthplace city</search>
<results>
1. Wolfgang Amadeus Mozart: Born 27 January 1756 in Salzburg...
</results>
<search>Salzburg river</search>
<results>
1. Salzburg: The city straddles the Salzach river...
</results>
<answer>Salzach</answer>"""


@dataclass
class Step:
    kind: str            # "search" | "answer" | "invalid"
    content: str
    results: list[str] = field(default_factory=list)


@dataclass
class Episode:
    question: Question
    steps: list[Step] = field(default_factory=list)
    transcript: str = ""
    answer: str | None = None
    retrieved_titles: list[str] = field(default_factory=list)
    # Character spans of what the MODEL wrote. The transcript also contains
    # <results> blocks the environment pasted in, and training on those would
    # teach the policy to hallucinate search results instead of reading them.
    generated_spans: list[tuple[int, int]] = field(default_factory=list)

    @property
    def did_search(self) -> bool:
        return any(s.kind == "search" for s in self.steps)

    @property
    def n_searches(self) -> int:
        return sum(1 for s in self.steps if s.kind == "search")


def format_results(index: BM25, query: str, k: int) -> tuple[str, list[str]]:
    """Rendered search results, plus the titles for the retrieval diagnostics."""
    hits = index.search(query, k=k)
    if not hits:
        return "<results>no matches</results>", []
    titles, lines = [], []
    for rank, (doc_id, _score) in enumerate(hits, 1):
        doc = index.docs[doc_id]
        titles.append(doc.title)
        # One line each: the model has to choose between them, and a small model
        # drowns if handed five full paragraphs.
        snippet = " ".join(doc.text.split())[:240]
        lines.append(f"{rank}. {doc.title}: {snippet}")
    return "<results>\n" + "\n".join(lines) + "\n</results>", titles


def parse_action(text: str) -> Step:
    """First tag wins; anything else is an invalid step that ends the episode."""
    s = SEARCH_RE.search(text)
    a = ANSWER_RE.search(text)
    if s and (not a or s.start() < a.start()):
        return Step("search", s.group(1).strip())
    if a:
        return Step("answer", a.group(1).strip())
    return Step("invalid", text.strip()[:200])


def initial_prompt(question: Question) -> str:
    return f"Question: {question.question}\n"


def run_episodes(questions: list[Question], index: BM25, batch_generate, *,
                 max_searches: int = 2, k: int = 3) -> list[Episode]:
    """Play many episodes in lockstep, one batched generation call per turn.

    Rollouts are the wall-clock cost of GRPO, and generating them one at a time
    wastes the GPU: measured shapes here put an unbatched 100-step run at ~8
    hours against roughly one with batching. Episodes finish at different turns,
    so each round only sends the ones still running.

    `batch_generate(list[str]) -> list[str]`.
    """
    episodes = [Episode(question=q) for q in questions]
    bodies = [initial_prompt(q) for q in questions]
    active = list(range(len(questions)))

    for _ in range(max_searches + 1):
        if not active:
            break
        chunks = batch_generate([bodies[i] for i in active])
        still: list[int] = []
        for i, chunk in zip(active, chunks, strict=True):
            ep = episodes[i]
            start = len(bodies[i])
            bodies[i] += chunk
            ep.generated_spans.append((start, len(bodies[i])))
            step = parse_action(chunk)
            ep.steps.append(step)

            if step.kind == "answer":
                ep.answer = step.content
                continue
            if step.kind == "invalid" or ep.n_searches > max_searches:
                continue

            rendered, titles = format_results(index, step.content, k)
            step.results = titles
            ep.retrieved_titles.extend(titles)
            bodies[i] += "\n" + rendered + "\n"
            still.append(i)
        active = still

    for ep, body in zip(episodes, bodies, strict=True):
        ep.transcript = body
    return episodes


def run_episode(question: Question, index: BM25, generate, *,
                max_searches: int = 2, k: int = 3) -> Episode:
    """Play one episode.

    `generate(prompt) -> str` is injected rather than imported so the same
    environment drives a real policy, a scripted baseline, or a test stub.
    """
    ep = Episode(question=question)
    body = initial_prompt(question)

    for _ in range(max_searches + 1):
        chunk = generate(body)
        start = len(body)
        body += chunk
        ep.generated_spans.append((start, len(body)))
        step = parse_action(chunk)
        ep.steps.append(step)

        if step.kind == "answer":
            ep.answer = step.content
            break
        if step.kind == "invalid":
            break
        if ep.n_searches > max_searches:
            break

        rendered, titles = format_results(index, step.content, k)
        step.results = titles
        ep.retrieved_titles.extend(titles)
        body += "\n" + rendered + "\n"

    ep.transcript = body
    return ep
