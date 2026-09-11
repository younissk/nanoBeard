"""Teacher data generation: the validators, and the leakage guard.

Two classes of bug matter here and neither shows up as an error at generation
time. A validator that lets a wrong worked example through teaches confident
wrong arithmetic. And a prompt source that overlaps the eval set turns the gates
into a measurement of the training data, which is worse than having no gates.

No network: the teacher is stubbed.
"""

from __future__ import annotations

import json

import pytest

from nanobeard.distill import prompts as P
from nanobeard.distill.generate import _json_object, gen_chat, gen_math, gen_tool
from nanobeard.distill.teacher import MIN_MAX_TOKENS, Reply, Teacher, Usage


class FakeTeacher:
    """Returns a queued reply per call."""

    def __init__(self, *replies: Reply):
        self.replies = list(replies)
        self.calls: list[tuple] = []

    def ask(self, system, user, tools=None, thinking=None):
        self.calls.append((system, user, tools, thinking))
        return self.replies.pop(0) if self.replies else Reply(content="", error="drained")


# ----- JSON extraction -----
def test_parses_a_bare_object():
    assert _json_object('{"a": 1}') == {"a": 1}


def test_parses_through_a_code_fence():
    # Models wrap JSON in ```json fences even when told not to.
    assert _json_object('```json\n{"a": 1}\n```') == {"a": 1}


def test_parses_with_prose_on_either_side():
    assert _json_object('Sure!\n{"a": 1}\nHope that helps') == {"a": 1}


def test_returns_none_for_unparseable():
    assert _json_object("no json here") is None
    assert _json_object("{broken") is None


def test_returns_none_for_a_bare_array():
    # The tool generator needs an object; an array would KeyError downstream.
    assert _json_object("[1, 2, 3]") is None


# ----- math validation -----
GOOD_MATH = {"question": "2+2?", "reference": "two plus two\n#### 4"}


def test_math_kept_when_the_answer_matches():
    t = FakeTeacher(Reply(content="Arr, 2 an' 2 be 4.\n#### 4"))
    row = gen_math(t, GOOD_MATH)
    assert row and row["kind"] == "math"


def test_math_dropped_when_the_answer_is_wrong():
    # The whole point: a confident pirate stating 5 is worse than no example.
    t = FakeTeacher(Reply(content="Arr, 'tis 5!\n#### 5"))
    assert gen_math(t, GOOD_MATH) is None


def test_math_dropped_without_a_final_marker():
    t = FakeTeacher(Reply(content="Arr, the answer be four, matey."))
    assert gen_math(t, GOOD_MATH) is None


def test_math_tolerates_comma_and_period_formatting():
    item = {"question": "?", "reference": "#### 1234"}
    t = FakeTeacher(Reply(content="#### 1,234."))
    assert gen_math(t, item) is not None


def test_math_dropped_on_teacher_error():
    t = FakeTeacher(Reply(content="", error="HTTP 500"))
    assert gen_math(t, GOOD_MATH) is None


# ----- tool validation -----
TOOLS = [{"type": "function", "function": {"name": "play_music", "parameters": {}}}]
POS = {"tools": TOOLS, "user": "play shanties", "expected": "play_music"}
NEG = {"tools": TOOLS, "user": "how are you?", "expected": None}


def _tool_reply(name, args, result, final):
    return Reply(content=json.dumps(
        {"tool_call": None if name is None else {"name": name, "arguments": args},
         "tool_result": result, "final": final}))


def test_tool_example_has_the_four_turn_shape():
    t = FakeTeacher(_tool_reply("play_music", {"track": "x"}, {"ok": True}, "Arr, playin' now!"))
    row = gen_tool(t, POS)
    assert row is not None
    roles = [m["role"] for m in row["messages"]]
    assert roles == ["system", "user", "assistant", "tool", "assistant"]
    # The call carries no prose and the prose turn carries no call: that split is
    # exactly what the model has to learn.
    assert row["messages"][2]["tool_calls"] and not row["messages"][2]["content"]
    assert row["messages"][4]["content"]


def test_tool_arguments_are_serialized_as_a_json_string():
    t = FakeTeacher(_tool_reply("play_music", {"track": "x"}, {"ok": 1}, "Arr!"))
    row = gen_tool(t, POS)
    assert row is not None
    args = row["messages"][2]["tool_calls"][0]["function"]["arguments"]
    assert json.loads(args) == {"track": "x"}


def test_tool_dropped_when_teacher_calls_a_different_tool():
    t = FakeTeacher(_tool_reply("get_tide", {}, {}, "Arr!"))
    assert gen_tool(t, POS) is None


def test_tool_dropped_when_teacher_refuses_to_call():
    t = FakeTeacher(_tool_reply(None, None, None, "Arr, I cannot."))
    assert gen_tool(t, POS) is None


def test_negative_case_kept_only_when_no_tool_is_called():
    t = FakeTeacher(_tool_reply(None, None, None, "Arr, I be well!"))
    row = gen_tool(t, NEG)
    assert row is not None
    assert row["kind"] == "tool_none"
    assert not any(m.get("tool_calls") for m in row["messages"])


def test_negative_case_dropped_when_teacher_calls_anyway():
    # Over-calling is the measured failure mode; it must not enter the data.
    t = FakeTeacher(_tool_reply("play_music", {}, {}, "Arr!"))
    assert gen_tool(t, NEG) is None


def test_tool_dropped_when_final_is_empty():
    t = FakeTeacher(_tool_reply("play_music", {}, {}, "   "))
    assert gen_tool(t, POS) is None


def test_chat_dropped_on_empty_content():
    assert gen_chat(FakeTeacher(Reply(content="")), "hi") is None


# ----- leakage guards -----
def test_training_tool_schemas_are_disjoint_from_the_eval_set():
    from nanobeard.evals.tools import load_scenarios

    eval_names = {t["function"]["name"] for s in load_scenarios() for t in s["tools"]}
    train_names = {t["function"]["name"] for t in P.TOOL_POOL}
    assert not (eval_names & train_names), eval_names & train_names


def test_training_tool_requests_are_disjoint_from_eval_prompts():
    from nanobeard.evals.tools import load_scenarios

    eval_users = {s["user"].strip().lower() for s in load_scenarios()}
    train_users = {u.strip().lower() for _, u in P.TOOL_REQUESTS}
    assert not (eval_users & train_users)


def test_tool_examples_include_negatives_and_distractors():
    import random

    cases = P.tool_examples(random.Random(0), 20)
    assert sum(1 for c in cases if c["expected"] is None) >= 3
    positives = [c for c in cases if c["expected"]]
    assert all(len(c["tools"]) > 1 for c in positives), "no distractor tools"
    assert all(any(t["function"]["name"] == c["expected"] for t in c["tools"]) for c in positives)


def test_situation_seeds_are_not_used_as_user_messages(monkeypatch):
    """personas.SITUATIONS are briefs for a *user simulator* ("You had an
    exhausting day and want to vent"), not messages a user sends. Fed in
    directly they produce examples where the assistant answers a stage
    direction — which is what the first sample batch actually did."""
    from nanobeard.rejection.personas import SITUATIONS

    fake = [{"instruction": f"Question {i}?", "context": ""} for i in range(50)]

    class FakeDS(list):
        def shuffle(self, seed=0):
            return self

    monkeypatch.setattr(
        "datasets.load_dataset", lambda *a, **k: FakeDS(fake), raising=False
    )
    got = P.chat_prompts(20)
    assert got, "no prompts produced"
    assert not (set(got) & set(SITUATIONS))
    assert SITUATIONS  # still used by selfplay, just not as chat prompts


# ----- teacher guards -----
def test_low_max_tokens_rejected_when_thinking_is_on(monkeypatch):
    monkeypatch.setenv("KIMI_API_KEY", "test-key")
    with pytest.raises(ValueError, match="content would come back empty"):
        Teacher(max_tokens=200, thinking=True)


def test_low_max_tokens_fine_without_thinking(monkeypatch):
    monkeypatch.setenv("KIMI_API_KEY", "test-key")
    Teacher(max_tokens=200, thinking=False)


def test_missing_key_is_a_clear_error(monkeypatch):
    monkeypatch.delenv("KIMI_API_KEY", raising=False)
    monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)
    monkeypatch.setattr("nanobeard.distill.teacher.load_env", lambda: None)
    with pytest.raises(RuntimeError, match="KIMI_API_KEY"):
        Teacher()


def test_usage_tracks_reasoning_tokens_separately():
    u = Usage()
    u.add({"prompt_tokens": 10, "completion_tokens": 100,
           "completion_tokens_details": {"reasoning_tokens": 88}})
    assert (u.calls, u.prompt_tokens, u.completion_tokens, u.reasoning_tokens) == (1, 10, 100, 88)
    assert "88%" in u.summary()


def test_min_max_tokens_is_documented_as_a_real_threshold():
    assert MIN_MAX_TOKENS >= 900
