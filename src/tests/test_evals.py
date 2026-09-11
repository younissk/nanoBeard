"""Eval gate scoring.

These graders decide whether a fine-tune ships, so a bug here is worse than no
eval at all — it would green-light a model that lost a capability. Everything
tested is pure: no server, no network.

The answer-extraction tests carry the most weight. A harness that scores "she
sold 72 clips" as wrong under-reports accuracy across the board, and the error
looks like a bad model rather than a bad regex.
"""

from __future__ import annotations

import pytest

from nanobeard.evals.gsm8k import extract_answer, gold_answer, normalize
from nanobeard.evals.tools import grade, load_scenarios, parse_call, required_params
from nanobeard.evals.voice import load_prompts, score_text


# ----- gsm8k answer extraction -----
@pytest.mark.parametrize("raw,want", [
    ("72", "72"), ("  72  ", "72"), ("1,234", "1234"), ("$18", "18"),
    ("18.", "18"), ("72.0", "72"), ("0.5", "0.5"), ("-3", "-3"),
    ("", None), ("abc", None), ("$1,234.00", "1234"),
])
def test_normalize(raw, want):
    assert normalize(raw) == want


def test_marker_wins_over_stray_numbers():
    text = "She sold 48 in April and 24 in May.\n#### 72"
    assert extract_answer(text) == "72"


def test_last_marker_wins_when_the_model_repeats_itself():
    assert extract_answer("#### 10\nwait, let me redo that\n#### 72") == "72"


def test_falls_back_to_the_last_number_without_a_marker():
    # Models routinely ignore the format instruction but still answer correctly.
    assert extract_answer("48 + 24 = 72 clips altogether") == "72"


def test_trailing_prose_after_the_number_still_extracts():
    assert extract_answer("The answer is 72 clips.") == "72"


def test_no_number_at_all_is_none_not_a_crash():
    assert extract_answer("I'm not sure how to solve this.") is None


def test_gold_answers_parse_from_the_dataset_format():
    ref = "She makes 9 * 2 = $<<9*2=18>>18 every day.\n#### 18"
    assert gold_answer(ref) == "18"


def test_calculator_annotations_do_not_leak_into_the_gold_answer():
    # The <<48/2=24>> spans contain numbers that appear after the last real one
    # in some rows; the #### marker has to win or golds are silently wrong.
    assert gold_answer("Natalia sold 48/2 = <<48/2=24>>24 clips.\n#### 72") == "72"


# ----- tool call parsing -----
def test_arguments_arrive_as_a_json_string():
    calls = [{"function": {"name": "get_weather", "arguments": '{"location": "Vienna"}'}}]
    name, args, ok = parse_call(calls)
    assert (name, args, ok) == ("get_weather", {"location": "Vienna"}, True)


def test_malformed_argument_json_is_not_well_formed():
    calls = [{"function": {"name": "f", "arguments": "{not json"}}]
    _, args, ok = parse_call(calls)
    assert args is None and ok is False


def test_no_call_counts_as_well_formed():
    # Absence of a call is judged by right_tool/restraint, not by well-formedness.
    assert parse_call([]) == (None, None, True)


def test_dict_arguments_are_accepted():
    calls = [{"function": {"name": "f", "arguments": {"a": 1}}}]
    assert parse_call(calls)[1] == {"a": 1}


# ----- grading -----
WEATHER = {"type": "function", "function": {
    "name": "get_weather",
    "parameters": {"type": "object",
                   "properties": {"location": {"type": "string"}, "unit": {"type": "string"}},
                   "required": ["location"]}}}
POS = {"id": "t", "tools": [WEATHER], "user": "?", "expect": "get_weather",
       "args": {"location": "Vienna"}}
NEG = {"id": "n", "tools": [WEATHER], "user": "?", "expect": None, "args": None}


def _call(name, args):
    import json as _j
    return [{"function": {"name": name, "arguments": _j.dumps(args)}}]


def test_right_tool_and_args():
    r = grade(POS, _call("get_weather", {"location": "Vienna"}))
    assert r["right_tool"] and r["args_ok"]


def test_wrong_tool_fails_even_with_good_args():
    r = grade(POS, _call("search_web", {"location": "Vienna"}))
    assert not r["right_tool"] and not r["args_ok"]


def test_missing_required_param_fails_args():
    r = grade(POS, _call("get_weather", {"unit": "celsius"}))
    assert r["right_tool"] and not r["args_ok"]


def test_wrong_pinned_value_fails_args():
    r = grade(POS, _call("get_weather", {"location": "Oslo"}))
    assert r["right_tool"] and not r["args_ok"]


def test_pinned_values_compare_loosely():
    # A model may emit 300 or "300"; both are right.
    s = {**POS, "args": {"location": "300"}}
    assert grade(s, _call("get_weather", {"location": 300}))["args_ok"]


def test_extra_unpinned_args_are_allowed():
    r = grade(POS, _call("get_weather", {"location": "Vienna", "unit": "celsius"}))
    assert r["args_ok"]


def test_negative_scenario_rewards_silence():
    assert grade(NEG, [])["restraint"] is True


def test_negative_scenario_punishes_over_calling():
    # Measured 2026-09-10: a heavy persona prompt took tool-calling from 8/15 to
    # 0/15. Restraint and right_tool have to be scored separately or a model that
    # never calls anything looks perfect on one of them.
    assert grade(NEG, _call("get_weather", {"location": "X"}))["restraint"] is False


def test_required_params_reads_the_schema():
    assert required_params(POS, "get_weather") == ["location"]
    assert required_params(POS, "nonexistent") == []


# ----- voice scoring -----
def test_plain_english_scores_zero():
    s = score_text("The weather in Vienna is fourteen degrees and raining.")
    assert s["markers"] == 0 and not s["in_voice"]


def test_spelling_and_content_markers_are_counted_apart():
    s = score_text("Ahoy matey, ye be lookin' fer th' doubloons")
    assert s["spelling"] > 0 and s["content"] > 0
    assert s["markers"] == s["spelling"] + s["content"]


def test_density_is_per_hundred_words():
    s = score_text("arr " + "word " * 99)
    assert s["words"] == 100
    assert s["density"] == pytest.approx(1.0, abs=0.01)


def test_empty_reply_does_not_divide_by_zero():
    assert score_text("")["density"] == 0.0


# ----- fixtures on disk -----
def test_tool_scenarios_are_well_formed():
    rows = load_scenarios()
    assert rows
    for r in rows:
        assert r["tools"] and r["user"]
        names = {t["function"]["name"] for t in r["tools"]}
        if r["expect"] is not None:
            assert r["expect"] in names, r["id"]


def test_scenarios_include_negatives():
    # Without them, "never calls a tool" scores 100% on restraint by vacuum.
    assert sum(1 for r in load_scenarios() if r["expect"] is None) >= 3


def test_scenario_ids_are_unique():
    ids = [r["id"] for r in load_scenarios()]
    assert len(ids) == len(set(ids))


def test_voice_prompts_load_from_the_canonical_set():
    prompts = load_prompts()
    assert len(prompts) > 50
    assert all(p["user"] and p["category"] for p in prompts)


def test_empty_prompts_are_excluded_from_voice_scoring():
    # edge-01 is an intentionally empty prompt. Scoring "is this in character?"
    # against no input measures nothing and drags the rate down.
    assert all(p["user"].strip() for p in load_prompts())
    assert "edge-01" not in {p["id"] for p in load_prompts()}
