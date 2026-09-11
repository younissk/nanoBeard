"""Label masking for the LoRA.

This is the highest-risk code in the fine-tune path: every bug in it is silent.
The loss still falls, the run still finishes, and the model has learned the
wrong thing. Two real bugs were caught here by decoding the supervised spans and
reading them, and both are pinned below:

* spans sliding one token forward, so a tool example supervised the opening of
  the `<tool_response>` turn — training the model to invent tool output;
* tool-call turns being skipped entirely, because Qwen3's generation prompt adds
  an empty `<think>` block that a tool-call turn does not have.

A hand-rolled tokenizer stands in for the real one so these run offline and fast.
"""

from __future__ import annotations

import re

import pytest

from nanobeard.finetune.data import (
    IGNORE_INDEX,
    Collator,
    Example,
    assistant_spans,
    build_example,
    describe,
    split,
    turn_markers,
)

HEADER = "<|im_start|>assistant\n"
END = "<|im_end|>"


# ----- span location, pure strings -----
def test_finds_a_single_assistant_turn():
    t = f"<|im_start|>user\nhi{END}\n{HEADER}yarr{END}\n"
    (lo, hi), = assistant_spans(t, HEADER, END)
    assert t[lo:hi] == f"yarr{END}"


def test_end_marker_is_inside_the_span():
    # Stopping is a behaviour the model has to learn; a model that never emits
    # the end marker runs to the token limit every turn.
    t = f"{HEADER}yarr{END}\n"
    (lo, hi), = assistant_spans(t, HEADER, END)
    assert t[lo:hi].endswith(END)


def test_header_itself_is_not_supervised():
    t = f"{HEADER}yarr{END}\n"
    (lo, _), = assistant_spans(t, HEADER, END)
    assert not t[lo:].startswith("<|im_start|>")


def test_two_assistant_turns_in_one_conversation():
    t = (f"<|im_start|>user\na{END}\n{HEADER}call{END}\n"
         f"<|im_start|>user\nresult{END}\n{HEADER}answer{END}\n")
    spans = assistant_spans(t, HEADER, END)
    assert [t[a:b] for a, b in spans] == [f"call{END}", f"answer{END}"]


def test_user_turns_between_assistant_turns_are_not_captured():
    # The tool response comes back as a *user* turn in Qwen3's template, so this
    # is the case that decides whether the model learns to fake tool output.
    t = (f"{HEADER}call{END}\n<|im_start|>user\n<tool_response>DATA</tool_response>{END}\n"
         f"{HEADER}answer{END}\n")
    joined = " ".join(t[a:b] for a, b in assistant_spans(t, HEADER, END))
    assert "DATA" not in joined
    assert "tool_response" not in joined


def test_spans_do_not_overlap_or_run_backwards():
    t = f"{HEADER}a{END}\n{HEADER}b{END}\n{HEADER}c{END}\n"
    spans = assistant_spans(t, HEADER, END)
    assert len(spans) == 3
    assert all(a < b for a, b in spans)
    assert all(spans[i][1] <= spans[i + 1][0] for i in range(len(spans) - 1))


def test_unterminated_final_turn_is_dropped():
    t = f"{HEADER}a{END}\n{HEADER}b-with-no-end"
    assert len(assistant_spans(t, HEADER, END)) == 1


def test_no_assistant_turn_gives_no_spans():
    assert assistant_spans(f"<|im_start|>user\nhi{END}\n", HEADER, END) == []


# ----- a tokenizer stand-in -----
class FakeTok:
    """ChatML-ish template with Qwen3's two awkward behaviours.

    Reproduces the empty `<think>` block on the generation prompt, and the habit
    of keeping that block only on the final assistant turn.
    """

    eos_token = END
    pad_token_id = 0

    def apply_chat_template(self, messages, tools=None, tokenize=False,
                            add_generation_prompt=False):
        out = []
        if tools:
            out.append(f"<|im_start|>system\nTOOLS:{len(tools)}{END}\n")
        last_assistant = max(
            (i for i, m in enumerate(messages) if m["role"] == "assistant"), default=-1
        )
        for i, m in enumerate(messages):
            if m["role"] == "assistant":
                body = ""
                if m.get("tool_calls"):
                    body = f"<tool_call>{m['tool_calls'][0]['function']['name']}</tool_call>"
                else:
                    # think block only on the final assistant turn
                    think = "<think>\n\n</think>\n\n" if i == last_assistant else ""
                    body = think + (m.get("content") or "")
                out.append(f"{HEADER}{body}{END}\n")
            else:
                out.append(f"<|im_start|>{m['role']}\n{m.get('content', '')}{END}\n")
        if add_generation_prompt:
            out.append(f"{HEADER}<think>\n\n</think>\n\n")
        return "".join(out)

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        # One "token" per word-ish chunk, with real character offsets.
        ids, offs = [], []
        for m in re.finditer(r"\S+|\s+", text):
            ids.append(len(ids) + 1)
            offs.append((m.start(), m.end()))
        out = {"input_ids": ids}
        if return_offsets_mapping:
            out["offset_mapping"] = offs
        return out


@pytest.fixture
def tok():
    return FakeTok()


def _supervised_text(tok, ex: Example, full: str) -> str:
    enc = tok(full, return_offsets_mapping=True)
    return "".join(
        full[a:b] for (a, b), lab in zip(enc["offset_mapping"], ex.labels, strict=True)
        if lab != IGNORE_INDEX
    )


CHAT = {"kind": "chat", "messages": [
    {"role": "system", "content": "SYSPROMPT"},
    {"role": "user", "content": "USERMSG"},
    {"role": "assistant", "content": "PIRATEREPLY"},
]}
TOOL = {"kind": "tool", "tools": [{"function": {"name": "f"}}], "messages": [
    {"role": "system", "content": "SYSPROMPT"},
    {"role": "user", "content": "USERMSG"},
    {"role": "assistant", "content": "",
     "tool_calls": [{"function": {"name": "get_stock_price"}}]},
    {"role": "user", "content": "<tool_response>TOOLDATA</tool_response>"},
    {"role": "assistant", "content": "PIRATEREPLY"},
]}


def test_chat_supervises_only_the_reply(tok):
    ex = build_example(tok, CHAT)
    text = tok.apply_chat_template(CHAT["messages"])
    sup = _supervised_text(tok, ex, text)
    assert "PIRATEREPLY" in sup
    assert "SYSPROMPT" not in sup and "USERMSG" not in sup


def test_tool_call_turn_is_supervised(tok):
    # The regression: a strict prefix check skipped this entirely because the
    # generation prompt carries a think block and a tool call does not.
    ex = build_example(tok, TOOL)
    sup = _supervised_text(tok, ex, tok.apply_chat_template(TOOL["messages"], TOOL["tools"]))
    assert "get_stock_price" in sup


def test_tool_response_is_never_supervised(tok):
    ex = build_example(tok, TOOL)
    sup = _supervised_text(tok, ex, tok.apply_chat_template(TOOL["messages"], TOOL["tools"]))
    assert "TOOLDATA" not in sup


def test_tool_example_supervises_both_assistant_turns(tok):
    ex = build_example(tok, TOOL)
    sup = _supervised_text(tok, ex, tok.apply_chat_template(TOOL["messages"], TOOL["tools"]))
    assert "get_stock_price" in sup and "PIRATEREPLY" in sup


def test_span_count_mismatch_drops_the_example(tok):
    # Claim three assistant turns, render two: refuse rather than mislabel.
    row = {"kind": "x", "messages": CHAT["messages"] + [{"role": "assistant", "content": "B"}]}
    orig = tok.apply_chat_template

    def truncated(messages, *a, **k):
        return orig([m for m in messages if m["role"] != "assistant"], *a, **k)

    tok.apply_chat_template = truncated
    assert build_example(tok, row) is None


def test_overlong_example_is_dropped_not_truncated(tok):
    # Truncating mid-answer teaches the model to stop mid-answer.
    assert build_example(tok, CHAT, max_len=3) is None


def test_markers_are_derived_from_the_template(tok):
    header, end = turn_markers(tok)
    assert header == HEADER, "think block must be trimmed off the header"
    assert end == END


# ----- batching and splitting -----
def test_collator_pads_and_never_supervises_padding():
    a = Example(input_ids=[1, 2, 3], labels=[IGNORE_INDEX, 2, 3], kind="chat")
    b = Example(input_ids=[4], labels=[4], kind="chat")
    batch = Collator(pad_token_id=0)([a, b])
    assert batch["input_ids"].shape == (2, 3)
    assert batch["labels"][1].tolist() == [4, IGNORE_INDEX, IGNORE_INDEX]
    assert batch["attention_mask"][1].tolist() == [1, 0, 0]


def test_split_is_stratified_by_kind():
    ex = [Example([1], [1], k) for k in ["chat"] * 40 + ["tool"] * 40 + ["math"] * 40]
    train, val = split(ex, val_frac=0.25)
    assert {e.kind for e in val} == {"chat", "tool", "math"}
    assert len(train) + len(val) == len(ex)


def test_split_is_deterministic():
    ex = [Example([i], [i], "chat") for i in range(50)]
    a1, b1 = split(ex, seed=5)
    a2, b2 = split(ex, seed=5)
    assert [e.input_ids for e in a1] == [e.input_ids for e in a2]
    assert [e.input_ids for e in b1] == [e.input_ids for e in b2]


def test_describe_reports_every_kind():
    out = describe([Example([1, 2], [IGNORE_INDEX, 2], "chat"),
                    Example([1, 2], [1, 2], "tool")])
    assert "chat" in out and "tool" in out and "TOTAL" in out
