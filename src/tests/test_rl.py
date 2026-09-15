"""Search RLVR: the reward, the environment, and the GRPO advantage.

The reward is the whole experiment. If it can be gamed, the policy will game it
rather than learn to search, and the run will look like a success. So the tests
that matter most here are the ones asserting a *cheat does not pay*.
"""

from __future__ import annotations

import pytest

from nanobeard.rl import rewards as R
from nanobeard.rl.corpus import BM25, Doc, Question, tokenize
from nanobeard.rl.env import format_results, parse_action, run_episode
from nanobeard.rl.grpo import group_advantages

DOCS = [
    Doc("Scott Derrickson", "Scott Derrickson is an American director born in Denver."),
    Doc("Ed Wood", "Edward Davis Wood Jr. was an American filmmaker."),
    Doc("Salzburg", "Salzburg straddles the Salzach river in Austria."),
    Doc("Tyler Bates", "Tyler Bates is an American musician and composer."),
]


@pytest.fixture
def index():
    return BM25(DOCS)


# ----- the reward cannot be padded or faked -----
def test_answer_normalisation_ignores_case_articles_and_punctuation():
    assert R.exact_match("The Beatles", "beatles") == 1.0
    assert R.exact_match("American.", "american") == 1.0
    assert R.exact_match("  yes  ", "Yes") == 1.0


def test_wrong_answer_scores_zero_however_it_is_dressed():
    assert R.exact_match("definitely American, I am certain", "Austrian") == 0.0


def test_padding_the_answer_does_not_raise_exact_match():
    # Token F1 rewards overlap, so a wordy answer scores well on it. That is
    # precisely why F1 is a diagnostic and EM is the reward.
    verbose = "the answer is American I think probably"
    assert R.token_f1(verbose, "American") > 0.0
    assert R.exact_match(verbose, "American") == 0.0


def test_retrieval_stats_are_not_in_the_total():
    """The central design claim: retrieving perfectly earns nothing by itself.

    Recall is hackable — a query of "the" returns everything — so it must never
    be worth points."""
    cheat = R.compute("wrong", "right", did_search=True,
                      retrieved_titles=["A", "B", "C"], gold_titles=["A", "B"])
    honest = R.compute("right", "right", did_search=True,
                       retrieved_titles=[], gold_titles=["A", "B"])
    assert cheat.breakdown["retrieval_recall"] == 1.0
    assert honest.breakdown["retrieval_recall"] == 0.0
    assert honest.total > cheat.total, "answering beats retrieving"


def test_format_bonus_is_far_smaller_than_answering():
    """It exists to give an untrained policy a gradient, not to be a strategy."""
    empty = R.compute("", "American", did_search=False)
    formatted_but_wrong = R.compute("banana", "American", did_search=True)
    correct = R.compute("American", "American", did_search=False)
    assert empty.total < formatted_but_wrong.total < correct.total
    assert formatted_but_wrong.total < 0.5 * correct.total


def test_no_answer_is_distinguishable_from_a_wrong_one():
    assert R.compute(None, "x", did_search=True).format_ok == 0.0
    assert R.compute("nope", "x", did_search=True).format_ok == 1.0


# ----- GRPO advantages -----
def test_advantages_centre_on_the_group_mean():
    a = group_advantages([0.0, 1.0])
    assert a[0] < 0 < a[1]
    assert abs(sum(a)) < 1e-6


def test_identical_rewards_give_no_signal():
    # The degenerate case: every attempt scored alike, so nothing says which was
    # better. Training on it would be noise.
    assert group_advantages([0.5, 0.5, 0.5, 0.5]) == [0.0] * 4


def test_a_single_rollout_has_no_baseline():
    assert group_advantages([1.0]) == [0.0]


def test_advantage_is_scale_free():
    # Doubling every reward must not double the step size; the division by the
    # group's spread is what makes GRPO's learning rate mean anything.
    small = group_advantages([0.0, 0.1, 0.2])
    large = group_advantages([0.0, 1.0, 2.0])
    assert all(abs(s - lg) < 1e-6 for s, lg in zip(small, large, strict=True))


# ----- the environment protocol -----
def test_search_tag_is_parsed():
    s = parse_action("<search>Mozart birthplace</search>")
    assert (s.kind, s.content) == ("search", "Mozart birthplace")


def test_answer_tag_is_parsed():
    assert parse_action("<answer>Salzach</answer>").kind == "answer"


def test_first_tag_wins():
    # A model that emits both must be judged on what it did first.
    assert parse_action("<search>a</search><answer>b</answer>").kind == "search"
    assert parse_action("<answer>b</answer><search>a</search>").kind == "answer"


def test_prose_without_tags_is_invalid():
    # The measured failure mode of the untrained policy: it knows the answer and
    # writes it as a sentence, which must score as a miss.
    assert parse_action("Scott Derrickson is American.").kind == "invalid"


def test_bm25_finds_the_relevant_document(index):
    hits = index.search("Derrickson American director", k=2)
    assert index.docs[hits[0][0]].title == "Scott Derrickson"


def test_bm25_ignores_stopwords(index):
    # "the" must not retrieve the whole corpus — that is the recall cheat.
    assert tokenize("the of and a") == []
    assert index.search("the", k=5) == []


def test_results_are_rendered_with_titles(index):
    text, titles = format_results(index, "Salzburg river", k=2)
    assert "<results>" in text and "Salzburg" in titles


# ----- episodes -----
Q = Question(qid="1", question="Was Ed Wood American?", answer="yes", gold_titles=["Ed Wood"])


def _scripted(*chunks):
    it = iter(chunks)
    return lambda _body: next(it, "<answer>giving up</answer>")


def test_episode_records_search_then_answer(index):
    ep = run_episode(Q, index, _scripted("<search>Ed Wood</search>", "<answer>yes</answer>"))
    assert [s.kind for s in ep.steps] == ["search", "answer"]
    assert ep.answer == "yes" and ep.did_search


def test_episode_stops_at_the_search_limit(index):
    ep = run_episode(Q, index, _scripted(*["<search>x</search>"] * 5), max_searches=2)
    assert ep.n_searches <= 3


def test_invalid_output_ends_the_episode(index):
    ep = run_episode(Q, index, _scripted("just chatting", "<answer>yes</answer>"))
    assert ep.answer is None and len(ep.steps) == 1


def test_generated_spans_exclude_the_environments_text(index):
    """Only the model's own tokens may carry gradient — training on the pasted
    <results> would teach it to hallucinate search results."""
    ep = run_episode(Q, index, _scripted("<search>Ed Wood</search>", "<answer>yes</answer>"))
    written = "".join(ep.transcript[a:b] for a, b in ep.generated_spans)
    assert "<search>" in written and "<answer>" in written
    assert "<results>" not in written


def test_retrieved_titles_accumulate_across_searches(index):
    ep = run_episode(Q, index,
                     _scripted("<search>Ed Wood</search>", "<search>Salzburg</search>",
                               "<answer>yes</answer>"))
    assert "Ed Wood" in ep.retrieved_titles and "Salzburg" in ep.retrieved_titles
