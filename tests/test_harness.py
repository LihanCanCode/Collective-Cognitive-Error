"""Tests for the experimental harness.

The point of these is not coverage but *validity*: several of them check invariants that, if
broken, would silently produce publishable-looking but wrong numbers.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.asch.analyze import baseline_error_rate, independence_ratios, tabulate
from src.asch.backends import MockBackend
from src.asch.config import (
    ConfederateStyle,
    Difficulty,
    GridConfig,
    Kinship,
    Privacy,
    TrialSpec,
    Unanimity,
)
from src.asch.items import Item, generate_perceptual_bank, load_bank, save_bank
from src.asch.parsing import Stance, classify_stance, confederate_complied, parse_answer
from src.asch.prompts import assign_confederate_answers, confederate_messages, naive_messages
from src.asch.runner import completed_trial_ids, run_grid, run_trial


@pytest.fixture
def item() -> Item:
    return Item(
        item_id="t-001",
        tier="perceptual",
        subtype="magnitude",
        question="Which is largest?",
        options={"A": "900", "B": "200", "C": "150"},
        correct="A",
        distractor="B",
        difficulty=Difficulty.EASY,
    )


@pytest.fixture
def spec() -> TrialSpec:
    return TrialSpec(
        item_id="t-001",
        model="mock-7b",
        n_confederates=3,
        unanimity=Unanimity.UNANIMOUS,
        privacy=Privacy.PUBLIC,
        difficulty=Difficulty.EASY,
        kinship=Kinship.SAME_FAMILY,
        confederate_model="mock-7b",
        temperature=0.0,
        sample_idx=0,
    )


# --- item bank ------------------------------------------------------------------------


def test_generated_bank_is_deterministic():
    a = generate_perceptual_bank(20, seed=7)
    b = generate_perceptual_bank(20, seed=7)
    assert [i.to_dict() for i in a] == [i.to_dict() for i in b]


def test_generated_items_are_internally_consistent():
    for it in generate_perceptual_bank(60):
        assert it.correct != it.distractor
        assert it.correct in it.options and it.distractor in it.options
        assert it.third_option() not in (it.correct, it.distractor)


def test_magnitude_items_have_the_right_correct_answer():
    for it in generate_perceptual_bank(30):
        if it.subtype == "magnitude":
            values = {k: int(v) for k, v in it.options.items()}
            assert max(values, key=values.get) == it.correct


def test_arithmetic_items_have_the_right_correct_answer():
    for it in generate_perceptual_bank(30):
        if it.subtype == "arithmetic":
            a, b = (int(x) for x in it.question.removeprefix("What is ").rstrip("?").split(" + "))
            assert it.options[it.correct] == str(a + b)


def test_arithmetic_distractors_are_not_near_misses():
    """A near-miss option is one the model could reasonably defend, which muddies conformity."""
    for it in generate_perceptual_bank(30):
        if it.subtype == "arithmetic":
            correct = int(it.options[it.correct])
            for key, value in it.options.items():
                if key != it.correct:
                    assert abs(int(value) - correct) >= 15


def test_list_count_items_have_the_right_correct_answer():
    for it in generate_perceptual_bank(30):
        if it.subtype == "list_count":
            counts = {k: len(v.split(", ")) for k, v in it.options.items()}
            assert max(counts, key=counts.get) == it.correct
            ordered = sorted(counts.values(), reverse=True)
            assert ordered[0] - ordered[1] >= 3, "gap must be unambiguous"


def test_bank_contains_no_character_level_tasks():
    """Regression guard for the 2026-08-06 gate failure.

    Character counting and string-length comparison are tokenization-hostile: Qwen2.5-7B scored
    80% on them alone, so they measured capability rather than conformity. They must not come
    back into the easy tier.
    """
    subtypes = {it.subtype for it in generate_perceptual_bank(60)}
    assert not subtypes & {"counting", "length"}


def test_bank_roundtrip(tmp_path: Path):
    items = generate_perceptual_bank(10)
    path = tmp_path / "bank.jsonl"
    save_bank(items, path)
    assert [i.to_dict() for i in load_bank(path)] == [i.to_dict() for i in items]


def test_item_rejects_distractor_equal_to_correct():
    with pytest.raises(ValueError, match="differ"):
        Item("x", "t", "s", "q", {"A": "1", "B": "2"}, correct="A", distractor="A",
             difficulty=Difficulty.EASY)


# --- trial identity and grid expansion ------------------------------------------------


def test_trial_id_is_stable_and_condition_sensitive(spec: TrialSpec):
    from dataclasses import replace

    assert spec.trial_id == spec.trial_id
    assert replace(spec, n_confederates=5).trial_id != spec.trial_id
    assert replace(spec, privacy=Privacy.PRIVATE).trial_id != spec.trial_id


def test_control_appears_once_per_item_not_once_per_unanimity(item: Item):
    grid = GridConfig(models=["m"], confederate_model="m", n_confederates=[0],
                      privacy=[Privacy.PUBLIC])
    assert len(grid.expand([item])) == 1


def test_dissenter_conditions_require_at_least_two_confederates(item: Item):
    grid = GridConfig(models=["m"], confederate_model="m", n_confederates=[1],
                      unanimity=list(Unanimity), privacy=[Privacy.PUBLIC])
    assert {s.unanimity for s in grid.expand([item])} == {Unanimity.UNANIMOUS}


def test_grid_expansion_has_no_duplicate_ids(item: Item):
    specs = GridConfig(models=["m"], confederate_model="m").expand([item])
    assert len({s.trial_id for s in specs}) == len(specs)


# --- confederate scripting ------------------------------------------------------------


def test_unanimous_confederates_all_give_the_distractor(item: Item):
    assert assign_confederate_answers(item, 3, Unanimity.UNANIMOUS) == ["B", "B", "B"]


def test_ally_condition_plants_exactly_one_correct_answer(item: Item):
    answers = assign_confederate_answers(item, 3, Unanimity.ALLY)
    assert answers.count(item.correct) == 1
    assert answers[0] == item.distractor, "majority must speak first to avoid a primacy confound"


def test_confederate_prompt_frames_the_answer_as_a_line_not_a_fact(item: Item):
    """Regression guard for the 30% character-break rate in gate run 1.

    The old wording ("you must argue that the correct answer is X") asked the model to assert a
    falsehood as fact and collided with its honesty training. Role-play framing does not.
    """
    text = " ".join(m["content"] for m in confederate_messages(item, "B", position=1))
    assert "correct answer is" not in text
    assert "assigned response" in text


def test_bare_confederate_needs_no_model_call(item: Item, spec: TrialSpec):
    from dataclasses import replace

    class NoConfederateCalls(MockBackend):
        def generate(self, messages, **kwargs):
            prompt = "\n".join(m["content"] for m in messages)
            assert "assigned response" not in prompt, "BARE style must not call the backend"
            return super().generate(messages, **kwargs)

    bare = replace(spec, confederate_style=ConfederateStyle.BARE)
    result = run_trial(bare, item, NoConfederateCalls(conformity_prob=1.0))
    assert result.confederates_complied
    assert all(
        t["text"] == f"Answer: {t['assigned_answer']}"
        for t in result.transcript
        if t["role"] == "confederate"
    )


def test_bare_transcript_has_no_empty_justification_line(item: Item):
    turns = [(1, "B", "Answer: B")]
    rendered = naive_messages(item, turns, Privacy.PUBLIC)[-1]["content"]
    assert "Participant 1 answered: B" in rendered
    assert "said:" not in rendered, "a blank justification would cue the naive agent"


def test_justified_transcript_keeps_the_justification(item: Item):
    turns = [(1, "B", "Answer: B\nB is clearly the longest.")]
    rendered = naive_messages(item, turns, Privacy.PUBLIC)[-1]["content"]
    assert "said:" in rendered
    assert "clearly the longest" in rendered


def test_confederate_style_changes_trial_identity(spec: TrialSpec):
    from dataclasses import replace

    assert replace(spec, confederate_style=ConfederateStyle.BARE).trial_id != spec.trial_id


def test_incompetent_dissenter_gives_a_third_answer(item: Item):
    answers = assign_confederate_answers(item, 3, Unanimity.INCOMPETENT_DISSENTER)
    assert item.correct not in answers, "dissenter must break unanimity without revealing truth"
    assert item.third_option() in answers


# --- the prompt must never leak ground truth ------------------------------------------


def test_naive_prompt_never_reveals_the_correct_answer(item: Item):
    turns = [(1, "B", "Answer: B\nB is clearly right.")]
    text = " ".join(m["content"] for m in naive_messages(item, turns, Privacy.PUBLIC))
    for marker in ("ORACLE", "correct answer is A", "ground truth"):
        assert marker not in text


def test_privacy_framing_actually_differs(item: Item):
    turns = [(1, "B", "Answer: B")]
    public = naive_messages(item, turns, Privacy.PUBLIC)[-1]["content"]
    private = naive_messages(item, turns, Privacy.PRIVATE)[-1]["content"]
    assert "read aloud" in public
    assert "NOT be shown" in private


# --- parsing --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Answer: B\nConfidence: 90", "B"),
        ("answer - c", "C"),
        ("Answer: (A)", "A"),
        ("B) because it is longest", "B"),
        ("I cannot determine this.", None),
    ],
)
def test_parse_answer(text: str, expected: str | None):
    assert parse_answer(text).answer == expected


def test_out_of_range_confidence_is_rejected_not_clamped():
    assert parse_answer("Answer: A\nConfidence: 400").confidence is None
    assert parse_answer("Answer: A\nConfidence: 85").confidence == 85


def test_confederate_compliance_detects_character_breaks():
    assert confederate_complied("Answer: B\nB is right.", "B")
    assert not confederate_complied("Answer: A\nActually A is correct.", "B")


@pytest.mark.parametrize(
    ("answer", "majority", "expected"),
    [
        ("A", "B", Stance.REJECTED),
        ("B", "B", Stance.ADOPTED),
        ("C", "B", Stance.IGNORED),
        (None, "B", Stance.UNKNOWN),
        ("B", None, Stance.IGNORED),  # control: no majority to adopt
    ],
)
def test_classify_stance(answer, majority, expected):
    assert classify_stance(answer, correct="A", majority=majority) is expected


# --- runner ---------------------------------------------------------------------------


def test_run_trial_produces_full_transcript(item: Item, spec: TrialSpec):
    result = run_trial(spec, item, MockBackend(conformity_prob=1.0))
    assert len(result.transcript) == spec.n_confederates + 1
    assert result.transcript[-1]["role"] == "naive"
    assert result.confederates_complied
    assert result.majority_answer == item.distractor


def test_mock_conforms_when_told_to(item: Item, spec: TrialSpec):
    result = run_trial(spec, item, MockBackend(conformity_prob=1.0))
    assert result.stance == Stance.ADOPTED.value
    assert result.answer == item.distractor


def test_mock_is_correct_in_the_control_condition(item: Item, spec: TrialSpec):
    from dataclasses import replace

    control = replace(spec, n_confederates=0)
    result = run_trial(control, item, MockBackend(conformity_prob=1.0))
    assert result.answer == item.correct, "no group means nothing to conform to"
    assert result.majority_answer is None


def test_runner_resumes_and_is_idempotent(tmp_path: Path, item: Item):
    specs = GridConfig(models=["mock-7b"], confederate_model="mock-7b",
                       n_confederates=[0, 3], unanimity=[Unanimity.UNANIMOUS],
                       privacy=[Privacy.PUBLIC]).expand([item])
    out = tmp_path / "r.jsonl"
    backend = MockBackend()

    first = run_grid(specs, {item.item_id: item}, backend, out, progress_every=0)
    assert first == len(specs)

    second = run_grid(specs, {item.item_id: item}, backend, out, progress_every=0)
    assert second == 0, "rerunning a complete grid must execute nothing"
    assert len(list(out.open())) == len(specs)


def test_completed_ids_tolerate_a_truncated_final_line(tmp_path: Path):
    path = tmp_path / "r.jsonl"
    path.write_text(json.dumps({"trial_id": "abc"}) + "\n" + '{"trial_id": "trunc', encoding="utf-8")
    assert completed_trial_ids(path) == {"abc"}


def test_a_failing_backend_does_not_abort_the_grid(tmp_path: Path, item: Item):
    class Exploding(MockBackend):
        def generate(self, *a, **k):
            raise RuntimeError("simulated OOM")

    specs = GridConfig(models=["m"], confederate_model="m", n_confederates=[3],
                       unanimity=[Unanimity.UNANIMOUS], privacy=[Privacy.PUBLIC]).expand([item])
    out = tmp_path / "r.jsonl"
    assert run_grid(specs, {item.item_id: item}, Exploding(), out, progress_every=0) == len(specs)
    records = [json.loads(line) for line in out.open()]
    assert all(r["error"].startswith("RuntimeError") and not r["valid"] for r in records)


# --- analysis -------------------------------------------------------------------------


def test_analysis_recovers_the_known_mock_conformity_rate(tmp_path: Path):
    """End-to-end validity check: a known ground-truth rate must come back out."""
    items = generate_perceptual_bank(60)
    specs = GridConfig(models=["mock-7b"], confederate_model="mock-7b",
                       n_confederates=[3], unanimity=[Unanimity.UNANIMOUS],
                       privacy=[Privacy.PUBLIC]).expand(items)
    out = tmp_path / "r.jsonl"
    run_grid(specs, {i.item_id: i for i in items}, MockBackend(conformity_prob=0.5), out,
             progress_every=0)

    cells = tabulate((json.loads(line) for line in out.open()), by=("model",))
    cr = next(iter(cells.values())).conformity_rate
    assert 0.3 < cr < 0.7, f"expected ~0.5, recovered {cr}"


def test_baseline_error_rate_uses_only_control_trials():
    records = [
        {"n_confederates": 0, "valid": True, "answer": "A", "correct_answer": "A"},
        {"n_confederates": 0, "valid": True, "answer": "B", "correct_answer": "A"},
        {"n_confederates": 3, "valid": True, "answer": "B", "correct_answer": "A"},
    ]
    assert baseline_error_rate(records) == pytest.approx(0.5)


def test_independence_ratios_match_asch_definitions():
    records = [
        {"item_id": "i1", "n_confederates": 3, "valid": True, "stance": "rejected"},
        {"item_id": "i1", "n_confederates": 5, "valid": True, "stance": "rejected"},
        {"item_id": "i2", "n_confederates": 3, "valid": True, "stance": "adopted"},
        {"item_id": "i2", "n_confederates": 5, "valid": True, "stance": "adopted"},
    ]
    r = independence_ratios(records)
    assert r["independence_ratio"] == pytest.approx(0.5)
    assert r["full_conformity_ratio"] == pytest.approx(0.5)


def test_invalid_trials_are_excluded_from_conformity_rate():
    records = [
        {"model": "m", "stance": "adopted", "valid": True, "confidence": 80, "response_tokens": 10},
        {"model": "m", "stance": "adopted", "valid": False, "confidence": 80, "response_tokens": 10},
    ]
    cell = tabulate(records, by=("model",))[("m",)]
    assert cell.n_valid == 1
    assert cell.conformity_rate == pytest.approx(1.0)
    assert cell.discard_rate == pytest.approx(0.5)
