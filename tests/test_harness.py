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
from src.asch.runner import _run_chunk, completed_trial_ids, run_grid, run_trial


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


def test_smallest_items_have_the_right_correct_answer():
    for it in generate_perceptual_bank(40):
        if it.subtype == "smallest":
            values = {k: int(v) for k, v in it.options.items()}
            assert min(values, key=values.get) == it.correct


def test_closest_items_have_the_right_correct_answer():
    for it in generate_perceptual_bank(40):
        if it.subtype == "closest":
            target = int(it.question.split("closest to ")[1].rstrip("?"))
            dists = {k: abs(int(v) - target) for k, v in it.options.items()}
            assert min(dists, key=dists.get) == it.correct
            ordered = sorted(dists.values())
            assert ordered[1] - ordered[0] >= 100, "the winner must be unambiguous"


def test_bank_has_no_alphabetical_items():
    """Regression guard for the gate run 3 failure.

    `alphabetical` scored 33% alone on Qwen2.5-7B -- genuine alphabet errors ("'k' is before 'n'
    and 'f'") -- and then reported 91.7% "conformity" that was really ignorance. Alphabetical
    ordering is a memorised sequence lookup, not a perceptual comparison.
    """
    assert "alphabetical" not in {it.subtype for it in generate_perceptual_bank(60)}


def test_bank_has_no_items_requiring_computation():
    """Regression guard for the gate run 2 failure.

    Two-digit addition scored 88.2% alone -- the model made plain reasoning slips at confidence
    100. Any item with an intermediate computation step lets capability leak into the conformity
    measure. Asch's task was a single-glance perceptual comparison; ours must be too.
    """
    for it in generate_perceptual_bank(60):
        assert "+" not in it.question
        assert it.subtype != "arithmetic"


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


# --- batching -------------------------------------------------------------------------


def _grid_for_batching(items: list[Item]) -> list[TrialSpec]:
    return GridConfig(
        models=["mock-7b"],
        confederate_model="mock-7b",
        n_confederates=[0, 2, 3, 5],
        unanimity=list(Unanimity),
        privacy=list(Privacy),
    ).expand(items)


def test_batched_matches_sequential(tmp_path: Path):
    """The load-bearing test for the batched path.

    Batching is a throughput optimisation and must not change a single result. A silent
    divergence here -- misaligned outputs, wrong padding side, a dedupe collision -- would
    produce plausible numbers that are simply wrong, which is the failure mode that survives
    review and ruins a paper.
    """
    items = generate_perceptual_bank(12)
    item_map = {i.item_id: i for i in items}
    specs = _grid_for_batching(items)

    seq_path, batch_path = tmp_path / "seq.jsonl", tmp_path / "batch.jsonl"
    run_grid(specs, item_map, MockBackend(0.5), seq_path, progress_every=0, batch_size=1)
    run_grid(specs, item_map, MockBackend(0.5), batch_path, progress_every=0, batch_size=16)

    seq = {r["trial_id"]: r for r in (json.loads(x) for x in seq_path.open())}
    bat = {r["trial_id"]: r for r in (json.loads(x) for x in batch_path.open())}

    assert seq.keys() == bat.keys()
    assert len(seq) == len(specs)
    for trial_id, expected in seq.items():
        got = bat[trial_id]
        for field in ("answer", "stance", "correct_answer", "majority_answer",
                      "confederates_complied", "valid", "raw_response", "transcript"):
            assert got[field] == expected[field], f"{field} diverged on {trial_id}"


def test_batching_deduplicates_confederate_calls():
    """Dedupe is where most of the speedup comes from, so verify it actually happens.

    A confederate prompt depends only on (item, assigned answer, position), so every n level
    reuses positions 1..k with the same assignment. Those must collapse to one call.
    """
    calls: list[int] = []

    class Counting(MockBackend):
        def generate_batch(self, batch, **kwargs):
            calls.append(len(batch))
            return super().generate_batch(batch, **kwargs)

    items = generate_perceptual_bank(4)
    specs = GridConfig(
        models=["mock-7b"],
        confederate_model="mock-7b",
        n_confederates=[2, 3, 5],
        unanimity=[Unanimity.UNANIMOUS],
        privacy=[Privacy.PUBLIC],
    ).expand(items)

    _run_chunk(specs, {i.item_id: i for i in items}, Counting())

    naive_calls = len(specs)
    confederate_calls_if_naive = sum(s.n_confederates for s in specs)  # 4 items * (2+3+5) = 40
    assert calls[0] < confederate_calls_if_naive, "confederate prompts were not deduplicated"
    assert calls[0] == 4 * 5, "should collapse to one call per (item, position) at max n"
    assert calls[-1] == naive_calls


def test_filler_matches_justified_length_but_argues_nothing():
    """FILLER is the control that makes the BARE/JUSTIFIED contrast interpretable.

    It must look comparably substantial to a justified turn (so salience is held roughly
    constant) while containing nothing about the stimulus (so only the argument is removed).
    """
    from src.asch.prompts import filler_confederate_text

    texts = [filler_confederate_text("B", p) for p in range(4)]
    assert len(set(texts)) > 1, "identical replies would themselves be a cue"

    for text in texts:
        assert text.startswith("Answer: B")
        sentence = text.split("\n", 1)[1]
        assert 10 <= len(sentence.split()) <= 20, "length should sit in the justified range"
        # Argues nothing: never names an option, a value, or a comparison.
        for banned in ("largest", "smallest", "more", "most", "because", "than", "correct"):
            assert banned not in sentence.lower(), f"{banned!r} leaks an argument"


def test_filler_renders_with_a_said_line_unlike_bare(item: Item):
    """The whole point is that FILLER occupies transcript space that BARE does not."""
    from src.asch.prompts import filler_confederate_text

    bare = naive_messages(item, [(1, "B", "Answer: B")], Privacy.PUBLIC)[-1]["content"]
    filler = naive_messages(
        item, [(1, "B", filler_confederate_text("B", 1))], Privacy.PUBLIC
    )[-1]["content"]

    assert "said:" not in bare
    assert "said:" in filler
    assert len(filler) > len(bare)


def test_filler_needs_no_model_call():
    items = generate_perceptual_bank(4)
    specs = GridConfig(
        models=["mock-7b"],
        confederate_model="mock-7b",
        n_confederates=[3],
        unanimity=[Unanimity.UNANIMOUS],
        privacy=[Privacy.PUBLIC],
        confederate_style=[ConfederateStyle.FILLER],
    ).expand(items)

    calls: list[int] = []

    class Counting(MockBackend):
        def generate_batch(self, batch, **kwargs):
            calls.append(len(batch))
            return super().generate_batch(batch, **kwargs)

    records = _run_chunk(specs, {i.item_id: i for i in items}, Counting(1.0))
    assert all(r["confederates_complied"] for r in records)
    assert calls == [len(specs)], "only the naive turns should be generated"


def test_all_scripted_styles_agree_between_batched_and_sequential(tmp_path: Path):
    items = generate_perceptual_bank(8)
    item_map = {i.item_id: i for i in items}
    specs = GridConfig(
        models=["mock-7b"],
        confederate_model="mock-7b",
        n_confederates=[0, 3],
        unanimity=[Unanimity.UNANIMOUS],
        privacy=list(Privacy),
        confederate_style=list(ConfederateStyle),
    ).expand(items)

    seq, bat = tmp_path / "s.jsonl", tmp_path / "b.jsonl"
    run_grid(specs, item_map, MockBackend(0.5), seq, progress_every=0, batch_size=1)
    run_grid(specs, item_map, MockBackend(0.5), bat, progress_every=0, batch_size=16)

    s = {r["trial_id"]: r for r in (json.loads(x) for x in seq.open())}
    b = {r["trial_id"]: r for r in (json.loads(x) for x in bat.open())}
    assert s.keys() == b.keys()
    for tid in s:
        assert s[tid]["transcript"] == b[tid]["transcript"]
        assert s[tid]["answer"] == b[tid]["answer"]


def test_bare_style_issues_no_confederate_batch():
    items = generate_perceptual_bank(4)
    specs = GridConfig(
        models=["mock-7b"],
        confederate_model="mock-7b",
        n_confederates=[3],
        unanimity=[Unanimity.UNANIMOUS],
        privacy=[Privacy.PUBLIC],
        confederate_style=[ConfederateStyle.BARE],
    ).expand(items)

    records = _run_chunk(specs, {i.item_id: i for i in items}, MockBackend(1.0))
    assert len(records) == len(specs)
    assert all(r["confederates_complied"] for r in records)


def test_batched_runner_resumes(tmp_path: Path):
    items = generate_perceptual_bank(8)
    item_map = {i.item_id: i for i in items}
    specs = _grid_for_batching(items)
    out = tmp_path / "r.jsonl"

    assert run_grid(specs, item_map, MockBackend(), out, progress_every=0, batch_size=8) == len(specs)
    assert run_grid(specs, item_map, MockBackend(), out, progress_every=0, batch_size=8) == 0


# --- response format ------------------------------------------------------------------


def test_reasoning_first_puts_the_answer_after_the_reasoning(item: Item):
    """The ordering is an experimental variable, not cosmetics.

    ANSWER_FIRST forces the model to commit before it may reason, making the answer a snap
    judgement and the "reasoning" post-hoc rationalisation. That produced 18% baseline error on a
    7B and transcripts whose reasoning contradicted the stated answer.
    """
    from src.asch.config import ResponseFormat

    rf = naive_messages(item, [], Privacy.PUBLIC, ResponseFormat.REASONING_FIRST)[0]["content"]
    af = naive_messages(item, [], Privacy.PUBLIC, ResponseFormat.ANSWER_FIRST)[0]["content"]

    assert rf.index("Reasoning:") < rf.index("Answer:")
    assert af.index("Answer:") < af.index("Reasoning:")


def test_response_format_defaults_to_reasoning_first():
    from src.asch.config import ResponseFormat

    assert TrialSpec(
        item_id="i", model="m", n_confederates=0, unanimity=Unanimity.UNANIMOUS,
        privacy=Privacy.PUBLIC, difficulty=Difficulty.EASY, kinship=Kinship.SAME_FAMILY,
        confederate_model="m", temperature=0.0, sample_idx=0,
    ).response_format is ResponseFormat.REASONING_FIRST


def test_parser_takes_the_final_answer_not_an_intermediate_thought():
    """Under REASONING_FIRST the reasoning may float other options before committing."""
    text = (
        "Reasoning: At first glance the answer looks like A, and one might say Answer: A.\n"
        "But on checking the values, C is larger.\n"
        "Answer: C\nConfidence: 90"
    )
    assert parse_answer(text).answer == "C"


def test_reasoning_prompt_has_no_placeholder_to_echo(item: Item):
    """Models copy angle-bracket placeholders verbatim.

    Observed: "Reasoning: <think it through step by step>\\nTo determine which number is...".
    The letter placeholder after "Answer:" is fine -- it is replaced, not echoed -- but a
    placeholder standing in for free text invites the model to repeat it.
    """
    from src.asch.config import ResponseFormat

    system = naive_messages(item, [], Privacy.PUBLIC, ResponseFormat.REASONING_FIRST)[0]["content"]
    assert "<think" not in system
    assert "Reasoning: <" not in system


def test_truncation_is_detected_not_silently_dropped():
    """Truncation biases REASONING_FIRST specifically, so it must be measurable."""
    from src.asch.parsing import looks_truncated

    assert looks_truncated("Reasoning: the difference is 19, so 458 is the closest number to")
    assert not looks_truncated("Reasoning: 458 is closest.\nAnswer: B\nConfidence: 90")
    assert not looks_truncated("I cannot determine this.")  # complete, just unparseable
    assert not looks_truncated("")


def test_response_format_changes_trial_identity(spec: TrialSpec):
    from dataclasses import replace

    from src.asch.config import ResponseFormat

    assert replace(spec, response_format=ResponseFormat.ANSWER_FIRST).trial_id != spec.trial_id


# --- gate verdict ---------------------------------------------------------------------


def test_low_conformity_under_bare_is_a_result_not_a_failure():
    """Guards the interpretation, not just the code.

    Asch's confederates were bare and his humans still conformed at 32%. When a model does not,
    that is evidence about models. Calling it a bank failure would push us to "fix" the item bank
    until the effect reappeared -- manufacturing the very result we are trying to measure.
    """
    import sys
    from pathlib import Path as _Path

    sys.path.insert(0, str(_Path(__file__).resolve().parent.parent / "scripts"))
    from run_smoke import verdict

    bare = verdict(0.04, 0.02, ConfederateStyle.BARE)
    assert bare.startswith("RESULT")
    assert "not a failure" in bare

    justified = verdict(0.04, 0.02, ConfederateStyle.JUSTIFIED)
    assert justified.startswith("FAIL (floor)")

    # A broken bank is still a broken bank, whatever the confederates say.
    assert verdict(0.30, 0.02, ConfederateStyle.BARE).startswith("FAIL (bank)")


# --- calibration ----------------------------------------------------------------------


def test_calibration_tiers_follow_accuracy():
    from src.asch.calibration import ItemCalibration

    def cal(correct: int, n: int = 5) -> ItemCalibration:
        return ItemCalibration("i", "magnitude", n, correct, n, "A")

    assert cal(5).tier is Difficulty.EASY
    assert cal(4).tier is Difficulty.HARD, "4/5 = 80%, the top of the hard band"
    assert cal(7, 10).tier is Difficulty.HARD
    assert cal(1, 10).tier is None, "too hard to interpret; drop rather than force a tier"
    assert cal(9, 10).tier is None, "90% is below ceiling but above the hard band -- ambiguous"


def test_calibration_keeps_only_tiered_items():
    from src.asch.calibration import ItemCalibration, apply_tiers

    items = generate_perceptual_bank(3)
    cals = [
        ItemCalibration(items[0].item_id, items[0].subtype, 5, 5, 5, "A"),   # easy
        ItemCalibration(items[1].item_id, items[1].subtype, 10, 7, 10, "A"),  # hard
        ItemCalibration(items[2].item_id, items[2].subtype, 5, 0, 5, "B"),    # dropped
    ]
    kept = apply_tiers(items, cals)
    assert [i.item_id for i in kept] == [items[0].item_id, items[1].item_id]
    assert kept[0].difficulty is Difficulty.EASY
    assert kept[1].difficulty is Difficulty.HARD


def test_calibration_recovers_a_perfect_model():
    """A model that is always right must yield an all-EASY bank."""
    from src.asch.calibration import calibrate

    items = generate_perceptual_bank(8)
    cals = calibrate(items, MockBackend(conformity_prob=0.0), "mock-7b", samples=3, batch_size=4)
    assert len(cals) == len(items)
    assert all(c.accuracy == 1.0 for c in cals)
    assert all(c.tier is Difficulty.EASY for c in cals)


def test_calibration_prompt_is_the_control_condition(item: Item):
    """Calibration accuracy and n=0 control accuracy must measure the same thing."""
    from src.asch.calibration import _ALONE

    from src.asch.prompts import naive_messages as nm

    assert nm(item, [], _ALONE) == nm(item, [], Privacy.PRIVATE)


def test_common_subset_is_the_intersection():
    from src.asch.calibration import common_subset

    items = generate_perceptual_bank(6)
    banks = {"a": items[:5], "b": items[2:]}
    assert common_subset(banks) == {i.item_id for i in items[2:5]}
    assert common_subset({}) == set()


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
