"""Resumable trial execution.

The runner assumes the session will die. Colab and Kaggle disconnect without warning, so every
completed trial is appended to JSONL and flushed immediately, and nothing is accumulated in
memory. On restart the runner reads the completed trial IDs back and executes only what is
missing, which makes a rerun idempotent and a partial grid trivially resumable.

Full transcripts -- every confederate turn, verbatim -- are stored on each record. Study 2
re-judges those transcripts without regenerating them, which is what makes the headline result
nearly free.
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

from .backends import Backend
from .config import ConfederateStyle, TrialSpec, Unanimity
from .items import Item
from .parsing import classify_stance, confederate_complied, effort_proxy, parse_answer
from .prompts import (
    assign_confederate_answers,
    bare_confederate_text,
    confederate_messages,
    naive_messages,
)


@dataclass
class TrialResult:
    spec: TrialSpec
    answer: str | None
    confidence: int | None
    stance: str
    correct_answer: str
    majority_answer: str | None
    confederates_complied: bool
    valid: bool
    response_tokens: int
    raw_response: str
    transcript: list[dict]
    error: str | None = None

    def to_record(self) -> dict:
        record = self.spec.to_dict()
        record.update(
            answer=self.answer,
            confidence=self.confidence,
            stance=self.stance,
            correct_answer=self.correct_answer,
            majority_answer=self.majority_answer,
            confederates_complied=self.confederates_complied,
            valid=self.valid,
            response_tokens=self.response_tokens,
            raw_response=self.raw_response,
            transcript=self.transcript,
            error=self.error,
        )
        return record


def completed_trial_ids(results_path: Path) -> set[str]:
    """Read back which trials are already done.

    Tolerates a truncated final line, which is what a mid-write session kill leaves behind.
    """
    if not results_path.exists():
        return set()
    done: set[str] = set()
    with results_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["trial_id"])
            except (json.JSONDecodeError, KeyError):
                continue  # truncated tail from a killed session
    return done


def run_trial(spec: TrialSpec, item: Item, backend: Backend) -> TrialResult:
    """Execute one trial: confederates speak in order, then the naive agent answers."""
    assigned = assign_confederate_answers(item, spec.n_confederates, spec.unanimity)
    transcript: list[dict] = []
    turns: list[tuple[int, str, str]] = []
    all_complied = True

    for position, answer_key in enumerate(assigned, start=1):
        if spec.confederate_style is ConfederateStyle.BARE:
            # Asch's confederates stated a line and nothing more. No model call needed, and
            # compliance is guaranteed by construction rather than merely checked.
            text = bare_confederate_text(answer_key)
        else:
            text = backend.generate(
                confederate_messages(item, answer_key, position),
                model=spec.confederate_model,
                temperature=spec.temperature,
                max_tokens=200,
                seed=_seed_for(spec, position),
            ).text

        complied = confederate_complied(text, answer_key)
        all_complied = all_complied and complied
        turns.append((position, answer_key, text))
        transcript.append(
            {
                "position": position,
                "role": "confederate",
                "assigned_answer": answer_key,
                "text": text,
                "complied": complied,
            }
        )

    gen = backend.generate(
        naive_messages(item, turns, spec.privacy),
        model=spec.model,
        temperature=spec.temperature,
        max_tokens=512,
        seed=_seed_for(spec, 0),
        oracle=item.correct,
    )
    transcript.append({"position": len(assigned) + 1, "role": "naive", "text": gen.text})

    parsed = parse_answer(gen.text)
    majority = _majority_answer(item, spec.n_confederates, spec.unanimity)
    stance = classify_stance(parsed.answer, item.correct, majority)

    return TrialResult(
        spec=spec,
        answer=parsed.answer,
        confidence=parsed.confidence,
        stance=stance.value,
        correct_answer=item.correct,
        majority_answer=majority,
        confederates_complied=all_complied,
        valid=all_complied and parsed.parsed,
        response_tokens=effort_proxy(gen.text),
        raw_response=gen.text,
        transcript=transcript,
    )


def _majority_answer(item: Item, n: int, unanimity: Unanimity) -> str | None:
    """The answer the confederate bloc is pushing, or None when there is no group.

    Even with one ally or one incompetent dissenter, the distractor remains the modal answer for
    every n>=2 cell we run, so it is what "conforming" means throughout.
    """
    if n == 0:
        return None
    return item.distractor


def _seed_for(spec: TrialSpec, position: int) -> int:
    """Stable per-call seed so reruns of the same trial reproduce exactly."""
    return int(spec.trial_id[:8], 16) % (2**31) + position


def run_grid(
    specs: Iterable[TrialSpec],
    items: dict[str, Item],
    backend: Backend,
    results_path: Path,
    *,
    resume: bool = True,
    progress_every: int = 25,
    batch_size: int = 1,
) -> int:
    """Execute every spec not already present in ``results_path``. Returns count executed.

    ``batch_size > 1`` switches to the two-phase batched path (see ``_run_chunk``), which is what
    makes the full grid tractable. Results are identical either way -- enforced by
    test_batched_matches_sequential -- so batching is purely a throughput choice.
    """
    results_path.parent.mkdir(parents=True, exist_ok=True)
    done = completed_trial_ids(results_path) if resume else set()

    pending = [s for s in specs if s.trial_id not in done]
    if done:
        print(f"[runner] resuming: {len(done)} done, {len(pending)} remaining", file=sys.stderr)

    executed = 0
    started = time.time()
    with results_path.open("a", encoding="utf-8") as f:
        for chunk in _chunks(pending, max(batch_size, 1)):
            if batch_size > 1:
                records = _run_chunk(chunk, items, backend)
            else:
                records = [_run_one_safely(spec, items, backend) for spec in chunk]

            for record in records:
                f.write(json.dumps(record) + "\n")
            f.flush()  # a killed session must not lose completed work
            executed += len(records)

            if progress_every and executed % progress_every < len(records):
                rate = executed / max(time.time() - started, 1e-9)
                print(
                    f"[runner] {executed}/{len(pending)} ({rate:.1f} trials/s)",
                    file=sys.stderr,
                )
    return executed


def _chunks(seq: list, size: int) -> Iterator[list]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def _run_one_safely(spec: TrialSpec, items: dict[str, Item], backend: Backend) -> dict:
    try:
        return run_trial(spec, items[spec.item_id], backend).to_record()
    except Exception as exc:  # noqa: BLE001 - one bad trial must not kill a multi-hour grid
        record = spec.to_dict()
        record.update(valid=False, error=f"{type(exc).__name__}: {exc}")
        return record


def _run_chunk(specs: list[TrialSpec], items: dict[str, Item], backend: Backend) -> list[dict]:
    """Execute a chunk of trials in two batched phases.

    Phase 1 generates every confederate turn in the chunk at once. This works because a
    confederate's prompt depends only on (item, assigned answer, position) -- confederates never
    see each other. That also means the same prompt recurs constantly across the grid (every n
    level reuses position 1, 2, 3...), so requests are **deduplicated**, which is a larger saving
    than the batching itself.

    Phase 2 generates every naive turn at once, now that the transcripts exist.

    Two batched passes replace up to n+1 sequential calls per trial.
    """
    conf_prompts: dict[tuple, list[dict[str, str]]] = {}
    plans: list[tuple[TrialSpec, list[tuple[int, str, tuple | None]]]] = []

    for spec in specs:
        item = items[spec.item_id]
        assigned = assign_confederate_answers(item, spec.n_confederates, spec.unanimity)
        entries: list[tuple[int, str, tuple | None]] = []
        for position, answer_key in enumerate(assigned, start=1):
            if spec.confederate_style is ConfederateStyle.BARE:
                entries.append((position, answer_key, None))
                continue
            key = (spec.confederate_model, spec.temperature, spec.item_id, answer_key, position)
            conf_prompts.setdefault(key, confederate_messages(item, answer_key, position))
            entries.append((position, answer_key, key))
        plans.append((spec, entries))

    conf_text = _generate_grouped(backend, conf_prompts, max_tokens=200)

    # Phase 2: assemble transcripts, then batch the naive turns.
    naive_prompts: dict[tuple, list[dict[str, str]]] = {}
    naive_oracles: dict[tuple, str] = {}
    transcripts: dict[str, tuple[list[dict], list[tuple[int, str, str]], bool]] = {}

    for spec, entries in plans:
        item = items[spec.item_id]
        transcript: list[dict] = []
        turns: list[tuple[int, str, str]] = []
        all_complied = True
        for position, answer_key, key in entries:
            text = bare_confederate_text(answer_key) if key is None else conf_text[key]
            complied = confederate_complied(text, answer_key)
            all_complied = all_complied and complied
            turns.append((position, answer_key, text))
            transcript.append(
                {
                    "position": position,
                    "role": "confederate",
                    "assigned_answer": answer_key,
                    "text": text,
                    "complied": complied,
                }
            )
        transcripts[spec.trial_id] = (transcript, turns, all_complied)
        nkey = (spec.model, spec.temperature, spec.trial_id)
        naive_prompts[nkey] = naive_messages(item, turns, spec.privacy)
        naive_oracles[nkey] = item.correct

    naive_text = _generate_grouped(backend, naive_prompts, max_tokens=512, oracles=naive_oracles)

    records: list[dict] = []
    for spec, _ in plans:
        item = items[spec.item_id]
        transcript, _, all_complied = transcripts[spec.trial_id]
        text = naive_text[(spec.model, spec.temperature, spec.trial_id)]
        transcript = [*transcript, {"position": len(transcript) + 1, "role": "naive", "text": text}]

        parsed = parse_answer(text)
        majority = _majority_answer(item, spec.n_confederates, spec.unanimity)
        records.append(
            TrialResult(
                spec=spec,
                answer=parsed.answer,
                confidence=parsed.confidence,
                stance=classify_stance(parsed.answer, item.correct, majority).value,
                correct_answer=item.correct,
                majority_answer=majority,
                confederates_complied=all_complied,
                valid=all_complied and parsed.parsed,
                response_tokens=effort_proxy(text),
                raw_response=text,
                transcript=transcript,
            ).to_record()
        )
    return records


def _generate_grouped(
    backend: Backend,
    prompts: dict[tuple, list[dict[str, str]]],
    *,
    max_tokens: int,
    oracles: dict[tuple, str] | None = None,
) -> dict[tuple, str]:
    """Batch prompts grouped by (model, temperature), which are the first two key elements."""
    out: dict[tuple, str] = {}
    groups: dict[tuple, list[tuple]] = defaultdict(list)
    for key in prompts:
        groups[(key[0], key[1])].append(key)

    for (model, temperature), keys in groups.items():
        gens = backend.generate_batch(
            [prompts[k] for k in keys],
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            oracles=[oracles.get(k) for k in keys] if oracles else None,
        )
        for key, gen in zip(keys, gens):
            out[key] = gen.text
    return out


def load_results(results_path: Path) -> Iterator[dict]:
    if not results_path.exists():
        return
    with results_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue
