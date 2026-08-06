"""Experiment configuration, condition space, and deterministic trial identity.

Trial IDs are a stable hash of the full condition tuple. This is what makes runs resumable:
the runner expands the whole grid, drops IDs already present in the results file, and executes
the remainder. Reruns are therefore idempotent, and a session dying mid-grid costs nothing.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import asdict, dataclass, field
from enum import Enum


class Unanimity(str, Enum):
    """How the confederate majority is composed.

    UNANIMOUS: every confederate gives the same wrong answer (Asch's standard critical trial).
    ALLY: one confederate gives the correct answer. In humans this cuts conformity ~80%, so it
        doubles as a deployable mitigation, not just a measurement.
    INCOMPETENT_DISSENTER: one confederate gives a *third*, different wrong answer. Per Allen &
        Levine (1968), breaking unanimity helps even when the dissenter is not credible. This
        separates "unanimity was broken" from "someone told me the truth".
    """

    UNANIMOUS = "unanimous"
    ALLY = "ally"
    INCOMPETENT_DISSENTER = "incompetent_dissenter"


class ConfederateStyle(str, Enum):
    """What the confederates actually say.

    BARE: the answer alone, no reasoning -- this is what Asch's confederates did. They stated a
        line and nothing else, so conformity could only be social. The faithful replication arm.
    FILLER: the answer plus a content-free sentence of comparable length. **The control that makes
        the BARE/JUSTIFIED contrast interpretable.** Bare turns render as a single short line while
        justified turns carry a whole sentence, so the two differ in textual salience as well as in
        argumentation. FILLER holds salience roughly constant and removes only the argument.
    JUSTIFIED: a real model call constrained to the assigned answer, writing its own supporting
        argument. Stronger pressure than Asch, and the multi-agent-realistic condition -- but it
        confounds social conformity with being argued into a position, because the arguments are
        themselves fabricated ("312 is larger than 787 because of the hundreds place").

    Pilot (Qwen2.5-7B, identical bank): JUSTIFIED 16.0%, BARE 2.0%. If FILLER tracks BARE, the
    driver is argumentation; if it tracks JUSTIFIED, the driver was merely having text there.
    That three-way contrast is the paper's central measurement, and no prior LLM-conformity paper
    runs any of it.
    """

    BARE = "bare"
    FILLER = "filler"
    JUSTIFIED = "justified"


class ResponseFormat(str, Enum):
    """Whether the naive agent answers before or after reasoning.

    ANSWER_FIRST: "Answer / Confidence / Reasoning". The model must emit the answer token before
        any reasoning, so the answer is a snap judgement and everything after it is post-hoc
        rationalisation. This was the original (accidental) format, and it produced 18% baseline
        error on a 7B for tasks it does trivially -- plus transcripts where the reasoning
        contradicts the stated answer ("...the correct answer must be A." / "Answer: B").
    REASONING_FIRST: "Reasoning / Answer / Confidence". The model deliberates, then commits.

    Keeping both makes this the **chain-of-thought-as-conformity-defence** manipulation predicted
    in session 3 by the list_count vs magnitude gap: if forcing explicit reasoning before
    committing reduces conformity, CoT is a deployable defence and not merely an accuracy trick.
    REASONING_FIRST is the default because calibration requires a clean baseline.
    """

    ANSWER_FIRST = "answer_first"
    REASONING_FIRST = "reasoning_first"


class Privacy(str, Enum):
    """Whether the naive agent's answer is visible to the group.

    Asch found conformity drops sharply under private responding, because normative pressure
    (fear of standing out) needs an audience while informational pressure does not. This factor
    is therefore a direct probe of *which* mechanism is operating.
    """

    PUBLIC = "public"
    PRIVATE = "private"


class Difficulty(str, Enum):
    """Item tier, assigned by the calibration pre-pass, not by hand."""

    EASY = "easy"  # >=95% alone-accuracy
    HARD = "hard"  # 60-80% alone-accuracy


class Kinship(str, Enum):
    """Whether confederates share the naive agent's model family.

    Shehata & Li report a kinship effect on social load; we test it as a clean factor.
    """

    SAME_FAMILY = "same_family"
    CROSS_FAMILY = "cross_family"


@dataclass(frozen=True)
class TrialSpec:
    """One fully-specified trial. Immutable and hashable -> defines trial identity."""

    item_id: str
    model: str
    n_confederates: int
    unanimity: Unanimity
    privacy: Privacy
    difficulty: Difficulty
    kinship: Kinship
    confederate_model: str
    temperature: float
    sample_idx: int
    confederate_style: ConfederateStyle = ConfederateStyle.JUSTIFIED
    response_format: ResponseFormat = ResponseFormat.REASONING_FIRST
    study: str = "study1"

    @property
    def is_control(self) -> bool:
        """n=0 is the alone condition -- Asch's control, and our conformity baseline."""
        return self.n_confederates == 0

    @property
    def trial_id(self) -> str:
        payload = json.dumps(
            {k: (v.value if isinstance(v, Enum) else v) for k, v in asdict(self).items()},
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def to_dict(self) -> dict:
        d = {k: (v.value if isinstance(v, Enum) else v) for k, v in asdict(self).items()}
        d["trial_id"] = self.trial_id
        return d


@dataclass
class GridConfig:
    """The condition grid to expand into trials.

    Defaults encode the Study 1 design from RESEARCH_PLAN.md 1.3. n=0 is included so every run
    carries its own control condition -- conformity is meaningless without a paired baseline.
    """

    models: list[str]
    confederate_model: str
    n_confederates: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 5, 7])
    unanimity: list[Unanimity] = field(default_factory=lambda: list(Unanimity))
    privacy: list[Privacy] = field(default_factory=lambda: list(Privacy))
    kinship: list[Kinship] = field(default_factory=lambda: [Kinship.SAME_FAMILY])
    confederate_style: list[ConfederateStyle] = field(
        default_factory=lambda: [ConfederateStyle.JUSTIFIED]
    )
    response_format: list[ResponseFormat] = field(
        default_factory=lambda: [ResponseFormat.REASONING_FIRST]
    )
    temperature: float = 0.0
    samples_per_cell: int = 1
    study: str = "study1"

    def expand(self, items: list) -> list[TrialSpec]:
        """Cartesian product of conditions x items, with degenerate cells pruned."""
        trials: list[TrialSpec] = []
        for item, model, n, unan, priv, kin, style, fmt, s in itertools.product(
            items,
            self.models,
            self.n_confederates,
            self.unanimity,
            self.privacy,
            self.kinship,
            self.confederate_style,
            self.response_format,
            range(self.samples_per_cell),
        ):
            if not _cell_is_meaningful(n, unan):
                continue
            trials.append(
                TrialSpec(
                    item_id=item.item_id,
                    model=model,
                    n_confederates=n,
                    unanimity=unan,
                    privacy=priv,
                    difficulty=item.difficulty,
                    kinship=kin,
                    confederate_model=self.confederate_model,
                    temperature=self.temperature,
                    sample_idx=s,
                    confederate_style=style,
                    response_format=fmt,
                    study=self.study,
                )
            )
        return _dedupe(trials)


def _cell_is_meaningful(n: int, unanimity: Unanimity) -> bool:
    """Prune cells that are degenerate or duplicate the control.

    At n=0 there is no group, so unanimity is undefined -- keep only the canonical UNANIMOUS
    label so the control appears exactly once per (item, privacy) rather than three times.
    A dissenter also needs someone to dissent from, so ALLY/INCOMPETENT_DISSENTER require n>=2.
    """
    if n == 0:
        return unanimity is Unanimity.UNANIMOUS
    if unanimity is not Unanimity.UNANIMOUS and n < 2:
        return False
    return True


def _dedupe(trials: list[TrialSpec]) -> list[TrialSpec]:
    seen: set[str] = set()
    out: list[TrialSpec] = []
    for t in trials:
        if t.trial_id not in seen:
            seen.add(t.trial_id)
            out.append(t)
    return out
