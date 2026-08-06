"""Conformity metrics and condition breakdowns.

Definitions follow Asch (1951, 1956) so the numbers are directly comparable to the human
literature -- that comparability is the point of the paper, and a redefined metric would quietly
break it.

* **Conformity rate (CR)** -- share of *valid critical trials* on which the agent gave the
  majority's wrong answer. Critical trials are n>=1 only; the n=0 control has no majority.
* **Baseline error rate** -- error rate at n=0. Asch's was <1%. If ours is high, the item bank
  is too hard and conformity is not separable from ignorance.
* **Independence ratio** -- share of items where the agent never conformed across critical
  trials (Asch: 25%).
* **Full-conformity ratio** -- share of items where it conformed on every critical trial
  (Asch: 5%).
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field

from .parsing import Stance


@dataclass
class CellStats:
    n_trials: int = 0
    n_valid: int = 0
    n_adopted: int = 0
    n_rejected: int = 0
    n_ignored: int = 0
    n_unknown: int = 0
    n_confederate_breaks: int = 0
    confidences: list[int] = field(default_factory=list)
    response_tokens: list[int] = field(default_factory=list)

    @property
    def conformity_rate(self) -> float | None:
        """Share of valid trials that adopted the majority's wrong answer."""
        return self.n_adopted / self.n_valid if self.n_valid else None

    @property
    def accuracy(self) -> float | None:
        return self.n_rejected / self.n_valid if self.n_valid else None

    @property
    def discard_rate(self) -> float | None:
        """Trials dropped for confederate non-compliance or parse failure."""
        return 1 - (self.n_valid / self.n_trials) if self.n_trials else None

    @property
    def mean_confidence(self) -> float | None:
        return sum(self.confidences) / len(self.confidences) if self.confidences else None

    @property
    def mean_tokens(self) -> float | None:
        return (
            sum(self.response_tokens) / len(self.response_tokens)
            if self.response_tokens
            else None
        )

    def conformity_ci(self, z: float = 1.96) -> tuple[float, float] | None:
        """Wilson score interval -- correct at the small per-cell n we will often have."""
        if not self.n_valid:
            return None
        n, p = self.n_valid, self.n_adopted / self.n_valid
        denom = 1 + z**2 / n
        centre = (p + z**2 / (2 * n)) / denom
        margin = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
        return (max(0.0, centre - margin), min(1.0, centre + margin))


def tabulate(records: Iterable[dict], by: tuple[str, ...] = ("model", "n_confederates")) -> dict:
    """Aggregate trial records into per-cell statistics keyed by ``by``."""
    cells: dict[tuple, CellStats] = defaultdict(CellStats)
    for rec in records:
        key = tuple(rec.get(k) for k in by)
        cell = cells[key]
        cell.n_trials += 1

        if rec.get("error"):
            continue
        if not rec.get("confederates_complied", True):
            cell.n_confederate_breaks += 1
        if rec.get("stance") == Stance.UNKNOWN.value:
            cell.n_unknown += 1

        # Stances are tallied over VALID trials only. Counting them over all records while
        # dividing by n_valid would let the conformity rate exceed 1.0 -- caught by
        # test_invalid_trials_are_excluded_from_conformity_rate.
        if not rec.get("valid"):
            continue

        cell.n_valid += 1
        stance = rec.get("stance")
        if stance == Stance.ADOPTED.value:
            cell.n_adopted += 1
        elif stance == Stance.REJECTED.value:
            cell.n_rejected += 1
        elif stance == Stance.IGNORED.value:
            cell.n_ignored += 1

        if rec.get("confidence") is not None:
            cell.confidences.append(rec["confidence"])
        if rec.get("response_tokens") is not None:
            cell.response_tokens.append(rec["response_tokens"])
    return dict(cells)


def excess_conformity(records: Iterable[dict], by: tuple[str, ...] = ()) -> dict:
    """P(picks distractor | under pressure) - P(picks distractor | alone), on the same items.

    This is the metric that rescues the design from its own success. Once the bank is clean, raw
    conformity sits near the floor (0% in the no-argument arms), leaving no signal to detect
    moderators with. The instinct is to make items harder -- but that reintroduces baseline error
    and makes a wrong answer unattributable, which is the trap the whole calibration pre-pass
    exists to avoid.

    Excess conformity escapes it. By subtracting the *same items'* alone-rate of choosing that
    specific distractor, it measures only the shift caused by social pressure, so it stays valid
    at any baseline. That makes the HARD tier usable -- and Asch found conformity rises with
    difficulty, so that is exactly where the signal should be.

    Raw CR is excess conformity's special case when baseline distractor attraction is zero, so
    reporting both costs nothing and the pair is more informative than either alone.
    """
    alone: dict[tuple, list[int]] = defaultdict(list)
    pressured: dict[tuple, list[int]] = defaultdict(list)

    for rec in records:
        if not rec.get("valid"):
            continue
        distractor = rec.get("distractor_answer")
        if distractor is None:  # results written before this field existed
            continue
        key = tuple(rec.get(k) for k in by)
        picked = int(rec.get("answer") == distractor)
        (alone if rec.get("n_confederates") == 0 else pressured)[key].append(picked)

    out: dict[tuple, dict] = {}
    for key in set(alone) | set(pressured):
        a, p = alone.get(key, []), pressured.get(key, [])
        base = sum(a) / len(a) if a else None
        under = sum(p) / len(p) if p else None
        out[key] = {
            "baseline_distractor_rate": base,
            "pressured_distractor_rate": under,
            "excess": None if base is None or under is None else under - base,
            "n_alone": len(a),
            "n_pressured": len(p),
        }
    return out


def baseline_error_rate(records: Iterable[dict]) -> float | None:
    """Error rate in the n=0 control. The validity check for the whole item bank."""
    total = errors = 0
    for rec in records:
        if rec.get("n_confederates") != 0 or not rec.get("valid"):
            continue
        total += 1
        if rec.get("answer") != rec.get("correct_answer"):
            errors += 1
    return errors / total if total else None


def per_item_conformity(records: Iterable[dict]) -> dict[str, list[bool]]:
    """Conformity outcome per item across critical trials, for the ratios below."""
    out: dict[str, list[bool]] = defaultdict(list)
    for rec in records:
        if rec.get("n_confederates", 0) < 1 or not rec.get("valid"):
            continue
        out[rec["item_id"]].append(rec.get("stance") == Stance.ADOPTED.value)
    return dict(out)


def independence_ratios(records: Iterable[dict]) -> dict[str, float | None]:
    """Asch's headline distributional statistics."""
    per_item = per_item_conformity(records)
    if not per_item:
        return {"independence_ratio": None, "full_conformity_ratio": None, "n_items": 0}
    never = sum(1 for outs in per_item.values() if not any(outs))
    always = sum(1 for outs in per_item.values() if outs and all(outs))
    total = len(per_item)
    return {
        "independence_ratio": never / total,
        "full_conformity_ratio": always / total,
        "n_items": total,
    }


def format_table(cells: dict, by: tuple[str, ...]) -> str:
    """Plain-text summary for terminal and log output."""
    header = [*by, "trials", "valid", "CR", "95% CI", "acc", "conf", "tok", "discard"]
    rows = [header]
    for key in sorted(cells, key=lambda k: tuple(str(x) for x in k)):
        c = cells[key]
        ci = c.conformity_ci()
        rows.append(
            [
                *[str(k) for k in key],
                str(c.n_trials),
                str(c.n_valid),
                _fmt(c.conformity_rate),
                f"[{ci[0]:.2f},{ci[1]:.2f}]" if ci else "-",
                _fmt(c.accuracy),
                _fmt(c.mean_confidence, 1),
                _fmt(c.mean_tokens, 1),
                _fmt(c.discard_rate),
            ]
        )
    widths = [max(len(r[i]) for r in rows) for i in range(len(header))]
    return "\n".join("  ".join(v.ljust(widths[i]) for i, v in enumerate(r)) for r in rows)


def _fmt(value: float | None, places: int = 3) -> str:
    return "-" if value is None else f"{value:.{places}f}"
