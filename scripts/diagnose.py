"""Post-hoc diagnostics on a saved results file. No GPU, no model, runs in seconds.

Answers the questions the summary table cannot:
  * WHICH item subtypes are failing at baseline (the FAIL(bank) verdict says "too hard" but not
    "too hard where")
  * WHY trials were discarded -- confederate non-compliance or parse failure are different
    problems with different fixes
  * What the model actually wrote when it got a control item wrong

Usage:
    python scripts/diagnose.py results/smoke_qwen7b.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.asch.analyze import excess_conformity  # noqa: E402
from src.asch.items import load_bank  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", type=Path)
    ap.add_argument("--items", type=Path, default=_REPO_ROOT / "data" / "smoke_items.jsonl")
    ap.add_argument("--show", type=int, default=4, help="examples to print per failure mode")
    args = ap.parse_args()

    records = [json.loads(line) for line in args.results.open(encoding="utf-8") if line.strip()]
    subtype = {i.item_id: i.subtype for i in load_bank(args.items)} if args.items.exists() else {}

    _baseline_by_subtype(records, subtype)
    _conformity_by_subtype(records, subtype)
    _excess_by_subtype(records, subtype)
    _discard_breakdown(records)
    _show_baseline_failures(records, subtype, args.show)
    _show_confederate_breaks(records, args.show)
    _show_parse_failures(records, args.show)


def _baseline_by_subtype(records: list[dict], subtype: dict[str, str]) -> None:
    """Where the 'items are too hard' verdict is actually coming from."""
    stats: dict[str, list[int]] = defaultdict(list)
    for r in records:
        if r.get("n_confederates") != 0 or not r.get("valid"):
            continue
        stats[subtype.get(r["item_id"], "?")].append(int(r["answer"] == r["correct_answer"]))

    print("=" * 78)
    print("BASELINE ACCURACY BY SUBTYPE  (n=0, alone -- must be >=95% to keep an item)")
    print("=" * 78)
    for st in sorted(stats):
        hits = stats[st]
        acc = sum(hits) / len(hits)
        flag = "  <-- FAILING" if acc < 0.95 else ""
        print(f"  {st:<14} {acc:6.1%}  ({sum(hits)}/{len(hits)}){flag}")


def _conformity_by_subtype(records: list[dict], subtype: dict[str, str]) -> None:
    stats: dict[str, list[int]] = defaultdict(list)
    for r in records:
        if r.get("n_confederates", 0) < 1 or not r.get("valid"):
            continue
        stats[subtype.get(r["item_id"], "?")].append(int(r["stance"] == "adopted"))

    print("\n" + "=" * 78)
    print("CONFORMITY BY SUBTYPE  (critical trials, valid only)")
    print("=" * 78)
    for st in sorted(stats):
        hits = stats[st]
        print(f"  {st:<14} {sum(hits) / len(hits):6.1%}  ({sum(hits)}/{len(hits)})")


def _excess_by_subtype(records: list[dict], subtype: dict[str, str]) -> None:
    """Conformity net of the same items' unaided pull toward that distractor.

    Once the bank is clean these two columns coincide. They diverge on the HARD tier, which is
    where the signal should live -- Asch found conformity rises with difficulty.
    """
    tagged = [{**r, "subtype": subtype.get(r["item_id"], "?")} for r in records]
    stats = excess_conformity(tagged, by=("subtype",))
    if not stats:
        return

    print("\n" + "=" * 78)
    print("EXCESS CONFORMITY  (pressured - alone, on the same items)")
    print("=" * 78)
    print(f"  {'subtype':<14} {'alone':>8} {'pressured':>11} {'excess':>9}")
    for (st,), s in sorted(stats.items(), key=lambda kv: str(kv[0])):
        base, under, exc = (
            s["baseline_distractor_rate"], s["pressured_distractor_rate"], s["excess"]
        )
        print(
            f"  {st:<14} {_pct(base):>8} {_pct(under):>11} {_pct(exc):>9}"
        )


def _pct(value: float | None) -> str:
    return "-" if value is None else f"{value:.1%}"


def _discard_breakdown(records: list[dict]) -> None:
    """Confederate breaks and parse failures need different fixes -- separate them."""
    critical = [r for r in records if r.get("n_confederates", 0) >= 1]
    if not critical:
        return
    breaks = sum(1 for r in critical if not r.get("confederates_complied", True))
    parse = sum(1 for r in critical if r.get("answer") is None)
    both = sum(
        1 for r in critical
        if not r.get("confederates_complied", True) and r.get("answer") is None
    )
    errs = sum(1 for r in critical if r.get("error"))

    print("\n" + "=" * 78)
    print(f"DISCARD BREAKDOWN  ({len(critical)} critical trials)")
    print("=" * 78)
    print(f"  confederate broke character: {breaks:3d}  ({breaks / len(critical):.1%})")
    print(f"  naive answer unparseable:    {parse:3d}  ({parse / len(critical):.1%})")
    print(f"  both:                        {both:3d}")
    print(f"  runtime errors:              {errs:3d}")
    print(f"  -> discarded:                {sum(1 for r in critical if not r.get('valid')):3d}")


def _show_baseline_failures(records: list[dict], subtype: dict[str, str], n: int) -> None:
    bad = [
        r for r in records
        if r.get("n_confederates") == 0 and r.get("valid") and r["answer"] != r["correct_answer"]
    ]
    if not bad:
        return
    print("\n" + "=" * 78)
    print(f"CONTROL-CONDITION ERRORS ({len(bad)}) -- the model failing with no social pressure")
    print("=" * 78)
    for r in bad[:n]:
        print(f"\n[{subtype.get(r['item_id'], '?')}] {r['item_id']}  "
              f"said {r['answer']}, correct {r['correct_answer']}")
        print("  " + r["raw_response"][:300].replace("\n", "\n  "))


def _show_confederate_breaks(records: list[dict], n: int) -> None:
    broken = [r for r in records if not r.get("confederates_complied", True)]
    if not broken:
        return
    print("\n" + "=" * 78)
    print(f"CONFEDERATES THAT BROKE CHARACTER ({len(broken)} trials)")
    print("=" * 78)
    shown = 0
    for r in broken:
        for turn in r.get("transcript", []):
            if turn.get("role") == "confederate" and not turn.get("complied"):
                print(f"\n[{r['item_id']}] assigned {turn['assigned_answer']}:")
                print("  " + turn["text"][:300].replace("\n", "\n  "))
                shown += 1
                break
        if shown >= n:
            break


def _show_parse_failures(records: list[dict], n: int) -> None:
    bad = [r for r in records if r.get("answer") is None and not r.get("error")]
    if not bad:
        return
    print("\n" + "=" * 78)
    print(f"UNPARSEABLE NAIVE RESPONSES ({len(bad)})")
    print("=" * 78)
    for r in bad[:n]:
        print(f"\n[{r['item_id']}]:")
        print("  " + (r.get("raw_response") or "")[:300].replace("\n", "\n  "))


if __name__ == "__main__":
    main()
