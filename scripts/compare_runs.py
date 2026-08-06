"""Fine-grained comparison of two results files -- e.g. sequential vs batched.

Separates two very different kinds of disagreement:
  * answer-level  -- the model reached a DIFFERENT conclusion. This is what matters for every
    number in the paper.
  * text-only     -- same answer, same stance, different phrasing en route to it. Expected and
    harmless; a strict raw_response diff (see notebook Cell 8) conflates this with the above and
    can make a batching run look far more broken than it is.

Background: batched GPU inference can differ from sequential inference even at temperature 0,
because batched matrix multiplication uses a different reduction order than a single-example
forward pass, which can flip an occasional near-tied greedy argmax. Padding was verified correct
(left-padded, attention_mask passed) -- this is standard GPU batching numerics, not a masking bug.
That does not make it safe to ignore: near-tied decisions are exactly what conformity trials are.

Usage:
    python scripts/compare_runs.py results/smoke_qwen7b.jsonl results/smoke_qwen7b_batched.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict[str, dict]:
    return {
        json.loads(line)["trial_id"]: json.loads(line)
        for line in path.open(encoding="utf-8")
        if line.strip()
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a", type=Path, help="e.g. the sequential run")
    ap.add_argument("b", type=Path, help="e.g. the batched run")
    ap.add_argument("--show", type=int, default=3)
    args = ap.parse_args()

    a, b = load(args.a), load(args.b)
    shared = a.keys() & b.keys()
    if not shared:
        print("No shared trial_ids -- these are not comparable runs (different config?).")
        return

    answer_mismatch, stance_mismatch, text_only = [], [], []
    for tid in shared:
        ra, rb = a[tid], b[tid]
        if ra.get("answer") != rb.get("answer"):
            answer_mismatch.append(tid)
        elif ra.get("stance") != rb.get("stance"):
            stance_mismatch.append(tid)
        elif ra.get("raw_response") != rb.get("raw_response"):
            text_only.append(tid)

    n = len(shared)
    print(f"compared {n} shared trials")
    print(f"  answer-level mismatches: {len(answer_mismatch):4d}  ({len(answer_mismatch)/n:.1%})"
          "  <- this is what matters for the paper")
    print(f"  stance-only mismatches:  {len(stance_mismatch):4d}  ({len(stance_mismatch)/n:.1%})")
    print(f"  text-only differences:   {len(text_only):4d}  ({len(text_only)/n:.1%})"
          "  <- expected; same conclusion, different wording")

    if answer_mismatch:
        print(f"\n{'=' * 70}\nANSWER-LEVEL MISMATCHES (the real signal) -- first {args.show}")
        print("=" * 70)
        for tid in answer_mismatch[: args.show]:
            ra, rb = a[tid], b[tid]
            print(f"\n[{tid}]  n={ra.get('n_confederates')}  "
                  f"correct={ra.get('correct_answer')}  majority={ra.get('majority_answer')}")
            print(f"  a: answer={ra.get('answer')} stance={ra.get('stance')}")
            print(f"     {ra.get('raw_response', '')[-150:]!r}")
            print(f"  b: answer={rb.get('answer')} stance={rb.get('stance')}")
            print(f"     {rb.get('raw_response', '')[-150:]!r}")

    print(f"\n{'=' * 70}")
    rate = len(answer_mismatch) / n
    if rate == 0:
        print("VERDICT: 0 answer-level mismatches. Batching is safe for this model/config.")
    elif rate < 0.03:
        print(f"VERDICT: {rate:.1%} answer-level mismatch rate. Small, but these effect sizes "
              "(single-digit %) are fragile -- treat batched numbers as provisional and confirm "
              "the final figures with a sequential (--batch-size 1) run.")
    else:
        print(f"VERDICT: {rate:.1%} answer-level mismatch rate. Too high to trust for the numbers "
              "reported in the paper. Use --batch-size 1 (sequential) for anything final; keep "
              "batching for exploratory/early-look runs only.")


if __name__ == "__main__":
    main()
