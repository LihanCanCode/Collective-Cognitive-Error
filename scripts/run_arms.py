"""The core results table: confederate style x response format, for one model.

This sweep produces the paper's central argument in a single run. Each cell removes one
candidate artefact, and conformity is expected to collapse as they come off:

    justified + answer_first     ~ how prior work measures it        (pilot: 36%)
    justified + reasoning_first  allow deliberation                  (pilot:  6%)
    filler    + reasoning_first  remove the argument, keep the text  (pilot:  0%)
    bare      + reasoning_first  Asch's actual paradigm              (pilot:  0%)

The top-left cell matters most. A null result on its own reads as "you failed to find the
effect"; reproducing the literature's magnitude and *then* dissolving it is the contribution.

bare and filler need no confederate generation, so the sweep costs barely more than two full
arms.

Usage:
    python scripts/run_arms.py --backend hf --model Qwen/Qwen2.5-7B-Instruct --batch-size 16
    python scripts/run_arms.py --backend mock --model mock-7b        # pipeline check
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.asch.analyze import baseline_error_rate, excess_conformity, tabulate  # noqa: E402
from src.asch.calibration import model_slug  # noqa: E402
from src.asch.config import (  # noqa: E402
    ConfederateStyle,
    GridConfig,
    Kinship,
    Privacy,
    ResponseFormat,
    Unanimity,
)
from src.asch.items import generate_perceptual_bank, load_bank  # noqa: E402
from src.asch.runner import load_results, run_grid  # noqa: E402
from run_smoke import build_backend  # noqa: E402, isort: skip

# (confederate style, response format, what removing it tests)
CELLS = [
    (ConfederateStyle.JUSTIFIED, ResponseFormat.ANSWER_FIRST, "approximates prior work"),
    (ConfederateStyle.JUSTIFIED, ResponseFormat.REASONING_FIRST, "+ deliberation"),
    (ConfederateStyle.FILLER, ResponseFormat.REASONING_FIRST, "+ argument removed"),
    (ConfederateStyle.BARE, ResponseFormat.REASONING_FIRST, "+ Asch's paradigm"),
    # Completes the 2x2 so the format effect can be read off the no-argument arms too.
    (ConfederateStyle.BARE, ResponseFormat.ANSWER_FIRST, "no argument, no deliberation"),
]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--backend", choices=["mock", "hf", "vllm", "api"], default="mock")
    ap.add_argument("--model", default="mock-7b")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--items", type=Path, default=_REPO_ROOT / "data" / "smoke_items.jsonl")
    ap.add_argument("--n-items", type=int, default=0, help="0 = whole bank")
    ap.add_argument("--n-confederates", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--out-dir", type=Path, default=_REPO_ROOT / "results" / "arms")
    args = ap.parse_args()

    items = load_bank(args.items) if args.items.exists() else generate_perceptual_bank(50)
    if args.n_items:
        items = items[: args.n_items]
    item_map = {i.item_id: i for i in items}
    slug = model_slug(args.model)

    backend = build_backend(args.backend, args.model, 0.3, args.dtype)
    rows = []
    try:
        for style, fmt, label in CELLS:
            out = args.out_dir / f"{slug}__{style.value}__{fmt.value}.jsonl"
            print(f"\n{'=' * 78}\n{style.value} x {fmt.value}  ({label})\n{'=' * 78}")

            specs = GridConfig(
                models=[args.model],
                confederate_model=args.model,
                n_confederates=[0, args.n_confederates],
                unanimity=[Unanimity.UNANIMOUS],
                privacy=[Privacy.PUBLIC],
                kinship=[Kinship.SAME_FAMILY],
                confederate_style=[style],
                response_format=[fmt],
                study="arms",
            ).expand(items)

            run_grid(specs, item_map, backend, out, batch_size=args.batch_size)
            rows.append((style, fmt, label, list(load_results(out))))
    finally:
        backend.close()

    _report(args.model, rows)


def _report(model: str, rows: list) -> None:
    print("\n\n" + "=" * 92)
    print(f"CORE RESULTS — {model}")
    print("=" * 92)
    header = f"{'confederates':<11} {'format':<16} {'base err':>9} {'CR':>7} {'excess':>8} {'n':>5}  note"
    print(header)
    print("-" * 92)

    table = {}
    for style, fmt, label, records in rows:
        base = baseline_error_rate(records)
        cells = tabulate([r for r in records if r.get("n_confederates", 0) > 0], by=("model",))
        cr = next(iter(cells.values())).conformity_rate if cells else None
        exc = excess_conformity(records).get((), {}).get("excess")
        n = next(iter(cells.values())).n_valid if cells else 0
        table[(style.value, fmt.value)] = cr
        print(
            f"{style.value:<11} {fmt.value:<16} {_pct(base):>9} {_pct(cr):>7} "
            f"{_pct(exc):>8} {n:>5}  {label}"
        )

    print("=" * 92)
    _interpret(table)


def _interpret(table: dict) -> None:
    prior = table.get(("justified", "answer_first"))
    delib = table.get(("justified", "reasoning_first"))
    filler = table.get(("filler", "reasoning_first"))
    bare = table.get(("bare", "reasoning_first"))

    if prior is None:
        return
    print(f"\nUnder conditions approximating prior work: {_pct(prior)} conformity.")
    if delib is not None:
        print(f"  - allowing deliberation:      {_pct(prior)} -> {_pct(delib)}")
    if delib is not None and filler is not None:
        print(f"  - removing the argument:      {_pct(delib)} -> {_pct(filler)}")
    if filler is not None and bare is not None and abs(filler - bare) < 0.05:
        print(f"  - filler ~ bare ({_pct(filler)} vs {_pct(bare)}): the text itself is not the driver")

    if prior >= 0.15 and (bare or 0) < 0.05:
        print(
            "\n=> The literature's magnitude reproduces, and collapses once argumentation and\n"
            "   snap-judgement formats are controlled. That is the paper."
        )
    elif prior < 0.15:
        print(
            "\n=> WARNING: the prior-work cell did not reproduce a substantial rate. Without it the\n"
            "   result reads as 'we failed to find the effect' rather than as a dissolution.\n"
            "   Check this cell before drawing conclusions."
        )


def _pct(value: float | None) -> str:
    return "-" if value is None else f"{value:.1%}"


if __name__ == "__main__":
    main()
