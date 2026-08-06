"""Per-model calibration pre-pass -- run this before any conformity grid.

Asks every candidate item alone, 5 samples at temperature 0.7, and sorts it into:
  * EASY  (>=95% correct)      -> the conformity bank; a wrong answer here under pressure IS
                                  conformity, because the model demonstrably knows the answer
  * HARD  (60-80% correct)     -> the task-difficulty factor for Study 1
  * dropped (anything else)    -> ambiguous, would blur both

Item banks are therefore **per-model**. Cross-model comparisons use the intersection.

Usage:
    python scripts/calibrate.py --backend hf --model Qwen/Qwen2.5-7B-Instruct
    python scripts/calibrate.py --backend mock --model mock-7b     # pipeline check, no GPU
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.asch.calibration import (  # noqa: E402
    DEFAULT_SAMPLES,
    DEFAULT_TEMPERATURE,
    apply_tiers,
    calibrate,
    model_slug,
    save_calibration,
    summarise,
)
from src.asch.items import generate_perceptual_bank, load_bank, save_bank  # noqa: E402
from run_smoke import build_backend  # noqa: E402, isort: skip


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--backend", choices=["mock", "hf", "vllm", "api"], default="mock")
    ap.add_argument("--model", default="mock-7b")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--items", type=Path, default=_REPO_ROOT / "data" / "smoke_items.jsonl")
    ap.add_argument("--n-items", type=int, default=0, help="0 = use the whole bank")
    ap.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    ap.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out-dir", type=Path, default=_REPO_ROOT / "data" / "calibrated")
    args = ap.parse_args()

    items = load_bank(args.items) if args.items.exists() else generate_perceptual_bank(200)
    if args.n_items:
        items = items[: args.n_items]

    slug = model_slug(args.model)
    print(f"[calibrate] {len(items)} items x {args.samples} samples on {args.model} "
          f"({len(items) * args.samples} generations)")

    backend = build_backend(args.backend, args.model, 0.3, args.dtype)
    try:
        calibrations = calibrate(
            items,
            backend,
            args.model,
            samples=args.samples,
            temperature=args.temperature,
            batch_size=args.batch_size,
        )
    finally:
        backend.close()

    print("\n" + summarise(calibrations) + "\n")

    report_path = args.out_dir / f"{slug}.calibration.jsonl"
    bank_path = args.out_dir / f"{slug}.items.jsonl"
    save_calibration(calibrations, report_path)

    kept = apply_tiers(items, calibrations)
    save_bank(kept, bank_path)

    print(f"report -> {report_path}")
    print(f"bank   -> {bank_path}  ({len(kept)}/{len(items)} items kept)")

    if not kept:
        print("\nNO ITEMS SURVIVED. The bank is too hard for this model — regenerate easier "
              "items before running any conformity grid.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
