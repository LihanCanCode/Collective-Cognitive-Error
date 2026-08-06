"""Generate the 50-item perceptual smoke-test bank.

Deterministic given the seed, so the bank is reproducible from the seed alone.

Both invocation forms work, from any working directory:
    python scripts/make_smoke_bank.py
    python -m scripts.make_smoke_bank
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# See run_smoke.py -- keeps imports working regardless of invocation form or cwd.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.asch.items import generate_perceptual_bank, save_bank  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    # 50 is the gate's size -- enough to spot a broken bank, not enough to measure with.
    # Observed effects (0% / 6% / 20%) need ~129 items per arm for 80% power, so the main bank
    # is 200. See scripts/run_arms.py, which prints the required n alongside every p-value.
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=20260806)
    ap.add_argument("--out", type=Path, default=_REPO_ROOT / "data" / "smoke_items.jsonl")
    args = ap.parse_args()

    items = generate_perceptual_bank(n=args.n, seed=args.seed)
    save_bank(items, args.out)
    print(f"Wrote {len(items)} items to {args.out}")

    by_subtype: dict[str, int] = {}
    for item in items:
        by_subtype[item.subtype] = by_subtype.get(item.subtype, 0) + 1
    for subtype, count in sorted(by_subtype.items()):
        print(f"  {subtype}: {count}")


if __name__ == "__main__":
    main()
