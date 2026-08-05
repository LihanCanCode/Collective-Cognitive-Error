"""Generate the 50-item perceptual smoke-test bank.

Deterministic given the seed, so the bank is reproducible from the seed alone.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.asch.items import generate_perceptual_bank, save_bank


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=20260806)
    ap.add_argument("--out", type=Path, default=Path("data/smoke_items.jsonl"))
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
