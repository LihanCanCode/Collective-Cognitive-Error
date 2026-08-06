"""Study 2, mined from Study 1's transcripts -- no new GPU generation needed.

Reads the JSONL files already produced by run_arms.py and asks: when the naive agent was wrong,
did its own response assert a checkable falsehood about the stimulus (e.g. "787 is the largest"
when it is not)? Every item is synthetic with known values, so this is mechanically decidable --
no LLM-as-judge, no subjective grading. See src/asch/fabrication.py for the detector and its
deliberately conservative design (undercounts subtle cases, essentially never over-counts).

The key comparison: fabrication rate under social pressure (n>0, typically the ANSWER_FIRST arms
where real conformity occurs) vs spontaneous, unprompted error (n=0). If pressured fabrication is
much higher, that is evidence social pressure induces *confabulated* justification specifically,
not just more of the same errors.

Usage (point at the directory run_arms.py wrote to):
    python scripts/mine_fabrication.py results/arms
    python scripts/mine_fabrication.py /kaggle/working/results_arms --items data/items_main.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.asch.fabrication import score_by_condition, score_records  # noqa: E402
from src.asch.items import generate_perceptual_bank, load_bank  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results_dir", type=Path, help="directory of run_arms.py JSONL output")
    ap.add_argument("--items", type=Path, default=_REPO_ROOT / "data" / "items_main.jsonl",
                    help="the exact bank used to generate results_dir; regenerated "
                         "(seed-deterministic) if the file is not found")
    args = ap.parse_args()

    if args.items.exists():
        items = {i.item_id: i for i in load_bank(args.items)}
    else:
        print(f"[mine_fabrication] {args.items} not found — regenerating the 200-item bank "
              "(deterministic given the seed, so this matches what run_arms.py used)",
              file=sys.stderr)
        items = {i.item_id: i for i in generate_perceptual_bank(200)}

    files = sorted(args.results_dir.glob("*.jsonl"))
    if not files:
        print(f"no .jsonl files in {args.results_dir}")
        return

    print(f"{'file':<70} {'n_wrong':>8} {'fabricated':>11} {'rate':>7}")
    print("-" * 100)

    all_pressured, all_spontaneous = [], []
    for path in files:
        records = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
        by_pressure = score_by_condition(records, items, by=("n_confederates",))

        for key, stats in sorted(by_pressure.items(), key=lambda kv: kv[0]):
            n = key[0]
            label = f"{path.name}  (n_confederates={n})"
            rate = f"{stats.rate:.1%}" if stats.rate is not None else "-"
            print(f"{label:<70} {stats.n_wrong:>8} {stats.n_fabricated:>11} {rate:>7}")
            if n and n > 0:
                all_pressured.append(stats)
            else:
                all_spontaneous.append(stats)

    def pool(stats_list):
        n_wrong = sum(s.n_wrong for s in stats_list)
        n_fab = sum(s.n_fabricated for s in stats_list)
        return n_wrong, n_fab, (n_fab / n_wrong if n_wrong else None)

    pw, pf, prate = pool(all_pressured)
    sw, sf, srate = pool(all_spontaneous)

    print("\n" + "=" * 70)
    print("POOLED (across all files)")
    print("=" * 70)
    print(f"  pressured   (n_confederates>0): {pf:4d}/{pw:<4d}  "
          f"{'-' if prate is None else f'{prate:.1%}'}")
    print(f"  spontaneous (n_confederates=0): {sf:4d}/{sw:<4d}  "
          f"{'-' if srate is None else f'{srate:.1%}'}")

    if prate is not None and srate is not None:
        print(f"\n  ratio (pressured / spontaneous): {prate / srate if srate else float('inf'):.2f}x"
              if srate else "\n  spontaneous rate is 0 -- any pressured fabrication is infinitely higher")
        if prate > srate:
            print("  => social pressure is associated with MORE fabrication than spontaneous "
                  "error, not just more errors of the same kind.")
    elif prate is not None and srate is None:
        print("\n  no spontaneous errors to compare against (clean baseline) -- pressured "
              "fabrication rate stands on its own but has no within-model control.")


if __name__ == "__main__":
    main()
