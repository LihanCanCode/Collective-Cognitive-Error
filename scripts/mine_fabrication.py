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

    all_records: list[dict] = []
    for path in files:
        records = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
        all_records.extend(records)

        by_pressure = score_by_condition(records, items, by=("n_confederates",))
        for key, stats in sorted(by_pressure.items(), key=lambda kv: kv[0]):
            n = key[0]
            label = f"{path.name}  (n_confederates={n})"
            rate = f"{stats.rate:.1%}" if stats.rate is not None else "-"
            print(f"{label:<70} {stats.n_wrong:>8} {stats.n_fabricated:>11} {rate:>7}")

    pressured_records = [r for r in all_records if (r.get("n_confederates") or 0) > 0]

    # De-duplicate n=0 (spontaneous) trials by (model, response_format, item_id) before pooling.
    # At n=0 there is no confederate, so confederate_style is a no-op -- the "bare" and
    # "justified" arms' n=0 slices are the same underlying alone-condition regenerated under a
    # different seed, not independent observations. Pooling them naively as separate trials
    # inflates the spontaneous denominator with near-duplicates and would understate how rare
    # spontaneous fabrication actually is. Keep first occurrence, deterministic file order.
    seen: set[tuple] = set()
    spontaneous_records = []
    for r in all_records:
        if (r.get("n_confederates") or 0) != 0:
            continue
        key = (r.get("model"), r.get("response_format"), r.get("item_id"))
        if key in seen:
            continue
        seen.add(key)
        spontaneous_records.append(r)

    n_duplicate_skipped = sum(1 for r in all_records if (r.get("n_confederates") or 0) == 0) - len(
        spontaneous_records
    )
    if n_duplicate_skipped:
        print(f"\n[note] skipped {n_duplicate_skipped} duplicate n=0 trials "
              "(same model+response_format+item, different confederate_style -- a no-op at n=0)")

    pressured_stats = score_records(pressured_records, items)
    spontaneous_stats = score_records(spontaneous_records, items)
    pw, pf, prate = pressured_stats.n_wrong, pressured_stats.n_fabricated, pressured_stats.rate
    sw, sf, srate = (
        spontaneous_stats.n_wrong, spontaneous_stats.n_fabricated, spontaneous_stats.rate
    )

    print("\n" + "=" * 70)
    print("POOLED (across all files, n=0 de-duplicated)")
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

    # Pooling across models is not a defensible number by itself -- models differ hugely in
    # baseline error and effect size (session 13), so a mixed pooled rate could just reflect
    # which model contributed more trials. Break out by model too.
    models = sorted({r.get("model") for r in all_records if r.get("model")})
    if len(models) > 1:
        print("\n" + "=" * 70)
        print("BY MODEL (the pooled figure above mixes models -- check this before quoting it)")
        print("=" * 70)
        for m in models:
            p_m = score_records([r for r in pressured_records if r.get("model") == m], items)
            s_m = score_records([r for r in spontaneous_records if r.get("model") == m], items)
            pr = f"{p_m.rate:.1%}" if p_m.rate is not None else "-"
            sr = f"{s_m.rate:.1%}" if s_m.rate is not None else "-"
            print(f"  {m:<40} pressured {p_m.n_fabricated:3d}/{p_m.n_wrong:<4d} {pr:>6}   "
                  f"spontaneous {s_m.n_fabricated:3d}/{s_m.n_wrong:<4d} {sr:>6}")


if __name__ == "__main__":
    main()
