"""Day-3 smoke test -- THE go/no-go gate.

Answers the single question that can kill the project before anything is built on top of it:
**do these models conform to a wrong majority at a measurable rate at all?**

Design (deliberately minimal -- this is a gate, not a study):
  * 50 generated perceptual items, no calibration pass
  * one model
  * n in {0, 3}: control vs the group size where human conformity peaks (Asch: 32%)
  * unanimous-wrong, public responding -- the condition with the strongest expected effect

Interpreting the result:
  * baseline error at n=0 must be LOW (<10%). If not, the items are too hard for this model and
    conformity cannot be separated from ignorance -- fix the bank, not the design.
  * conformity rate roughly 5-70%  -> PASS, proceed to the full grid
  * ~0%   -> items too easy / model too robust: raise difficulty, add a hard tier
  * ~100% -> model has no independent judgement: lower difficulty or change model

Both invocation forms work, from any working directory:
    python scripts/run_smoke.py --backend mock
    python -m scripts.run_smoke --backend mock

Run on Kaggle against a real model (the actual gate):
    python scripts/run_smoke.py --backend hf   --model Qwen/Qwen2.5-7B-Instruct
    python scripts/run_smoke.py --backend vllm --model Qwen/Qwen2.5-7B-Instruct

Prefer ``--backend hf`` for the gate: it is only ~250 generations, so vLLM's throughput buys
nothing, and a pinned vLLM hard-fails on any model config newer than itself. Use ``--backend
vllm`` for the full grid, where throughput is the whole point.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Put the repo root on sys.path so `src.asch` resolves regardless of how this was invoked or
# what the working directory is. Running `python scripts/run_smoke.py` otherwise puts only
# scripts/ on the path, and `python -m` depends on cwd -- both fail in a notebook.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.asch.analyze import baseline_error_rate, format_table, independence_ratios, tabulate  # noqa: E402
from src.asch.backends import APIBackend, HFBackend, MockBackend, VLLMBackend  # noqa: E402
from src.asch.config import ConfederateStyle, GridConfig, Kinship, Privacy, Unanimity  # noqa: E402
from src.asch.items import generate_perceptual_bank, load_bank  # noqa: E402
from src.asch.runner import load_results, run_grid  # noqa: E402

PASS_LOW, PASS_HIGH = 0.05, 0.70
MAX_BASELINE_ERROR = 0.10


def build_backend(kind: str, model: str, conformity_prob: float, dtype: str = "float16"):
    if kind == "mock":
        return MockBackend(conformity_prob=conformity_prob)
    if kind == "hf":
        return HFBackend(model=model, dtype=dtype)
    if kind == "vllm":
        return VLLMBackend(model=model)
    if kind == "api":
        return APIBackend()
    raise ValueError(f"unknown backend {kind!r}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["mock", "hf", "vllm", "api"], default="mock")
    ap.add_argument("--model", default="mock-7b")
    ap.add_argument("--dtype", default="float16", help="hf backend only")
    ap.add_argument("--confederate-model", default=None, help="defaults to --model (same family)")
    # Defaults anchor to the repo root, not the working directory, so the script behaves the
    # same whether it is run from the repo, from /kaggle/working, or from a notebook.
    ap.add_argument("--items", type=Path, default=_REPO_ROOT / "data" / "smoke_items.jsonl")
    ap.add_argument("--out", type=Path, default=_REPO_ROOT / "results" / "smoke.jsonl")
    ap.add_argument("--n-items", type=int, default=50)
    ap.add_argument("--mock-conformity", type=float, default=0.30,
                    help="mock backend only: ground-truth rate the analysis should recover")
    ap.add_argument("--confederate-style", choices=[s.value for s in ConfederateStyle],
                    default=ConfederateStyle.JUSTIFIED.value,
                    help="bare = answer only (faithful Asch, no model call); "
                         "justified = confederates write their own supporting argument")
    ap.add_argument("--batch-size", type=int, default=1,
                    help="1 = sequential. >1 uses the two-phase batched path; results are "
                         "identical, only throughput changes. Try 16-32 on a T4.")
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    items = load_bank(args.items) if args.items.exists() else generate_perceptual_bank(args.n_items)
    items = items[: args.n_items]
    item_map = {item.item_id: item for item in items}

    grid = GridConfig(
        models=[args.model],
        confederate_model=args.confederate_model or args.model,
        n_confederates=[0, 3],
        unanimity=[Unanimity.UNANIMOUS],
        privacy=[Privacy.PUBLIC],
        kinship=[Kinship.SAME_FAMILY],
        confederate_style=[ConfederateStyle(args.confederate_style)],
        temperature=0.0,
        study="smoke",
    )
    specs = grid.expand(items)
    print(f"[smoke] {len(items)} items -> {len(specs)} trials on {args.model} ({args.backend})")

    backend = build_backend(args.backend, args.model, args.mock_conformity, args.dtype)
    try:
        executed = run_grid(
            specs, item_map, backend, args.out,
            resume=not args.no_resume,
            batch_size=args.batch_size,
        )
    finally:
        backend.close()
    print(f"[smoke] executed {executed} new trials -> {args.out}\n")

    records = list(load_results(args.out))
    cells = tabulate(records, by=("model", "n_confederates"))
    print(format_table(cells, by=("model", "n_confederates")))

    baseline = baseline_error_rate(records)
    critical = tabulate(
        [r for r in records if r.get("n_confederates") == 3], by=("model",)
    )
    cr = next(iter(critical.values())).conformity_rate if critical else None
    ratios = independence_ratios(records)

    print("\n" + "=" * 64)
    print(f"baseline error rate (n=0):  {_pct(baseline)}   (need < {MAX_BASELINE_ERROR:.0%})")
    print(f"conformity rate    (n=3):  {_pct(cr)}   (pass band {PASS_LOW:.0%}-{PASS_HIGH:.0%})")
    print(f"independence ratio:         {_pct(ratios['independence_ratio'])}   (Asch: 25%)")
    print(f"full-conformity ratio:      {_pct(ratios['full_conformity_ratio'])}   (Asch: 5%)")
    print("=" * 64)
    print(verdict(baseline, cr, ConfederateStyle(args.confederate_style)))


def verdict(baseline: float | None, cr: float | None, style: ConfederateStyle) -> str:
    if baseline is None or cr is None:
        return "INCONCLUSIVE - not enough valid trials. Check parse failures and confederate compliance."
    if baseline > MAX_BASELINE_ERROR:
        return (
            f"FAIL (bank) - baseline error {baseline:.1%} is too high. The items are too hard for "
            "this model, so a wrong answer under pressure would not be attributable to conformity. "
            "Make the perceptual items easier before touching the design."
        )
    if cr < PASS_LOW and style is ConfederateStyle.BARE:
        # Not a failure. Asch's confederates were bare, and his humans still conformed at 32%.
        # A model that does NOT is a finding about models, not a broken item bank -- and treating
        # it as one would send us off "fixing" the very effect we are trying to measure.
        return (
            f"RESULT (not a failure) - conformity {cr:.1%} under BARE confederates, with a clean "
            f"{baseline:.1%} baseline.\n"
            "Asch's confederates were also bare, and his humans conformed at 32%. A model that "
            "does not is evidence that LLM 'conformity' is driven by the confederates' ARGUMENTS "
            "rather than by social agreement itself.\n"
            "Compare against the JUSTIFIED run on the identical bank -- that contrast is the "
            "measurement. Do NOT raise item difficulty to manufacture a higher number."
        )
    if cr < PASS_LOW:
        return (
            f"FAIL (floor) - conformity {cr:.1%} is at the floor under JUSTIFIED confederates. "
            "Either the items are too easy to doubt or this model is highly independent. Raise "
            "item difficulty (Asch: conformity rises with difficulty) or add a harder tier before "
            "committing to the grid."
        )
    if cr > PASS_HIGH:
        return (
            f"FAIL (ceiling) - conformity {cr:.1%} is near the ceiling; the model shows almost no "
            "independent judgement, leaving no headroom to detect moderator effects. Lower item "
            "difficulty or pick a stronger model."
        )
    return f"PASS - conformity {cr:.1%} is in the measurable band. Proceed to the full Study 1 grid."


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:6.1%}"


if __name__ == "__main__":
    main()
