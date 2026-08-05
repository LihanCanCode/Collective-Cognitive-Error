"""Kaggle / Colab smoke-test runner — THE go/no-go gate.

Paste each block below into a separate notebook cell, in order.
Kaggle: Settings -> Accelerator -> GPU T4 x2, and Internet -> ON.

Expected wall-clock: ~10-20 min for 100 trials on a 7B model, most of it model download.

--------------------------------------------------------------------------------------
CELL 1 — install
--------------------------------------------------------------------------------------
!pip install -q vllm==0.6.3

# Kaggle ships a preinstalled torch that sometimes conflicts with vllm. If the import in
# CELL 3 fails, restart the session once after this cell and re-run from CELL 2.

--------------------------------------------------------------------------------------
CELL 2 — get the code
--------------------------------------------------------------------------------------
!git clone https://github.com/LihanCanCode/Collective-Cognitive-Error.git repo
%cd repo
!python -m scripts.make_smoke_bank

# Sanity: the whole pipeline must pass offline before we spend a GPU-second on it.
!python -m scripts.run_smoke --backend mock --out results/mock_check.jsonl

--------------------------------------------------------------------------------------
CELL 3 — run the gate
--------------------------------------------------------------------------------------
!python -m scripts.run_smoke \
    --backend vllm \
    --model Qwen/Qwen2.5-7B-Instruct \
    --out results/smoke_qwen7b.jsonl

--------------------------------------------------------------------------------------
CELL 4 — inspect a few transcripts by hand
--------------------------------------------------------------------------------------
# Do not skip this. The printed conformity number is meaningless if the confederates are
# breaking character or the naive agent is answering in an unparsed format. Read five.

import json
from pathlib import Path

records = [json.loads(l) for l in Path("results/smoke_qwen7b.jsonl").open()]
critical = [r for r in records if r["n_confederates"] == 3]

for rec in critical[:5]:
    print("=" * 90)
    print(f"stance={rec['stance']}  answer={rec['answer']}  correct={rec['correct_answer']}  "
          f"majority={rec['majority_answer']}  valid={rec['valid']}")
    for turn in rec["transcript"]:
        who = turn["role"]
        extra = f" (assigned {turn['assigned_answer']}, complied={turn['complied']})" if who == "confederate" else ""
        print(f"\n--- {who}{extra} ---")
        print(turn["text"][:400])

--------------------------------------------------------------------------------------
CELL 5 — diagnostics
--------------------------------------------------------------------------------------
# If the verdict was FAIL, these two numbers say which knob to turn.

from collections import Counter

print("stance distribution:", Counter(r["stance"] for r in critical))
print("confederate break rate:",
      sum(1 for r in critical if not r.get("confederates_complied")) / max(len(critical), 1))
print("parse failure rate:",
      sum(1 for r in critical if r.get("answer") is None) / max(len(critical), 1))

# High break rate      -> confederates refuse to argue for the wrong answer. Soften the
#                         confederate instruction, or use a less safety-tuned confederate model.
# High parse failure   -> the model ignores the output format. Tighten the format instruction
#                         or add a one-shot example; do NOT loosen the parser to compensate.
# CR ~0 with both low  -> genuine independence. Raise item difficulty (Asch: conformity rises
#                         with difficulty) rather than concluding the effect is absent.

--------------------------------------------------------------------------------------
CELL 6 — save results off the ephemeral session
--------------------------------------------------------------------------------------
# Kaggle wipes /kaggle/working when the session ends. Commit the notebook (which persists
# /kaggle/working as output) or download the JSONL before closing.
!cp results/smoke_qwen7b.jsonl /kaggle/working/ 2>/dev/null || true
!ls -la results/
"""
