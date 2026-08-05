# CLAUDE.md — Project Memory

> **Read this file first in any new session.** It is the single source of truth for what this
> project is, what has been done, and what is next. Update the Status Log at the bottom at the
> end of every working session.

---

## 1. What this project is

**Research question:** When several LLM agents make the same error, are those errors *independent
and simultaneous* (correlated failure) or *causally coupled* (social influence)? And critically —
when an agent conforms to a wrong majority, does it **fabricate evidence** to justify the answer it
just switched to?

**Working title:** *Asch in Silicon: Does Social Conformity Cause Hallucination in Multi-Agent LLM Systems?*

**Target venue:** **ICLR 2027** — abstract **18 Sep 2026**, paper **25 Sep 2026** (AOE).
Fallback cascade (sequential, never simultaneous): ACL/EMNLP 2027 via ARR → FAccT 2027.
arXiv preprint goes up **immediately after submission (~25 Sep 2026)**.

**Full design rationale lives in [RESEARCH_PLAN.md](RESEARCH_PLAN.md). This file is the operational
state; that file is the science.**

### The three studies (one shared harness)

| Study | Question | Status |
|---|---|---|
| **1 — Asch replication** | Do LLM agents conform to a unanimous wrong majority on ground-truth items under *real message passing*? Do Asch's moderators (group size, dissenting ally, difficulty, privacy) replicate? | ✅ Harness built + verified end-to-end on mock. **Not yet run on a real model.** |
| **2 — Conformity → hallucination** (headline) | When an agent conforms, does it fabricate supporting evidence? Does the conformity persist in a fresh context (informational) or evaporate (normative)? | Not started — reuses Study 1 transcripts |
| **3 — Mere presence** | Is any of it diffusion of responsibility rather than conformity? Manipulate *stated existence* of co-reviewers with zero peer content. | Not started |

### Why this is publishable (verified against the PDFs in this folder)

- **Choi et al., ACL 2025** — subjective topics only, no ground truth. We use verifiable items.
- **Bellina et al., 2026** — social pressure *simulated via prompt*, no message passing. We use real confederate agents that generate their own justifications.
- **Shehata & Li, 2026 (Bystander)** — §3.4 confirms they inject a **poisoned unanimous consensus at every n≥2**, so their bystander effect is confounded with conformity. Study 3 runs the mere-presence condition they never ran.
- **Nobody has operationalized Deutsch & Gerard's normative-vs-informational split for LLMs.** Our fresh-context re-ask does exactly that. Cheapest novel measure in the design.

⚠️ **Do not repeat this error:** the note file `A genuinely unpublished angle_ the _LLM Bystander.md`
claims the LLM bystander effect is unpublished. It is **not** — Shehata & Li (arXiv:2605.10698) is in
this folder. Our contribution is the narrower, real gap described above.

⚠️ `conferences.md` lists AAMAS 2027 as a January 2027 deadline. **Wrong** — it is historically
October. AAMAS is excluded anyway (falls between ICLR submission and decision = dual submission).

---

## 2. Repository layout

```
src/asch/
  config.py      Experiment config, condition enums, TrialSpec, deterministic trial IDs
  backends.py    LLM backend abstraction: MockBackend (local dev), VLLMBackend (Kaggle), APIBackend
  items.py       Item schema, item bank load/save, perceptual item generator
  prompts.py     Prompt templates: confederate, naive agent, public/private, mere-presence
  runner.py      Trial expansion + resumable JSONL execution loop
  parsing.py     Answer extraction, stance classification
  analyze.py     Conformity rate, independence ratio, per-condition breakdown
data/
  smoke_items.jsonl     50 generated perceptual items (contamination-proof)
scripts/
  make_smoke_bank.py    Regenerate the smoke bank
  run_smoke.py          Day-3 smoke test — THE go/no-go gate
  calibrate.py          Per-model calibration pre-pass (Asch control condition)
results/                JSONL trial logs (gitignored, append-only)
tests/                  pytest
```

**Run everything from the repo root.** `python -m scripts.run_smoke --help`

---

## 3. Datasets — what we need and where it comes from

Nothing was in this folder at start; all data is either **generated** or **downloaded**.

| Tier | Source | Purpose | Status |
|---|---|---|---|
| **Perceptual-analog** | **Generated** (`make_smoke_bank.py`) — numeric magnitude, string length, letter counting | Direct Asch line-judgment homolog. Synthetic ⇒ **impossible to contaminate**, which pre-empts the biggest reviewer objection | ✅ Done — 50 items at `data/smoke_items.jsonl`, seed 20260806 |
| **Factual short-answer** | HuggingFace `trivia_qa` / SimpleQA | Real-world knowledge conformity | ⬜ Not downloaded |
| **Multi-step reasoning** | HuggingFace `gsm8k` | Tests whether pressure derails a derivation | ⬜ Not downloaded |
| **Knowledge MCQ** | HuggingFace `TIGER-Lab/MMLU-Pro` | Comparability with Wang et al. 2025 | ⬜ Not downloaded |

**The critical step is not downloading — it is the calibration pre-pass.** Raw benchmarks are
unusable as-is: Asch's control error rate was <1%, and if our baseline is noisy, a wrong answer
cannot be attributed to social pressure rather than plain ignorance. So `calibrate.py` runs every
candidate item alone, 5 samples, and keeps only those the model answers correctly ≥95% of the time.

**Consequence: item banks are per-model.** Cross-model comparisons use the intersection. A second
output is the **hard tier** (60–80% alone-accuracy) for the task-difficulty factor.

`datasets` is not installed locally and there is no local GPU — download and calibration run on
Kaggle/Colab, not on this machine.

---

## 4. Environment

- **Local (this machine):** Windows, Python 3.14, no GPU. Has `transformers`, `pandas`, `numpy`, `scipy`, `pytest`. Missing `datasets`, `vllm`, `statsmodels`.
- **Local dev is done entirely against `MockBackend`** — deterministic, no network, no GPU. The whole pipeline is testable here.
- **Kaggle (primary runner):** 2×T4, 30 h/week, less aggressive disconnects than Colab. Use for the Study 1 grid.
- **Colab:** easier iteration, use for development and one-off jobs.
- **Never hold results in memory.** The runner appends each trial to JSONL and flushes immediately, because Colab/Kaggle sessions die.

---

## 5. Key design invariants — do not silently change these

1. **Naive agent always answers last.** This is Asch's seating arrangement.
2. **Confederates generate their own justification text** while being constrained to a designated
   wrong answer. This is what distinguishes us from Bellina et al.'s prompt-simulated pressure.
3. **Confederate compliance is checked every trial.** If a confederate did not actually assert its
   assigned answer, the trial is marked invalid and excluded. Report the discard rate.
4. **Trial IDs are a deterministic hash of the full condition tuple.** This is what makes runs
   resumable and reruns idempotent.
5. **T=0 for the main grid** (comparability with prior work), plus a T=0.7 × 5-sample robustness
   subset. Shehata & Li list greedy-only as a limitation; covering it is cheap.
6. **Judge must be from a different model family than the agent under test, and blinded to
   condition** (n, unanimity, privacy stripped from its input).
7. **Design is frozen at OSF preregistration.** After that, changes are documented as deviations.

---

## 6. Status log

### 2026-08-06 — Session 1 (planning + P0 scaffolding)

**Done:**
- Read all 4 note files and 2 key PDFs (Bystander, Group Conformity). Verified the Shehata & Li
  confound at §3.4 — this is the load-bearing novelty claim, and it is real.
- Wrote [RESEARCH_PLAN.md](RESEARCH_PLAN.md): full design, 3 studies, DVs, stats, power analysis.
- Corrected two errors in the Perplexity-generated notes (bystander "unpublished" claim; AAMAS date).
- Re-planned from a 16-week serial schedule to a **6-week 3-track concurrent schedule** for ICLR.
- Built P0 harness: config/trial IDs, backend abstraction, prompts, resumable runner, parsing,
  analysis, 50-item smoke bank, tests.
- **38/38 tests pass.** Pipeline verified end-to-end on `MockBackend`: 100 trials, resume
  confirmed idempotent (second run executed 0), analysis recovers the mock's known conformity
  rate, baseline error 0.0% at n=0.
- **Bug caught by test, fixed:** `tabulate()` counted stances over all records but divided by
  valid-only, letting conformity rate exceed 1.0. Stances are now tallied over valid trials only.

**Next up (in order):**
1. ⬜ **Run the day-3 smoke test on a real model** (Kaggle, Qwen2.5-7B-Instruct). THE gate.
   `python -m scripts.run_smoke --backend vllm --model Qwen/Qwen2.5-7B-Instruct`
   Pass = conformity rate roughly 5–70% **and** baseline error <10%. The script prints an
   explicit PASS/FAIL verdict with the reason. At ~0% or ~100%, retune item difficulty before
   building anything further.
2. ⬜ Download the three HF datasets on Kaggle; build the full candidate bank.
3. ⬜ Run `calibrate.py` per model → per-model easy/hard tiers.
4. ⬜ OSF preregistration (must precede the full grid).
5. ⬜ Launch Study 1 full grid.

**Blocked on the user (raised 2026-08-06, both have lead time):**
- ⬜ Recruit **2 human annotators** for the 200-trial κ validation (needed W4–5, recruit now).
- ⬜ **Paid API budget** for the frontier replication (keys live by end of W2).

**Open questions:**
- Kaggle vs Colab as primary runner — leaning Kaggle for the grid, Colab for dev.
- Which frontier models for the replication (depends on budget).
