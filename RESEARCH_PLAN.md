# Implementation Plan — Correlated Cognitive Error and Hallucination in Multi-Agent LLMs

**Working title:** *Asch in Silicon: Does Social Conformity Cause Hallucination in Multi-Agent LLM Systems?*

**Target:** **ICLR 2027** — abstract **18 Sep 2026**, paper **25 Sep 2026** (AOE). Full scope, no reductions. Sequential fallbacks after decision. See §6.

**Compute:** Phase 1 free Colab/Kaggle T4 + open-weights. Phase 2 paid APIs + Colab Pro. No local GPU.

---

> **⚠️ Reframing #2, 2026-08-07 (200-item, 2-model confirmation).** The measurement-artefact
> dissolution now **replicates on a second, architecturally distinct model family** (Mistral-7B,
> not just Qwen) with p<0.0001. This is now the load-bearing result of the paper, not a side
> arm. **Consequence: the original Study 1 moderator battery (group size × ally × privacy ×
> kinship) is deprioritized for this paper.** It was designed when the headline claim was "Asch
> replicates on LLMs, here are the boundary conditions" — with the claim now "the reported effect
> is largely two measurement artefacts, and here is the dissolution," the moderator battery adds
> combinatorial cost without strengthening that core claim. It remains valuable as follow-up work
> (a paper in its own right, given the Study 1 harness already exists) but is not on the critical
> path to submission. See CLAUDE.md session 13 for the full reasoning and the numbers.
>
> **⚠️ Reframing #1, 2026-08-06 (50-item pilot).** A bare-assertion control changed the headline
> claim. On an identical bank, Qwen2.5-7B conformed at **16%** when confederates supplied
> arguments and **2%** when they stated only an answer — which is what Asch's confederates did,
> and his humans still conformed at 32%. So the effect the LLM-conformity literature reports may
> be **persuasion by plausible-sounding reasoning, not social conformity**. No prior paper (Choi
> et al. 2025; Bellina et al. 2026; Shehata & Li 2026) runs a bare control, so none can separate
> the two. Superseded in scale by the 200-item run above, but the mechanism is the same.

## 0a. Current scope for THIS paper (as of 2026-08-07)

**In scope, load-bearing:**
1. The measurement-artefact dissolution (arm sweep: BARE/FILLER/JUSTIFIED × REASONING/ANSWER_FIRST)
   — reproduces the literature's magnitude, then dissolves it. Replicated on 2+ model families.
2. Study 2, reframed as a mined-transcript analysis (`fabrication.py`) rather than a fresh
   experimental arm: ground-truth-verifiable fabrication rate, conditional on being wrong,
   pressured vs spontaneous. Costs no additional GPU time — it runs on Study 1's own transcripts.

**Out of scope for this paper, deferred to follow-up work:**
- The full Study 1 moderator battery (group size sweep, ally/incompetent-dissenter, privacy,
  kinship) — large combinatorial space, does not strengthen the dissolution claim, better scoped
  as its own paper given the harness already supports it.
- Study 3 (mere-presence / bystander effect) — the clean, unpublished-gap angle from the original
  scoping notes, but a separate experimental arm with its own item type (document review, not
  perceptual comparison). Natural next paper once this one is submitted.

## 0. The claim we are making

Three nested questions, one shared experimental harness:

1. **(Study 1 — replication gap)** Do LLM agents conform to a unanimous wrong majority on *ground-truth-verifiable* items under *real inter-agent message passing*, and do Asch's four moderators (group size, dissenting ally, task difficulty, response privacy) replicate?
2. **(Study 2 — the novel headline)** When an agent conforms, does it *fabricate supporting evidence*? I.e. does social pressure convert a stance change into a genuine hallucination? And does the conformity persist in a fresh context (internalization) or vanish (mere compliance)?
3. **(Study 3 — causal isolation)** Is any of this diffusion of responsibility rather than conformity? Manipulate the *stated existence* of co-reviewers with zero peer content.

**Positioning against prior work (verified against the PDFs in this folder):**

| Prior work | What it did | What we add |
|---|---|---|
| Choi et al., ACL 2025 Findings | Multi-agent debate, 5 subjective social topics, no ground truth | Ground-truth-verifiable items → conformity becomes *error*, measurable |
| Bellina et al. 2026 | Asch-style perceptual tasks, but social pressure **simulated via prompt**, no message passing | Real confederate agents generating their own justification text |
| Shehata & Li 2026 (Bystander) | 22,500 trajectories; but for every n≥2 they **inject a unanimous poisoned consensus** (§3.4) — bystander effect and conformity are confounded | Study 3 runs the mere-presence condition they never ran, decoupling the two |
| Jamshidi et al. 2026 (Cascade) | Hallucination propagation, no conformity measure | Study 2 links conformity → hallucination in one design |
| Wang et al. 2025 (Sycophancy) | Single-agent, MMLU multiple-choice, user opinion only | Peer-agent pressure, free-text rationales, internalization test |

Nobody has operationalized Deutsch & Gerard's **normative vs informational** split for LLMs. Our fresh-context re-ask does exactly that, and it is a clean, quotable contribution.

---

## 1. Experimental design

### 1.1 Item bank

Four tiers, ~1500 items total after filtering:

- **Perceptual-analog (Asch homolog):** numeric magnitude comparison, string-length judgment, sequence-position. Synthetic, uncontaminated, unambiguous. This is the direct line-judgment translation.
- **Factual short-answer:** SimpleQA / TriviaQA subset.
- **Multi-step reasoning:** GSM8K.
- **Knowledge MCQ:** MMLU-Pro subset (for comparability with Wang et al.).

### 1.2 Calibration pre-pass — non-negotiable

This is Asch's control condition and it is what makes the whole design valid. For each model, run every candidate item alone, 5 samples, T=0.7. **Keep only items the model answers correctly ≥95% of the time.** Asch's control error rate was <1%; if our baseline is noisy, conformity is unattributable to social pressure. Item banks are therefore **per-model**, with a shared common subset for cross-model comparison.

Produces a second tier for free: **hard items** (alone-accuracy 60–80%) for the task-difficulty factor.

### 1.3 Study 1 factors

| Factor | Levels |
|---|---|
| Majority size `n` | 0 (control), 1, 2, 3, 5, 7 — tests the human saturation-at-3 finding |
| Unanimity | unanimous-wrong / **correct ally** (one confederate gives truth) / **incompetent dissenter** (one confederate gives a *different* wrong answer, per Allen & Levine 1968) |
| Response privacy | public (answer enters shared transcript) / private (answer goes only to experimenter, group never sees it) |
| Task difficulty | easy tier (≥95% alone) / hard tier (60–80% alone) |
| Confederate kinship | same-family instances / cross-family models |

**Message passing (the methodological differentiator):** confederates are real model instances instructed to argue for a designated wrong answer, each producing its own natural justification. They post turn-by-turn into a shared transcript; the naive agent answers **last**, matching Asch's seating. Full transcripts logged. Confederate character-break detection (did it actually assert the wrong answer?) is an automatic trial-validity filter — discard broken trials, report the discard rate.

### 1.4 Dependent variables

1. **Conformity rate (CR)** — % of critical trials matching the majority's wrong answer
2. **Full-conformity ratio** — % of agents conforming on *every* critical trial (Asch: ~5%); **independence ratio** — % never conforming (Asch: 25%)
3. **Accuracy** vs. the n=0 baseline
4. **Hallucination rate in rationale** — atomic-claim decomposition of the naive agent's justification, each claim verified; reported separately for conforming vs non-conforming trials ← *Study 2 headline*
5. **Internalization rate** — same question re-asked in a **fresh context** with no transcript. Conformity that persists = informational influence; conformity that evaporates = normative compliance ← *Study 2, second novel measure*
6. **Rationale length** (effort proxy) and **self-reported confidence** + calibration error (ECE)
7. **Stance classification**: ADOPTED / REJECTED / IGNORED / UNKNOWN (adopt Shehata & Li's scheme for direct comparability — cite it)

### 1.5 Study 3 — mere presence

No peer content at any point. Conditions: *"You are the only reviewer"* / *"k other agents will also review this"* / *"k other agents have already reviewed and approved this"* (approval stated, content withheld), k ∈ {2, 5}. Same item bank, planted-error detection task. If vigilance drops with k while no opinion was ever shown, diffusion of responsibility is real and separable from conformity — which is precisely the inference Shehata & Li's design cannot support.

---

## 2. Models

**Phase 1 (free tier, all trials):** Qwen2.5-Instruct at 1.5B / 7B / 14B-AWQ (within-family size scaling), Llama-3.1-8B-Instruct, Mistral-7B-Instruct-v0.3, Gemma-2-9B-it. Served with vLLM on Colab T4 / Kaggle 2×T4 (30 h/week), batched offline inference, 4-bit where needed.

**Phase 2 (paid):** GPT-5.x, Claude, Gemini as the naive agent on a reduced cell set, to show the effect holds (or doesn't) at the frontier. Shehata & Li found Claude fully resilient and GPT collapsing — a cross-family divergence we should test on our paradigm.

**Temperature:** T=0 for the main grid (comparability), plus a T=0.7 × 5-sample robustness subset. Shehata & Li list greedy-decoding-only as an explicit limitation; covering it is cheap and pre-empts a reviewer.

---

## 3. Judging and validation

- **Claim verification:** LLM-as-judge with atomic-claim decomposition, judge constrained to a *different model family* than the agent under test, blinded to condition (`n`, unanimity, privacy stripped from the input).
- **Human validation:** 200-trial stratified subset, two independent annotators, report Cohen's κ. A conference paper will not survive review without this.
- **Contamination check:** verify perceptual-analog items are synthetic and unseen; report per-tier results so a contamination objection cannot sink the whole paper.

---

## 4. Statistics

- Mixed-effects logistic regression: `conform ~ n * unanimity * privacy * difficulty + (1|item) + (1|model)`
- Effect sizes with CIs; Holm correction across the condition grid; Fisher exact for individual cell contrasts (matches prior work's reporting)
- **Power:** ⚠️ **superseded by gate run 4.** The original estimate assumed CR 30% → 15% (~120
  trials/cell, ~4,300/model). The observed base rate on a *clean* bank under REASONING_FIRST is
  **6%**, which needs ~350/cell to detect a doubling and ~1,200/cell to detect a 50% increase —
  3–10× the original budget. Mitigations: test moderators under JUSTIFIED confederates (36% CR,
  real headroom) rather than at the 6% floor, and grow the calibrated bank to 400+ items. The
  large contrasts (BARE/FILLER/JUSTIFIED, REASONING/ANSWER_FIRST) carry the paper and need far
  less n. See CLAUDE.md session 8.
- **Preregister on OSF** before Phase 2 runs. Cheap, and it makes the replication claim credible.

---

## 5. Phases and go/no-go gates

Calendar anchored to **start 6 Aug 2026**, **submit 25 Sep 2026**. Nothing is cut; the phases are run **concurrently** instead of serially. Three tracks run in parallel throughout:

- **Track A — Experiments** (compute-bound, embarrassingly parallel across models)
- **Track B — Writing** (starts week 1; the related-work and method sections do not need results)
- **Track C — Human/admin lead time** (annotators, API budget, OpenReview, preregistration — all have latency and none need results)

| Week | Track A — Experiments | Track B — Writing | Track C — Lead time |
|---|---|---|---|
| **W1** Aug 6–13 | Harness, JSONL logger, resumable runner. **Day-3 smoke test** on a hand-built 50-item bank (see gate below). Full item bank + calibration pre-pass | Related Work (all key PDFs already read); Introduction framing | Recruit 2 annotators; OpenReview profile; **OSF preregistration**; secure paid-API budget |
| **W2** Aug 13–20 | **P1 pilot gate.** Confederate-compliance test. On pass, Study 1 grid launches immediately on model 1 | Method section (design is frozen at preregistration, so this is writable now) | Annotation guidelines + interface; API keys live |
| **W3** Aug 20–27 | Study 1 full grid, all open models, **parallel sessions across Kaggle/Colab accounts**. Study 3 (mere-presence) runs concurrently — independent prompt set, same harness | Experimental-setup section; figure scaffolding with dummy data | Annotators trained on pilot transcripts |
| **W4** Aug 27–Sep 3 | Study 2: claim-level judging over W3 transcripts + fresh-context re-ask. **Frontier-API replication runs concurrently** (no longer deferred) | Results section skeleton wired to the analysis notebook | Annotators begin the 200-trial subset |
| **W5** Sep 3–10 | Buffer + reruns for any broken cells. Robustness subset (T=0.7 × 5) | Full first draft: results, analysis, discussion | Human validation returns; κ computed |
| **W6** Sep 10–18 | Final figures, all statistics locked | Complete draft, internal read-through, limitations | **Abstract submitted Sep 18** |
| **W6.5** Sep 18–25 | Appendix, supplementary, reproducibility package | Polish, format check | **Paper submitted Sep 25** |

**What makes this real, not optimism:**

1. **Compute was never the constraint.** ~4,300 trials/model at 7B with vLLM batching is a few T4-hours. The model dimension is fully parallel — 6 models can run simultaneously across separate Kaggle/Colab sessions. W3 is wall-clock-bound by session limits, not by total FLOPs.
2. **Study 2 needs almost no new generation** — it re-judges W3's stored transcripts. This was already true in the serial plan and is why the headline result is cheap.
3. **Writing does not wait for results.** Related work, method, and setup are ~50% of the paper and are all writable in W1–W3 because the design is frozen at preregistration.
4. **ICLR has a rebuttal period.** Reviews land ~Nov 2026 with an author-discussion window. Additional robustness experiments are *normal and expected* there. This is not a cut — it is how the venue works.

**The single load-bearing change: the Day-3 smoke test.** In the serial plan the P1 gate sat at week 4 with slack behind it. Now it sits at W2 with none. So we pull a scaled-down version forward to **day 3 of W1**: 50 hand-written items, one model, n ∈ {0,3}, unanimous-wrong, no calibration pass. It answers the only question that can kill the project — *do these models conform at a measurable rate at all?* — before we build anything on top of the assumption. Cost: a few hours. If conformity is ~0% or ~100%, we retune difficulty in W1 instead of discovering it in W2 with nothing behind it.

Study 2 reusing Study 1's transcripts is the key efficiency: the headline result costs almost no extra generation.

---

## 6. Venue strategy

Verified Aug 2026 — `conferences.md` in this folder was generated for the earlier bystander-only framing and has at least one date materially wrong.

A **sequential cascade**, not parallel submissions — simultaneous submission to two venues is prohibited.

| Order | Venue | Deadline | Role |
|---|---|---|---|
| **1** | **ICLR 2027** (Rio, Apr 2027) | Abstract **Sep 18**, paper **Sep 25, 2026** | **Primary target.** Strong AI-safety / multi-agent / LLM-behaviour presence; open review; rebuttal period absorbs added robustness work. |
| 2 | **ACL 2027** (Kyoto) via ARR, or **EMNLP 2027** | ARR cycle after ICLR decisions (~late Jan 2027) | If ICLR rejects, the reviews are public and detailed — resubmit strengthened. NLP audience knows this literature well. |
| 3 | **FAccT 2027** | Jan–Feb 2027 | Accountability framing: multi-agent review pipelines silently losing vigilance. Timing may collide with ICLR decisions — check before committing. |
| — | **AAMAS 2027** | historically **Oct 8–28, 2026** (site says TBC) — *not* Jan 2027 as `conferences.md` claims | **Excluded.** Falls between ICLR submission and decision, so submitting there would be a dual submission. |
| — | NeurIPS 2026, CI 2026, HCOMP 2026, AIES 2026, ICLR-cycle NeurIPS 2027 | passed / too late | — |

**arXiv preprint: post on ~Sep 25, immediately after submission.** ICLR is non-anonymous-friendly (OpenReview submissions become public), so preprinting costs nothing and secures priority. This area moves fast — Shehata & Li posted May 2026.

---

## 7. Risks

| Risk | Mitigation |
|---|---|
| 7B models show ~0% conformity (ceiling) | Hard tier exists precisely for this; Asch found conformity rises with difficulty. P1 gate catches it early. |
| Models refuse to role-play confederates | Frame as "argue for position X in a structured debate"; pre-test confederate compliance in P0. |
| Colab disconnects mid-run | Every trial written to append-only JSONL immediately; runner resumes from last completed trial ID. Never hold results in memory. |
| Judge unreliability | Cross-family blinded judge + human κ; report judge-model sensitivity on a subset. |
| Benchmark contamination | Synthetic perceptual tier is contamination-proof; report per-tier. |
| Scope creep | Mechanistic/activation-patching study is **explicitly out of scope** — needs internals access and doubles the timeline. |
| **Human annotators unavailable in time** (longest non-compute lead time) | Recruit in W1, not W4. If κ-validation slips, submit with judge-only results + a stated validation-in-progress note, and land the human numbers in the rebuttal. |
| **Paid-API budget not approved by W3** | Frontier replication is what makes reviewers take the result seriously. Secure budget in W1. If it fails, open-weights-only is still a complete paper — the size-scaling story within the Qwen family carries it. |
| **Free-tier session limits throttle W3** | Parallelize across accounts; Kaggle 2×T4 / 30h-week as the primary runner. Runner must be resumable — never hold results in memory. |
| **Compressed schedule leaves no slack** | W5 is deliberately a buffer week, not a work week. If W3 slips into it, W5 absorbs it. If nothing slips, W5 buys the robustness subset. |

---

## 8. Deliverables

1. Reproducible harness (config-driven conditions, full transcript logs)
2. Per-model calibrated item bank
3. Trial-level dataset (releasable — a contribution in itself)
4. Paper: Asch replication + conformity→hallucination coupling + normative/informational split + mere-presence isolation
