# Asch in Silicon

Does social conformity *cause* hallucination in multi-agent LLM systems?

When several LLM agents make the same error, are those errors **independent and simultaneous**
(correlated failure) or **causally coupled** (social influence)? And when an agent conforms to a
wrong majority, does it fabricate evidence to justify the answer it just switched to?

Three studies over one shared harness:

1. **Asch replication** — do LLM agents conform to a unanimous wrong majority on
   ground-truth-verifiable items under *real inter-agent message passing*? Do Asch's moderators
   (group size, dissenting ally, task difficulty, response privacy) replicate?
2. **Conformity → hallucination** — when an agent conforms, does it fabricate supporting
   evidence? Does the conformity survive in a fresh context (informational influence) or
   evaporate (normative)? — the Deutsch & Gerard (1955) split, not previously operationalized
   for LLMs.
3. **Mere presence** — is any of it diffusion of responsibility rather than conformity?
   Manipulates the *stated existence* of co-reviewers with zero peer content.

See [RESEARCH_PLAN.md](RESEARCH_PLAN.md) for the full design and [CLAUDE.md](CLAUDE.md) for
current project state.

## Quick start

```bash
pip install -r requirements.txt

# Regenerate the perceptual item bank (deterministic given the seed)
python -m scripts.make_smoke_bank

# Verify the pipeline offline — no GPU, no network, no API key
python -m scripts.run_smoke --backend mock

# The real gate, on a GPU
python -m scripts.run_smoke --backend vllm --model Qwen/Qwen2.5-7B-Instruct

pytest tests/ -q
```

## Layout

| Path | Purpose |
|---|---|
| `src/asch/config.py` | Condition space, deterministic trial IDs, grid expansion |
| `src/asch/backends.py` | `MockBackend` (offline), `VLLMBackend` (Kaggle/Colab), `APIBackend` |
| `src/asch/items.py` | Item schema and the generated perceptual item bank |
| `src/asch/prompts.py` | Confederate scripting, naive-agent transcript, Study 2/3 prompts |
| `src/asch/runner.py` | Resumable, append-only trial execution |
| `src/asch/parsing.py` | Answer extraction, stance classification, compliance checks |
| `src/asch/analyze.py` | Asch-comparable conformity metrics with Wilson intervals |

## Design notes

- **The perceptual item bank is generated, not downloaded.** It is the direct translation of
  Asch's line-judgement task, and because the items never existed before, no model can have
  memorised them — which removes the contamination objection from the tier carrying the core
  replication claim.
- **Ground truth never enters a prompt.** It is passed out of band (`Backend.generate(oracle=)`)
  purely so the mock backend can simulate a competent agent; real backends ignore it. A test
  asserts the naive prompt contains no leak.
- **Confederates are real model calls** constrained to assert a designated wrong answer, writing
  their own justifications. Compliance is checked per trial and non-compliant trials are
  discarded, not silently kept.
- **The runner assumes the session will die.** Every trial is appended and flushed immediately;
  restarting skips completed trial IDs.
