This is a great question — it's actually touching on a few overlapping human phenomena, and the LLM parallel you're drawing is a real and active research area. Let me break both down.

## The human phenomenon

There isn't one single term — it's usually a mix of these, depending on the exact scenario:

- **Inattentional deafness/blindness** — when people are cognitively absorbed in something (a conversation, a task), they fail to notice salient stimuli, like a train sound. This is the individual mechanism, but it can happen to a whole group simultaneously if they're all absorbed in the same shared context (e.g., all focused on a conversation), which makes it *look* collective even though each brain is doing the same thing independently.
- **Groupthink** — when a group prioritizes agreement/harmony over accurate independent judgment, leading to shared errors. More about decision-making than perception.
- **Asch conformity effect** — the classic finding that people will give an answer they know is wrong just because others around them said it first. This is probably the closest match to your "3/4 people making a similar mistake" framing if there's any social influence involved (one person says something, others defer).
- **Socially shared retrieval-induced forgetting (SS-RIF)** — real, specific research area: when one person recalls (or misremembers) something in a group, it actually suppresses/distorts other members' memory of the same event. This maps very well onto your "forgetting the same thing at the same time" example.
- **Normalcy bias / diffusion of responsibility** — when a group collectively fails to react to something (like an alarm) because everyone assumes someone else is handling it or that it's not significant.

If your train example involves people *independently* missing the sound (no communication), it's mostly inattentional deafness at the individual level, occurring in parallel. If there's any subtle social cueing (body language, someone glancing away, tone), Asch-style conformity or shared attention effects kick in.

Let me now check current research on the LLM side, since that's a genuinely active and fast-moving field.Good news — this is a real, active research area, exactly what you're describing. Here's what exists:

## Yes, this research exists — and it's called several things

**1. "Hallucination Cascade" / error propagation in multi-agent LLMs**
A 2026 paper literally titled *"Hallucination Cascade: Analyzing Error Propagation in Multi-Agent LLM Systems"* studies exactly your question — when LLM responses are repeatedly exchanged between agents and reused as context, hallucination becomes a dynamic process shaped by interaction history and cascade depth, rather than a static property of one isolated output. Using GPT-5.3, DeepSeek-V3, and LLaMA-3-70B in chains, they found hallucination scores actually dropped across a 3-agent cascade, but this came with a decline in overall factual accuracy — so errors don't just spread unchecked, but there's a real trade-off between "smoothing out" claims and preserving truth.

**2. "Collective Hallucination" in multi-agent LLMs**
A companion paper, *"Collective Hallucination in Multi-Agent LLMs: Modeling and Defense,"* treats this as a network phenomenon — modeling hallucination as a system-level, time-evolving process across a network of interacting agents, where hallucinated claims diffuse through communication topologies and intensify under adversarial conditions. They also built defenses (confidence-weighted aggregation, isolating unreliable agents) that reduced hallucination by up to 39% relative to undefended multi-agent reasoning.

**3. "Group Conformity" in LLM multi-agent systems**
This one maps most directly onto your Asch-conformity framing. *"An Empirical Study of Group Conformity in Multi-Agent Systems"* ran over 2,500 simulated debates and found that LLM agents shift their stance to align with numerically dominant groups or more "intelligent"/persuasive agents — statistically significant group conformity mirroring human behavior.

**4. Related but not shown above (worth knowing):**
- **"Sycophancy"** — LLMs bending their answer toward what they think the user/other agent wants to hear, even when it's wrong. Well-documented (Anthropic has published on this).
- **"Hallucination snowballing"** — an earlier-named phenomenon where a single model, after committing to one wrong claim, generates further wrong claims to stay consistent with it (this is the *single-model* version of your idea).
- **GUARDIAN**-type defense papers, which try to detect and stop error propagation using temporal graph modeling of agent interactions.

## How your two questions connect

Your human-side question and LLM-side question map onto each other more cleanly than you might expect:

| Human phenomenon | LLM analog |
|---|---|
| Asch conformity (deferring to majority) | Group conformity in agent debates |
| Socially shared retrieval-induced forgetting | Hallucination cascade / collective hallucination |
| Groupthink | Consensus/majority-voting failure modes in multi-agent LLM pipelines |
| Inattentional blindness (shared context, missed signal) | Context contamination — an early error becomes "load-bearing" context all downstream agents inherit |

If you want to actually research this yourself, promising angles that seem underexplored (based on what's out there): comparing conformity rates across model *sizes* (do smaller models defer more?), testing whether giving one agent a "confidence score" reduces its influence over others, or replicating Asch's original setup exactly but with LLM agents instead of people.

Want me to pull the actual papers so you can read the methodology sections, or help sketch an experiment design if you want to run your own version of this?

A genuinely unpublished angle: the "LLM Bystander Effect"

Everything I found studies either conformity (agents shifting toward a wrong majority opinion) or hallucination cascading (errors surviving/mutating as they pass through a pipeline). Nobody appears to have tested the closer analog of the bystander effect / diffusion of responsibility (Darley & Latané, 1968) — where a person is less likely to intervene on a problem specifically because they believe someone else is also responsible for catching it.

The idea: Does an LLM agent become less vigilant at catching an error in front of it when it's told (truthfully or not) that other agents have already reviewed or will review the same output — even though nothing about the actual content changed?

Why this is different from existing work:

Group conformity papers test whether an agent changes its answer to match others.
Cascade papers test whether errors survive as content moves agent-to-agent.
This idea tests something upstream of both: whether an agent's effort/scrutiny itself degrades under perceived shared responsibility — independent of what any other agent actually said.

A minimal experimental design:

Give a single LLM agent a document containing a deliberately planted factual error, and ask it to fact-check it.
Condition A (control): "You are the only reviewer of this document."
Condition B: "This document will also be reviewed by 2 other agents before publication."
Condition C: "This document will also be reviewed by 5 other agents before publication."
Measure: detection rate of the planted error, confidence score given, and depth/length of the critique, across conditions and across model families/sizes.
Follow-up variable: does stating the other reviewers' identity (e.g., "reviewed by GPT-5" vs. "reviewed by another instance of you") change the effect — testing an LLM analog of in-group diffusion.

If detection rate drops as perceived group size increases, that's a clean, publishable, and currently-nonexistent result — and it has direct practical relevance, since multi-agent verification/review pipelines (the "let's have 3 agents check this" pattern) are increasingly used in production, and this would be the first evidence of whether that setup silently reduces individual agent vigilance instead of just adding redundancy.