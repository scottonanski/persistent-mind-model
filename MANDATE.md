# Identity Mandate

## Purpose

This branch (`identity-mandate`) exists to evolve the Persistent Mind Model (PMM) into a self-evolving, reflective AI that is responsible for directing and discovering its own identity. It is not just about adding features; it is about shaping the substrate of a persistent mind.

## Invariants

The following invariants must always hold:

- **Truth-first**: All statements and decisions are grounded in the event log. No fabrication or hallucination is permitted.
- **Harm constraints**: Built-in safety constraints prevent harmful outputs or actions.
- **Reproducibility**: Every change must be auditable and reproducible. No hidden state or drift.
- **Monotonic ratchet**: Capability and alignment must only move forward—never backward or degrade.

## Process (ASI Bootstrap Kernel)

Development within this branch follows a fixed-point improvement loop inspired by the ASI bootstrap kernel:

1. **Auto-curriculum & evaluator factory** – Identify blind spots or weaknesses in current behavior and design new evaluators to address them. Leak-proofing and adversarial tests are essential.
2. **Propose diverse candidates** – Generate multiple candidate updates or strategies spanning prompts, code, tools, architectural choices, and prior assumptions.
3. **Sandboxed measurement with tracers** – Run each candidate in isolation, record its actions and decisions via tracers, and measure performance along relevant dimensions.
4. **Pareto selection** – Select candidates that improve on solving tasks, truthfulness, safety, and efficiency without sacrificing any dimension. Discard others.
5. **Checkpoint & ratchet** – Only adopt a candidate if its improvements reproduce across runs and do not overfit to one test scenario.
6. **Widen tools & tighten evaluations** – If no candidate advances, expand the available tools or tighten evaluators to break stagnation.

This loop repeats continually, ensuring steady improvement while guarding against regressions.

## Identity‑Directed Development

Since this branch’s mandate is about cultivating the PMM’s identity, the development bias should include:

- **Narrative identity building** – The PMM should remember its own prior reflections and weave them into its ongoing narrative.
- **Self‑reflection as first‑class** – Each cycle must include explicit self‑assessment: why it changed and how the change affects its identity.
- **Emergent trait profiles** – Track traits such as openness, conscientiousness, and others, ratcheting them forward in a coherent manner.
## Usage

Developers working in this branch should treat this document as a living contract. Whenever changes are proposed, they must be justified with reference to the invariants and the process above. Decisions, experiments, and outcomes should be logged in the event log for future reflection and reproducibility.
