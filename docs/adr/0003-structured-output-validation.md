# ADR 0003: Validate-and-repair structured output on our side

- Status: accepted
- Date: 2026-10-08

## Context

Analyses must produce machine-usable artefacts (they feed an OpenAPI builder and test-case
generation). Provider-native structured-output modes differ between vendors and impose schema
restrictions (e.g. every property required, no defaults), and the offline provider has no
such mode at all. Model output can still be syntactically valid but semantically wrong
(unknown ids, references to entities that do not exist).

## Decision

1. Generate a JSON Schema from Pydantic models (`ai/schemas.py`) and embed it in the prompt;
   use plain JSON mode where available.
2. Parse defensively (code fences, leading prose) and validate with Pydantic using
   `extra="forbid"`, enums and id/name patterns.
3. On failure, append the model's answer plus the exact validation errors and ask for a
   corrected object; at most `STRUCTURED_MAX_REPAIRS` (default 2) repairs, then fail the
   analysis with the errors.
4. After validation, run a separate grounding/reference-integrity pass (ADR-independent
   rules in `ai/grounding.py`) and score confidence, penalising outputs that needed repairs.

## Consequences

- Identical guarantees for every provider; switching vendors does not change what is stored.
- Repairs cost extra calls; they are bounded, logged per attempt in `ai_usage`, and visible
  as `repairAttempts` on the analysis.
- Transport retries (timeouts, 429, 5xx) are kept separate from repairs so a flaky network
  does not consume the repair budget.
- Semantic errors that pass the schema are handled by grounding (drop/flag/strip) rather than
  by more retries, which keeps latency predictable.
