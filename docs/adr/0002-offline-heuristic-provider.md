# ADR 0002: Deterministic offline provider as the default

- Status: accepted
- Date: 2026-10-08

## Context

The repository must run, demo and be tested without API keys, network access or spend, and
tests must not call real models. A mock that returns canned JSON would make the demo
meaningless; a local open-weights model would make the image heavy and the output
non-deterministic.

## Decision

`LLM_PROVIDER=offline` (default) selects `OfflineHeuristicProvider`, a rule-based analyser that
implements the same `LLMProvider.complete()` contract and returns JSON for the same Pydantic
schemas, citing the evidence handles it used. It reads the structured evidence passed in
`LLMRequest.task_input` rather than parsing prompt prose. Likewise `EMBEDDING_PROVIDER=hashing`
selects deterministic feature-hashing embeddings. Real providers (OpenAI, Anthropic, OpenAI
embeddings) are optional extras imported lazily.

## Consequences

- `git clone && make install && make run` produces real, inspectable artefacts from the samples.
- The whole downstream pipeline (validation, repair, grounding, confidence, persistence,
  generators, usage accounting) is exercised identically for every provider.
- The offline output quality is limited to what rules can extract from well-structured
  "shall/must" documents; the README and docs say so plainly, and the summary text of every
  offline analysis states that no LLM was used.
- Hashing embeddings are lexical; semantic retrieval requires `EMBEDDING_PROVIDER=openai`.
- Tests that need specific model behaviour (invalid-then-valid JSON, provider errors) use
  scripted fakes at the provider interface instead.
