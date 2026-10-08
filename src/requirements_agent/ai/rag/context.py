"""Context construction: numbered evidence blocks within a token budget.

Each retrieved chunk becomes an evidence block with a short handle (`E1`, `E2`, ...) and its
real chunk id. Models cite the short handle (LLMs copy `E7` far more reliably than a UUID); the
grounding step maps handles back to chunk ids and rejects anything that was not provided.

Blocks are admitted greedily in relevance order while they fit the budget (a block that does
not fit is skipped, and smaller lower-ranked blocks may still fit), then emitted in reading
order (document, position) so the model sees coherent text.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass

from requirements_agent.ai.rag.retriever import RetrievedChunk
from requirements_agent.ingestion.tokens import estimate_tokens


@dataclass(frozen=True, slots=True)
class EvidenceBlock:
    label: str
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    document_title: str
    section_path: str
    ordinal: int
    text: str
    score: float

    def render(self) -> str:
        section = self.section_path or "-"
        return (
            f'[{self.label}] chunk={self.chunk_id} | doc="{self.document_title}" | '
            f'section="{section}" | score={self.score:.3f}\n{self.text}'
        )


@dataclass(frozen=True, slots=True)
class EvidenceContext:
    blocks: list[EvidenceBlock]
    text: str
    token_count: int
    skipped_for_budget: int

    @property
    def labels(self) -> dict[str, EvidenceBlock]:
        return {block.label: block for block in self.blocks}

    @property
    def max_score(self) -> float:
        return max((b.score for b in self.blocks), default=0.0)


class ContextBuilder:
    SEPARATOR = "\n\n"

    def __init__(self, max_tokens: int) -> None:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        self.max_tokens = max_tokens

    def build(self, chunks: Sequence[RetrievedChunk]) -> EvidenceContext:
        admitted: list[RetrievedChunk] = []
        used = 0
        skipped = 0
        for chunk in sorted(chunks, key=lambda c: c.rank):
            # Placeholder label of max width so the estimate is an upper bound.
            cost = estimate_tokens(self._render(f"E{len(chunks)}", chunk)) + 1
            if used + cost > self.max_tokens:
                skipped += 1
                continue
            admitted.append(chunk)
            used += cost

        admitted.sort(key=lambda c: (c.document_title, str(c.document_id), c.ordinal))
        blocks = [
            EvidenceBlock(
                label=f"E{i}",
                chunk_id=c.chunk_id,
                document_id=c.document_id,
                document_title=c.document_title,
                section_path=c.section_path,
                ordinal=c.ordinal,
                text=c.text,
                score=c.score,
            )
            for i, c in enumerate(admitted, start=1)
        ]
        text = self.SEPARATOR.join(block.render() for block in blocks)
        return EvidenceContext(
            blocks=blocks,
            text=text,
            token_count=estimate_tokens(text),
            skipped_for_budget=skipped,
        )

    @staticmethod
    def _render(label: str, chunk: RetrievedChunk) -> str:
        return EvidenceBlock(
            label=label,
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            document_title=chunk.document_title,
            section_path=chunk.section_path,
            ordinal=chunk.ordinal,
            text=chunk.text,
            score=chunk.score,
        ).render()
