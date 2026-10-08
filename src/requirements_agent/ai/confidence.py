"""Confidence scoring for generated artefacts.

    confidence = clip01( (w_r * R + w_s * S + w_m * M) * P_repair * G )

* R - retrieval strength: mean cosine similarity of the item's cited evidence blocks, divided
  by the best similarity in the whole evidence set (embedder-agnostic: hashing and OpenAI
  embeddings live on different absolute scales).
* S - lexical support (evidence coverage): share of the item's content terms (stop words
  removed, 5-character prefix match so "approve" ~ "approval") that occur in the cited
  evidence text. Low support means the item says things its citations do not.
* M - model-reported confidence, clipped to [0, 1] (0.5 when absent).
* P_repair - max(0.5, 1 - 0.15 * repairs): output that needed schema repairs is less trusted.
* G - 1.0 when grounded, 0.5 when the item was kept but flagged as uncited.

Default weights w_r = 0.35, w_s = 0.35, w_m = 0.30. The score is a *ranking/triage signal*
for reviewers, not a calibrated probability.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from requirements_agent.ai.embeddings.hashing import tokenize
from requirements_agent.ai.grounding import GroundedItem
from requirements_agent.ai.rag.context import EvidenceContext

_TEXT_FIELDS = (
    "title",
    "description",
    "role",
    "goal",
    "benefit",
    "summary",
    "metric",
    "name",
    "path",
)


@dataclass(frozen=True, slots=True)
class ConfidenceWeights:
    retrieval: float = 0.35
    support: float = 0.35
    model: float = 0.30
    repair_penalty: float = 0.15
    repair_floor: float = 0.5
    ungrounded_factor: float = 0.5


@dataclass(frozen=True, slots=True)
class ConfidenceBreakdown:
    retrieval: float
    support: float
    model: float
    repair_factor: float
    grounding_factor: float
    final: float

    def to_dict(self) -> dict[str, float]:
        return {
            "retrieval": self.retrieval,
            "support": self.support,
            "model": self.model,
            "repairFactor": self.repair_factor,
            "groundingFactor": self.grounding_factor,
            "final": self.final,
        }


def _clip(value: float) -> float:
    return min(1.0, max(0.0, value))


def _stems(terms: Iterable[str]) -> set[str]:
    return {t[:5] for t in terms if not t.isdigit()}


def item_text(payload: dict[str, Any]) -> str:
    parts: list[str] = [str(payload[k]) for k in _TEXT_FIELDS if payload.get(k)]
    for criterion in payload.get("acceptanceCriteria") or []:
        parts.extend(str(v) for v in criterion.values())
    for field in payload.get("fields") or []:
        parts.append(str(field.get("name", "")).replace("_", " "))
    return " ".join(parts)


class ConfidenceScorer:
    def __init__(self, weights: ConfidenceWeights | None = None) -> None:
        self.weights = weights or ConfidenceWeights()

    def score(
        self, item: GroundedItem, context: EvidenceContext, repairs: int
    ) -> ConfidenceBreakdown:
        w = self.weights
        blocks = context.labels
        cited = [blocks[label] for label in item.evidence_labels if label in blocks]

        max_score = context.max_score
        if cited and max_score > 0:
            retrieval = _clip(sum(b.score for b in cited) / len(cited) / max_score)
        else:
            retrieval = 0.0

        item_terms = _stems(tokenize(item_text(item.payload)))
        evidence_terms = _stems(tokenize(" ".join(b.text for b in cited)))
        support = len(item_terms & evidence_terms) / len(item_terms) if item_terms else 0.0

        model = _clip(item.model_confidence)
        repair_factor = max(w.repair_floor, 1.0 - w.repair_penalty * max(0, repairs))
        grounding_factor = 1.0 if item.grounded else w.ungrounded_factor
        raw = w.retrieval * retrieval + w.support * support + w.model * model
        final = _clip(raw * repair_factor * grounding_factor)
        return ConfidenceBreakdown(
            retrieval=round(retrieval, 4),
            support=round(support, 4),
            model=round(model, 4),
            repair_factor=round(repair_factor, 4),
            grounding_factor=grounding_factor,
            final=round(final, 3),
        )
