"""Grounding check: every artefact must be traceable to evidence that was actually provided.

Rules (applied after schema validation, before persistence):

1. Citations are resolved against the evidence handed to the model. `E3`, `[E3]`, `e3` and the
   full chunk UUID are accepted; anything else is an *unknown citation*: it is stripped and
   reported (a model citing evidence it never saw is a hallucination signal).
2. Requirements and user stories (FR/NFR/US) with no valid citation are **dropped**.
   Derived design artefacts (entities, endpoints, tasks) may legitimately synthesise several
   requirements, so they are **kept but flagged** `grounded=false` (and their confidence is
   halved by the scorer).
3. Duplicate ids (two FR-001, two `GET /invoices`) keep the first occurrence only.
4. Cross-references must point at items that exist in the same analysis: story ->
   requirement ids, endpoint -> entity models / story ids, entity relationship targets, task
   dependencies. Unknown references are removed and reported.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import Any

from requirements_agent.ai.rag.context import EvidenceContext
from requirements_agent.ai.schemas import AnalysisOutput
from requirements_agent.domain.enums import ItemType, Section

REQUIRES_EVIDENCE = frozenset(
    {ItemType.FUNCTIONAL_REQUIREMENT, ItemType.NON_FUNCTIONAL_REQUIREMENT, ItemType.USER_STORY}
)

_SECTION_FIELDS: list[tuple[Section, str, ItemType]] = [
    (Section.FUNCTIONAL_REQUIREMENTS, "functional_requirements", ItemType.FUNCTIONAL_REQUIREMENT),
    (
        Section.NON_FUNCTIONAL_REQUIREMENTS,
        "non_functional_requirements",
        ItemType.NON_FUNCTIONAL_REQUIREMENT,
    ),
    (Section.USER_STORIES, "user_stories", ItemType.USER_STORY),
    (Section.ENTITIES, "entities", ItemType.ENTITY),
    (Section.API_ENDPOINTS, "api_endpoints", ItemType.API_ENDPOINT),
    (Section.IMPLEMENTATION_TASKS, "implementation_tasks", ItemType.IMPLEMENTATION_TASK),
]
_HANDLE = re.compile(r"^\[?\s*[Ee](\d{1,4})\s*\]?$")


@dataclass(slots=True)
class GroundedItem:
    item_type: ItemType
    ref_code: str
    title: str
    payload: dict[str, Any]
    evidence_chunk_ids: list[str]
    evidence_labels: list[str]
    model_confidence: float
    grounded: bool = True
    issues: list[str] = field(default_factory=list)


@dataclass(slots=True)
class GroundingReport:
    kept: int = 0
    dropped: list[dict[str, str]] = field(default_factory=list)
    flagged: list[dict[str, str]] = field(default_factory=list)
    unknown_citations: list[dict[str, str]] = field(default_factory=list)
    removed_references: list[dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kept": self.kept,
            "dropped": self.dropped,
            "flagged": self.flagged,
            "unknownCitations": self.unknown_citations,
            "removedReferences": self.removed_references,
        }


class GroundingChecker:
    def check(
        self, output: AnalysisOutput, context: EvidenceContext, sections: set[Section]
    ) -> tuple[list[GroundedItem], GroundingReport]:
        report = GroundingReport()
        handles = {block.label.upper(): str(block.chunk_id) for block in context.blocks}
        chunk_ids = {str(block.chunk_id): block.label for block in context.blocks}
        items: list[GroundedItem] = []
        seen: set[tuple[ItemType, str]] = set()

        for section, attribute, item_type in _SECTION_FIELDS:
            if section not in sections:
                continue
            for model in getattr(output, attribute):
                payload = model.model_dump(by_alias=True, mode="json")
                ref = _ref_code(item_type, payload)
                if (item_type, ref) in seen:
                    report.dropped.append(_entry(item_type, ref, "duplicate id"))
                    continue
                seen.add((item_type, ref))

                resolved, labels = [], []
                for citation in payload.get("evidence", []):
                    chunk_id = _resolve(citation, handles, chunk_ids)
                    if chunk_id is None:
                        report.unknown_citations.append(
                            {"itemType": item_type.value, "ref": ref, "citation": str(citation)}
                        )
                    elif chunk_id not in resolved:
                        resolved.append(chunk_id)
                        labels.append(chunk_ids[chunk_id])

                item = GroundedItem(
                    item_type=item_type,
                    ref_code=ref,
                    title=_title(item_type, payload),
                    payload=payload,
                    evidence_chunk_ids=resolved,
                    evidence_labels=labels,
                    model_confidence=float(payload.get("confidence", 0.5)),
                )
                if not resolved:
                    if item_type in REQUIRES_EVIDENCE:
                        report.dropped.append(_entry(item_type, ref, "no valid evidence citation"))
                        continue
                    item.grounded = False
                    item.issues.append("no valid evidence citation")
                    report.flagged.append(_entry(item_type, ref, "no valid evidence citation"))
                payload["evidence"] = resolved
                items.append(item)

        self._check_references(items, report)
        report.kept = len(items)
        return items, report

    # ------------------------------------------------------------------ references
    def _check_references(self, items: list[GroundedItem], report: GroundingReport) -> None:
        by_type: dict[ItemType, set[str]] = {t: set() for t in ItemType}
        for item in items:
            by_type[item.item_type].add(item.ref_code)
        entities = by_type[ItemType.ENTITY]
        known_any = set().union(*by_type.values())

        def keep(item: GroundedItem, key: str, allowed: set[str]) -> None:
            values = item.payload.get(key) or []
            kept = [v for v in values if v in allowed]
            for value in values:
                if value not in allowed:
                    report.removed_references.append(
                        {
                            "itemType": item.item_type.value,
                            "ref": item.ref_code,
                            "field": key,
                            "value": str(value),
                        }
                    )
            item.payload[key] = kept

        for item in items:
            if item.item_type is ItemType.USER_STORY:
                keep(
                    item,
                    "relatedRequirements",
                    by_type[ItemType.FUNCTIONAL_REQUIREMENT]
                    | by_type[ItemType.NON_FUNCTIONAL_REQUIREMENT],
                )
            elif item.item_type is ItemType.API_ENDPOINT:
                keep(item, "relatedStories", by_type[ItemType.USER_STORY])
                for key in ("requestModel", "responseModel"):
                    model = item.payload.get(key)
                    if model and model not in entities:
                        item.payload[key] = None
                        item.issues.append(f"{key} '{model}' is not a defined entity")
                        report.removed_references.append(
                            {
                                "itemType": item.item_type.value,
                                "ref": item.ref_code,
                                "field": key,
                                "value": model,
                            }
                        )
            elif item.item_type is ItemType.ENTITY:
                relationships = item.payload.get("relationships") or []
                valid = [r for r in relationships if r.get("target") in entities]
                for rel in relationships:
                    if rel not in valid:
                        report.removed_references.append(
                            {
                                "itemType": item.item_type.value,
                                "ref": item.ref_code,
                                "field": "relationships",
                                "value": str(rel.get("target")),
                            }
                        )
                item.payload["relationships"] = valid
            elif item.item_type is ItemType.IMPLEMENTATION_TASK:
                keep(item, "dependsOn", by_type[ItemType.IMPLEMENTATION_TASK] - {item.ref_code})
                keep(item, "relatedItems", known_any)


def _resolve(citation: Any, handles: dict[str, str], chunk_ids: dict[str, str]) -> str | None:
    text = str(citation).strip()
    match = _HANDLE.match(text)
    if match:
        return handles.get(f"E{int(match.group(1))}")
    try:
        normalised = str(uuid.UUID(text))
    except ValueError:
        return None
    return normalised if normalised in chunk_ids else None


def _ref_code(item_type: ItemType, payload: dict[str, Any]) -> str:
    if item_type is ItemType.ENTITY:
        return str(payload["name"])
    if item_type is ItemType.API_ENDPOINT:
        return f"{payload['method']} {payload['path']}"
    return str(payload["id"])


def _title(item_type: ItemType, payload: dict[str, Any]) -> str:
    if item_type is ItemType.NON_FUNCTIONAL_REQUIREMENT:
        text = str(payload["description"])
    elif item_type is ItemType.ENTITY:
        text = str(payload["name"])
    elif item_type is ItemType.API_ENDPOINT:
        text = str(payload["summary"])
    else:
        text = str(payload["title"])
    return text if len(text) <= 500 else text[:497] + "..."


def _entry(item_type: ItemType, ref: str, reason: str) -> dict[str, str]:
    return {"itemType": item_type.value, "ref": ref, "reason": reason}
