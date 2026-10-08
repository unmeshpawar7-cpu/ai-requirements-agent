from __future__ import annotations

import json
import uuid
from typing import Any

import pytest

from requirements_agent.ai.confidence import ConfidenceScorer, ConfidenceWeights, item_text
from requirements_agent.ai.grounding import GroundedItem, GroundingChecker
from requirements_agent.ai.prompts.templates import RenderedPrompt
from requirements_agent.ai.providers.base import LLMRequest, LLMResponse, ProviderError
from requirements_agent.ai.rag.context import EvidenceBlock, EvidenceContext
from requirements_agent.ai.schemas import AnalysisOutput, clip_unit
from requirements_agent.ai.structured import (
    StructuredGenerator,
    StructuredOutputError,
    extract_json,
)
from requirements_agent.ai.usage import InMemoryUsageRecorder
from requirements_agent.domain.enums import ItemType, Section

PROMPT = RenderedPrompt(name="requirements_analysis", version=3, system="sys", user="user")


class ScriptedProvider:
    """Mock LLM: returns the scripted responses in order and records every request."""

    name = "scripted"
    model = "scripted-1"

    def __init__(self, responses: list[str | Exception]) -> None:
        self.responses = list(responses)
        self.requests: list[LLMRequest] = []

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return LLMResponse(text=item, model=self.model, input_tokens=100, output_tokens=50)


VALID = json.dumps(
    {
        "summary": "Invoice approval portal.",
        "functionalRequirements": [
            {
                "id": "FR-001",
                "title": "Approve invoices",
                "description": "Approvers approve.",
                "priority": "must",
                "evidence": ["E1"],
                "confidence": 0.9,
            }
        ],
    }
)
INVALID = json.dumps(
    {"summary": "x", "functionalRequirements": [{"id": "REQ-1", "title": "A", "unknown": 1}]}
)


# -------------------------------------------------------------------- structured output
async def test_invalid_then_valid_output_is_repaired_with_feedback() -> None:
    provider = ScriptedProvider([INVALID, VALID])
    recorder = InMemoryUsageRecorder()
    result = await StructuredGenerator(provider, recorder, max_repairs=2).generate(
        PROMPT, AnalysisOutput
    )
    assert result.attempts == 2
    assert result.repairs == 1
    assert result.value.functional_requirements[0].id == "FR-001"
    assert (result.input_tokens, result.output_tokens) == (200, 100)

    repair_request = provider.requests[1]
    assert [m.role for m in repair_request.messages] == ["user", "assistant", "user"]
    feedback = repair_request.messages[-1].content
    assert "functionalRequirements.0.id" in feedback
    assert "functionalRequirements.0.unknown" in feedback  # extra fields are rejected
    assert repair_request.response_schema["title"] == "AnalysisOutput"

    assert [e.success for e in recorder.events] == [False, True]
    assert recorder.events[0].error_type == "SchemaValidationError"
    assert {(e.prompt_name, e.prompt_version) for e in recorder.events} == {
        ("requirements_analysis", 3)
    }


async def test_gives_up_after_bounded_repairs() -> None:
    provider = ScriptedProvider(["not json at all", INVALID, INVALID])
    recorder = InMemoryUsageRecorder()
    with pytest.raises(StructuredOutputError) as info:
        await StructuredGenerator(provider, recorder, max_repairs=2).generate(
            PROMPT, AnalysisOutput
        )
    assert info.value.attempts == 3
    assert info.value.errors
    assert len(provider.requests) == 3
    assert recorder.events[0].error_type == "InvalidJSON"


async def test_provider_errors_are_recorded_and_propagated() -> None:
    provider = ScriptedProvider([ProviderError("boom")])
    recorder = InMemoryUsageRecorder()
    with pytest.raises(ProviderError):
        await StructuredGenerator(provider, recorder).generate(PROMPT, AnalysisOutput)
    assert recorder.events[0].success is False
    assert recorder.events[0].error_type == "ProviderError"


def test_extract_json_tolerates_fences_and_prose() -> None:
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Here you go: {"a": {"b": 2}} thanks') == {"a": {"b": 2}}
    with pytest.raises(ValueError, match="no JSON"):
        extract_json("nothing here")


def test_confidence_is_clipped() -> None:
    assert clip_unit(1.7) == 1.0
    assert clip_unit(-2) == 0.0
    assert clip_unit("high") == 0.5
    assert clip_unit(float("nan")) == 0.5


# ---------------------------------------------------------------------------- grounding
def make_context(texts: list[str], scores: list[float] | None = None) -> EvidenceContext:
    scores = scores or [0.5] * len(texts)
    blocks = [
        EvidenceBlock(
            label=f"E{i}",
            chunk_id=uuid.uuid4(),
            document_id=uuid.uuid4(),
            document_title="Doc",
            section_path="S",
            ordinal=i,
            text=text,
            score=score,
        )
        for i, (text, score) in enumerate(zip(texts, scores, strict=True), start=1)
    ]
    return EvidenceContext(blocks=blocks, text="", token_count=0, skipped_for_budget=0)


def analysis(**sections: Any) -> AnalysisOutput:
    return AnalysisOutput.model_validate({"summary": "Summary", **sections})


def fr(id_: str, evidence: list[str]) -> dict[str, Any]:
    return {
        "id": id_,
        "title": f"Requirement {id_}",
        "description": "Approvers approve invoices.",
        "evidence": evidence,
    }


def test_grounding_drops_uncited_requirements_and_strips_unknown_citations() -> None:
    context = make_context(["Approvers approve invoices.", "Suppliers submit invoices."])
    chunk_one = str(context.blocks[1].chunk_id)
    output = analysis(
        functionalRequirements=[
            fr("FR-001", ["E1", "E99"]),  # one valid, one hallucinated handle
            fr("FR-002", []),  # uncited -> dropped
            fr("FR-003", ["made-up-id"]),  # only unknown -> dropped
            fr("FR-004", [chunk_one, "[e2]"]),  # full chunk id and loose handle both resolve
            fr("FR-001", ["E1"]),  # duplicate id -> dropped
        ]
    )
    items, report = GroundingChecker().check(output, context, {Section.FUNCTIONAL_REQUIREMENTS})
    assert [i.ref_code for i in items] == ["FR-001", "FR-004"]
    assert items[0].evidence_chunk_ids == [str(context.blocks[0].chunk_id)]
    assert items[0].payload["evidence"] == items[0].evidence_chunk_ids
    assert items[1].evidence_labels == ["E2"]
    reasons = {(d["ref"], d["reason"]) for d in report.dropped}
    assert reasons == {
        ("FR-002", "no valid evidence citation"),
        ("FR-003", "no valid evidence citation"),
        ("FR-001", "duplicate id"),
    }
    assert {c["citation"] for c in report.unknown_citations} == {"E99", "made-up-id"}
    assert report.to_dict()["kept"] == 2


def test_grounding_flags_derived_items_and_checks_cross_references() -> None:
    context = make_context(["Each invoice has an amount and a status."])
    output = analysis(
        functionalRequirements=[fr("FR-001", ["E1"])],
        userStories=[
            {
                "id": "US-001",
                "title": "Approve",
                "role": "approver",
                "goal": "approve invoices",
                "acceptanceCriteria": [{"given": "g1", "when": "w1", "then": "t1"}],
                "relatedRequirements": ["FR-001", "FR-404"],
                "evidence": ["E1"],
            }
        ],
        entities=[
            {
                "name": "Invoice",
                "description": "An invoice",
                "evidence": [],
                "fields": [{"name": "amount", "type": "decimal"}],
                "relationships": [{"target": "Ghost", "kind": "many-to-one"}],
            }
        ],
        apiEndpoints=[
            {
                "method": "GET",
                "path": "/invoices",
                "summary": "List invoices",
                "responseModel": "Invoice",
                "requestModel": "Unknown",
                "evidence": ["E1"],
                "relatedStories": ["US-001", "US-999"],
            }
        ],
        implementationTasks=[
            {
                "id": "T-001",
                "title": "Build it",
                "description": "Build the thing",
                "dependsOn": ["T-001", "T-002"],
                "relatedItems": ["Invoice", "FR-001", "X"],
                "evidence": ["E1"],
            }
        ],
    )
    items, report = GroundingChecker().check(output, context, set(Section))
    by_ref = {i.ref_code: i for i in items}
    assert by_ref["Invoice"].grounded is False  # derived artefact kept but flagged
    assert by_ref["Invoice"].payload["relationships"] == []
    assert by_ref["US-001"].payload["relatedRequirements"] == ["FR-001"]
    endpoint = by_ref["GET /invoices"]
    assert endpoint.payload["requestModel"] is None
    assert endpoint.payload["responseModel"] == "Invoice"
    assert endpoint.payload["relatedStories"] == ["US-001"]
    assert by_ref["T-001"].payload["dependsOn"] == []
    assert by_ref["T-001"].payload["relatedItems"] == ["Invoice", "FR-001"]
    assert report.flagged == [
        {"itemType": "entity", "ref": "Invoice", "reason": "no valid evidence citation"}
    ]
    removed = {(r["field"], r["value"]) for r in report.removed_references}
    assert {
        ("relatedRequirements", "FR-404"),
        ("requestModel", "Unknown"),
        ("relationships", "Ghost"),
        ("dependsOn", "T-002"),
    } <= removed


def test_grounding_ignores_unrequested_sections() -> None:
    context = make_context(["text"])
    output = analysis(functionalRequirements=[fr("FR-001", ["E1"])])
    items, _ = GroundingChecker().check(output, context, {Section.USER_STORIES})
    assert items == []


# --------------------------------------------------------------------------- confidence
def grounded_item(
    labels: list[str],
    *,
    model: float = 0.8,
    grounded: bool = True,
    description: str = "Approvers approve invoices quickly",
) -> GroundedItem:
    return GroundedItem(
        item_type=ItemType.FUNCTIONAL_REQUIREMENT,
        ref_code="FR-001",
        title="t",
        payload={"title": "Approve invoices", "description": description},
        evidence_chunk_ids=[],
        evidence_labels=labels,
        model_confidence=model,
        grounded=grounded,
    )


def test_confidence_formula_components() -> None:
    context = make_context(
        ["Approvers must approve invoices quickly.", "Unrelated text about trucks."], [0.8, 0.4]
    )
    scorer = ConfidenceScorer()
    strong = scorer.score(grounded_item(["E1"]), context, repairs=0)
    assert strong.retrieval == pytest.approx(1.0)
    assert strong.support == pytest.approx(1.0)
    assert strong.final == pytest.approx(0.35 + 0.35 + 0.3 * 0.8, abs=1e-3)

    weak = scorer.score(grounded_item(["E2"]), context, repairs=0)
    assert weak.retrieval == pytest.approx(0.5)
    assert weak.support < 0.5
    assert weak.final < strong.final

    repaired = scorer.score(grounded_item(["E1"]), context, repairs=2)
    assert repaired.repair_factor == pytest.approx(0.7)
    assert repaired.final == pytest.approx(strong.final * 0.7, abs=1e-3)

    flagged = scorer.score(grounded_item([], grounded=False), context, repairs=0)
    assert flagged.retrieval == 0.0
    assert flagged.grounding_factor == 0.5
    assert 0.0 <= flagged.final <= 0.5

    floor = ConfidenceScorer(ConfidenceWeights()).score(grounded_item(["E1"]), context, repairs=10)
    assert floor.repair_factor == 0.5
    assert set(strong.to_dict()) == {
        "retrieval",
        "support",
        "model",
        "repairFactor",
        "groundingFactor",
        "final",
    }


def test_item_text_includes_criteria_and_fields() -> None:
    payload = {
        "title": "Story",
        "acceptanceCriteria": [{"given": "an invoice", "when": "approved", "then": "exported"}],
        "fields": [{"name": "due_date"}],
    }
    text = item_text(payload)
    assert "exported" in text
    assert "due date" in text
