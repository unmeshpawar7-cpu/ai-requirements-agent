from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from requirements_agent.ai.embeddings.hashing import HashingEmbeddingProvider
from requirements_agent.ai.embeddings.openai_embeddings import OpenAIEmbeddingProvider
from requirements_agent.ai.factory import build_embedding_provider, build_llm_provider
from requirements_agent.ai.offline.analyser import HeuristicRequirementsAnalyser
from requirements_agent.ai.providers.anthropic_provider import AnthropicProvider
from requirements_agent.ai.providers.base import (
    ChatMessage,
    LLMRequest,
    ProviderError,
    ProviderUnavailableError,
    is_transient_error,
)
from requirements_agent.ai.providers.offline_provider import OfflineHeuristicProvider
from requirements_agent.ai.providers.openai_provider import OpenAIProvider
from requirements_agent.ai.schemas import AnalysisOutput, TestCaseGenerationOutput
from requirements_agent.ai.usage import (
    DatabaseUsageRecorder,
    InMemoryUsageRecorder,
    PriceTable,
    TrackedEmbeddingProvider,
    UsageEvent,
    ai_call_context,
    current_ai_context,
)
from requirements_agent.domain.enums import AIOperation
from requirements_agent.ingestion.chunker import Chunker
from requirements_agent.ingestion.normalise import normalise_text
from requirements_agent.services.jobs import AsyncioJobRunner
from tests.conftest import SAMPLES, make_settings

ALL_SECTIONS = [
    "functional_requirements",
    "non_functional_requirements",
    "user_stories",
    "entities",
    "api_endpoints",
    "implementation_tasks",
]


def request(task: str = "requirements_analysis", task_input: dict | None = None) -> LLMRequest:
    return LLMRequest(
        task=task,
        system="system",
        messages=[ChatMessage("user", "hello")],
        response_schema={},
        schema_name="X",
        task_input=task_input or {},
    )


class TransientError(Exception):
    status_code = 503


# ------------------------------------------------------------------- remote providers
class FakeOpenAI:
    def __init__(self, failures: int = 0) -> None:
        self.failures = failures
        self.calls: list[dict[str, Any]] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))
        self.embeddings = SimpleNamespace(create=self._embed)

    async def _create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.failures:
            self.failures -= 1
            raise TransientError("unavailable")
        return SimpleNamespace(
            model="gpt-test",
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content='{"ok": true}'), finish_reason="stop"
                )
            ],
            usage=SimpleNamespace(prompt_tokens=11, completion_tokens=7),
        )

    async def _embed(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        data = [SimpleNamespace(index=i, embedding=[3.0, 4.0]) for i in range(len(kwargs["input"]))]
        return SimpleNamespace(data=list(reversed(data)))


async def test_openai_provider_maps_request_and_retries_transient_errors() -> None:
    client = FakeOpenAI(failures=1)
    provider = OpenAIProvider(api_key=None, model="gpt-test", client=client, max_attempts=3)
    response = await provider.complete(request())
    assert response.text == '{"ok": true}'
    assert (response.input_tokens, response.output_tokens) == (11, 7)
    assert len(client.calls) == 2
    sent = client.calls[-1]
    assert sent["messages"][0] == {"role": "system", "content": "system"}
    assert sent["response_format"] == {"type": "json_object"}


async def test_openai_provider_does_not_retry_client_errors() -> None:
    class BadRequestError(Exception):
        status_code = 400

    async def fail(**_: Any) -> Any:
        raise BadRequestError("bad")

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=fail)))
    provider = OpenAIProvider(api_key=None, client=client, max_attempts=3)
    with pytest.raises(BadRequestError):
        await provider.complete(request())


async def test_openai_embeddings_reorder_and_normalise() -> None:
    client = FakeOpenAI()
    provider = OpenAIEmbeddingProvider(api_key=None, dimension=2, client=client, batch_size=2)
    vectors = await provider.embed(["a", "b", "c"])
    assert vectors == [[0.6, 0.8]] * 3
    assert client.calls[0]["dimensions"] == 2
    assert len(client.calls) == 2  # batched


async def test_anthropic_provider_extracts_text_blocks() -> None:
    calls: list[dict[str, Any]] = []

    async def create(**kwargs: Any) -> Any:
        calls.append(kwargs)
        return SimpleNamespace(
            model="claude-test",
            stop_reason="end_turn",
            content=[
                SimpleNamespace(type="text", text='{"a": '),
                SimpleNamespace(type="text", text="1}"),
            ],
            usage=SimpleNamespace(input_tokens=20, output_tokens=5),
        )

    provider = AnthropicProvider(
        api_key=None, client=SimpleNamespace(messages=SimpleNamespace(create=create))
    )
    response = await provider.complete(request())
    assert response.text == '{"a": 1}'
    assert calls[0]["system"] == "system"
    assert calls[0]["messages"] == [{"role": "user", "content": "hello"}]

    async def empty(**_: Any) -> Any:
        return SimpleNamespace(content=[], usage=None)

    empty_provider = AnthropicProvider(
        api_key=None, client=SimpleNamespace(messages=SimpleNamespace(create=empty))
    )
    with pytest.raises(ProviderError):
        await empty_provider.complete(request())


def test_providers_require_keys_and_factory_defaults(tmp_path: Path) -> None:
    with pytest.raises(ProviderUnavailableError):
        OpenAIProvider(api_key=None)
    with pytest.raises(ProviderUnavailableError):
        AnthropicProvider(api_key=None)
    with pytest.raises(ProviderUnavailableError):
        OpenAIEmbeddingProvider(api_key=None)
    settings = make_settings(tmp_path)
    assert isinstance(build_llm_provider(settings), OfflineHeuristicProvider)
    assert isinstance(build_embedding_provider(settings), HashingEmbeddingProvider)
    with pytest.raises(ProviderUnavailableError):
        build_llm_provider(make_settings(tmp_path, llm_provider="anthropic"))


def test_transient_error_classification() -> None:
    assert is_transient_error(TransientError())
    assert is_transient_error(type("RateLimitError", (Exception,), {})())
    assert not is_transient_error(ValueError("x"))


# -------------------------------------------------------------------- offline provider
def evidence_for(sample: str) -> list[dict[str, Any]]:
    text = normalise_text((SAMPLES / sample).read_text())
    return [
        {"label": f"E{i}", "text": c.text, "section_path": c.section_path, "document_title": sample}
        for i, c in enumerate(Chunker(350, 50).split(text), start=1)
    ]


@pytest.mark.parametrize("sample", ["field-service-work-orders.md", "invoice-approval-portal.txt"])
async def test_offline_provider_emits_schema_valid_grounded_output(sample: str) -> None:
    evidence = evidence_for(sample)
    provider = OfflineHeuristicProvider()
    response = await provider.complete(
        request(task_input={"evidence": evidence, "sections": ALL_SECTIONS, "focus": "approvals"})
    )
    output = AnalysisOutput.model_validate_json(response.text)
    labels = {e["label"] for e in evidence}
    cited_items = [
        *output.functional_requirements,
        *output.non_functional_requirements,
        *output.user_stories,
        *output.entities,
        *output.api_endpoints,
    ]
    assert output.functional_requirements and output.user_stories and output.entities
    assert all(item.evidence and set(item.evidence) <= labels for item in cited_items)
    assert all(0 <= item.confidence <= 1 for item in cited_items)
    assert "Offline heuristic analysis" in output.summary
    assert response.tokens_estimated
    # Deterministic: identical input -> identical output.
    again = await provider.complete(
        request(task_input={"evidence": evidence, "sections": ALL_SECTIONS, "focus": "approvals"})
    )
    assert again.text == response.text


def test_offline_analyser_rules_on_small_input() -> None:
    evidence = [
        {
            "label": "E1",
            "section_path": "SRS > 2 Functional Requirements",
            "text": (
                "- The system shall allow approvers to approve invoices.\n"
                "- Each invoice has an invoice number, supplier, total amount, due date and "
                "status (draft, approved, paid).\n"
                "- Suppliers can view their invoices.\n"
                "- The portal must respond within 2 seconds.\n"
                "- As a clerk, I want to export invoices so that payments run daily.\n"
                "- Given an approved invoice, when the clerk exports it, "
                "then it is marked exported."
            ),
        },
        {"label": "E2", "section_path": "SRS > 5 Open Issues", "text": "Is SSO required?"},
    ]
    output = HeuristicRequirementsAnalyser().analyse(evidence, ALL_SECTIONS)
    assert [fr.priority for fr in output.functional_requirements][:1] == ["must"]
    nfr = output.non_functional_requirements[0]
    assert (nfr.category, nfr.metric) == ("performance", "within 2 seconds")
    invoice = next(e for e in output.entities if e.name == "Invoice")
    fields = {f.name: f for f in invoice.fields}
    assert fields["total_amount"].type == "decimal"
    assert fields["due_date"].type == "date"
    assert fields["status"].enum_values == ["draft", "approved", "paid"]
    explicit = next(s for s in output.user_stories if s.role == "clerk")
    assert explicit.benefit == "payments run daily"
    assert explicit.acceptance_criteria[0].then == "it is marked exported"
    derived = next(s for s in output.user_stories if s.role == "approver")
    assert derived.related_requirements
    routes = {(e.method, e.path) for e in output.api_endpoints}
    assert ("POST", "/invoices/{id}/approve") in routes
    assert ("GET", "/invoices") in routes
    assert any("SSO" in q for q in output.open_questions)


async def test_offline_provider_generates_test_cases_and_rejects_unknown_tasks() -> None:
    stories = [
        {
            "id": "US-001",
            "title": "Approve invoices",
            "role": "approver",
            "goal": "approve invoices",
            "evidence": ["c1"],
            "confidence": 0.8,
            "acceptanceCriteria": [
                {
                    "given": "an invoice over 10,000 EUR",
                    "when": "the approver approves it",
                    "then": "a second approval is required",
                },
                {
                    "given": "an invoice",
                    "when": "the approver rejects it",
                    "then": "it is rejected",
                },
            ],
        },
    ]
    provider = OfflineHeuristicProvider()
    response = await provider.complete(request("test_case_generation", {"stories": stories}))
    cases = TestCaseGenerationOutput.model_validate_json(response.text).test_cases
    assert [c.type for c in cases] == ["positive", "negative", "edge", "positive"]
    assert cases[1].when == ["the user approves it"]
    assert all(c.story_id == "US-001" for c in cases)
    with pytest.raises(ProviderError):
        await provider.complete(request("summarise"))


# ------------------------------------------------------------------------- usage + jobs
def test_price_table_estimates() -> None:
    prices = PriceTable({"openai:m": {"input_per_1k": 1.0, "output_per_1k": 2.0}})
    assert prices.estimate("openai", "m", 500, 1000) == pytest.approx(2.5)
    assert prices.estimate("openai", "unknown", 500, 1000) == 0.0


async def test_tracked_embedder_records_success_and_failure() -> None:
    recorder = InMemoryUsageRecorder()
    tracked = TrackedEmbeddingProvider(HashingEmbeddingProvider(32), recorder)
    project_id = uuid.uuid4()
    with ai_call_context(project_id=project_id):
        assert current_ai_context().project_id == project_id
        await tracked.embed(["approve invoices quickly"])
    assert current_ai_context().project_id is None
    assert recorder.events[0].operation is AIOperation.EMBEDDING
    assert recorder.events[0].input_tokens > 0

    class Broken(HashingEmbeddingProvider):
        async def embed(self, texts):  # type: ignore[no-untyped-def]
            raise RuntimeError("down")

    failing = TrackedEmbeddingProvider(Broken(32), recorder)
    with pytest.raises(RuntimeError):
        await failing.embed(["x"])
    assert recorder.events[-1].success is False
    assert recorder.events[-1].error_type == "RuntimeError"


async def test_database_usage_recorder_never_raises() -> None:
    class BrokenFactory:
        def __call__(self) -> Any:
            raise RuntimeError("db down")

    recorder = DatabaseUsageRecorder(BrokenFactory(), PriceTable({}))  # type: ignore[arg-type]
    await recorder.record(UsageEvent(AIOperation.CHAT, "offline", "m", 1, 1, 1.0, success=True))


async def test_job_runner_runs_isolates_failures_and_shuts_down() -> None:
    runner = AsyncioJobRunner(concurrency=1)
    done: list[str] = []

    async def ok() -> None:
        done.append("ok")

    async def crash() -> None:
        raise RuntimeError("boom")

    runner.submit("ok", ok)
    runner.submit("crash", crash)
    await runner.wait_idle()
    assert done == ["ok"]
    assert runner.pending == 0

    async def slow() -> None:
        await asyncio.sleep(30)

    runner.submit("slow", slow)
    await runner.shutdown(timeout=0.05)
    assert runner.pending == 0
    with pytest.raises(RuntimeError, match="shutting down"):
        runner.submit("late", ok)


def test_analysis_schema_rejects_bad_ids_and_extra_fields() -> None:
    bad = {
        "summary": "Summary",
        "functionalRequirements": [
            {"id": "FR-1", "title": "Title", "description": "Desc", "extra": True}
        ],
    }
    with pytest.raises(ValueError, match="extra"):
        AnalysisOutput.model_validate(bad)
    assert json.loads(AnalysisOutput(summary="Summary").model_dump_json())["userStories"] == []
