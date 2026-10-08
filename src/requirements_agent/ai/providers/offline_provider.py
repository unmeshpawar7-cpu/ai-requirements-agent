"""`LLMProvider` implementation backed by the deterministic heuristic analyser.

It honours the provider contract exactly (same request, raw JSON text response, token
estimates), so the structured-output validation, grounding and confidence stages run unchanged
whether the answer came from GPT, Claude or these rules. It reads the structured
`task_input` rather than the rendered prompt text, and ignores repair messages (its output is
already schema-valid).
"""

from __future__ import annotations

from requirements_agent.ai.offline.analyser import HeuristicRequirementsAnalyser
from requirements_agent.ai.offline.test_cases import derive_test_cases
from requirements_agent.ai.providers.base import LLMRequest, LLMResponse, ProviderError
from requirements_agent.ingestion.tokens import estimate_tokens


class OfflineHeuristicProvider:
    name = "offline"
    model = "heuristic-v1"

    def __init__(self) -> None:
        self._analyser = HeuristicRequirementsAnalyser()

    async def complete(self, request: LLMRequest) -> LLMResponse:
        payload = request.task_input
        if request.task == "requirements_analysis":
            result = self._analyser.analyse(
                evidence=payload.get("evidence", []),
                sections=payload.get("sections", []),
                project_name=payload.get("project_name", ""),
                focus=payload.get("focus"),
            )
        elif request.task == "test_case_generation":
            result = derive_test_cases(payload.get("stories", []))
        else:
            raise ProviderError(f"Offline provider does not support task '{request.task}'")
        text = result.model_dump_json(by_alias=True)
        prompt_text = request.system + "".join(m.content for m in request.messages)
        return LLMResponse(
            text=text,
            model=self.model,
            input_tokens=estimate_tokens(prompt_text),
            output_tokens=estimate_tokens(text),
            finish_reason="stop",
            tokens_estimated=True,
        )
