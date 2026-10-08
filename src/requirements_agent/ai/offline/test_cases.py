"""Rule-based test-case derivation from acceptance criteria (offline provider).

For every acceptance criterion:
* one `positive` case that restates Given/When/Then as executable steps;
* for the first criterion of each story, one `negative` authorisation case (an actor without
  the story's role attempts the action) - flagged in its preconditions as an assumption;
* one `edge` case when the criterion mentions a number (limits, time windows, amounts),
  probing the boundary value.
"""

from __future__ import annotations

import re
from typing import Any

from requirements_agent.ai.schemas import TestCaseGenerationOutput, UserStory

_NUMBER = re.compile(r"\b\d+(?:[.,]\d+)*(?:\s*%|\s+[a-z]+)?", re.IGNORECASE)


def derive_test_cases(stories: list[dict[str, Any]]) -> TestCaseGenerationOutput:
    cases: list[dict[str, Any]] = []

    def add(case: dict[str, Any]) -> None:
        cases.append({"id": f"TC-{len(cases) + 1:03d}", **case})

    for raw in stories:
        story = UserStory.model_validate(raw)
        role = story.role
        for index, criterion in enumerate(story.acceptance_criteria):
            add(
                {
                    "title": f"{story.title} - criterion {index + 1} holds",
                    "storyId": story.id,
                    "criterionIndex": index,
                    "type": "positive",
                    "priority": "high" if index == 0 else "medium",
                    "preconditions": [f"A user with the '{role}' role exists"],
                    "given": [criterion.given],
                    "when": [criterion.when],
                    "then": [criterion.then],
                    "confidence": 0.8,
                }
            )
            if index == 0:
                when = re.sub(
                    rf"\bthe {re.escape(role)}\b",
                    "the user",
                    criterion.when,
                    count=1,
                    flags=re.IGNORECASE,
                )
                add(
                    {
                        "title": f"{story.title} - rejected without the {role} role",
                        "storyId": story.id,
                        "criterionIndex": index,
                        "type": "negative",
                        "priority": "medium",
                        "preconditions": [
                            f"Assumption: the action is restricted to the '{role}' role"
                        ],
                        "given": [f"a signed-in user who does not have the '{role}' role"],
                        "when": [when],
                        "then": [
                            "the system rejects the request with an authorisation error",
                            "no data is changed",
                        ],
                        "confidence": 0.5,
                    }
                )
            number = _NUMBER.search(f"{criterion.given} {criterion.when} {criterion.then}")
            if number:
                value = number.group(0).strip()
                add(
                    {
                        "title": f"{story.title} - boundary at {value}",
                        "storyId": story.id,
                        "criterionIndex": index,
                        "type": "edge",
                        "priority": "medium",
                        "preconditions": [],
                        "given": [criterion.given],
                        "when": [f"{criterion.when} exactly at the boundary value {value}"],
                        "then": [f"the outcome matches the criterion: {criterion.then}"],
                        "confidence": 0.55,
                    }
                )
    return TestCaseGenerationOutput.model_validate({"testCases": cases})
