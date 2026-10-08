"""Deterministic Gherkin (`.feature`) export of generated test cases.

One `Feature` per user story (story narrative as the feature description), one `Scenario` per
test case, tagged with the test-case id, the story id, the case type and priority so runners
such as behave / pytest-bdd / Cucumber can select subsets.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class StoryInfo:
    id: str
    title: str
    role: str
    goal: str
    benefit: str | None = None


@dataclass(frozen=True, slots=True)
class ScenarioInfo:
    id: str
    story_id: str
    title: str
    case_type: str
    priority: str
    given: Sequence[str]
    when: Sequence[str]
    then: Sequence[str]
    preconditions: Sequence[str] = field(default_factory=tuple)


_KEYWORD_PREFIX = re.compile(r"^(?:given|when|then|and|but)\s+", re.IGNORECASE)


def _line(text: str) -> str:
    """Single-line, keyword-free step text (models sometimes repeat 'Given ...')."""
    collapsed = re.sub(r"\s+", " ", str(text)).strip()
    return _KEYWORD_PREFIX.sub("", collapsed).rstrip(".")


def _tag(text: str) -> str:
    return "@" + re.sub(r"[^A-Za-z0-9_.-]+", "-", text).strip("-")


def _steps(keyword: str, steps: Iterable[str]) -> list[str]:
    lines = []
    for index, step in enumerate(s for s in steps if _line(s)):
        lines.append(f"    {keyword if index == 0 else 'And'} {_line(step)}")
    return lines


def export_feature(
    stories: Sequence[StoryInfo], scenarios: Sequence[ScenarioInfo], *, header: str = ""
) -> str:
    stories_by_id = {s.id: s for s in stories}
    grouped: dict[str, list[ScenarioInfo]] = {}
    for scenario in scenarios:
        grouped.setdefault(scenario.story_id, []).append(scenario)

    lines: list[str] = []
    if header:
        lines += [f"# {_line(h)}" for h in header.splitlines() if h.strip()]
        lines.append("")
    for story_id in sorted(grouped):
        story = stories_by_id.get(story_id)
        title = _line(story.title) if story else story_id
        lines += [_tag(story_id), f"Feature: {title}"]
        if story:
            lines.append(f"  As a {_line(story.role)}")
            goal = _line(story.goal)
            lines.append(f"  I want {goal if goal.lower().startswith('to ') else 'to ' + goal}")
            if story.benefit:
                lines.append(f"  So that {_line(story.benefit)}")
        for scenario in sorted(grouped[story_id], key=lambda s: s.id):
            lines.append("")
            for precondition in scenario.preconditions:
                lines.append(f"  # Precondition: {_line(precondition)}")
            tags = [scenario.id, scenario.case_type, f"priority-{scenario.priority}"]
            lines.append("  " + " ".join(_tag(t) for t in tags))
            lines.append(f"  Scenario: {_line(scenario.title)}")
            lines += _steps("Given", scenario.given)
            lines += _steps("When", scenario.when)
            lines += _steps("Then", scenario.then)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
