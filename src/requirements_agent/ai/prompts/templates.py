"""Prompt template parsing, validation, rendering and seed loading.

Template format (stored verbatim in `prompt_templates.template`)::

    <<<SYSTEM>>>
    ...system prompt with $placeholders...
    <<<USER>>>
    ...user prompt with $placeholders...

Placeholders use `string.Template` syntax (`$name`), which leaves JSON braces alone; a literal
dollar sign is written `$$`. Seed files additionally start with a small `---` front-matter block
(name, version, description) that is not part of the stored template.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from importlib import resources
from string import Template

SYSTEM_MARKER = "<<<SYSTEM>>>"
USER_MARKER = "<<<USER>>>"


class PromptTemplateError(ValueError):
    """The template is malformed or does not match the prompt's variable contract."""


@dataclass(frozen=True, slots=True)
class PromptSpec:
    name: str
    required_variables: frozenset[str]
    optional_variables: frozenset[str] = frozenset()

    @property
    def allowed_variables(self) -> frozenset[str]:
        return self.required_variables | self.optional_variables


PROMPT_SPECS: dict[str, PromptSpec] = {
    "requirements_analysis": PromptSpec(
        name="requirements_analysis",
        required_variables=frozenset({"evidence", "schema", "sections"}),
        optional_variables=frozenset({"project_name", "focus", "evidence_ids"}),
    ),
    "test_case_generation": PromptSpec(
        name="test_case_generation",
        required_variables=frozenset({"stories", "schema"}),
        optional_variables=frozenset({"project_name", "story_ids"}),
    ),
}


@dataclass(frozen=True, slots=True)
class RenderedPrompt:
    name: str
    version: int
    system: str
    user: str


@dataclass(frozen=True, slots=True)
class SeedTemplate:
    name: str
    version: int
    description: str | None
    template: str
    checksum: str


def checksum(template: str) -> str:
    return hashlib.sha256(template.encode("utf-8")).hexdigest()


def split_template(template: str) -> tuple[str, str]:
    if template.count(SYSTEM_MARKER) != 1 or template.count(USER_MARKER) != 1:
        raise PromptTemplateError(
            f"Template must contain exactly one {SYSTEM_MARKER} and one {USER_MARKER} marker"
        )
    before_user, user = template.split(USER_MARKER)
    if SYSTEM_MARKER not in before_user:
        raise PromptTemplateError(f"{SYSTEM_MARKER} must come before {USER_MARKER}")
    system = before_user.split(SYSTEM_MARKER, 1)[1]
    if not system.strip() or not user.strip():
        raise PromptTemplateError("System and user sections must not be empty")
    return system.strip(), user.strip()


def validate_template(name: str, template: str) -> None:
    """Raise `PromptTemplateError` if the template breaks the prompt's variable contract."""
    spec = PROMPT_SPECS.get(name)
    if spec is None:
        raise PromptTemplateError(f"Unknown prompt name '{name}'")
    system, user = split_template(template)
    identifiers: set[str] = set()
    for part in (system, user):
        tmpl = Template(part)
        if not tmpl.is_valid():
            raise PromptTemplateError("Template contains an invalid placeholder (use $$ for '$')")
        identifiers.update(tmpl.get_identifiers())
    missing = spec.required_variables - identifiers
    unknown = identifiers - spec.allowed_variables
    if missing:
        raise PromptTemplateError(f"Template is missing required placeholders: {sorted(missing)}")
    if unknown:
        raise PromptTemplateError(f"Template uses unknown placeholders: {sorted(unknown)}")


def render(name: str, version: int, template: str, variables: dict[str, str]) -> RenderedPrompt:
    spec = PROMPT_SPECS[name]
    values = dict.fromkeys(spec.optional_variables, "") | variables
    system, user = split_template(template)
    try:
        return RenderedPrompt(
            name=name,
            version=version,
            system=Template(system).substitute(values),
            user=Template(user).substitute(values),
        )
    except (KeyError, ValueError) as exc:
        raise PromptTemplateError(f"Cannot render prompt '{name}' v{version}: {exc}") from exc


_FRONT_MATTER = re.compile(r"\A---\n(?P<meta>.*?)\n---\n", re.DOTALL)


def parse_seed_file(content: str) -> SeedTemplate:
    match = _FRONT_MATTER.match(content)
    if not match:
        raise PromptTemplateError("Seed prompt file must start with a --- front-matter block")
    meta: dict[str, str] = {}
    for line in match.group("meta").splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            meta[key.strip()] = value.strip()
    body = content[match.end() :].strip() + "\n"
    try:
        name, version = meta["name"], int(meta["version"])
    except (KeyError, ValueError) as exc:
        raise PromptTemplateError("Front matter needs 'name' and integer 'version'") from exc
    validate_template(name, body)
    return SeedTemplate(
        name=name,
        version=version,
        description=meta.get("description"),
        template=body,
        checksum=checksum(body),
    )


def load_seed_templates() -> list[SeedTemplate]:
    """Read every `ai/prompts/<name>/v<N>.md` shipped with the package."""
    seeds: list[SeedTemplate] = []
    root = resources.files("requirements_agent.ai.prompts")
    for directory in sorted(root.iterdir(), key=lambda p: p.name):
        if not directory.is_dir() or directory.name.startswith("_"):
            continue
        for file in sorted(directory.iterdir(), key=lambda p: p.name):
            if re.fullmatch(r"v\d+\.md", file.name):
                seed = parse_seed_file(file.read_text(encoding="utf-8"))
                if seed.name != directory.name or f"v{seed.version}.md" != file.name:
                    raise PromptTemplateError(
                        f"{directory.name}/{file.name}: front matter does not match path"
                    )
                seeds.append(seed)
    return seeds
