"""Small, dependency-free English text helpers used by the offline heuristic analyser.

These are deliberately simple rules (no POS tagger, no model). They are good enough to turn
well-written "shall/must" specifications into structured artefacts, and they fail in
predictable ways on free-form prose; see docs/ai-design.md for the known limitations.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass

from requirements_agent.ingestion.chunker import detect_heading

_HEADING = re.compile(r"^#{1,6}\s+")
_LIST_MARKER = re.compile(r"^(?:[-+*]\s+|\d+[.)]\s+|[a-z][.)]\s+)")
_REQ_ID_PREFIX = re.compile(r"^(?:\[?[A-Z]{1,6}[-_ ]?\d{1,4}(?:\.\d+)*\]?\s*[:.)\-\u2013]?\s+)")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")
_ABBREVIATIONS = ("e.g.", "i.e.", "etc.", "vs.", "approx.", "incl.", "no.", "Mr.", "Mrs.", "Dr.")

DETERMINERS = frozenset(
    [
        "a",
        "an",
        "the",
        "each",
        "every",
        "any",
        "all",
        "their",
        "its",
        "his",
        "her",
        "our",
        "your",
        "this",
        "that",
        "these",
        "those",
        "new",
        "existing",
        "multiple",
        "several",
        "one",
    ]
)
PREPOSITIONS = frozenset(
    [
        "to",
        "for",
        "from",
        "with",
        "by",
        "in",
        "on",
        "at",
        "of",
        "via",
        "within",
        "into",
        "onto",
        "about",
        "against",
        "between",
        "through",
        "during",
        "before",
        "after",
        "when",
        "if",
        "once",
        "upon",
        "based",
        "so",
        "unless",
        "until",
        "while",
        "where",
        "which",
        "that",
        "including",
        "using",
        "without",
        "per",
        "across",
        "over",
        "under",
    ]
)
NON_ACTOR_HEADS = frozenset(
    [
        "system",
        "application",
        "platform",
        "portal",
        "app",
        "service",
        "api",
        "solution",
        "software",
        "database",
        "it",
        "they",
        "this",
        "data",
        "report",
        "reports",
        "module",
        "interface",
        "page",
        "screen",
        "form",
        "tool",
        "engine",
        "process",
        "feature",
        "notification",
        "notifications",
        "email",
        "emails",
        "record",
        "records",
        "information",
        "field",
        "fields",
        "value",
        "values",
        "job",
        "jobs",
    ]
)
ROLE_HINTS = frozenset(
    [
        "user",
        "users",
        "admin",
        "admins",
        "administrator",
        "administrators",
        "manager",
        "managers",
        "customer",
        "customers",
        "supplier",
        "suppliers",
        "vendor",
        "vendors",
        "approver",
        "approvers",
        "technician",
        "technicians",
        "dispatcher",
        "dispatchers",
        "clerk",
        "clerks",
        "auditor",
        "auditors",
        "analyst",
        "analysts",
        "operator",
        "operators",
        "agent",
        "agents",
        "staff",
        "employee",
        "employees",
        "controller",
        "controllers",
        "owner",
        "owners",
        "requester",
        "requesters",
        "reviewer",
        "reviewers",
        "planner",
        "planners",
        "supervisor",
        "supervisors",
        "member",
        "members",
        "guest",
        "guests",
        "client",
        "clients",
        "patient",
        "patients",
        "engineer",
        "engineers",
        "coordinator",
        "coordinators",
        "officer",
        "officers",
        "accountant",
        "accountants",
        "buyer",
        "buyers",
        "lead",
        "leads",
        "director",
        "directors",
    ]
)
ROLE_SUFFIXES = ("er", "or", "ist", "ant", "ent")

_IRREGULAR_PAST_PARTICIPLE = {
    "send": "sent",
    "see": "seen",
    "set": "set",
    "get": "retrieved",
    "write": "written",
    "take": "taken",
    "give": "given",
    "make": "made",
    "find": "found",
    "build": "built",
    "pay": "paid",
    "read": "read",
    "choose": "chosen",
    "run": "run",
    "hold": "held",
    "keep": "kept",
    "override": "overridden",
    "put": "put",
    "upload": "uploaded",
    "show": "shown",
    "draw": "drawn",
    "sign": "signed",
    "submit": "submitted",
    "cancel": "cancelled",
    "log": "logged",
    "plan": "planned",
    "ship": "shipped",
}
_IRREGULAR_THIRD_PERSON = {"be": "is", "have": "has", "do": "does", "go": "goes"}


@dataclass(frozen=True, slots=True)
class Statement:
    """One requirement-like statement: a sentence or list item from an evidence block."""

    text: str
    heading: str  # nearest heading inside the evidence block (may be empty)


def iter_statements(block_text: str) -> Iterator[Statement]:
    """Yield sentences and list items from an evidence block, skipping heading lines."""
    heading = ""
    paragraph: list[str] = []

    def flush() -> Iterator[Statement]:
        if paragraph:
            joined = " ".join(paragraph)
            for sentence in split_sentences(joined):
                cleaned = clean_statement(sentence)
                if cleaned:
                    yield Statement(cleaned, heading)
            paragraph.clear()

    for raw_line in block_text.split("\n"):
        line = raw_line.strip()
        if not line:
            yield from flush()
            continue
        if _HEADING.match(line) or (not paragraph and detect_heading(line)):
            yield from flush()
            heading = _HEADING.sub("", line).strip()
            continue
        if _LIST_MARKER.match(line):
            yield from flush()
            paragraph.append(_LIST_MARKER.sub("", line))
            continue
        paragraph.append(line)
    yield from flush()


def split_sentences(text: str) -> list[str]:
    protected = text
    for i, abbreviation in enumerate(_ABBREVIATIONS):
        protected = protected.replace(abbreviation, f"\x00{i}\x00")
    parts = _SENTENCE_SPLIT.split(protected)
    restored = []
    for part in parts:
        for i, abbreviation in enumerate(_ABBREVIATIONS):
            part = part.replace(f"\x00{i}\x00", abbreviation)
        restored.append(part.strip())
    return [p for p in restored if p]


def clean_statement(text: str) -> str:
    text = _REQ_ID_PREFIX.sub("", text.strip())
    text = re.sub(r"\s+", " ", text).strip()
    text = text.strip("*_ ")
    return text if len(text) >= 12 else ""


def normalise_key(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", "", text.lower()).strip()


def words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z][A-Za-z'-]*", text)


def singularize(word: str) -> str:
    lower = word.lower()
    if len(lower) <= 3 or lower.endswith(("ss", "us", "is")):
        return word
    if lower.endswith("ies"):
        return word[:-3] + "y"
    if lower.endswith(("ches", "shes", "xes", "sses", "zes")):
        return word[:-2]
    if lower.endswith("s"):
        return word[:-1]
    return word


def pluralize(word: str) -> str:
    lower = word.lower()
    if lower.endswith("y") and len(lower) > 1 and lower[-2] not in "aeiou":
        return word[:-1] + "ies"
    if lower.endswith(("s", "x", "z", "ch", "sh")):
        return word + "es"
    return word + "s"


def singularize_phrase(phrase: str) -> str:
    parts = phrase.split()
    if not parts:
        return phrase
    return " ".join([*parts[:-1], singularize(parts[-1])])


def is_plural(phrase: str) -> bool:
    last = phrase.split()[-1].lower() if phrase.split() else ""
    return last.endswith("s") and not last.endswith(("ss", "us", "is")) and len(last) > 3


def strip_determiners(phrase: str) -> str:
    parts = phrase.split()
    while parts and parts[0].lower() in DETERMINERS:
        parts.pop(0)
    return " ".join(parts)


def pascal_case(phrase: str) -> str:
    return "".join(w[:1].upper() + w[1:].lower() for w in words(phrase))


def snake_case(phrase: str, max_words: int = 4) -> str:
    tokens = [w.lower().replace("'", "") for w in words(phrase)][:max_words]
    name = "_".join(t.replace("-", "_") for t in tokens if t)
    name = re.sub(r"[^a-z0-9_]", "", name).strip("_")
    return name[:63]


def kebab_case(phrase: str) -> str:
    return "-".join(w.lower() for w in words(phrase))


def camel_case(*parts: str) -> str:
    tokens = [w for part in parts for w in words(part)]
    if not tokens:
        return ""
    first, rest = tokens[0].lower(), [t[:1].upper() + t[1:].lower() for t in tokens[1:]]
    return re.sub(r"[^A-Za-z0-9]", "", first + "".join(rest))


def third_person(verb: str) -> str:
    lower = verb.lower()
    if lower in _IRREGULAR_THIRD_PERSON:
        return _IRREGULAR_THIRD_PERSON[lower]
    if lower.endswith("y") and len(lower) > 1 and lower[-2] not in "aeiou":
        return verb[:-1] + "ies"
    if lower.endswith(("s", "x", "z", "ch", "sh", "o")):
        return verb + "es"
    return verb + "s"


def past_participle(verb: str) -> str:
    lower = verb.lower()
    if lower in _IRREGULAR_PAST_PARTICIPLE:
        return _IRREGULAR_PAST_PARTICIPLE[lower]
    if lower.endswith("e"):
        return verb + "d"
    if lower.endswith("y") and len(lower) > 1 and lower[-2] not in "aeiou":
        return verb[:-1] + "ied"
    return verb + "ed"


def looks_like_actor(phrase: str) -> bool:
    tokens = [w.lower() for w in words(phrase)]
    if not tokens or len(tokens) > 3:
        return False
    head = tokens[-1]
    if head in NON_ACTOR_HEADS:
        return False
    return head in ROLE_HINTS or singularize(head) in ROLE_HINTS or head.endswith(ROLE_SUFFIXES)


def truncate_words(text: str, max_words: int, max_chars: int = 200) -> str:
    tokens = text.split()
    clipped = " ".join(tokens[:max_words])
    return clipped[:max_chars].rstrip(" ,;:")


def capitalise(text: str) -> str:
    return text[:1].upper() + text[1:] if text else text
