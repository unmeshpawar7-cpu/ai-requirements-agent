"""Deterministic, rule-based requirement analyser (the `offline` LLM provider's brain).

This is NOT a language model. It applies documented heuristics to the same numbered evidence
blocks an LLM would receive and emits JSON for the same `AnalysisOutput` schema, citing the
evidence handle of every statement it used. It is good at well-structured "shall/must"
specifications and weak at implicit requirements, synonyms and anything needing domain
knowledge. Rules, in order:

* "As a <role>, I want <goal> [so that <benefit>]" -> user story (Given/When/Then lines nearby
  become its acceptance criteria).
* Statements in "out of scope" sections are ignored; statements in "open issues" sections and
  questions become open questions; vague terms ("fast", "user-friendly", "TBD") raise one too.
* Statements with a modal verb (shall/must/should/will/may/can/needs to) or list items under a
  "functional" heading -> functional requirement, unless quality-attribute keywords classify it
  as a non-functional requirement (category = best keyword match, metric = first number+unit).
* FRs whose actor is a role ("Dispatchers shall be able to X", "The system shall allow
  approvers to X") also yield a derived user story with one template-based acceptance criterion.
* "<Each/The> <thing> has/includes/records <a, b and c>" and "<Thing>: a, b, c" -> entity with
  typed fields (types inferred from field names, enum values from parentheses); fields that
  name another entity become many-to-one relationships.
* Verbs applied to entities (create/submit -> POST, view/list -> GET, update -> PATCH,
  delete -> DELETE, approve/assign/... -> POST action sub-resource) -> endpoint suggestions.
* Implementation tasks are derived from the generated entities, endpoints, NFR categories and
  stories.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from requirements_agent.ai.offline import text as tx
from requirements_agent.ai.schemas import (
    AnalysisOutput,
    ApiEndpoint,
    Entity,
    FunctionalRequirement,
    ImplementationTask,
    NonFunctionalRequirement,
    UserStory,
)

logger = logging.getLogger(__name__)

MAX_CITATIONS = 4
MAX_ENTITIES = 10

_STORY = re.compile(
    r"^as an? (?P<role>[^,]{2,60}?),?\s+i (?:want|need|would like)(?: to)?\s+(?P<goal>.+?)"
    r"(?:,?\s+so that\s+(?P<benefit>.+?))?\.?$",
    re.IGNORECASE,
)
_GHERKIN = re.compile(
    r"^given\s+(?P<given>.+?),?\s+when\s+(?P<when>.+?),?\s+then\s+(?P<then>.+?)\.?$",
    re.IGNORECASE,
)
_MODAL = re.compile(
    r"\b(shall|must|should|will|may|can|could|is required to|are required to|needs to|"
    r"need to|has to|have to)\b",
    re.IGNORECASE,
)
_PRIORITY = {
    "shall": "must",
    "must": "must",
    "is required to": "must",
    "are required to": "must",
    "needs to": "must",
    "need to": "must",
    "has to": "must",
    "have to": "must",
    "should": "should",
    "will": "should",
    "may": "could",
    "can": "could",
    "could": "could",
}
_MODAL_CONFIDENCE = {"must": 0.85, "should": 0.75, "could": 0.6}
_SYSTEM_SUBJECT = re.compile(
    r"^(?:the\s+)?(?:system|application|platform|portal|app|solution|service|software|api)\s+"
    r"(?:shall|must|should|will|can|may|needs to|is required to)\s+"
    r"(?:allow|enable|let|permit|provide (?:the ability for|functionality for))\s+"
    r"(?P<actor>[a-z][a-z -]{1,40}?)\s+to\s+(?P<goal>.+?)\.?$",
    re.IGNORECASE,
)
_ACTOR_SUBJECT = re.compile(
    r"^(?:the\s+|an?\s+|all\s+|each\s+)?(?P<actor>[a-z][a-z -]{1,40}?)\s+"
    r"(?:shall|must|should|will|can|may|need to|needs to)\s+(?:be able to\s+)?"
    r"(?P<goal>(?!be\b|not\b)[a-z].+?)\.?$",
    re.IGNORECASE,
)
_CONDITION = re.compile(
    r"\b(?P<kw>when|if|once|after|upon|whenever)\s+(?P<cond>[^,]+?)(?:,|$)", re.IGNORECASE
)
_BENEFIT = re.compile(r",?\s+(?:so that|in order to)\s+(?P<benefit>.+?)\.?$", re.IGNORECASE)
_METRIC = re.compile(
    r"(?:(?:within|under|below|less than|at least|at most|no more than|up to|minimum of|"
    r"maximum of|more than|over|<=|>=|<|>)\s+)?\d+(?:[.,]\d+)*\s*"
    r"(?:%|percent|ms\b|milliseconds?|seconds?|secs?\b|minutes?|mins?\b|hours?|days?|weeks?|"
    r"months?|years?|concurrent users|users|requests(?: per (?:second|minute))?|"
    r"work orders|invoices|records|transactions|MB|GB)",
    re.IGNORECASE,
)

NFR_KEYWORDS: dict[str, tuple[str, ...]] = {
    "performance": (
        "response time",
        "latency",
        "throughput",
        "p95",
        "p99",
        "performance",
        "load within",
        "page load",
        "milliseconds",
        "within 2 seconds",
        "render",
        "fast",
    ),
    "security": (
        "encrypt",
        "tls",
        "https",
        "authenticat",
        "authoriz",
        "authoris",
        "role-based",
        "rbac",
        "password",
        "mfa",
        "multi-factor",
        "single sign-on",
        "sso",
        "owasp",
        "pii",
        "sensitive",
        "access control",
        "least privilege",
        "vulnerab",
        "secure",
    ),
    "availability": (
        "uptime",
        "availability",
        "available 24",
        "99.",
        "failover",
        "downtime",
        "disaster recovery",
        "rto",
        "rpo",
        "maintenance window",
    ),
    "scalability": (
        "concurrent",
        "scalab",
        "scale",
        "peak",
        "horizontal",
        "volume of",
        "growth",
    ),
    "usability": (
        "usab",
        "accessib",
        "wcag",
        "intuitive",
        "mobile-friendly",
        "responsive",
        "offline",
        "low connectivity",
        "locali",
        "language",
        "screen reader",
        "user-friendly",
        "easy to use",
        "number of clicks",
    ),
    "reliability": (
        "backup",
        "restore",
        "data loss",
        "resilien",
        "fault",
        "recover",
        "durab",
        "idempot",
        "retry",
        "retries",
    ),
    "maintainability": (
        "maintainab",
        "logging",
        "monitoring",
        "observab",
        "metrics",
        "documented",
        "modular",
        "configurable",
        "tracing",
        "health check",
    ),
    "compliance": (
        "gdpr",
        "sox",
        "hipaa",
        "retention",
        "retain",
        "regulat",
        "compliance",
        "comply",
        "audit trail",
        "legal",
        "pci",
        "statutory",
    ),
}
_STRONG_NFR = {"security", "availability", "compliance", "performance", "scalability"}
_VAGUE_TERMS = (
    "fast",
    "quickly",
    "user-friendly",
    "easy to",
    "intuitive",
    "flexible",
    "appropriate",
    "adequate",
    "as needed",
    "as required",
    "etc",
    "tbd",
    "to be determined",
    "to be confirmed",
    "tbc",
    "reasonable",
    "sufficient",
    "optimal",
    "robust",
    "seamless",
    "efficient",
    "timely",
    "periodically",
    "regularly",
    "and/or",
    "where possible",
    "if possible",
)
_SKIP_SUBJECTS = re.compile(
    r"^(?:this|the) (?:document|section|specification|srs|chapter)\b", re.IGNORECASE
)

_ENTITY_DEFINITION = re.compile(
    r"^(?:each|every|a|an|the)?\s*(?P<entity>[a-z][a-z -]{1,40}?)\s+(?:record\s+)?"
    r"(?:(?:shall|must|should|will)\s+(?:have|include|contain|capture|record|store|hold)|"
    r"has|have|includes?|contains?|consists of|captures?|records?|stores?|holds?)\s+"
    r"(?:at least\s+|at a minimum\s+)?(?:the following(?: fields| attributes| data)?:?\s+|"
    r"an?\s+|the\s+)?(?P<fields>[a-z].+?)\.?$",
    re.IGNORECASE,
)
_ENTITY_COLON = re.compile(
    r"^(?P<entity>[A-Za-z][A-Za-z -]{1,40}?)\s*(?:record|entity)?\s*:\s*(?P<fields>.+?)\.?$"
)
_ONE_TO_MANY = re.compile(
    r"\b(?:an?|each|every|the)\s+(?P<parent>[a-z][a-z -]{1,30}?)\s+(?:can |may |shall |must )?"
    r"(?:have|has|contain|contains|include|includes|reference|references)\s+"
    r"(?:one or more|multiple|many|several|zero or more|a list of|any number of)\s+"
    r"(?P<child>[a-z][a-z -]{1,30}?)\b(?:[.,;]|\s+(?:and|with|that|which)\b|$)",
    re.IGNORECASE,
)
_NON_ENTITY_HEADS = tx.NON_ACTOR_HEADS | {
    "following",
    "ability",
    "option",
    "support",
    "access",
    "time",
    "way",
    "need",
    "list",
    "number",
    "set",
    "step",
    "type",
    "status",
    "user interface",
    "ui",
    "screen",
    "summary",
    "dashboard",
    "approval",
    "approvals",
    "history",
}

_VERB_ROUTES: dict[str, tuple[str, str | None]] = {
    # verb -> (HTTP method, action sub-resource or None)
    "create": ("POST", None),
    "add": ("POST", None),
    "submit": ("POST", None),
    "register": ("POST", None),
    "raise": ("POST", None),
    "log": ("POST", None),
    "capture": ("POST", None),
    "enter": ("POST", None),
    "upload": ("POST", None),
    "view": ("GET", None),
    "see": ("GET", None),
    "list": ("GET", None),
    "search": ("GET", None),
    "filter": ("GET", None),
    "browse": ("GET", None),
    "track": ("GET", None),
    "monitor": ("GET", None),
    "display": ("GET", None),
    "find": ("GET", None),
    "update": ("PATCH", None),
    "edit": ("PATCH", None),
    "modify": ("PATCH", None),
    "change": ("PATCH", None),
    "delete": ("DELETE", None),
    "remove": ("DELETE", None),
    "approve": ("POST", "approve"),
    "reject": ("POST", "reject"),
    "assign": ("POST", "assign"),
    "reassign": ("POST", "reassign"),
    "complete": ("POST", "complete"),
    "close": ("POST", "close"),
    "cancel": ("POST", "cancel"),
    "escalate": ("POST", "escalate"),
    "dispatch": ("POST", "dispatch"),
    "schedule": ("POST", "schedule"),
    "reschedule": ("POST", "reschedule"),
    "reopen": ("POST", "reopen"),
    "archive": ("POST", "archive"),
    "export": ("POST", "export"),
    "sign": ("POST", "sign"),
    "verify": ("POST", "verify"),
    "match": ("POST", "match"),
    "acknowledge": ("POST", "acknowledge"),
    "accept": ("POST", "accept"),
    "decline": ("POST", "decline"),
    "hold": ("POST", "hold"),
    "release": ("POST", "release"),
    "pay": ("POST", "pay"),
    "delegate": ("POST", "delegate"),
}
_COLLECTION_GET_VERBS = {"list", "search", "filter", "browse", "monitor", "find"}
_ITEM_GET_VERBS = {"view", "see", "track", "display"}


# --------------------------------------------------------------------------- data classes
@dataclass(slots=True)
class _Stmt:
    text: str
    labels: list[str]
    section: str
    heading: str

    @property
    def lower(self) -> str:
        return self.text.lower()

    @property
    def context(self) -> str:
        return f"{self.section} > {self.heading}".lower()


@dataclass(slots=True)
class _EntityDraft:
    phrase: str  # singular, lowercase
    description: str
    labels: list[str] = field(default_factory=list)
    fields: dict[str, dict[str, Any]] = field(default_factory=dict)
    relationships: dict[str, str] = field(default_factory=dict)
    defined: bool = False
    mentions: int = 0

    @property
    def name(self) -> str:
        return tx.pascal_case(self.phrase)


# --------------------------------------------------------------------------- analyser
class HeuristicRequirementsAnalyser:
    def analyse(
        self,
        evidence: list[dict[str, Any]],
        sections: list[str],
        project_name: str = "",
        focus: str | None = None,
    ) -> AnalysisOutput:
        statements = self._collect(evidence)
        open_questions: list[str] = []
        stories: list[dict[str, Any]] = []
        frs: list[dict[str, Any]] = []
        nfrs: list[dict[str, Any]] = []
        criteria_by_story: dict[int, list[dict[str, str]]] = defaultdict(list)

        for stmt in statements:
            context = stmt.context
            if "out of scope" in context or "non-goal" in context:
                continue
            if any(k in context for k in ("open issue", "open question", "assumption")) or (
                stmt.text.endswith("?")
            ):
                open_questions.append(f"{stmt.text} ({', '.join(stmt.labels)})")
                continue
            gherkin = _GHERKIN.match(stmt.text)
            if gherkin and stories:
                criteria_by_story[len(stories) - 1].append(
                    {k: _clean_clause(gherkin.group(k)) for k in ("given", "when", "then")}
                )
                continue
            story = _STORY.match(stmt.text)
            if story:
                stories.append(self._explicit_story(stmt, story))
                continue
            if _SKIP_SUBJECTS.match(stmt.text):
                continue
            self._vague(stmt, open_questions)
            category, metric = self._nfr_signal(stmt)
            modal = _MODAL.search(stmt.text)
            in_functional = "functional" in context and "non-functional" not in context
            in_nfr_section = (
                "non-functional" in context
                or "quality" in context
                or (category is not None and category in context)
            )
            if category and (
                in_nfr_section
                or (category in _STRONG_NFR and metric)
                or (category in _STRONG_NFR and not in_functional and modal)
            ):
                nfrs.append(self._nfr(stmt, category, metric, len(nfrs) + 1))
                continue
            if modal or in_functional:
                if _entity_definition(stmt.text, stmt.context) or (
                    not modal and _ONE_TO_MANY.search(stmt.text)
                ):
                    continue  # data definitions are handled by entity extraction
                frs.append(self._fr(stmt, modal.group(1).lower() if modal else None, len(frs) + 1))

        for index, criteria in criteria_by_story.items():
            stories[index]["acceptanceCriteria"] = criteria
            stories[index]["confidence"] = 0.9
        derived = self._derived_stories(frs, start=len(stories) + 1, explicit=stories)
        stories.extend(derived)
        for i, story in enumerate(stories, start=1):
            story["id"] = f"US-{i:03d}"

        entities = self._entities(statements)
        endpoints = self._endpoints(entities, statements, stories)
        tasks = self._tasks(entities, endpoints, nfrs, stories, sections)

        wanted = set(sections)
        output = {
            "summary": self._summary(evidence, frs, nfrs, stories, entities, endpoints, focus),
            "functionalRequirements": frs if "functional_requirements" in wanted else [],
            "nonFunctionalRequirements": nfrs if "non_functional_requirements" in wanted else [],
            "userStories": stories if "user_stories" in wanted else [],
            "entities": entities if "entities" in wanted else [],
            "apiEndpoints": endpoints if "api_endpoints" in wanted else [],
            "implementationTasks": tasks if "implementation_tasks" in wanted else [],
            "openQuestions": _dedupe(open_questions)[:15],
        }
        for key, model in ITEM_MODELS.items():
            output[key] = safe_items(model, output[key])
        return AnalysisOutput.model_validate(output)

    # ------------------------------------------------------------------ collection
    @staticmethod
    def _collect(evidence: list[dict[str, Any]]) -> list[_Stmt]:
        merged: dict[str, _Stmt] = {}
        order: list[str] = []
        for block in evidence:
            label = block["label"]
            for statement in tx.iter_statements(block["text"]):
                key = tx.normalise_key(statement.text)
                if not key:
                    continue
                if key in merged:
                    if label not in merged[key].labels:
                        merged[key].labels.append(label)
                    continue
                merged[key] = _Stmt(
                    text=statement.text,
                    labels=[label],
                    section=block.get("section_path", ""),
                    heading=statement.heading,
                )
                order.append(key)
        return [merged[k] for k in order]

    # ------------------------------------------------------------------ requirements
    @staticmethod
    def _nfr_signal(stmt: _Stmt) -> tuple[str | None, str | None]:
        lower = stmt.lower
        scores = {
            category: sum(1 for kw in keywords if kw in lower)
            for category, keywords in NFR_KEYWORDS.items()
        }
        for category in scores:
            if category in stmt.context:
                scores[category] += 1
        best = max(scores, key=lambda c: (scores[c], -list(NFR_KEYWORDS).index(c)))
        metric_match = _METRIC.search(stmt.text)
        metric = metric_match.group(0).strip() if metric_match else None
        return (best if scores[best] > 0 else None), metric

    @staticmethod
    def _fr(stmt: _Stmt, modal: str | None, number: int) -> dict[str, Any]:
        priority = _PRIORITY.get(modal or "", "should")
        confidence = _MODAL_CONFIDENCE[priority] if modal else 0.55
        return {
            "id": f"FR-{number:03d}",
            "title": _title(stmt.text),
            "description": stmt.text[:2000],
            "priority": priority,
            "evidence": stmt.labels[:MAX_CITATIONS],
            "confidence": confidence,
        }

    @staticmethod
    def _nfr(stmt: _Stmt, category: str, metric: str | None, number: int) -> dict[str, Any]:
        return {
            "id": f"NFR-{number:03d}",
            "category": category,
            "description": stmt.text[:2000],
            "metric": metric[:300] if metric else None,
            "evidence": stmt.labels[:MAX_CITATIONS],
            "confidence": 0.85 if metric else 0.65,
        }

    @staticmethod
    def _vague(stmt: _Stmt, open_questions: list[str]) -> None:
        lower = stmt.lower
        for term in _VAGUE_TERMS:
            if re.search(rf"(?<![a-z]){re.escape(term)}(?![a-z])", lower):
                open_questions.append(
                    f"Clarify '{term}' with a measurable criterion: \"{stmt.text}\" "
                    f"({', '.join(stmt.labels)})"
                )
                return

    # ------------------------------------------------------------------ stories
    @staticmethod
    def _explicit_story(stmt: _Stmt, match: re.Match[str]) -> dict[str, Any]:
        role = match.group("role").strip()
        goal = match.group("goal").strip().rstrip(",")
        benefit = match.group("benefit")
        return {
            "id": "US-000",
            "title": tx.capitalise(tx.truncate_words(goal, 10)),
            "role": role[:100],
            "goal": goal[:500],
            "benefit": benefit.strip()[:500] if benefit else None,
            "acceptanceCriteria": [_criterion(role, goal, stmt.text)],
            "relatedRequirements": [],
            "evidence": stmt.labels[:MAX_CITATIONS],
            "confidence": 0.8,
        }

    @staticmethod
    def _derived_stories(
        frs: list[dict[str, Any]], *, start: int, explicit: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        seen_goals = {tx.normalise_key(s["goal"]) for s in explicit}
        stories: list[dict[str, Any]] = []
        for fr in frs:
            sentence = fr["description"]
            match = _SYSTEM_SUBJECT.match(sentence) or _ACTOR_SUBJECT.match(sentence)
            if not match:
                continue
            actor = tx.strip_determiners(match.group("actor").strip())
            if not tx.looks_like_actor(actor):
                continue
            role = tx.singularize_phrase(
                " ".join(w if w.isupper() else w.lower() for w in actor.split())
            )
            goal = re.split(r",\s+and\s+|;\s+", match.group("goal").strip())[0]
            benefit_match = _BENEFIT.search(goal)
            benefit = benefit_match.group("benefit") if benefit_match else None
            if benefit_match:
                goal = goal[: benefit_match.start()].strip()
            if tx.normalise_key(goal) in seen_goals or len(goal) < 3:
                continue
            seen_goals.add(tx.normalise_key(goal))
            stories.append(
                {
                    "id": f"US-{start + len(stories):03d}",
                    "title": fr["title"],
                    "role": role[:100],
                    "goal": goal[:500],
                    "benefit": benefit[:500] if benefit else None,
                    "acceptanceCriteria": [_criterion(role, goal, sentence)],
                    "relatedRequirements": [fr["id"]],
                    "evidence": fr["evidence"],
                    "confidence": 0.65,
                }
            )
        return stories

    # ------------------------------------------------------------------ entities
    def _entities(self, statements: list[_Stmt]) -> list[dict[str, Any]]:
        drafts: dict[str, _EntityDraft] = {}

        for stmt in statements:
            definition = _entity_definition(stmt.text, stmt.context)
            if not definition:
                continue
            phrase, raw_fields = definition
            draft = drafts.setdefault(
                _entity_key(phrase), _EntityDraft(phrase=phrase, description=stmt.text[:1000])
            )
            draft.defined = True
            _add_labels(draft.labels, stmt.labels)
            for raw in _split_fields(raw_fields):
                parsed = _parse_field(raw)
                if parsed and parsed["name"] not in draft.fields:
                    draft.fields[parsed["name"]] = parsed

        # Mentions (for evidence and ranking) and one-to-many statements.
        for stmt in statements:
            lower = stmt.lower
            for draft in drafts.values():
                if _mentions(lower, draft.phrase):
                    draft.mentions += 1
                    if len(draft.labels) < MAX_CITATIONS:
                        _add_labels(draft.labels, stmt.labels[:1])
            for match in _ONE_TO_MANY.finditer(stmt.text):
                parent = _entity_key(tx.strip_determiners(match.group("parent").lower()))
                child = _entity_key(tx.strip_determiners(match.group("child").lower()))
                if parent in drafts and child in drafts and parent != child:
                    drafts[parent].relationships[drafts[child].name] = "one-to-many"
                    drafts[child].relationships.setdefault(drafts[parent].name, "many-to-one")

        ranked = sorted(drafts.values(), key=lambda d: (-len(d.fields), -d.mentions, d.phrase))
        selected = [d for d in ranked if d.fields][:MAX_ENTITIES]
        names = {d.phrase: d.name for d in selected}

        entities: list[dict[str, Any]] = []
        for draft in selected:
            fields: list[dict[str, Any]] = [
                {"name": "id", "type": "uuid", "required": True, "description": "Primary key"}
            ]
            for spec in draft.fields.values():
                target = _references_entity(spec["name"], names, exclude=draft.phrase)
                if target:
                    fk = f"{tx.snake_case(_phrase_of(target, names))}_id"
                    draft.relationships.setdefault(target, "many-to-one")
                    if all(f["name"] != fk for f in fields):
                        fields.append(
                            {
                                "name": fk,
                                "type": "uuid",
                                "required": spec["required"],
                                "description": f"Reference to {target} ({spec['name']})",
                            }
                        )
                    continue
                if all(f["name"] != spec["name"] for f in fields):
                    fields.append(spec)
            fields.append(
                {
                    "name": "created_at",
                    "type": "datetime",
                    "required": True,
                    "description": "Creation timestamp",
                }
            )
            relationships = [
                {"target": target, "kind": kind}
                for target, kind in sorted(draft.relationships.items())
                if target in names.values() and target != draft.name
            ]
            entities.append(
                {
                    "name": draft.name,
                    "description": draft.description,
                    "fields": fields[:60],
                    "relationships": relationships[:20],
                    "evidence": draft.labels[:MAX_CITATIONS],
                    "confidence": 0.8 if len(draft.fields) >= 3 else 0.6,
                }
            )
        return entities

    # ------------------------------------------------------------------ endpoints
    @staticmethod
    def _endpoints(
        entities: list[dict[str, Any]], statements: list[_Stmt], stories: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        endpoints: dict[tuple[str, str], dict[str, Any]] = {}
        for entity in entities:
            name = entity["name"]
            phrase = " ".join(re.findall(r"[A-Z][a-z0-9]*", name)).lower()
            resource = "/" + tx.kebab_case(tx.pluralize(phrase))
            field_names = {f["name"] for f in entity["fields"]}
            for stmt in statements:
                lower = stmt.lower
                if not _mentions(lower, phrase):
                    continue
                for verb, plural_object in _verbs_applied_to(lower, phrase):
                    method, action = _VERB_ROUTES[verb]
                    specs = _route_specs(
                        verb, method, action, resource, phrase, name, field_names, plural_object
                    )
                    for spec in specs:
                        key = (spec["method"], spec["path"])
                        existing = endpoints.get(key)
                        if existing is None:
                            spec["evidence"] = list(stmt.labels[:MAX_CITATIONS])
                            spec["relatedStories"] = _related_stories(stories, phrase, verb)
                            endpoints[key] = spec
                        else:
                            _add_labels(existing["evidence"], stmt.labels)
                            del existing["evidence"][MAX_CITATIONS:]

        method_order = {"GET": 0, "POST": 1, "PUT": 2, "PATCH": 3, "DELETE": 4}
        return sorted(endpoints.values(), key=lambda e: (e["path"], method_order[e["method"]]))

    # ------------------------------------------------------------------ tasks
    @staticmethod
    def _tasks(
        entities: list[dict[str, Any]],
        endpoints: list[dict[str, Any]],
        nfrs: list[dict[str, Any]],
        stories: list[dict[str, Any]],
        sections: list[str],
    ) -> list[dict[str, Any]]:
        tasks: list[dict[str, Any]] = []
        entity_task: dict[str, str] = {}

        def add(task: dict[str, Any]) -> str:
            task_id = f"T-{len(tasks) + 1:03d}"
            tasks.append({"id": task_id, **task})
            return task_id

        if "entities" in sections or "api_endpoints" in sections:
            for entity in entities:
                size = len(entity["fields"])
                entity_task[entity["name"]] = add(
                    {
                        "title": f"Model {entity['name']} entity and migration",
                        "description": (
                            f"Create the {entity['name']} table/ORM model with "
                            f"{size} fields and its relationships, plus a database migration."
                        ),
                        "estimate": "S" if size <= 6 else ("M" if size <= 12 else "L"),
                        "dependsOn": [],
                        "relatedItems": [entity["name"]],
                        "evidence": entity["evidence"],
                        "confidence": entity["confidence"],
                    }
                )
        if "api_endpoints" in sections:
            by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for endpoint in endpoints:
                by_model[endpoint["responseModel"] or endpoint["requestModel"] or "Misc"].append(
                    endpoint
                )
            for model, group in sorted(by_model.items()):
                routes = [f"{e['method']} {e['path']}" for e in group]
                evidence: list[str] = []
                for e in group:
                    _add_labels(evidence, e["evidence"])
                add(
                    {
                        "title": f"Implement {model} API endpoints",
                        "description": "Implement, validate and test: " + ", ".join(routes),
                        "estimate": "S" if len(group) <= 2 else ("M" if len(group) <= 5 else "L"),
                        "dependsOn": [entity_task[model]] if model in entity_task else [],
                        "relatedItems": routes,
                        "evidence": evidence[:MAX_CITATIONS],
                        "confidence": 0.6,
                    }
                )
        if "non_functional_requirements" in sections:
            by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for nfr in nfrs:
                by_category[nfr["category"]].append(nfr)
            for category, group in sorted(by_category.items()):
                evidence = []
                for n in group:
                    _add_labels(evidence, n["evidence"])
                metrics = [n["metric"] for n in group if n["metric"]]
                add(
                    {
                        "title": f"Implement {category} controls",
                        "description": (
                            f"Address {', '.join(n['id'] for n in group)}"
                            + (f" (targets: {', '.join(metrics)})" if metrics else "")
                            + "; add automated checks that verify them."
                        ),
                        "estimate": "M",
                        "dependsOn": [],
                        "relatedItems": [n["id"] for n in group],
                        "evidence": evidence[:MAX_CITATIONS],
                        "confidence": 0.6,
                    }
                )
        if "user_stories" in sections and stories:
            evidence = []
            for s in stories:
                _add_labels(evidence, s["evidence"])
            add(
                {
                    "title": "Automate acceptance tests for user stories",
                    "description": "Turn the Given/When/Then criteria of "
                    + ", ".join(s["id"] for s in stories)
                    + " into automated acceptance tests.",
                    "estimate": "S" if len(stories) <= 3 else ("M" if len(stories) <= 10 else "L"),
                    "dependsOn": [],
                    "relatedItems": [s["id"] for s in stories],
                    "evidence": evidence[:MAX_CITATIONS],
                    "confidence": 0.55,
                }
            )
        return tasks

    @staticmethod
    def _summary(
        evidence: list[dict[str, Any]],
        frs: list[Any],
        nfrs: list[Any],
        stories: list[Any],
        entities: list[Any],
        endpoints: list[Any],
        focus: str | None,
    ) -> str:
        documents = sorted({b.get("document_title", "") for b in evidence} - {""})
        text = (
            f"Offline heuristic analysis (rule-based, no LLM) of {len(evidence)} evidence block(s)"
            f" from {len(documents)} document(s): found {len(frs)} functional and {len(nfrs)} "
            f"non-functional requirement(s), {len(stories)} user stor"
            f"{'y' if len(stories) == 1 else 'ies'}, {len(entities)} entit"
            f"{'y' if len(entities) == 1 else 'ies'} and {len(endpoints)} endpoint suggestion(s)."
        )
        if focus:
            text += f" Retrieval was steered by the focus '{focus[:200]}'."
        return text


# --------------------------------------------------------------------------- helpers
def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _add_labels(target: list[str], labels: list[str]) -> None:
    for label in labels:
        if label not in target:
            target.append(label)


def _clean_clause(text: str) -> str:
    return text.strip().strip(",.;").strip()[:500] or "n/a"


def _title(sentence: str) -> str:
    text = sentence
    modal = _MODAL.search(text)
    if modal:
        text = text[modal.end() :]
    text = re.sub(
        r"^\s*(?:be able to\s+|(?:allow|enable|let|permit)\s+(?:the\s+)?"
        r"(?:[a-z]+\s+){0,2}?[a-z]+\s+to\s+|provide\s+|support\s+)",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.split(r"[,;:(]|\s+(?:so that|in order to|when|if|unless)\s+", text)[0]
    title = tx.truncate_words(text.strip().rstrip("."), 9)
    if len(title.split()) < 2:
        title = tx.truncate_words(sentence.rstrip("."), 9)
    return tx.capitalise(title)[:200] or "Untitled requirement"


def _criterion(role: str, goal: str, sentence: str) -> dict[str, str]:
    """Template-based Given/When/Then rewrite of a single requirement sentence."""
    condition = _CONDITION.search(goal) or _CONDITION.search(sentence)
    action = goal
    if condition and condition.start() > 0 and condition.group(0) in goal:
        action = goal[: condition.start()].strip().rstrip(",")
    given = (
        _clean_clause(condition.group("cond"))
        if condition
        else f"an authenticated {role} with access to the feature"
    )
    tokens = action.split()
    verb = tokens[0].lower() if tokens else "use"
    rest = " ".join(tokens[1:])
    when = f"the {role} {tx.third_person(verb)} {rest}".strip()
    obj = _object_phrase(rest)
    if verb in _COLLECTION_GET_VERBS | _ITEM_GET_VERBS:
        then = f"the system displays the {obj}" if obj else "the system displays the results"
    elif verb in {"receive", "get"}:
        then = f"the {role} receives {rest}" if rest else f"the {role} is notified"
    elif obj:
        aux = "were" if tx.is_plural(obj) else "was"
        then = f"the system confirms that the {obj} {aux} {tx.past_participle(verb)}"
    else:
        then = f"the system confirms the {verb} action to the {role}"
    return {"given": given[:500], "when": _clean_clause(when), "then": _clean_clause(then)}


def _object_phrase(rest: str) -> str:
    tokens = []
    for token in rest.split():
        bare = token.lower().strip(",.;")
        if bare in tx.PREPOSITIONS and bare != "of":
            break
        tokens.append(token.strip(",.;"))
    return tx.strip_determiners(" ".join(tokens[:6]))


def _entity_key(phrase: str) -> str:
    return " ".join(tx.singularize(w) for w in phrase.lower().split())


def _entity_definition(sentence: str, context: str = "") -> tuple[str, str] | None:
    match = _ENTITY_DEFINITION.match(sentence)
    if not match and not any(k in context for k in ("actor", "role", "user", "stakeholder")):
        # "Thing: a, b, c" definition lists (but not "Dispatcher: creates ..." actor lists).
        match = _ENTITY_COLON.match(sentence)
    if not match:
        return None
    phrase = tx.singularize_phrase(tx.strip_determiners(match.group("entity").lower().strip()))
    tokens = phrase.split()
    if (
        not tokens
        or len(tokens) > 3
        or phrase in _NON_ENTITY_HEADS
        or tokens[-1] in _NON_ENTITY_HEADS
    ):
        return None
    if any(t in tx.PREPOSITIONS or t in {"shall", "must", "should", "will"} for t in tokens):
        return None
    fields = match.group("fields")
    if "," not in fields and " and " not in fields:
        return None  # a single object is a requirement, not a data definition
    return phrase, fields


def _split_fields(raw: str) -> list[str]:
    parts: list[str] = []
    depth = 0
    current = ""
    for char in raw:
        if char == "(":
            depth += 1
        elif char == ")":
            depth = max(0, depth - 1)
        if char in ",;" and depth == 0:
            parts.append(current)
            current = ""
        else:
            current += char
    parts.append(current)
    result: list[str] = []
    for part in parts:
        part = re.sub(r"^\s*(?:and|or)\s+", "", part.strip())
        head, paren, tail = part.partition("(")
        pieces = [p.strip() for p in re.split(r"\s+and\s+", head) if p.strip()]
        if paren and pieces:
            pieces[-1] = f"{pieces[-1]} ({tail}"
        result.extend(pieces)
    return [p for p in result if p]


_ENUM_HINT = re.compile(r"^(?P<name>[^(]+?)\s*\((?P<values>[^)]+)\)")


def _parse_field(raw: str) -> dict[str, Any] | None:
    text = raw.strip().rstrip(".")
    optional = bool(re.search(r"\b(?:optional|if any|where applicable)\b", text, re.IGNORECASE))
    enum_values: list[str] = []
    match = _ENUM_HINT.match(text)
    if match:
        text = match.group("name")
        candidates = [v.strip() for v in re.split(r",|/|\bor\b", match.group("values"))]
        if len(candidates) >= 2:
            enum_values = [v for v in (tx.snake_case(c, 4) for c in candidates) if v][:30]
    text = re.sub(r"\b(?:optional|if any|where applicable)\b", "", text, flags=re.IGNORECASE)
    phrase = tx.strip_determiners(text.lower().strip())
    name = tx.snake_case(phrase)
    if not name or not name[0].isalpha() or len(phrase.split()) > 5:
        return None
    field_type = "enum" if enum_values else _infer_type(name)
    return {
        "name": name,
        "type": field_type,
        "required": not optional,
        "description": raw.strip()[:300],
        "enumValues": enum_values if field_type == "enum" else [],
    }


def _infer_type(name: str) -> str:
    parts = set(name.split("_"))
    if name.endswith("_id") or name == "id":
        return "uuid"
    if "email" in parts:
        return "email"
    if parts & {"timestamp", "time", "datetime"} or name.endswith("_at"):
        return "datetime"
    if parts & {"date", "deadline", "dob"}:
        return "date"
    if parts & {"amount", "total", "price", "cost", "rate", "balance", "tax", "fee", "subtotal"}:
        return "decimal"
    if parts & {"count", "quantity", "qty", "duration", "minutes", "hours", "days", "age"}:
        return "integer"
    if parts & {"status", "priority", "type", "category", "state", "level", "severity", "stage"}:
        return "enum"
    if name.startswith(("is_", "has_")) or parts & {"flag", "enabled"}:
        return "boolean"
    if parts & {"notes", "description", "comments", "comment", "remarks", "details", "reason"}:
        return "text"
    return "string"


def _mentions(lower_text: str, phrase: str) -> bool:
    singular = re.escape(phrase)
    plural = re.escape(tx.pluralize(phrase))
    return re.search(rf"\b(?:{plural}|{singular})\b", lower_text) is not None


def _references_entity(field_name: str, names: dict[str, str], exclude: str) -> str | None:
    words = field_name.replace("_id", "").replace("_", " ")
    for phrase, name in sorted(names.items(), key=lambda kv: -len(kv[0])):
        if phrase == exclude:
            continue
        if re.search(rf"\b{re.escape(phrase)}s?\b", words):
            return name
    return None


def _phrase_of(name: str, names: dict[str, str]) -> str:
    return next((p for p, n in names.items() if n == name), name)


def _verbs_applied_to(lower_text: str, phrase: str) -> list[tuple[str, bool]]:
    """(verb, object_is_plural) for route-table verbs whose direct object is the entity.

    The object may be preceded by at most two modifier words (e.g. "approve *pending*
    invoices", "view *their assigned* work orders"); prepositions in between break the link,
    so "create work orders *from* customer requests" does not target Customer.
    """
    plural = tx.pluralize(phrase)
    found: list[tuple[str, bool]] = []
    pattern = re.compile(
        rf"\b(?P<verb>{'|'.join(sorted(_VERB_ROUTES, key=len, reverse=True))})(?:s|es)?\s+"
        rf"(?P<mods>(?:[a-z'-]+\s+){{0,2}}?)(?P<obj>{re.escape(plural)}|{re.escape(phrase)})\b"
    )
    for match in pattern.finditer(lower_text):
        if any(w in tx.PREPOSITIONS for w in match.group("mods").split()):
            continue
        item = (match.group("verb"), match.group("obj") == plural and plural != phrase)
        if item not in found:
            found.append(item)
    return found


def _route_specs(
    verb: str,
    method: str,
    action: str | None,
    resource: str,
    phrase: str,
    model: str,
    field_names: set[str],
    plural_object: bool,
) -> list[dict[str, Any]]:
    item = f"{resource}/{{id}}"
    plural_phrase = tx.pluralize(phrase)
    plural_model = tx.pascal_case(plural_phrase)
    specs: list[dict[str, Any]] = []
    if action:
        specs.append(
            _endpoint(
                "POST",
                f"{item}/{action}",
                f"{tx.capitalise(action)} a {phrase}",
                tx.camel_case(action, phrase),
                None,
                model,
                False,
                200,
            )
        )
    elif method == "POST":
        specs.append(
            _endpoint(
                "POST",
                resource,
                f"Create a {phrase}",
                tx.camel_case("create", phrase),
                model,
                model,
                False,
                201,
            )
        )
    elif method == "PATCH":
        specs.append(
            _endpoint(
                "PATCH",
                item,
                f"Update a {phrase}",
                tx.camel_case("update", phrase),
                model,
                model,
                False,
                200,
            )
        )
    elif method == "DELETE":
        specs.append(
            _endpoint(
                "DELETE",
                item,
                f"Delete a {phrase}",
                tx.camel_case("delete", phrase),
                None,
                None,
                False,
                204,
            )
        )
    else:
        if verb in _COLLECTION_GET_VERBS or plural_object:
            params = [
                {"name": "page", "type": "integer", "required": False},
                {"name": "pageSize", "type": "integer", "required": False},
            ]
            if verb == "search":
                params.append(
                    {
                        "name": "q",
                        "type": "string",
                        "required": False,
                        "description": "Free-text search",
                    }
                )
            if "status" in field_names:
                params.append({"name": "status", "type": "string", "required": False})
            spec = _endpoint(
                "GET",
                resource,
                f"List {plural_phrase}",
                tx.camel_case("list", plural_phrase),
                None,
                model,
                True,
                200,
            )
            spec["queryParams"] = params
            specs.append(spec)
        else:
            specs.append(
                _endpoint(
                    "GET",
                    item,
                    f"Get a {phrase} by id",
                    tx.camel_case("get", phrase),
                    None,
                    model,
                    False,
                    200,
                )
            )
    del plural_model
    return specs


def _endpoint(
    method: str,
    path: str,
    summary: str,
    operation_id: str,
    request_model: str | None,
    response_model: str | None,
    collection: bool,
    status: int,
) -> dict[str, Any]:
    return {
        "method": method,
        "path": path,
        "summary": summary,
        "operationId": operation_id or None,
        "requestModel": request_model,
        "responseModel": response_model,
        "responseIsCollection": collection,
        "successStatus": status,
        "queryParams": [],
        "relatedStories": [],
        "evidence": [],
        "confidence": 0.65 if method == "POST" and "/{id}/" in path else 0.6,
    }


def _related_stories(stories: list[dict[str, Any]], phrase: str, verb: str) -> list[str]:
    related = []
    for story in stories:
        goal = story["goal"].lower()
        if verb in goal and _mentions(goal, phrase):
            related.append(story["id"])
    return related[:10]


def safe_items(model: type[Any], items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop items that would not validate (defensive: the analyser must always emit valid JSON)."""
    valid = []
    for item in items:
        try:
            model.model_validate(item)
            valid.append(item)
        except ValidationError:
            logger.debug("offline analyser dropped an invalid %s item", model.__name__)
    return valid


ITEM_MODELS = {
    "functionalRequirements": FunctionalRequirement,
    "nonFunctionalRequirements": NonFunctionalRequirement,
    "userStories": UserStory,
    "entities": Entity,
    "apiEndpoints": ApiEndpoint,
    "implementationTasks": ImplementationTask,
}
