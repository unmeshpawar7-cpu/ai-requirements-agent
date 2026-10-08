"""Deterministic OpenAPI 3.1 document builder.

Input: the *validated and grounded* entity and endpoint suggestions of an analysis. No AI is
involved here: the same input always yields byte-identical JSON (sorted paths, fixed method
order, sorted schemas), so specs can be diffed between analyses.

Conventions applied to every generated API:
* entity `X` -> component schemas `X` (read model), `XCreate` (no read-only fields),
  `XUpdate` (all optional, at least one property) and `XPage` (paginated list);
* `decimal` fields are strings with a decimal pattern (no binary floating point for money);
* bearer JWT security, a shared `Error` envelope and reusable error responses;
* traceability extensions: `x-evidence` (chunk ids), `x-confidence`, `x-related-stories`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from requirements_agent.ai.schemas import ApiEndpoint, Entity, EntityField

OPENAPI_VERSION = "3.1.0"
READ_ONLY_FIELDS = frozenset({"id", "created_at", "updated_at"})
_METHOD_ORDER = {"GET": 0, "POST": 1, "PUT": 2, "PATCH": 3, "DELETE": 4}
_PATH_PARAM = re.compile(r"{([A-Za-z_][A-Za-z0-9_]*)}")

_TYPE_MAP: dict[str, dict[str, Any]] = {
    "string": {"type": "string"},
    "text": {"type": "string"},
    "integer": {"type": "integer", "format": "int64"},
    "number": {"type": "number"},
    "decimal": {"type": "string", "format": "decimal", "pattern": r"^-?\d+(\.\d+)?$"},
    "boolean": {"type": "boolean"},
    "date": {"type": "string", "format": "date"},
    "datetime": {"type": "string", "format": "date-time"},
    "uuid": {"type": "string", "format": "uuid"},
    "email": {"type": "string", "format": "email"},
    "enum": {"type": "string"},
}

_ERROR_RESPONSES = {
    "BadRequest": "The request is malformed",
    "Unauthorized": "Missing or invalid bearer token",
    "NotFound": "The resource does not exist",
    "ValidationError": "The request body failed validation",
}


@dataclass(slots=True)
class OpenAPIBuildResult:
    document: dict[str, Any]
    warnings: list[str] = field(default_factory=list)


class OpenAPIBuilder:
    def build(
        self,
        *,
        title: str,
        version: str = "0.1.0",
        description: str = "",
        entities: list[Entity],
        endpoints: list[ApiEndpoint],
        extensions: dict[str, Any] | None = None,
    ) -> OpenAPIBuildResult:
        warnings: list[str] = []
        entity_names = {e.name for e in entities}
        schemas: dict[str, Any] = {"Error": _error_schema()}
        for entity in sorted(entities, key=lambda e: e.name):
            schemas.update(_entity_schemas(entity))

        paths: dict[str, dict[str, Any]] = {}
        used_operation_ids: set[str] = set()
        tags: dict[str, str] = {}
        ordered = sorted(endpoints, key=lambda e: (e.path, _METHOD_ORDER[e.method]))
        for endpoint in ordered:
            operation = self._operation(endpoint, entity_names, used_operation_ids, warnings)
            tag = operation["tags"][0]
            tags.setdefault(tag, _tag_description(tag, entities))
            path_item = paths.setdefault(endpoint.path, {})
            method = endpoint.method.lower()
            if method in path_item:
                warnings.append(f"duplicate operation {endpoint.method} {endpoint.path} skipped")
                continue
            path_item[method] = operation

        document: dict[str, Any] = {
            "openapi": OPENAPI_VERSION,
            "info": {
                "title": title,
                "version": version,
                "description": description
                or "Generated from requirement documents by ai-requirements-agent.",
            },
            "tags": [{"name": n, "description": d} for n, d in sorted(tags.items())],
            "paths": paths,
            "components": {
                "schemas": schemas,
                "responses": {
                    name: {
                        "description": text,
                        "content": {
                            "application/json": {"schema": {"$ref": "#/components/schemas/Error"}}
                        },
                    }
                    for name, text in _ERROR_RESPONSES.items()
                },
                "securitySchemes": {
                    "bearerAuth": {"type": "http", "scheme": "bearer", "bearerFormat": "JWT"}
                },
            },
            "security": [{"bearerAuth": []}],
        }
        if extensions:
            document["x-generated"] = extensions
        return OpenAPIBuildResult(document=document, warnings=warnings)

    # ------------------------------------------------------------------ operations
    def _operation(
        self,
        endpoint: ApiEndpoint,
        entity_names: set[str],
        used_ids: set[str],
        warnings: list[str],
    ) -> dict[str, Any]:
        request_model = self._checked(endpoint.request_model, entity_names, endpoint, warnings)
        response_model = self._checked(endpoint.response_model, entity_names, endpoint, warnings)
        tag = response_model or request_model or _tag_from_path(endpoint.path)

        operation: dict[str, Any] = {
            "tags": [tag],
            "summary": endpoint.summary,
            "operationId": _unique_operation_id(endpoint, used_ids),
        }
        parameters = [
            {
                "name": name,
                "in": "path",
                "required": True,
                "schema": {"type": "string", "format": "uuid"}
                if name == "id" or name.lower().endswith("id")
                else {"type": "string"},
            }
            for name in _PATH_PARAM.findall(endpoint.path)
        ]
        for param in endpoint.query_params:
            schema = dict(_TYPE_MAP[param.type])
            if param.name == "pageSize":
                schema.update({"minimum": 1, "maximum": 100, "default": 20})
            elif param.name == "page":
                schema.update({"minimum": 1, "default": 1})
            entry: dict[str, Any] = {
                "name": param.name,
                "in": "query",
                "required": param.required,
                "schema": schema,
            }
            if param.description:
                entry["description"] = param.description
            parameters.append(entry)
        if parameters:
            operation["parameters"] = parameters

        has_body = endpoint.method in {"POST", "PUT", "PATCH"} and request_model is not None
        if has_body:
            suffix = "Update" if endpoint.method == "PATCH" else "Create"
            operation["requestBody"] = {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {"$ref": f"#/components/schemas/{request_model}{suffix}"}
                    }
                },
            }

        responses: dict[str, Any] = {}
        status = str(endpoint.success_status)
        if endpoint.success_status == 204:
            responses[status] = {"description": "No content"}
        else:
            if response_model:
                name = (
                    f"{response_model}Page" if endpoint.response_is_collection else response_model
                )
                schema: dict[str, Any] = {"$ref": f"#/components/schemas/{name}"}
            else:
                schema = {"type": "object"}
            responses[status] = {
                "description": "Successful response",
                "content": {"application/json": {"schema": schema}},
            }
        if has_body or endpoint.method in {"POST", "PUT", "PATCH"}:
            responses["400"] = {"$ref": "#/components/responses/BadRequest"}
        responses["401"] = {"$ref": "#/components/responses/Unauthorized"}
        if "{" in endpoint.path:
            responses["404"] = {"$ref": "#/components/responses/NotFound"}
        if has_body:
            responses["422"] = {"$ref": "#/components/responses/ValidationError"}
        operation["responses"] = responses

        operation["x-evidence"] = list(endpoint.evidence)
        operation["x-confidence"] = round(endpoint.confidence, 3)
        if endpoint.related_stories:
            operation["x-related-stories"] = list(endpoint.related_stories)
        return operation

    @staticmethod
    def _checked(
        model: str | None, entity_names: set[str], endpoint: ApiEndpoint, warnings: list[str]
    ) -> str | None:
        if model is None or model in entity_names:
            return model
        warnings.append(
            f"{endpoint.method} {endpoint.path}: unknown model '{model}' replaced by a generic "
            "object"
        )
        return None


# ---------------------------------------------------------------------- schema helpers
def field_schema(spec: EntityField) -> dict[str, Any]:
    schema = dict(_TYPE_MAP[spec.type])
    if spec.type == "enum" and spec.enum_values:
        schema["enum"] = list(spec.enum_values)
    if spec.description:
        schema["description"] = spec.description
    if spec.name in READ_ONLY_FIELDS:
        schema["readOnly"] = True
    return schema


def _entity_schemas(entity: Entity) -> dict[str, Any]:
    properties = {f.name: field_schema(f) for f in entity.fields}
    required = sorted(f.name for f in entity.fields if f.required)
    writable = {
        name: {k: v for k, v in schema.items() if k != "readOnly"}
        for name, schema in properties.items()
        if name not in READ_ONLY_FIELDS
    }
    writable_required = [n for n in required if n not in READ_ONLY_FIELDS]

    read_model: dict[str, Any] = {
        "type": "object",
        "description": entity.description,
        "properties": properties,
    }
    if required:
        read_model["required"] = required
    if entity.relationships:
        read_model["x-relationships"] = [
            {"target": r.target, "kind": r.kind} for r in entity.relationships
        ]

    create: dict[str, Any] = {"type": "object", "properties": writable}
    if writable_required:
        create["required"] = writable_required
    update: dict[str, Any] = {"type": "object", "properties": writable, "minProperties": 1}

    page = {
        "type": "object",
        "required": ["items", "page", "pageSize", "totalCount", "totalPages"],
        "properties": {
            "items": {"type": "array", "items": {"$ref": f"#/components/schemas/{entity.name}"}},
            "page": {"type": "integer", "minimum": 1},
            "pageSize": {"type": "integer", "minimum": 1, "maximum": 100},
            "totalCount": {"type": "integer", "minimum": 0},
            "totalPages": {"type": "integer", "minimum": 0},
        },
    }
    return {
        entity.name: read_model,
        f"{entity.name}Create": create,
        f"{entity.name}Update": update,
        f"{entity.name}Page": page,
    }


def _error_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "required": ["success", "message"],
        "properties": {
            "success": {"type": "boolean", "const": False},
            "message": {"type": "string"},
            "errors": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "field": {"type": ["string", "null"]},
                        "message": {"type": "string"},
                    },
                    "required": ["message"],
                },
            },
            "traceId": {"type": ["string", "null"]},
        },
    }


def _tag_from_path(path: str) -> str:
    first = next((seg for seg in path.split("/") if seg and not seg.startswith("{")), "default")
    return "".join(part.capitalize() for part in first.split("-")) or "Default"


def _tag_description(tag: str, entities: list[Entity]) -> str:
    entity = next((e for e in entities if e.name == tag), None)
    return entity.description if entity else f"Operations on {tag}"


def _unique_operation_id(endpoint: ApiEndpoint, used: set[str]) -> str:
    base = endpoint.operation_id or _derive_operation_id(endpoint)
    candidate, counter = base, 2
    while candidate in used:
        candidate = f"{base}{counter}"
        counter += 1
    used.add(candidate)
    return candidate


def _derive_operation_id(endpoint: ApiEndpoint) -> str:
    words = [endpoint.method.lower()]
    for segment in endpoint.path.split("/"):
        if not segment:
            continue
        if segment.startswith("{"):
            words.append("By" + segment.strip("{}").capitalize())
        else:
            words.extend(part.capitalize() for part in segment.split("-"))
    return re.sub(r"[^A-Za-z0-9]", "", "".join(words))
