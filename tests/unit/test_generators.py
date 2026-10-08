from __future__ import annotations

import json

from openapi_spec_validator import validate

from requirements_agent.ai.schemas import ApiEndpoint, Entity
from requirements_agent.generators.gherkin import ScenarioInfo, StoryInfo, export_feature
from requirements_agent.generators.markdown_docs import render_markdown
from requirements_agent.generators.openapi_builder import OpenAPIBuilder

ENTITIES = [
    Entity.model_validate(
        {
            "name": "Invoice",
            "description": "Supplier invoice",
            "fields": [
                {"name": "id", "type": "uuid"},
                {"name": "invoice_number", "type": "string"},
                {"name": "total_amount", "type": "decimal"},
                {"name": "status", "type": "enum", "enumValues": ["submitted", "approved"]},
                {"name": "due_date", "type": "date", "required": False},
                {"name": "created_at", "type": "datetime"},
            ],
            "relationships": [{"target": "Supplier", "kind": "many-to-one"}],
            "evidence": ["chunk-1"],
        }
    ),
    Entity.model_validate(
        {
            "name": "Supplier",
            "description": "A vendor",
            "evidence": ["chunk-2"],
            "fields": [{"name": "id", "type": "uuid"}, {"name": "email", "type": "email"}],
        }
    ),
]
ENDPOINTS = [
    ApiEndpoint.model_validate(e)
    for e in [
        {
            "method": "POST",
            "path": "/invoices",
            "summary": "Submit an invoice",
            "operationId": "createInvoice",
            "requestModel": "Invoice",
            "responseModel": "Invoice",
            "successStatus": 201,
            "evidence": ["chunk-1"],
            "confidence": 0.8,
        },
        {
            "method": "GET",
            "path": "/invoices",
            "summary": "List invoices",
            "responseModel": "Invoice",
            "responseIsCollection": True,
            "queryParams": [
                {"name": "page", "type": "integer"},
                {"name": "pageSize", "type": "integer"},
                {"name": "status", "type": "string", "description": "Filter"},
            ],
            "relatedStories": ["US-001"],
            "evidence": ["chunk-1"],
        },
        {
            "method": "POST",
            "path": "/invoices/{id}/approve",
            "summary": "Approve",
            "responseModel": "Invoice",
            "evidence": ["chunk-1"],
        },
        {
            "method": "PATCH",
            "path": "/invoices/{id}",
            "summary": "Update",
            "requestModel": "Invoice",
            "responseModel": "Invoice",
        },
        {"method": "DELETE", "path": "/invoices/{id}", "summary": "Delete", "successStatus": 204},
        {
            "method": "GET",
            "path": "/suppliers/{supplierId}",
            "summary": "Get supplier",
            "operationId": "createInvoice",
            "responseModel": "Ghost",
        },
    ]
]


def build() -> dict:
    result = OpenAPIBuilder().build(
        title="Portal API",
        entities=ENTITIES,
        endpoints=list(reversed(ENDPOINTS)),
        extensions={"analysisId": "a1"},
    )
    return {"document": result.document, "warnings": result.warnings}


def test_openapi_document_is_valid_and_complete() -> None:
    built = build()
    document = built["document"]
    validate(document)  # openapi-spec-validator, OpenAPI 3.1
    assert document["openapi"] == "3.1.0"
    schemas = document["components"]["schemas"]
    assert {"Invoice", "InvoiceCreate", "InvoiceUpdate", "InvoicePage", "Supplier", "Error"} <= set(
        schemas
    )
    invoice = schemas["Invoice"]
    assert invoice["properties"]["id"]["readOnly"] is True
    assert invoice["properties"]["status"]["enum"] == ["submitted", "approved"]
    assert invoice["properties"]["total_amount"]["format"] == "decimal"
    assert "due_date" not in invoice["required"]
    assert "id" not in schemas["InvoiceCreate"]["properties"]
    assert schemas["InvoiceUpdate"]["minProperties"] == 1

    paths = document["paths"]
    create = paths["/invoices"]["post"]
    assert create["requestBody"]["content"]["application/json"]["schema"]["$ref"].endswith(
        "InvoiceCreate"
    )
    assert "201" in create["responses"] and "422" in create["responses"]
    assert create["x-evidence"] == ["chunk-1"]
    listing = paths["/invoices"]["get"]
    assert listing["responses"]["200"]["content"]["application/json"]["schema"]["$ref"].endswith(
        "InvoicePage"
    )
    page_size = next(p for p in listing["parameters"] if p["name"] == "pageSize")
    assert page_size["schema"]["maximum"] == 100
    assert paths["/invoices/{id}"]["delete"]["responses"]["204"] == {"description": "No content"}
    assert paths["/invoices/{id}"]["patch"]["requestBody"]["content"]["application/json"]["schema"][
        "$ref"
    ].endswith("InvoiceUpdate")
    assert document["x-generated"] == {"analysisId": "a1"}


def test_openapi_handles_unknown_models_and_duplicate_operation_ids() -> None:
    built = build()
    supplier = built["document"]["paths"]["/suppliers/{supplierId}"]["get"]
    assert supplier["responses"]["200"]["content"]["application/json"]["schema"] == {
        "type": "object"
    }
    assert supplier["operationId"] == "createInvoice2"
    assert any("Ghost" in w for w in built["warnings"])
    approve = built["document"]["paths"]["/invoices/{id}/approve"]["post"]
    assert approve["operationId"] == "postInvoicesByIdApprove"


def test_openapi_output_is_deterministic() -> None:
    first = json.dumps(build()["document"])
    second = OpenAPIBuilder().build(
        title="Portal API",
        entities=list(reversed(ENTITIES)),
        endpoints=ENDPOINTS,
        extensions={"analysisId": "a1"},
    )
    assert first == json.dumps(second.document)


def test_markdown_docs_render_endpoints_and_models() -> None:
    markdown = render_markdown(build()["document"])
    assert markdown.startswith("# Portal API")
    assert "| POST | `/invoices` | Submit an invoice | 0.8 |" in markdown
    assert "### `GET /invoices`" in markdown
    assert "User stories: US-001" in markdown
    assert "| `total_amount` | string (decimal) | yes |" in markdown
    assert "many-to-one -> `Supplier`" in markdown
    empty = render_markdown(OpenAPIBuilder().build(title="E", entities=[], endpoints=[]).document)
    assert "No endpoints were generated" in empty


def test_gherkin_export() -> None:
    stories = [
        StoryInfo("US-001", "Approve invoices", "approver", "approve invoices", "pay on time")
    ]
    scenarios = [
        ScenarioInfo(
            "TC-002",
            "US-001",
            "Rejected without role",
            "negative",
            "medium",
            ["a viewer"],
            ["they approve"],
            ["access is denied", "nothing changes"],
            ["Assumption: role restricted"],
        ),
        ScenarioInfo(
            "TC-001",
            "US-001",
            "Approve\nhappy path",
            "positive",
            "high",
            ["Given an invoice", "it is pending"],
            ["the approver approves it."],
            ["it is approved"],
        ),
        ScenarioInfo("TC-003", "US-404", "Orphan", "positive", "low", ["g"], ["w"], ["t"]),
    ]
    feature = export_feature(stories, scenarios, header="Generated by test")
    assert feature.startswith("# Generated by test\n")
    assert (
        "@US-001\nFeature: Approve invoices\n  As a approver\n  I want to approve invoices"
        in feature
    )
    assert "  So that pay on time" in feature
    assert feature.index("TC-001") < feature.index("TC-002")  # sorted scenarios
    assert "  @TC-001 @positive @priority-high\n  Scenario: Approve happy path" in feature
    assert (
        "    Given an invoice\n    And it is pending\n    When the approver approves it" in feature
    )
    assert "  # Precondition: Assumption: role restricted" in feature
    assert "Feature: US-404" in feature  # unknown story still exported, titled by id
