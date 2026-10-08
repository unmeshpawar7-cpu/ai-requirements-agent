# API examples

Real requests and responses captured by `scripts/capture_api_examples.py`, which runs the
application in-process (SQLite, `LLM_PROVIDER=offline`, `EMBEDDING_PROVIDER=hashing`) against
`samples/field-service-work-orders.md`. Ids and timestamps are from that run. Long arrays and
strings are trimmed and marked as such; complete outputs are in `samples/generated/`.

Because the offline provider is a rule-based heuristic (not an LLM), the artefacts below show
what the pipeline *structure* produces without any API key. With `LLM_PROVIDER=openai` or
`anthropic` the same endpoints return model-generated content that goes through the same
validation, grounding and confidence scoring.

```bash
export TOKEN=...   # from POST /api/auth/token
```

### Register (first account becomes admin)

```bash
curl -s -X POST 'http://localhost:8000/api/auth/register' \
  -H 'Content-Type: application/json' \
  -d '{"email": "ana@example.com", "password": "correct-horse-battery", "fullName": "Ana"}'
```

Response: `201 Created` - `Location: /api/users/01b601d2-e666-4a2b-a808-52fb12d09b6c`

```json
{
  "success": true,
  "data": {
    "id": "01b601d2-e666-4a2b-a808-52fb12d09b6c",
    "email": "ana@example.com",
    "fullName": "Ana",
    "role": "admin",
    "isActive": true,
    "createdAt": "2026-10-08T08:23:59.760348Z"
  },
  "message": "Account created"
}
```

### Get an access token

```bash
curl -s -X POST 'http://localhost:8000/api/auth/token' \
  -H 'Content-Type: application/json' \
  -d '{"email": "ana@example.com", "password": "correct-horse-battery"}'
```

Response: `200 OK`

```json
{
  "success": true,
  "data": {
    "accessToken": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIwMWI2MDFkMi1lNjY2LTRhMmItYTgwOC01MmZiMTJkMDliNmMiLCJyb2xlIjoiYWRtaW4iLCJpc3MiOiJhaS1yZXF1aXJlbWVudHMtYWdlbnQiLCJhdWQiOiJhaS1yZXF1aXJlbWVudHMtYWdlbnQtYXBpIiwiaWF0IjoxNzkxNDQ3ODM5LCJuYmYiOjE3OTE0NDc4MzksImV4cCI6MTc5MTQ1MTQzOSwianRpIjo... [91 chars omitted]",
    "tokenType": "bearer",
    "expiresIn": 3600
  },
  "message": null
}
```

### Create a project

```bash
curl -s -X POST 'http://localhost:8000/api/projects' \
  -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"name": "FieldOps", "description": "Field-service work order management"}'
```

Response: `201 Created` - `Location: /api/projects/271d8d38-3253-4e36-a806-7eca2a683961`

```json
{
  "success": true,
  "data": {
    "id": "271d8d38-3253-4e36-a806-7eca2a683961",
    "name": "FieldOps",
    "description": "Field-service work order management",
    "status": "active",
    "ownerId": "01b601d2-e666-4a2b-a808-52fb12d09b6c",
    "documentCount": 0,
    "analysisCount": 0,
    "createdAt": "2026-10-08T08:23:59.810088Z",
    "updatedAt": "2026-10-08T08:23:59.810091Z",
    "archivedAt": null
  },
  "message": "Project created"
}
```

### Upload a requirement document (processed in the background)

```bash
curl -s -X POST 'http://localhost:8000/api/projects/271d8d38-3253-4e36-a806-7eca2a683961/documents' \
  -H "Authorization: Bearer $TOKEN" \
  -F 'file=@samples/field-service-work-orders.md'
```

Response: `202 Accepted` - `Location: /api/documents/1cf56989-f18a-4b52-8656-faffd74757db`

```json
{
  "success": true,
  "data": {
    "id": "1cf56989-f18a-4b52-8656-faffd74757db",
    "projectId": "271d8d38-3253-4e36-a806-7eca2a683961",
    "title": "field-service-work-orders.md",
    "filename": "field-service-work-orders.md",
    "contentType": "text/markdown",
    "source": "upload",
    "sizeBytes": 4424,
    "checksum": "1bd3c5ffb79c8aaed697004a5c3d653f3663f2ae292b944205e72feec770c05f",
    "status": "pending",
    "errorMessage": null,
    "charCount": 0,
    "tokenEstimate": 0,
    "chunkCount": 0,
    "createdAt": "2026-10-08T08:23:59.825725Z",
    "processedAt": null
  },
  "message": "Document accepted for processing"
}
```

### Poll the document until it is ready

```bash
curl -s -X GET 'http://localhost:8000/api/documents/1cf56989-f18a-4b52-8656-faffd74757db' \
  -H "Authorization: Bearer $TOKEN"
```

Response: `200 OK`

```json
{
  "success": true,
  "data": {
    "id": "1cf56989-f18a-4b52-8656-faffd74757db",
    "projectId": "271d8d38-3253-4e36-a806-7eca2a683961",
    "title": "field-service-work-orders.md",
    "filename": "field-service-work-orders.md",
    "contentType": "text/markdown",
    "source": "upload",
    "sizeBytes": 4424,
    "checksum": "1bd3c5ffb79c8aaed697004a5c3d653f3663f2ae292b944205e72feec770c05f",
    "status": "ready",
    "errorMessage": null,
    "charCount": 4423,
    "tokenEstimate": 1106,
    "chunkCount": 14,
    "createdAt": "2026-10-08T08:23:59.825725Z",
    "processedAt": "2026-10-08T08:23:59.857082Z"
  },
  "message": null
}
```

### Duplicate upload is rejected

The SHA-256 checksum is unique per project.

```bash
curl -s -X POST 'http://localhost:8000/api/projects/271d8d38-3253-4e36-a806-7eca2a683961/documents' \
  -H "Authorization: Bearer $TOKEN" \
  -F 'file=@samples/field-service-work-orders.md'
```

Response: `409 Conflict`

```json
{
  "success": false,
  "message": "This document was already uploaded to the project (document 1cf56989-f18a-4b52-8656-faffd74757db)",
  "errors": [],
  "traceId": "36ed7f715d0441eabcfdd35e146af8e5"
}
```

### Semantic search (top-k + MMR)

```bash
curl -s -X POST 'http://localhost:8000/api/projects/271d8d38-3253-4e36-a806-7eca2a683961/search' \
  -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"query": "dispatch board load time", "topK": 2, "useMmr": true}'
```

Response: `200 OK`

```json
{
  "success": true,
  "data": {
    "query": "dispatch board load time",
    "embeddingProvider": "hashing:hashing-v1",
    "vectorBackend": "memory",
    "results": [
      {
        "rank": 1,
        "score": 0.272438,
        "chunkId": "d1769b56-ef36-4cb9-901a-0783913a15a6",
        "documentId": "1cf56989-f18a-4b52-8656-faffd74757db",
        "documentTitle": "field-service-work-orders.md",
        "ordinal": 9,
        "sectionPath": "FieldOps Work Order Management - Software Requirements Specification > 3 Non-Functional Requirements > 3.1 Performance",
        "text": "## 3 Non-Functional Requirements\n\n### 3.1 Performance\n\n- The dispatch board shall load within 2 seconds for up to 500 open work orders.\n- Search results should be returned in under 500 ms at the 95th percentile."
      },
      {
        "rank": 2,
        "score": 0.195679,
        "chunkId": "78a8f02c-bf40-46d1-837b-01529802e68f",
        "documentId": "1cf56989-f18a-4b52-8656-faffd74757db",
        "documentTitle": "field-service-work-orders.md",
        "ordinal": 5,
        "sectionPath": "FieldOps Work Order Management - Software Requirements Specification > 2 Functional Requirements > 2.2 Scheduling and dispatch",
        "text": "### 2.2 Scheduling and dispatch\n\n- The system shall allow dispatchers to assign work orders to technicians based on skills and availability.\n- Each technician has a name, email, phone number, skills, home depot and working hours.\n- Dispatchers shall be able to reassign a work ord... [256 chars omitted]"
      }
    ]
  },
  "message": null
}
```

### Start an analysis (202, runs in the background)

```bash
curl -s -X POST 'http://localhost:8000/api/projects/271d8d38-3253-4e36-a806-7eca2a683961/analyses' \
  -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"sections": ["functional_requirements", "non_functional_requirements", "user_stories", "entities", "api_endpoints", "implementation_tasks"], "focus": "work order scheduling and dispatch"}'
```

Response: `202 Accepted` - `Location: /api/analyses/426f6a14-4c71-4645-98ad-54eecff839d6`

```json
{
  "success": true,
  "data": {
    "id": "426f6a14-4c71-4645-98ad-54eecff839d6",
    "projectId": "271d8d38-3253-4e36-a806-7eca2a683961",
    "status": "queued",
    "sections": [
      "functional_requirements",
      "non_functional_requirements",
      "... 4 more item(s) omitted"
    ],
    "focus": "work order scheduling and dispatch",
    "promptName": "requirements_analysis",
    "promptVersion": 1,
    "provider": "offline",
    "model": "heuristic-v1",
    "confidence": null,
    "itemCounts": {},
    "errorMessage": null,
    "createdAt": "2026-10-08T08:23:59.918436Z",
    "completedAt": null
  },
  "message": "Analysis queued"
}
```

### AI endpoints are rate limited per user

This capture runs with RATE_LIMIT_AI_CAPACITY=1 to show the 429 envelope.

```bash
curl -s -X POST 'http://localhost:8000/api/projects/271d8d38-3253-4e36-a806-7eca2a683961/analyses' \
  -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"sections": ["functional_requirements", "non_functional_requirements", "user_stories", "entities", "api_endpoints", "implementation_tasks"], "focus": "work order scheduling and dispatch"}'
```

Response: `429 Too Many Requests`

```json
{
  "success": false,
  "message": "Rate limit exceeded; retry in 60s",
  "errors": [],
  "traceId": "305417cbc4ae455bb212e41bd8c4f0a1"
}
```

### Get the analysis (grounded artefacts grouped by type)

Arrays are trimmed to one entry here; the full response is in `samples/generated/field-service/analysis.json`.

```bash
curl -s -X GET 'http://localhost:8000/api/analyses/426f6a14-4c71-4645-98ad-54eecff839d6' \
  -H "Authorization: Bearer $TOKEN"
```

Response: `200 OK`

```json
{
  "success": true,
  "data": {
    "id": "426f6a14-4c71-4645-98ad-54eecff839d6",
    "projectId": "271d8d38-3253-4e36-a806-7eca2a683961",
    "status": "completed",
    "sections": [
      "functional_requirements",
      "... 5 more item(s) omitted"
    ],
    "focus": "work order scheduling and dispatch",
    "promptName": "requirements_analysis",
    "promptVersion": 1,
    "provider": "offline",
    "model": "heuristic-v1",
    "confidence": 0.695,
    "itemCounts": {
      "functional_requirement": 14,
      "non_functional_requirement": 9,
      "user_story": 10,
      "entity": 3,
      "api_endpoint": 8,
      "implementation_task": 11
    },
    "errorMessage": null,
    "createdAt": "2026-10-08T08:23:59.918436Z",
    "completedAt": "2026-10-08T08:23:59.987964Z",
    "summary": "Offline heuristic analysis (rule-based, no LLM) of 14 evidence block(s) from 1 document(s): found 14 functional and 9 non-functional requirement(s), 10 user stories, 3 entities and 8 endpoint suggestion(s). Retrieval was steered by the focus 'work order scheduling and dispatch'.",
    "openQuestions": [
      "Clarify 'easy to' with a measurable criterion: \"The mobile app should be easy to use with gloves on.\" (E13)",
      "... 2 more item(s) omitted"
    ],
    "embeddingModel": "hashing:hashing-v1",
    "retrievalParams": {
      "topKPerQuery": 8,
      "maxEvidence": 24,
      "minScore": 0.02,
      "mmrLambda": 0.7,
      "contextTokens": 6000,
      "vectorBackend": "memory"
    },
    "evidence": [
      {
        "label": "E1",
        "chunkId": "a48cefce-b27f-47c7-8b62-f7c33d09ceff",
        "documentId": "1cf56989-f18a-4b52-8656-faffd74757db",
        "documentTitle": "field-service-work-orders.md",
        "sectionPath": "FieldOps Work Order Management - Software Requirements Specification",
        "score": 0.293502
      },
      "... 13 more item(s) omitted"
    ],
    "groundingReport": {
      "kept": 55,
      "dropped": [],
      "flagged": [],
      "unknownCitations": [],
      "removedReferences": [],
      "evidenceBlocks": 14,
      "contextTokens": 1884,
      "skippedForBudget": 0
    },
    "repairAttempts": 0,
    "inputTokens": 6023,
    "outputTokens": 4951,
    "startedAt": "2026-10-08T08:23:59.929474Z",
    "items": {
      "functional_requirement": [
        {
          "id": "95a90501-d81c-43d3-a7bd-fff095108377",
          "itemType": "functional_requirement",
          "refCode": "FR-001",
          "title": "Create work orders from customer service requests",
          "confidence": 0.813,
          "grounded": true,
          "groundingIssues": [],
          "evidenceChunkIds": [
            "ee22d4e5-a034-456b-816b-7f23d16796a4"
          ],
          "confidenceBreakdown": {
            "retrieval": 0.5955,
            "support": 1.0,
            "model": 0.85,
            "repairFactor": 1.0,
            "groundingFactor": 1.0,
            "final": 0.813
          },
          "payload": {
            "evidence": [
              "ee22d4e5-a034-456b-816b-7f23d16796a4"
            ],
            "confidence": 0.85,
            "id": "FR-001",
            "title": "Create work orders from customer service requests",
            "description": "The system shall allow dispatchers to create work orders from customer service requests.",
            "priority": "must"
          }
        },
        "... 13 more item(s) omitted"
      ],
      "non_functional_requirement": [
        {
          "id": "01e86b0e-552a-4e7e-8c2a-a1d75e50d1d4",
          "itemType": "non_functional_requirement",
          "refCode": "NFR-001",
          "title": "The dispatch board shall load within 2 seconds for up to 500 open work orders.",
          "confidence": 0.826,
          "grounded": true,
          "groundingIssues": [],
          "evidenceChunkIds": [
            "d1769b56-ef36-4cb9-901a-0783913a15a6"
          ],
          "confidenceBreakdown": {
            "retrieval": 0.6325,
            "support": 1.0,
            "model": 0.85,
            "repairFactor": 1.0,
            "groundingFactor": 1.0,
            "final": 0.826
          },
          "payload": {
            "evidence": [
              "d1769b56-ef36-4cb9-901a-0783913a15a6"
            ],
            "confidence": 0.85,
            "id": "NFR-001",
            "category": "performance",
            "description": "The dispatch board shall load within 2 seconds for up to 500 open work orders.",
            "metric": "within 2 seconds"
          }
        },
        "... 8 more item(s) omitted"
      ],
      "user_story": [
        {
          "id": "fd34477e-0730-4f75-bc2a-ff93f8f34bef",
          "itemType": "user_story",
          "refCode": "US-001",
          "title": "Receive a notification when the technician is on the way",
          "confidence": 0.791,
          "grounded": true,
          "groundingIssues": [],
          "evidenceChunkIds": [
            "5c94e7d0-10c6-48f3-be28-181bb4621c1e"
          ],
          "confidenceBreakdown": {
            "retrieval": 0.4899,
            "support": 1.0,
            "model": 0.9,
            "repairFactor": 1.0,
            "groundingFactor": 1.0,
            "final": 0.791
          },
          "payload": {
            "evidence": [
              "5c94e7d0-10c6-48f3-be28-181bb4621c1e"
            ],
            "confidence": 0.9,
            "id": "US-001",
            "title": "Receive a notification when the technician is on the way",
            "role": "customer",
            "goal": "receive a notification when the technician is on the way",
            "benefit": "I can make sure someone is at the site",
            "acceptanceCriteria": [
              {
                "given": "a work order is in progress",
                "when": "the technician marks it as en route",
                "then": "the customer receives an SMS with the estimated arrival time"
              }
            ],
            "relatedRequirements": []
          }
        },
        "... 9 more item(s) omitted"
      ],
      "entity": [
        {
          "id": "f70bf7f0-6ce8-43c4-bd25-f581f22cd2cd",
          "itemType": "entity",
          "refCode": "WorkOrder",
          "title": "WorkOrder",
          "confidence": 0.779,
          "grounded": true,
          "groundingIssues": [],
          "evidenceChunkIds": [
            "ee22d4e5-a034-456b-816b-7f23d16796a4",
            "... 3 more item(s) omitted"
          ],
          "confidenceBreakdown": {
            "retrieval": 0.6135,
            "support": 0.9259,
            "model": 0.8,
            "repairFactor": 1.0,
            "groundingFactor": 1.0,
            "final": 0.779
          },
          "payload": {
            "evidence": [
              "ee22d4e5-a034-456b-816b-7f23d16796a4",
              "... 3 more item(s) omitted"
            ],
            "confidence": 0.8,
            "name": "WorkOrder",
            "description": "Each work order has a reference number, customer, site address, priority (low, normal, high, emergency), status (new, scheduled, in progress, on hold, completed, cancelled), problem description, scheduled date and assigned technician.",
            "fields": [
              {
                "name": "id",
                "type": "uuid",
                "required": true,
                "description": "Primary key",
                "enumValues": []
              },
              "... 9 more item(s) omitted"
            ],
            "relationships": [
              {
                "target": "PartUsed",
                "kind": "one-to-many",
                "description": null
              },
              "... 1 more item(s) omitted"
            ]
          }
        },
        "... 2 more item(s) omitted"
      ],
      "api_endpoint": [
        {
          "id": "208a288f-67c5-4687-9f08-257b2803b2d4",
          "itemType": "api_endpoint",
          "refCode": "GET /work-orders",
          "title": "List work orders",
          "confidence": 0.567,
          "grounded": true,
          "groundingIssues": [],
          "evidenceChunkIds": [
            "bcfb8a83-a19d-4cbb-bb61-41d98d95c45e"
          ],
          "confidenceBreakdown": {
            "retrieval": 0.439,
            "support": 0.6667,
            "model": 0.6,
            "repairFactor": 1.0,
            "groundingFactor": 1.0,
            "final": 0.567
          },
          "payload": {
            "evidence": [
              "bcfb8a83-a19d-4cbb-bb61-41d98d95c45e"
            ],
            "confidence": 0.6,
            "method": "GET",
            "path": "/work-orders",
            "summary": "List work orders",
            "operationId": "listWorkOrders",
            "requestModel": null,
            "responseModel": "WorkOrder",
            "responseIsCollection": true,
            "successStatus": 200,
            "queryParams": [
              {
                "name": "page",
                "type": "integer",
                "required": false,
                "description": null
              },
              "... 2 more item(s) omitted"
            ],
            "relatedStories": [
              "US-007",
              "... 1 more item(s) omitted"
            ]
          }
        },
        "... 7 more item(s) omitted"
      ],
      "implementation_task": [
        {
          "id": "71b642b5-884e-4254-9118-e891e6633f56",
          "itemType": "implementation_task",
          "refCode": "T-001",
          "title": "Model WorkOrder entity and migration",
          "confidence": 0.518,
          "grounded": true,
          "groundingIssues": [],
          "evidenceChunkIds": [
            "ee22d4e5-a034-456b-816b-7f23d16796a4",
            "... 3 more item(s) omitted"
          ],
          "confidenceBreakdown": {
            "retrieval": 0.6135,
            "support": 0.1818,
            "model": 0.8,
            "repairFactor": 1.0,
            "groundingFactor": 1.0,
            "final": 0.518
          },
          "payload": {
            "evidence": [
              "ee22d4e5-a034-456b-816b-7f23d16796a4",
              "... 3 more item(s) omitted"
            ],
            "confidence": 0.8,
            "id": "T-001",
            "title": "Model WorkOrder entity and migration",
            "description": "Create the WorkOrder table/ORM model with 10 fields and its relationships, plus a database migration.",
            "estimate": "M",
            "dependsOn": [],
            "relatedItems": [
              "WorkOrder"
            ]
          }
        },
        "... 10 more item(s) omitted"
      ]
    }
  },
  "message": null
}
```

### Generated OpenAPI 3.1 document

Excerpt (two paths, two schemas); the full document is in `samples/generated/field-service/openapi.json` and passes openapi-spec-validator.

```bash
curl -s -X GET 'http://localhost:8000/api/analyses/426f6a14-4c71-4645-98ad-54eecff839d6/openapi.json' \
  -H "Authorization: Bearer $TOKEN"
```

Response: `200 OK`

```json
{
  "openapi": "3.1.0",
  "info": {
    "title": "FieldOps API",
    "version": "0.1.0",
    "description": "Draft API generated from the requirement documents of project 'FieldOps'. Review before implementing."
  },
  "paths": {
    "/work-orders": {
      "get": {
        "tags": [
          "WorkOrder"
        ],
        "summary": "List work orders",
        "operationId": "listWorkOrders",
        "parameters": [
          {
            "name": "page",
            "in": "query",
            "required": false,
            "schema": {
              "type": "integer",
              "format": "int64",
              "minimum": 1,
              "default": 1
            }
          },
          {
            "name": "pageSize",
            "in": "query",
            "required": false,
            "schema": {
              "type": "integer",
              "format": "int64",
              "minimum": 1,
              "maximum": 100,
              "default": 20
            }
          },
          {
            "name": "status",
            "in": "query",
            "required": false,
            "schema": {
              "type": "string"
            }
          }
        ],
        "responses": {
          "200": {
            "description": "Successful response",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/WorkOrderPage"
                }
              }
            }
          },
          "401": {
            "$ref": "#/components/responses/Unauthorized"
          }
        },
        "x-evidence": [
          "bcfb8a83-a19d-4cbb-bb61-41d98d95c45e"
        ],
        "x-confidence": 0.6,
        "x-related-stories": [
          "US-007",
          "US-008"
        ]
      },
      "post": {
        "tags": [
          "WorkOrder"
        ],
        "summary": "Create a work order",
        "operationId": "createWorkOrder",
        "requestBody": {
          "required": true,
          "content": {
            "application/json": {
              "schema": {
                "$ref": "#/components/schemas/WorkOrderCreate"
              }
            }
          }
        },
        "responses": {
          "201": {
            "description": "Successful response",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/WorkOrder"
                }
              }
            }
          },
          "400": {
            "$ref": "#/components/responses/BadRequest"
          },
          "401": {
            "$ref": "#/components/responses/Unauthorized"
          },
          "422": {
            "$ref": "#/components/responses/ValidationError"
          }
        },
        "x-evidence": [
          "84cc8071-140d-4331-9ee6-826bce68586b",
          "ee22d4e5-a034-456b-816b-7f23d16796a4"
        ],
        "x-confidence": 0.6,
        "x-related-stories": [
          "US-002"
        ]
      }
    },
    "/work-orders/{id}": {
      "patch": {
        "tags": [
          "WorkOrder"
        ],
        "summary": "Update a work order",
        "operationId": "updateWorkOrder",
        "parameters": [
          {
            "name": "id",
            "in": "path",
            "required": true,
            "schema": {
              "type": "string",
              "format": "uuid"
            }
          }
        ],
        "requestBody": {
          "required": true,
          "content": {
            "application/json": {
              "schema": {
                "$ref": "#/components/schemas/WorkOrderUpdate"
              }
            }
          }
        },
        "responses": {
          "200": {
            "description": "Successful response",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/WorkOrder"
                }
              }
            }
          },
          "400": {
            "$ref": "#/components/responses/BadRequest"
          },
          "401": {
            "$ref": "#/components/responses/Unauthorized"
          },
          "404": {
            "$ref": "#/components/responses/NotFound"
          },
          "422": {
            "$ref": "#/components/responses/ValidationError"
          }
        },
        "x-evidence": [
          "ee22d4e5-a034-456b-816b-7f23d16796a4"
        ],
        "x-confidence": 0.6,
        "x-related-stories": [
          "US-003"
        ]
      }
    }
  },
  "components": {
    "schemas": {
      "Error": {
        "type": "object",
        "required": [
          "success",
          "message"
        ],
        "properties": {
          "success": {
            "type": "boolean",
            "const": false
          },
          "message": {
            "type": "string"
          },
          "errors": {
            "type": "array",
            "items": {
              "type": "object",
              "properties": {
                "field": {
                  "type": [
                    "string",
                    "null"
                  ]
                },
                "message": {
                  "type": "string"
                }
              },
              "required": [
                "message"
              ]
            }
          },
          "traceId": {
            "type": [
              "string",
              "null"
            ]
          }
        }
      },
      "PartUsed": {
        "type": "object",
        "description": "Each part used has a part number, description, quantity and unit cost.",
        "properties": {
          "id": {
            "type": "string",
            "format": "uuid",
            "description": "Primary key",
            "readOnly": true
          },
          "part_number": {
            "type": "string",
            "description": "part number"
          },
          "description": {
            "type": "string",
            "description": "description"
          },
          "quantity": {
            "type": "integer",
            "format": "int64",
            "description": "quantity"
          },
          "unit_cost": {
            "type": "string",
            "format": "decimal",
            "pattern": "^-?\\d+(\\.\\d+)?$",
            "description": "unit cost"
          },
          "created_at": {
            "type": "string",
            "format": "date-time",
            "description": "Creation timestamp",
            "readOnly": true
          }
        },
        "required": [
          "created_at",
          "description",
          "id",
          "... 3 more item(s) omitted"
        ],
        "x-relationships": [
          {
            "target": "WorkOrder",
            "kind": "many-to-one"
          }
        ]
      }
    }
  },
  "x-generated": {
    "generator": "ai-requirements-agent",
    "analysisId": "426f6a14-4c71-4645-98ad-54eecff839d6",
    "prompt": "requirements_analysis@v1",
    "provider": "offline:heuristic-v1",
    "warnings": []
  }
}
```

### Generated Markdown API docs

```bash
curl -s -X GET 'http://localhost:8000/api/analyses/426f6a14-4c71-4645-98ad-54eecff839d6/api-docs.md' \
  -H "Authorization: Bearer $TOKEN"
```

Response: `200 OK`

```markdown
# FieldOps API

Version `0.1.0` - OpenAPI `3.1.0`

Draft API generated from the requirement documents of project 'FieldOps'. Review before implementing.

> Generated by ai-requirements-agent. Every operation lists the evidence chunk ids it
> was derived from and a confidence score; review before implementation.

## Authentication

All operations require `Authorization: Bearer <JWT>`.

## Endpoints

| Method | Path | Summary | Confidence |
| --- | --- | --- | --- |
| GET | `/work-orders` | List work orders | 0.6 |
| POST | `/work-orders` | Create a work order | 0.6 |
| PATCH | `/work-orders/{id}` | Update a work order | 0.6 |
| POST | `/work-orders/{id}/assign` | Assign a work order | 0.65 |
| POST | `/work-orders/{id}/cancel` | Cancel a work order | 0.65 |
| POST | `/work-orders/{id}/complete` | Complete a work order | 0.65 |
| POST | `/work-orders/{id}/reassign` | Reassign a work order | 0.65 |
| POST | `/work-orders/{id}/schedule` | Schedule a work order | 0.65 |

## WorkOrder

### `GET /work-orders`

... (209 more lines)
```

### Generate test cases from acceptance criteria

```bash
curl -s -X POST 'http://localhost:8000/api/analyses/426f6a14-4c71-4645-98ad-54eecff839d6/test-cases' \
  -H "Authorization: Bearer $TOKEN"
```

Response: `201 Created` - `Location: /api/analyses/426f6a14-4c71-4645-98ad-54eecff839d6/test-cases`

```json
{
  "success": true,
  "data": {
    "analysisId": "426f6a14-4c71-4645-98ad-54eecff839d6",
    "generated": 19,
    "dropped": [],
    "testCases": [
      {
        "id": "df2dc431-38c9-4f7d-b9d5-17b3a4f13a34",
        "refCode": "TC-001",
        "storyRef": "US-001",
        "title": "Receive a notification when the technician is on the way - criterion 1 holds",
        "caseType": "positive",
        "priority": "high",
        "confidence": 0.8,
        "promptName": "test_case_generation",
        "promptVersion": 1,
        "payload": {
          "id": "TC-001",
          "title": "Receive a notification when the technician is on the way - criterion 1 holds",
          "storyId": "US-001",
          "criterionIndex": 0,
          "type": "positive",
          "priority": "high",
          "preconditions": [
            "A user with the 'customer' role exists"
          ],
          "given": [
            "a work order is in progress"
          ],
          "when": [
            "the technician marks it as en route"
          ],
          "then": [
            "the customer receives an SMS with the estimated arrival time"
          ],
          "confidence": 0.8
        },
        "createdAt": "2026-10-08T08:24:00.045375Z"
      },
      {
        "id": "9f4f7fe0-db2d-4228-8193-5999d0b998c6",
        "refCode": "TC-002",
        "storyRef": "US-002",
        "title": "Create work orders from customer service requests - criterion 1 holds",
        "caseType": "positive",
        "priority": "high",
        "confidence": 0.8,
        "promptName": "test_case_generation",
        "promptVersion": 1,
        "payload": {
          "id": "TC-002",
          "title": "Create work orders from customer service requests - criterion 1 holds",
          "storyId": "US-002",
          "criterionIndex": 0,
          "type": "positive",
          "priority": "high",
          "preconditions": [
            "A user with the 'dispatcher' role exists"
          ],
          "given": [
            "an authenticated dispatcher with access to the feature"
          ],
          "when": [
            "the dispatcher creates work orders from customer service requests"
          ],
          "then": [
            "the system confirms that the work orders were created"
          ],
          "confidence": 0.8
        },
        "createdAt": "2026-10-08T08:24:00.045382Z"
      },
      "... 17 more item(s) omitted"
    ]
  },
  "message": null
}
```

### Export test cases as Gherkin

```bash
curl -s -X GET 'http://localhost:8000/api/analyses/426f6a14-4c71-4645-98ad-54eecff839d6/test-cases.feature' \
  -H "Authorization: Bearer $TOKEN"
```

Response: `200 OK`

```gherkin
# Generated by ai-requirements-agent from analysis 426f6a14-4c71-4645-98ad-54eecff839d6
# Prompt test_case_generation v1

@US-001
Feature: Receive a notification when the technician is on the way
  As a customer
  I want to receive a notification when the technician is on the way
  So that I can make sure someone is at the site

  # Precondition: A user with the 'customer' role exists
  @TC-001 @positive @priority-high
  Scenario: Receive a notification when the technician is on the way - criterion 1 holds
    Given a work order is in progress
    When the technician marks it as en route
    Then the customer receives an SMS with the estimated arrival time

@US-002
Feature: Create work orders from customer service requests
  As a dispatcher
  I want to create work orders from customer service requests

  # Precondition: A user with the 'dispatcher' role exists
  @TC-002 @positive @priority-high
  Scenario: Create work orders from customer service requests - criterion 1 holds
    Given an authenticated dispatcher with access to the feature
... (170 more lines)
```

### AI usage report (admin)

```bash
curl -s -X GET 'http://localhost:8000/api/ai/usage' \
  -H "Authorization: Bearer $TOKEN"
```

Response: `200 OK`

```json
{
  "success": true,
  "data": {
    "totalCalls": 5,
    "totalFailures": 0,
    "totalInputTokens": 9664,
    "totalOutputTokens": 7223,
    "totalEstimatedCostUsd": 0.0,
    "groups": [
      {
        "operation": "embedding",
        "provider": "hashing",
        "model": "hashing-v1",
        "promptName": null,
        "promptVersion": null,
        "calls": 3,
        "failures": 0,
        "inputTokens": 1634,
        "outputTokens": 0,
        "estimatedCostUsd": 0.0,
        "avgLatencyMs": 3.44
      },
      {
        "operation": "chat",
        "provider": "offline",
        "model": "heuristic-v1",
        "promptName": "requirements_analysis",
        "promptVersion": 1,
        "calls": 1,
        "failures": 0,
        "inputTokens": 6023,
        "outputTokens": 4951,
        "estimatedCostUsd": 0.0,
        "avgLatencyMs": 15.96
      },
      {
        "operation": "chat",
        "provider": "offline",
        "model": "heuristic-v1",
        "promptName": "test_case_generation",
        "promptVersion": 1,
        "calls": 1,
        "failures": 0,
        "inputTokens": 2007,
        "outputTokens": 2272,
        "estimatedCostUsd": 0.0,
        "avgLatencyMs": 0.92
      }
    ]
  },
  "message": null
}
```

### Validation error envelope

```bash
curl -s -X POST 'http://localhost:8000/api/projects/271d8d38-3253-4e36-a806-7eca2a683961/search' \
  -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"query": "", "topK": 500}'
```

Response: `422 Unprocessable Entity`

```json
{
  "success": false,
  "message": "Validation failed",
  "errors": [
    {
      "field": "query",
      "message": "String should have at least 2 characters"
    },
    {
      "field": "topK",
      "message": "Input should be less than or equal to 50"
    }
  ],
  "traceId": "651f61079b674b25bf371ef72332b2bb"
}
```

### Missing token

```bash
curl -s -X GET 'http://localhost:8000/api/projects'
```

Response: `401 Unauthorized`

```json
{
  "success": false,
  "message": "Missing bearer token",
  "errors": [],
  "traceId": "5759ce2c52f84b299a028eeef72146f7"
}
```
