.DEFAULT_GOAL := help
PYTHON ?= python3
VENV := .venv
BIN := $(VENV)/bin

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-14s %s\n", $$1, $$2}'

install: ## Create the virtualenv and install the app with dev extras
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -e ".[dev]"

lint: ## Ruff lint + format check
	$(BIN)/ruff check src tests scripts
	$(BIN)/ruff format --check src tests scripts

format: ## Apply ruff formatting and safe fixes
	$(BIN)/ruff check --fix src tests scripts
	$(BIN)/ruff format src tests scripts

test: ## Run the test suite (Postgres tests run when TEST_DATABASE_URL is set)
	$(BIN)/pytest

coverage: ## Run tests with a coverage report
	$(BIN)/pytest --cov --cov-report=term-missing --cov-report=xml

run: ## Run the API locally with SQLite + offline AI (http://localhost:8000/docs)
	DATABASE_AUTO_CREATE=true $(BIN)/uvicorn requirements_agent.main:create_app --factory --reload

migrate: ## Apply Alembic migrations to DATABASE_URL
	$(BIN)/alembic upgrade head

examples: ## Regenerate docs/api-examples.md by running the app in-process
	$(BIN)/python scripts/capture_api_examples.py

docker-up: ## Start API + pgvector with docker compose
	docker compose up --build -d

docker-down: ## Stop the compose stack
	docker compose down

.PHONY: help install lint format test coverage run migrate examples docker-up docker-down
