# syntax=docker/dockerfile:1.7
# ---------------------------------------------------------------------------- build stage
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

COPY pyproject.toml README.md ./
COPY src ./src

# Optional provider SDKs, e.g. --build-arg EXTRAS=openai,anthropic
ARG EXTRAS=""
RUN pip install --upgrade pip \
 && if [ -n "${EXTRAS}" ]; then pip install ".[${EXTRAS}]"; else pip install .; fi

# -------------------------------------------------------------------------- runtime stage
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:${PATH}"

RUN groupadd --system app \
 && useradd --system --gid app --home-dir /app --shell /usr/sbin/nologin app

WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY alembic.ini ./alembic.ini
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod 0755 /usr/local/bin/entrypoint.sh && chown -R app:app /app

USER app
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health/live', timeout=3).status == 200 else 1)"

ENTRYPOINT ["entrypoint.sh"]
CMD ["uvicorn", "requirements_agent.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
