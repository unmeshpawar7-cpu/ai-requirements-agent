#!/bin/sh
# Apply database migrations (PostgreSQL) before starting the API.
set -e
if [ "${RUN_MIGRATIONS:-true}" = "true" ]; then
  echo "Running database migrations..."
  alembic -c /app/alembic.ini upgrade head
fi
exec "$@"
