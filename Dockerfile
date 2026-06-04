# ===========================================================================
# Customer Intelligence Platform — Dockerfile
# ===========================================================================
# Builds a production-ready container for the FastAPI endpoint.
#
# Usage:
#   docker build -t clv-platform .
#   docker run -p 8000:8000 clv-platform
#
# With env override (e.g. custom artifacts path):
#   docker run -p 8000:8000 -e CLV_BASE_DIR=/app clv-platform
#
# Multi-stage build:
#   Stage 1 (builder) — installs all dependencies into a venv
#   Stage 2 (runtime) — copies only the venv + app code, no build tools
#   Result: ~40% smaller final image
# ===========================================================================

# ---------------------------------------------------------------------------
# Stage 1 — Builder
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder

# Prevents .pyc files and enables unbuffered stdout (better Docker logs)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# Install build dependencies (needed to compile some wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Create isolated virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Install FastAPI + uvicorn (may already be in requirements.txt — safe to repeat)
RUN pip install fastapi uvicorn[standard]


# ---------------------------------------------------------------------------
# Stage 2 — Runtime
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    CLV_BASE_DIR=/app

# Create non-root user for security
RUN groupadd --gid 1001 appgroup && \
    useradd  --uid 1001 --gid appgroup --shell /bin/bash --create-home appuser

WORKDIR /app

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Copy application code
COPY src/          ./src/
COPY api/          ./api/
COPY artifacts/    ./artifacts/

# Ensure artifact directories exist (in case of missing files)
RUN mkdir -p artifacts/models artifacts/graphs

# Set ownership to non-root user
RUN chown -R appuser:appgroup /app

USER appuser

# Expose FastAPI port
EXPOSE 8000

# Health check — Docker will mark container unhealthy if /health fails
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" \
    || exit 1

# Start the API server
# - host 0.0.0.0 required to accept connections from outside the container
# - workers=2: 2 Gunicorn workers (adjust based on CPU count)
# - timeout-keep-alive=65: slightly above common load balancer timeout
CMD ["uvicorn", "api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2", \
     "--timeout-keep-alive", "65", \
     "--log-level", "info"]