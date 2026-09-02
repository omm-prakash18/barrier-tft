# ─────────────────────────────────────────────────────────────────
# Stage 1: builder — install dependencies
# ─────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY src/ ./src/

# Install project + dependencies into /install prefix
RUN pip install --no-cache-dir --prefix=/install ".[dev]"

# ─────────────────────────────────────────────────────────────────
# Stage 2: runtime — lean image, non-root user
# ─────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Security: run as non-root
RUN useradd --create-home --shell /bin/bash appuser
WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy source and entrypoint
COPY --chown=appuser:appuser src/      ./src/
COPY --chown=appuser:appuser tests/    ./tests/
COPY --chown=appuser:appuser main.py   ./main.py

# Ensure Python finds src package
ENV PYTHONPATH="/app"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Default: run the full pipeline
USER appuser
CMD ["python", "main.py"]

# ─────────────────────────────────────────────────────────────────
# Build & run:
#   docker build -t barrier-tft .
#   docker run --rm barrier-tft
#
# Run tests only:
#   docker run --rm barrier-tft python -m pytest tests/ -v
# ─────────────────────────────────────────────────────────────────
