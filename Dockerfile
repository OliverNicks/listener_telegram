# ==========================================
# Stage 1: Builder (Compiles & installs dependencies)
# ==========================================
FROM python:3.11-slim AS builder

# Set build-time environment flags
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# Install minimal build tools needed for C-extensions/wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create a clean standalone virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# CACHE LAYER: Copy ONLY requirements.txt first
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt


# ==========================================
# Stage 2: Final Minimal Runtime Image
# ==========================================
FROM python:3.11-slim AS runner

# Runtime environment flags
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

# Create non-root system user and group (UID 10001)
RUN groupadd -g 10001 appgroup && \
    useradd -u 10001 -g appgroup -s /bin/bash -m appuser

# Set working directory
WORKDIR /app

# Pre-create stateful directories with correct ownership for appuser
RUN mkdir -p /app/logs /app/data /app/models /app/config && \
    chown -R appuser:appgroup /app

# Copy ONLY virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Copy application source code with non-root user permissions
COPY --chown=appuser:appgroup src/ /app/src/
COPY --chown=appuser:appgroup bots/ /app/bots/
COPY --chown=appuser:appgroup config/ /app/config/
COPY --chown=appuser:appgroup pyproject.toml /app/pyproject.toml

# Switch to the non-root user
USER appuser

# Default startup command
CMD ["python", "-m", "bots.bot_b.listener"]