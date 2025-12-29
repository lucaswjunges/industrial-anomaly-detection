# Industrial IoT Anomaly Detection - Production Dockerfile
# Multi-stage build for optimal image size
# Author: Lucas William Junges
# Date: December 2024

# ========================================
# Stage 1: Builder
# ========================================
FROM python:3.9-slim as builder

LABEL maintainer="Lucas William Junges"
LABEL description="Industrial IoT Anomaly Detection Pipeline"

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ========================================
# Stage 2: Runtime
# ========================================
FROM python:3.9-slim

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy application code
COPY src/ /app/src/
COPY train_simple.py /app/
COPY train_nasa.py /app/
COPY evaluate_simple.py /app/
COPY evaluate_nasa.py /app/
COPY requirements.txt /app/

# Create directories for data and models
RUN mkdir -p /app/data/raw /app/data/processed /app/data/results /app/models

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV TF_CPP_MIN_LOG_LEVEL=2

# Create non-root user for security
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

# Default command: show usage
CMD ["python", "-c", "print('\\n=== Industrial IoT Anomaly Detection ===\\n\\nAvailable commands:\\n  python train_simple.py    - Train on synthetic data\\n  python train_nasa.py      - Train on NASA bearing data\\n  python evaluate_simple.py - Evaluate synthetic models\\n  python evaluate_nasa.py   - Evaluate NASA models\\n\\nFor interactive shell: docker run -it <image> /bin/bash\\n')"]

# Metadata
LABEL version="1.0.0"
LABEL description="Production-ready anomaly detection for industrial IoT"
