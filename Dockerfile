# ── Stage 1: Build & cache dependencies + embedding model ────────
FROM python:3.12-slim AS builder

WORKDIR /app

# Install build essentials for compiled deps (numpy, scipy, faiss)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml README.md LICENSE ./
COPY chicory/ chicory/

# Install the package with API + LLM dependencies
RUN pip install --no-cache-dir ".[api,all-llm]"

# Pre-download the embedding model so it's baked into the image
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# ── Stage 2: Runtime ─────────────────────────────────────────────
FROM python:3.12-slim

WORKDIR /app

# Copy installed Python packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy cached embedding model from builder
COPY --from=builder /root/.cache/huggingface /root/.cache/huggingface

# Copy application code
COPY chicory/ chicory/
COPY pyproject.toml README.md LICENSE ./

# Install the package itself (metadata only, deps already present)
RUN pip install --no-cache-dir --no-deps .

# Create data directory
RUN mkdir -p /root/.chicory

EXPOSE 8000

# Prevent HuggingFace from trying to download anything at runtime
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

CMD ["chicory-api"]
