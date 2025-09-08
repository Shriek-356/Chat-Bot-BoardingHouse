# ===== Build stage =====
FROM python:3.11-slim AS builder
ENV PIP_NO_CACHE_DIR=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --upgrade pip && pip wheel -r requirements.txt -w /wheels

# ===== Runtime stage =====
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    SENTENCE_TRANSFORMERS_HOME=/root/.cache/sentence_transformers
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates && rm -rf /var/lib/apt/lists/*

# quan trọng: copy requirements.txt vào runtime
COPY requirements.txt .
# copy các wheel đã build
COPY --from=builder /wheels /wheels
# cài từ wheels (không cần internet)
RUN pip install --no-index --find-links=/wheels -r requirements.txt


# copy source
COPY . .
EXPOSE 8000

# dùng curl cho healthcheck (đã cài ở trên)
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8000/docs >/dev/null || exit 1

CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]

# Pre-download embedding model để cache vào image
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/multilingual-e5-small')"
