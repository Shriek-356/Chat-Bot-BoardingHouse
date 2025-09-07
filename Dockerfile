# ===== Build stage =====
FROM python:3.11-slim AS builder
ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    SENTENCE_TRANSFORMERS_HOME=/root/.cache/sentence_transformers

# system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip && pip wheel -r requirements.txt -w /wheels

# preload model vào cache để image run nhanh, ổn định offline
RUN python - <<'PY'
from sentence_transformers import SentenceTransformer
SentenceTransformer("intfloat/multilingual-e5-base")
PY

# ===== Runtime stage =====
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    SENTENCE_TRANSFORMERS_HOME=/root/.cache/sentence_transformers

# optional: perf TLS
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /wheels /wheels
RUN pip install --no-index --find-links=/wheels -r /wheels/requirements.txt

# copy app
COPY . .

# cổng uvicorn
EXPOSE 8000

# healthcheck đơn giản
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD wget -qO- http://127.0.0.1:8000/docs >/dev/null || exit 1

# chạy uvicorn nhiều worker (tối ưu CPU)
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
