FROM python:3.12-slim

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app:/app/thirdparty/dart \
    HF_HOME=/root/.cache/huggingface

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    patch \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt requirements-ml.txt ./

RUN pip install --upgrade pip \
    && pip install --index-url "${TORCH_INDEX_URL}" torch torchvision \
    && pip install -r requirements.txt -r requirements-ml.txt

COPY core/ ./core/
COPY adapters/ ./adapters/
COPY service/ ./service/
COPY thirdparty/ ./thirdparty/

# Если thirdparty/dart на хосте был чистым (без патча), применяем dart-cpu.patch при сборке
RUN if [ -d /app/thirdparty/dart ] && [ -f /app/thirdparty/dart-cpu.patch ]; then \
      cd /app/thirdparty/dart && \
      (patch -p1 --dry-run --silent < /app/thirdparty/dart-cpu.patch && patch -p1 < /app/thirdparty/dart-cpu.patch || true); \
    fi

RUN mkdir -p /app/weights

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)" || exit 1

CMD ["uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "8000"]
