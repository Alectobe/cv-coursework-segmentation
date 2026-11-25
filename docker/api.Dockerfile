# Образ сервиса инференса (CPU; GPU не нужен).
FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app/src:/app \
    MODEL_PATH=/app/artifacts/final/model.pt

WORKDIR /app

RUN pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu

COPY requirements/api.txt requirements/api.txt
RUN pip install -r requirements/api.txt

COPY src/polypseg/ src/polypseg/
COPY api/ api/
COPY scripts/download_weights.py scripts/download_weights.py

EXPOSE 8000

# Контейнер считается здоровым только когда /health отвечает 200, то есть модель загружена
HEALTHCHECK --interval=10s --timeout=5s --start-period=60s --retries=6 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
