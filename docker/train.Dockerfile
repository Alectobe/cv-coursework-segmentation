# Образ для подготовки данных, обучения и оценки (GPU).
FROM python:3.12-slim-bookworm

# Индекс колёс PyTorch: cu128 - RTX 20xx-50xx, cu126 - для более старых GPU
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app/src:/app \
    HF_HOME=/app/artifacts/hf_cache \
    MPLCONFIGDIR=/tmp/matplotlib \
    NO_ALBUMENTATIONS_UPDATE=1

WORKDIR /app

# torch ставится отдельным слоем: он самый тяжёлый и меняется реже всего
RUN pip install torch==2.8.0 torchvision==0.23.0 --index-url ${TORCH_INDEX_URL}

COPY requirements/train.txt requirements/train.txt
RUN pip install -r requirements/train.txt

COPY src/ src/
COPY scripts/ scripts/
COPY configs/ configs/
