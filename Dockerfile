FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/home/prism/.cache/huggingface

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-ml.txt ./

# CPU-only PyTorch keeps the image smaller than the default CUDA build.
# Override TORCH_INDEX_URL to install a different wheel set.
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
RUN pip install --upgrade pip \
    && pip install --index-url "${TORCH_INDEX_URL}" torch==2.14.1 \
    && pip install -r requirements.txt -r requirements-ml.txt

COPY . .

RUN useradd --create-home --uid 10001 prism \
    && mkdir -p /app/benchmarks/results /home/prism/.cache/huggingface \
    && chown -R prism:prism /app /home/prism

USER prism

EXPOSE 8000

CMD ["sh", "-c", "alembic upgrade head && exec uvicorn app.main:app --host 0.0.0.0 --port 8000"]
