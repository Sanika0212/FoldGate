# Serving image: the router plus ONNX Runtime, no model weights and no training stack.
#
# The exported graphs arrive at runtime, in the directory FOLDGATE_ARTIFACTS points at (an
# init container fetches them in the Helm chart, KServe mounts them at /mnt/models).
FROM ghcr.io/astral-sh/uv:0.9.9-python3.12-bookworm-slim AS build

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app
COPY pyproject.toml uv.lock README.md LICENSE NOTICE ./
COPY src ./src

# Serving needs the router, ONNX Runtime, the tokenizer and the difficulty model, not torch.
RUN uv sync --frozen --no-dev --no-default-groups \
    --no-install-package torch \
    --no-install-package lightning \
    --no-install-package peft \
    --no-install-package mlflow

FROM python:3.12-slim-bookworm

RUN useradd --uid 10001 --create-home --shell /usr/sbin/nologin foldgate
WORKDIR /app
COPY --from=build --chown=10001:10001 /app /app

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/tmp/huggingface \
    FOLDGATE_ARTIFACTS=/models/champion

USER 10001
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=3s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/healthz')"

ENTRYPOINT ["foldgate"]
CMD ["serve", "--port", "8080", "--host", "0.0.0.0"]
