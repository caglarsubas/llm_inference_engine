# syntax=docker/dockerfile:1.7
# ---------------------------------------------------------------------------
# Inference engine image
#
# Single-stage on python:3.12-slim. llama-cpp-python compiles from source
# during install — that's the slow step (~3-5 min on a fresh build), cached
# afterwards. Default build is CPU-only; for CUDA hosts pass
# ``--build-arg CMAKE_ARGS="-DGGML_CUDA=on"`` to the build command. Metal is
# not available inside containers (Docker Desktop on macOS runs a Linux VM
# with no GPU passthrough); the engine degrades to CPU automatically.
#
# Image size is dominated by the compiled llama-cpp-python wheel. We don't
# install the [mlx] extra in the container — mlx-lm is Apple-Silicon-only
# and the composite registry handles "no MLX models present" cleanly.
# ---------------------------------------------------------------------------

FROM python:3.12-slim AS runtime

# Build args — override on `docker build` to switch acceleration.
ARG CMAKE_ARGS=""

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv

# Build deps for llama-cpp-python (gcc, cmake, etc.) + curl for uv install.
# We install build deps in the same layer we use them so layer pruning later
# is straightforward if image-size becomes a priority.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# uv comes in via the official install script; we keep it on PATH globally.
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && ln -s /root/.local/bin/uv /usr/local/bin/uv

WORKDIR /app

# Copy everything uv sync needs to install both deps and the project itself.
# README.md is required by hatchling (referenced as ``readme`` in
# pyproject.toml) and src/ is needed because the project is installed as a
# wheel from this source tree.
COPY pyproject.toml uv.lock* README.md ./
COPY src ./src

# Install the project's runtime deps. Default extras are empty; production
# deployments that want OTel should rebuild with `--build-arg EXTRAS=otel`.
ARG EXTRAS=""
RUN if [ -n "$EXTRAS" ]; then \
        CMAKE_ARGS="${CMAKE_ARGS}" uv sync --no-dev --extra ${EXTRAS} ; \
    else \
        CMAKE_ARGS="${CMAKE_ARGS}" uv sync --no-dev ; \
    fi

# Non-root runtime user. uvicorn doesn't need root and a dedicated UID limits
# the blast radius of any host-volume misconfigurations.
RUN useradd --create-home --uid 10001 engine \
    && chown -R engine:engine /app /opt/venv
USER engine

# Configurable inside compose; defaults sized for a CPU-only laptop run.
ENV HOST=0.0.0.0 \
    PORT=8080 \
    OLLAMA_MODELS_DIR=/models/ollama \
    MLX_MODELS_DIR=/models/mlx \
    AUTH_KEYS_FILE=/config/auth_keys.json \
    AUTO_EVAL_POLICIES_FILE=/config/auto_eval_policies.json

EXPOSE 8080

# Container-level healthcheck so the LB has something to gate on. /v1/health
# stays open even with auth on (round 5 design).
HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen(f'http://127.0.0.1:{__import__(\"os\").environ.get(\"PORT\", \"8080\")}/v1/health', timeout=3).status == 200 else 1)"

# Skip uv at runtime — the venv at /opt/venv is fully populated and uv was
# only needed during install. Calling uvicorn from the venv directly avoids
# the non-root user needing access to /root/.local/bin (where uv lives).
CMD ["sh", "-c", "/opt/venv/bin/uvicorn inference_engine.main:app --host $HOST --port $PORT"]
