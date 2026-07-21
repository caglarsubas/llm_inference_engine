# syntax=docker/dockerfile:1.7
# ---------------------------------------------------------------------------
# Orchestra inference engine - Debian runtime image.
#
# The builder contains compilers needed by llama-cpp-python. The final image
# contains only the locked virtual environment and runtime libraries. The base
# image is pinned by its multi-platform digest; update it deliberately through
# a reviewed pull request.
#
# Build with OpenTelemetry support (the published profile):
#   docker build --build-arg EXTRAS=otel -t inference-engine:local .
# ---------------------------------------------------------------------------

ARG PYTHON_IMAGE=python:3.12-slim@sha256:423ed6ab25b1921a477529254bfeeabf5855151dc2c3141699a1bfc852199fbf

FROM ${PYTHON_IMAGE} AS builder

ARG CMAKE_ARGS=""
ARG EXTRAS=""
ARG UV_VERSION=0.7.17

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_DOWNLOADS=never

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential \
      ca-certificates \
      cmake \
      git \
 && rm -rf /var/lib/apt/lists/* \
 && python -m pip install --no-cache-dir "uv==${UV_VERSION}"

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src

# Install the project non-editably so the runtime does not depend on /app/src.
# EXTRAS remains a build-time option; canonical published images pass "otel".
RUN if [ -n "${EXTRAS}" ]; then \
      CMAKE_ARGS="-DGGML_NATIVE=OFF ${CMAKE_ARGS}" uv sync --frozen --no-dev --no-editable --extra "${EXTRAS}"; \
    else \
      CMAKE_ARGS="-DGGML_NATIVE=OFF ${CMAKE_ARGS}" uv sync --frozen --no-dev --no-editable; \
    fi

FROM ${PYTHON_IMAGE} AS runtime

ARG VCS_REF="unknown"
ARG IMAGE_VERSION="dev"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH=/opt/venv/bin:$PATH \
    HOME=/tmp \
    TMPDIR=/tmp \
    HOST=0.0.0.0 \
    PORT=8080 \
    OLLAMA_MODELS_DIR=/models/ollama \
    MLX_MODELS_DIR=/models/mlx \
    HF_VLM_MODELS_DIR=/models/hf-vlm \
    AUTH_KEYS_FILE=/config/auth_keys.json \
    AUTO_EVAL_POLICIES_FILE=/config/auto_eval_policies.json \
    MODEL_ROUTING_POLICY_FILE=/config/model_routing_policy.json \
    MODEL_ROUTING_LAST_KNOWN_GOOD_FILE=/state/model_routing_policy.lkg.json \
    MODEL_ROUTING_TRUST_STORE_FILE=/config/model_routing_trust.json \
    MODEL_ROUTING_PRICING_FILE=/config/model_routing_pricing.json

LABEL org.opencontainers.image.title="Orchestra Inference Engine" \
      org.opencontainers.image.description="Tenant-deployed OpenAI-compatible Orchestra model plane" \
      org.opencontainers.image.source="https://github.com/caglarsubas/llm_inference_engine" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.version="${IMAGE_VERSION}" \
      io.prometa.image.variant="debian"

# GID 0 permissions support OpenShift arbitrary UIDs.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates libgomp1 \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --system --uid 10001 --gid 0 --no-create-home engine \
 && mkdir -p \
      /app \
      /state \
      /models/ollama/manifests \
      /models/ollama/blobs \
      /models/mlx \
      /models/hf-vlm \
 && chown 10001:0 /state \
 && chmod 0770 /state

WORKDIR /app
COPY --from=builder --chown=10001:0 /opt/venv /opt/venv

# COPY preserves the locked environment's group read/execute permissions and
# assigns GID 0. Only /state is group-writable (created above) so an arbitrary
# SCC UID can persist last-known-good policy without mutating code.

USER 10001
EXPOSE 8080

HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import os,ssl,urllib.request;t=os.environ.get('INFERENCE_ENGINE_SERVER_TLS_CERT_FILE');c=ssl.create_default_context(cafile=os.environ.get('INFERENCE_ENGINE_PROBE_TLS_CA_FILE') or t) if t else None;t and setattr(c,'check_hostname',False);p=os.environ.get('INFERENCE_ENGINE_PROBE_TLS_CERT_FILE');p and c.load_cert_chain(p,os.environ['INFERENCE_ENGINE_PROBE_TLS_KEY_FILE']);raise SystemExit(0 if urllib.request.urlopen(('%s://127.0.0.1:%s/v1/health' % ('https' if t else 'http',os.environ.get('PORT','8080'))),timeout=3,context=c).status == 200 else 1)"

CMD ["python", "-m", "inference_engine.server"]
