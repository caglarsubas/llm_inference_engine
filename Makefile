.PHONY: install install-metal install-mlx install-otel sync run dev run-otel list-models smoke download-mlx-model otel-up otel-down compose-build compose-up compose-up-scale compose-logs compose-down compose-ps compose-vllm-up compose-vllm-down compose-vllm-multigpu-up compose-vllm-multigpu-down compose-up-sticky compose-down-sticky obs-up obs-down obs-logs obs-load native-install native-uninstall native-up native-down native-restart native-status native-logs test lint clean

install:
	uv sync

install-metal:
	CMAKE_ARGS="-DGGML_METAL=on" uv sync --reinstall-package llama-cpp-python

install-mlx:
	uv sync --extra mlx

install-otel:
	uv sync --extra otel --extra dev

sync:
	uv sync

# Override with: make download-mlx-model MODEL=mlx-community/<repo>
MODEL ?= mlx-community/Llama-3.2-1B-Instruct-4bit
download-mlx-model:
	uv run python scripts/download_mlx_model.py $(MODEL)

run:
	uv run uvicorn inference_engine.main:app --host 127.0.0.1 --port 8080

dev:
	uv run uvicorn inference_engine.main:app --host 127.0.0.1 --port 8080 --reload

# Run with OTel tracing on. Requires `make otel-up` first (or any OTLP/gRPC collector on :4317).
run-otel:
	OTEL_ENABLED=true uv run uvicorn inference_engine.main:app --host 127.0.0.1 --port 8080

otel-up:
	docker compose -f docker-compose.otel.yml up -d
	@echo "Jaeger UI: http://127.0.0.1:16686"

otel-down:
	docker compose -f docker-compose.otel.yml down

# ---------------------------------------------------------------------------
# Containerized deployment — multi-replica behind nginx
# ---------------------------------------------------------------------------

# Default replica count when scaling. Override with REPLICAS=N.
REPLICAS ?= 2

compose-build:
	docker compose build engine

compose-up:
	docker compose up -d --scale engine=$(REPLICAS)
	@echo
	@echo "engine: replicas=$(REPLICAS); LB on http://127.0.0.1:$${LB_PORT:-8080}"
	@echo "    curl http://127.0.0.1:$${LB_PORT:-8080}/v1/health"
	@echo "    curl http://127.0.0.1:$${LB_PORT:-8080}/lb_health"

compose-up-scale: compose-up

compose-logs:
	docker compose logs -f --tail=100 engine nginx

compose-ps:
	docker compose ps

compose-down:
	docker compose down

# vLLM sidecar overlay — continuous chat batching on a CUDA host.
# Requires nvidia-container-toolkit and a real GPU; documented in README.
compose-vllm-up:
	docker compose -f docker-compose.yml -f docker-compose.vllm.yml up -d --scale engine=$(REPLICAS)
	@echo
	@echo "engine + vllm: replicas=$(REPLICAS); LB on http://127.0.0.1:$${LB_PORT:-8080}"
	@echo "    curl http://127.0.0.1:$${LB_PORT:-8080}/v1/models | jq '.data[]|select(.format==\"vllm\")'"

compose-vllm-down:
	docker compose -f docker-compose.yml -f docker-compose.vllm.yml down

# Multi-GPU vLLM — two vLLM services on two GPUs (device_ids '0' and '1' by
# default; override via VLLM_GPU_ID_PRIMARY / VLLM_GPU_ID_SECONDARY).
compose-vllm-multigpu-up:
	docker compose -f docker-compose.yml -f docker-compose.vllm-multigpu.yml up -d --scale engine=$(REPLICAS)
	@echo
	@echo "engine + 2 vLLM upstreams: replicas=$(REPLICAS); LB on http://127.0.0.1:$${LB_PORT:-8080}"
	@echo "    primary   GPU $${VLLM_GPU_ID_PRIMARY:-0}: http://vllm:8000          ($${VLLM_MODEL_PRIMARY:-meta-llama/Llama-3.2-1B-Instruct})"
	@echo "    secondary GPU $${VLLM_GPU_ID_SECONDARY:-1}: http://vllm-secondary:8000 ($${VLLM_MODEL_SECONDARY:-meta-llama/Llama-3.2-3B-Instruct})"

compose-vllm-multigpu-down:
	docker compose -f docker-compose.yml -f docker-compose.vllm-multigpu.yml down

# HAProxy overlay — header-based tenant stickiness instead of round-robin
# (round 26). Use this for multi-turn agent traffic where cache locality
# matters; default nginx overlay is fine for stateless workloads.
compose-up-sticky:
	docker compose -f docker-compose.yml -f docker-compose.haproxy.yml up -d --scale engine=$(REPLICAS)
	@echo
	@echo "engine + haproxy: replicas=$(REPLICAS); LB on http://127.0.0.1:$${LB_PORT:-8080}"
	@echo "    HAProxy stats: http://127.0.0.1:$${HAPROXY_STATS_PORT:-8404}/stats"
	@echo "    same Authorization Bearer token will hit the same replica every time"

compose-down-sticky:
	docker compose -f docker-compose.yml -f docker-compose.haproxy.yml down

# Observability stack — engine + OTel Collector + Prometheus + Grafana + Jaeger.
# Brings up the full dashboard surface; engine traffic is auto-instrumented.
obs-up:
	docker compose -f docker-compose.yml -f docker-compose.observability.yml up -d --scale engine=$(REPLICAS)
	@echo
	@echo "engine + observability: replicas=$(REPLICAS); LB on http://127.0.0.1:$${LB_PORT:-8080}"
	@echo "    Grafana   → http://127.0.0.1:$${GRAFANA_PORT:-3000}    (admin / admin)"
	@echo "    Prometheus→ http://127.0.0.1:$${PROMETHEUS_PORT:-9090}"
	@echo "    Jaeger UI → http://127.0.0.1:$${JAEGER_UI_PORT:-16686}"
	@echo
	@echo "  Open Grafana, navigate to: Dashboards → Inference Engine → Inference Engine — Overview"

obs-down:
	docker compose -f docker-compose.yml -f docker-compose.observability.yml down

obs-logs:
	docker compose -f docker-compose.yml -f docker-compose.observability.yml logs -f --tail=80 otel-collector grafana prometheus

# Drive a small synthetic load so Grafana panels have data immediately. Two
# tenants, two models, alternating — produces traffic-by-tenant + by-model
# breakdowns in the overview dashboard within ~30s of running.
obs-load:
	@for i in $$(seq 1 20); do \
	  curl -s -H "Authorization: Bearer sk-tenant-alpha" -X POST http://127.0.0.1:$${LB_PORT:-8080}/v1/chat/completions \
	    -H 'content-type: application/json' \
	    -d '{"model":"llama3.2:1b","messages":[{"role":"user","content":"ping"}],"max_tokens":4}' > /dev/null & \
	  curl -s -H "Authorization: Bearer sk-tenant-beta" -X POST http://127.0.0.1:$${LB_PORT:-8080}/v1/chat/completions \
	    -H 'content-type: application/json' \
	    -d '{"model":"llama3.2:3b","messages":[{"role":"user","content":"ping"}],"max_tokens":4}' > /dev/null & \
	  wait; \
	done
	@echo "  drove 40 requests across 2 tenants × 2 models — check Grafana"

# ---------------------------------------------------------------------------
# Native topology — engine + ollama as launchd agents (Metal-accelerated),
# observability stack in containers.  Recommended for on-prem Apple Silicon
# deployments where Docker's CPU-only container VM would tank inference
# latency (Metal isn't passed through to Linux containers on macOS).
# ---------------------------------------------------------------------------

native-install: install-metal install-otel
	./scripts/native-service.sh install

native-uninstall:
	./scripts/native-service.sh uninstall

native-up:
	./scripts/native-service.sh start
	docker compose -f docker-compose.native.yml up -d
	@echo
	@echo "engine (native, Metal): http://127.0.0.1:8080"
	@echo "ollama (native, Metal): http://127.0.0.1:11434"
	@echo "Grafana    → http://127.0.0.1:$${GRAFANA_PORT:-3030}    (admin / admin)"
	@echo "Prometheus → http://127.0.0.1:$${PROMETHEUS_PORT:-9090}"
	@echo "Jaeger UI  → http://127.0.0.1:$${JAEGER_UI_PORT:-16686}"

native-down:
	./scripts/native-service.sh stop
	docker compose -f docker-compose.native.yml down

native-restart:
	./scripts/native-service.sh restart

native-status:
	./scripts/native-service.sh status
	@echo
	@docker compose -f docker-compose.native.yml ps

# Tail engine logs by default; pass TARGET=ollama for the fallback runtime.
TARGET ?= engine
native-logs:
	./scripts/native-service.sh logs $(TARGET)

list-models:
	uv run python scripts/list_models.py

smoke:
	uv run python scripts/smoke_test.py

stress:
	uv run python scripts/stress_test.py --requests 20 --concurrency 8 --models llama3.2:1b

stress-cross-backend:
	uv run python scripts/stress_test.py --requests 20 --concurrency 8 \
	  --models llama3.2:1b,Llama-3.2-1B-Instruct-4bit:mlx

test:
	uv run pytest -v

lint:
	uv run ruff check src tests

clean:
	rm -rf .venv .pytest_cache .ruff_cache **/__pycache__
