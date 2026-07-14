from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
from redis import Redis
from redis.exceptions import ConnectionError as RedisConnectionError

from inference_engine.model_routing_runtime import (
    ModelRoutingEnforcementError,
    ModelRoutingRateLimiter,
    ModelRoutingRuntimeConfigError,
    RedisModelRoutingRateLimiter,
    build_model_routing_rate_limiter,
)


class FakeRedis:
    def __init__(self, result=None, error: Exception | None = None) -> None:
        self.result = [1, 0, 1] if result is None else result
        self.error = error
        self.eval_calls: list[tuple] = []
        self.closed = False

    def ping(self) -> bool:
        if self.error is not None:
            raise self.error
        return True

    def eval(self, *args):
        self.eval_calls.append(args)
        if self.error is not None:
            raise self.error
        return self.result

    def close(self) -> None:
        self.closed = True


def settings_for(**overrides):
    values = {
        "model_routing_rate_limit_scope": "process-replica",
        "model_routing_rate_limit_max_buckets": 100,
        "model_routing_rate_limit_redis_url": "",
        "model_routing_rate_limit_redis_url_file": "",
        "model_routing_rate_limit_allow_insecure_redis": False,
        "model_routing_rate_limit_key_prefix": "orchestra:model-routing",
        "model_routing_rate_limit_connect_timeout_seconds": 0.1,
        "model_routing_rate_limit_operation_timeout_seconds": 0.1,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def consume(limiter, *, limit: int = 2) -> None:
    limiter.consume(
        digest="sha256:policy-secret",
        route_id="sensitive-route",
        org_id="org-sensitive",
        tenant="tenant-sensitive",
        limit=limit,
        policy_id="policy-1",
    )


def test_process_scope_remains_the_dependency_free_default() -> None:
    limiter = build_model_routing_rate_limiter(settings_for())

    assert isinstance(limiter, ModelRoutingRateLimiter)
    assert limiter.scope == "process-replica"
    limiter.ping()
    limiter.close()


@pytest.mark.parametrize(
    ("overrides", "code"),
    [
        ({"model_routing_rate_limit_scope": "deployment-shared"}, "rate_limit_backend_source_invalid"),
        (
            {
                "model_routing_rate_limit_scope": "deployment-shared",
                "model_routing_rate_limit_redis_url": "redis://127.0.0.1:6379/0",
                "model_routing_rate_limit_redis_url_file": "/secret/url",
            },
            "rate_limit_backend_source_invalid",
        ),
        (
            {"model_routing_rate_limit_redis_url": "redis://127.0.0.1:6379/0"},
            "rate_limit_backend_config_unused",
        ),
        (
            {
                "model_routing_rate_limit_scope": "deployment-shared",
                "model_routing_rate_limit_redis_url": "redis://valkey.internal:6379/0",
            },
            "rate_limit_backend_tls_required",
        ),
        (
            {
                "model_routing_rate_limit_scope": "deployment-shared",
                "model_routing_rate_limit_redis_url": "redis://127.0.0.1:6379/0",
                "model_routing_rate_limit_key_prefix": "bad prefix",
            },
            "rate_limit_key_prefix_invalid",
        ),
    ],
)
def test_shared_scope_rejects_ambiguous_or_unsafe_configuration(overrides, code) -> None:
    with pytest.raises(ModelRoutingRuntimeConfigError) as raised:
        build_model_routing_rate_limiter(settings_for(**overrides))
    assert raised.value.code == code


def test_shared_scope_reads_a_mounted_url_and_allows_explicit_non_tls_test_mode(
    tmp_path: Path,
) -> None:
    url_file = tmp_path / "redis-url"
    url_file.write_text("redis://valkey.test.svc:6379/4\n", encoding="utf-8")

    limiter = build_model_routing_rate_limiter(
        settings_for(
            model_routing_rate_limit_scope="deployment-shared",
            model_routing_rate_limit_redis_url_file=str(url_file),
            model_routing_rate_limit_allow_insecure_redis=True,
        )
    )

    assert isinstance(limiter, RedisModelRoutingRateLimiter)
    assert limiter.scope == "deployment-shared"
    limiter.close()


def test_shared_limiter_uses_only_a_hashed_identity_key() -> None:
    client = FakeRedis()
    limiter = RedisModelRoutingRateLimiter(client, key_prefix="orchestra:test")

    consume(limiter)

    assert len(client.eval_calls) == 1
    call = client.eval_calls[0]
    assert call[1] == 1
    key = call[2]
    assert key.startswith("orchestra:test:rpm:")
    assert len(key.rsplit(":", 1)[-1]) == 64
    serialized = repr(call)
    for sensitive in (
        "policy-secret",
        "sensitive-route",
        "org-sensitive",
        "tenant-sensitive",
    ):
        assert sensitive not in serialized


def test_shared_limiter_returns_a_bounded_retry_after() -> None:
    limiter = RedisModelRoutingRateLimiter(
        FakeRedis(result=[0, 1_501, 2]),
        key_prefix="orchestra:test",
    )

    with pytest.raises(ModelRoutingEnforcementError) as raised:
        consume(limiter)

    assert raised.value.code == "rate_limit_exceeded"
    assert raised.value.retry_after_seconds == 2


def test_shared_limiter_fails_closed_when_backend_is_unavailable() -> None:
    limiter = RedisModelRoutingRateLimiter(
        FakeRedis(error=RedisConnectionError("unavailable")),
        key_prefix="orchestra:test",
    )

    with pytest.raises(ModelRoutingRuntimeConfigError) as startup_error:
        limiter.ping()
    assert startup_error.value.code == "rate_limit_backend_unavailable"

    with pytest.raises(ModelRoutingEnforcementError) as request_error:
        consume(limiter)
    assert request_error.value.code == "rate_limit_backend_unavailable"
    assert request_error.value.retry_after_seconds == 1


@pytest.mark.skipif(not os.getenv("TEST_VALKEY_URL"), reason="TEST_VALKEY_URL is not configured")
def test_shared_limiter_is_atomic_across_clients() -> None:
    url = os.environ["TEST_VALKEY_URL"]
    prefix = f"orchestra:test:{uuid4().hex}"
    first = RedisModelRoutingRateLimiter(Redis.from_url(url), key_prefix=prefix)
    second = RedisModelRoutingRateLimiter(Redis.from_url(url), key_prefix=prefix)
    first.ping()
    second.ping()

    def attempt(index: int) -> str:
        try:
            consume(first if index % 2 == 0 else second, limit=7)
            return "accepted"
        except ModelRoutingEnforcementError as exc:
            assert exc.code == "rate_limit_exceeded"
            return "denied"

    try:
        with ThreadPoolExecutor(max_workers=10) as pool:
            results = list(pool.map(attempt, range(20)))
    finally:
        first.close()
        second.close()

    assert results.count("accepted") == 7
    assert results.count("denied") == 13
