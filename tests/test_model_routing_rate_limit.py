from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
from redis import Redis
from redis.exceptions import ConnectionError as RedisConnectionError

import inference_engine.model_routing_runtime as model_routing_runtime
from inference_engine.model_routing_runtime import (
    ModelRoutingEnforcementError,
    ModelRoutingRateLimiter,
    ModelRoutingRuntimeConfigError,
    RedisModelRoutingRateLimiter,
    build_model_routing_rate_limiter,
)


class FakeRedis:
    def __init__(
        self,
        result=None,
        error: Exception | None = None,
        acknowledged_replicas: int = 1,
    ) -> None:
        self.result = [1, 0, 1] if result is None else result
        self.error = error
        self.acknowledged_replicas = acknowledged_replicas
        self.eval_calls: list[tuple] = []
        self.wait_calls: list[tuple[int, int]] = []
        self.closed = False

    class Pipeline:
        def __init__(self, client: FakeRedis) -> None:
            self.client = client
            self.arguments: tuple | None = None
            self.replicas: int | None = None
            self.timeout: int | None = None

        def eval(self, *args):
            self.arguments = args
            return self

        def wait(self, replicas: int, timeout: int):
            self.replicas = replicas
            self.timeout = timeout
            return self

        def execute(self):
            if self.arguments is None or self.replicas is None or self.timeout is None:
                raise AssertionError("incomplete fake pipeline")
            result = self.client.eval(*self.arguments)
            acknowledged = self.client.wait(self.replicas, self.timeout)
            return [result, acknowledged]

    def ping(self) -> bool:
        if self.error is not None:
            raise self.error
        return True

    def eval(self, *args):
        self.eval_calls.append(args)
        if self.error is not None:
            raise self.error
        return self.result

    def wait(self, replicas: int, timeout: int) -> int:
        self.wait_calls.append((replicas, timeout))
        if self.error is not None:
            raise self.error
        return self.acknowledged_replicas

    def pipeline(self, *, transaction: bool):
        assert transaction is False
        return self.Pipeline(self)

    def close(self) -> None:
        self.closed = True


def settings_for(**overrides):
    values = {
        "model_routing_rate_limit_scope": "process-replica",
        "model_routing_rate_limit_max_buckets": 100,
        "model_routing_rate_limit_redis_url": "",
        "model_routing_rate_limit_redis_url_file": "",
        "model_routing_rate_limit_sentinel_config_file": "",
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
        (
            {"model_routing_rate_limit_scope": "deployment-shared"},
            "rate_limit_backend_source_invalid",
        ),
        (
            {
                "model_routing_rate_limit_scope": "deployment-shared",
                "model_routing_rate_limit_redis_url": "redis://127.0.0.1:6379/0",
                "model_routing_rate_limit_redis_url_file": "/secret/url",
            },
            "rate_limit_backend_source_invalid",
        ),
        (
            {
                "model_routing_rate_limit_scope": "deployment-shared",
                "model_routing_rate_limit_redis_url": "redis://127.0.0.1:6379/0",
                "model_routing_rate_limit_sentinel_config_file": "/secret/sentinel.json",
            },
            "rate_limit_backend_source_invalid",
        ),
        (
            {"model_routing_rate_limit_redis_url": "redis://127.0.0.1:6379/0"},
            "rate_limit_backend_config_unused",
        ),
        (
            {"model_routing_rate_limit_sentinel_config_file": "/secret/sentinel.json"},
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


def sentinel_config(**overrides) -> dict:
    values = {
        "configVersion": 1,
        "serviceName": "orchestra-model-routing",
        "sentinels": [
            {"host": f"sentinel-{index}.tenant.svc.cluster.local", "port": 26379}
            for index in range(3)
        ],
        "minOtherSentinels": 1,
        "database": 0,
        "password": "data-secret",
        "sentinelPassword": "sentinel-secret",
        "tls": True,
        "caFile": "/etc/orchestra/ca/ca-bundle.crt",
        "requiredReplicaAcks": 1,
        "replicaAckTimeoutMilliseconds": 750,
    }
    values.update(overrides)
    return values


def write_sentinel_config(path: Path, **overrides) -> None:
    path.write_text(json.dumps(sentinel_config(**overrides)), encoding="utf-8")


def test_shared_scope_builds_tls_sentinel_client_and_requires_replica_acknowledgement(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_file = tmp_path / "sentinel.json"
    write_sentinel_config(config_file)
    captured: dict[str, object] = {}
    master = FakeRedis(acknowledged_replicas=1)
    discovery_clients = (FakeRedis(), FakeRedis(), FakeRedis())

    class FakeSentinel:
        def __init__(self, sentinels, **kwargs) -> None:
            captured["sentinels"] = sentinels
            captured["kwargs"] = kwargs
            self.sentinels = list(discovery_clients)

        def master_for(self, service_name: str, **kwargs):
            captured["service_name"] = service_name
            captured["master_kwargs"] = kwargs
            return master

    monkeypatch.setattr(model_routing_runtime, "Sentinel", FakeSentinel)

    limiter = build_model_routing_rate_limiter(
        settings_for(
            model_routing_rate_limit_scope="deployment-shared",
            model_routing_rate_limit_sentinel_config_file=str(config_file),
            model_routing_rate_limit_operation_timeout_seconds=1.0,
        )
    )
    limiter.ping()
    consume(limiter)
    limiter.close()

    assert captured["sentinels"] == [
        (f"sentinel-{index}.tenant.svc.cluster.local", 26379) for index in range(3)
    ]
    kwargs = captured["kwargs"]
    assert kwargs["min_other_sentinels"] == 1
    assert kwargs["password"] == "data-secret"
    assert kwargs["ssl"] is True
    assert kwargs["ssl_check_hostname"] is True
    assert kwargs["ssl_ca_certs"] == "/etc/orchestra/ca/ca-bundle.crt"
    assert kwargs["sentinel_kwargs"]["password"] == "sentinel-secret"
    assert captured["service_name"] == "orchestra-model-routing"
    assert captured["master_kwargs"] == {"check_connection": True}
    assert master.wait_calls == [(1, 750)]
    assert master.closed
    assert all(client.closed for client in discovery_clients)


@pytest.mark.parametrize(
    "overrides",
    [
        {"sentinels": [{"host": "sentinel.tenant", "port": 26379}] * 2},
        {
            "sentinels": [
                {"host": "sentinel-0.tenant", "port": 26379},
                {"host": "sentinel-0.tenant", "port": 26379},
                {"host": "sentinel-2.tenant", "port": 26379},
            ]
        },
        {"minOtherSentinels": 3},
        {"serviceName": "bad service"},
        {"unexpected": True},
    ],
)
def test_sentinel_source_rejects_invalid_or_ambiguous_topology(
    tmp_path: Path,
    overrides: dict,
) -> None:
    config_file = tmp_path / "sentinel.json"
    write_sentinel_config(config_file, **overrides)

    with pytest.raises(ModelRoutingRuntimeConfigError) as raised:
        build_model_routing_rate_limiter(
            settings_for(
                model_routing_rate_limit_scope="deployment-shared",
                model_routing_rate_limit_sentinel_config_file=str(config_file),
            )
        )

    assert raised.value.code == "rate_limit_sentinel_config_invalid"
    assert "data-secret" not in str(raised.value)
    assert "sentinel-secret" not in str(raised.value)


def test_sentinel_source_requires_tls_for_remote_discovery(tmp_path: Path) -> None:
    config_file = tmp_path / "sentinel.json"
    write_sentinel_config(config_file, tls=False, caFile=None)

    with pytest.raises(ModelRoutingRuntimeConfigError) as raised:
        build_model_routing_rate_limiter(
            settings_for(
                model_routing_rate_limit_scope="deployment-shared",
                model_routing_rate_limit_sentinel_config_file=str(config_file),
            )
        )

    assert raised.value.code == "rate_limit_backend_tls_required"


def test_sentinel_ack_timeout_must_fit_inside_operation_timeout(tmp_path: Path) -> None:
    config_file = tmp_path / "sentinel.json"
    write_sentinel_config(config_file, replicaAckTimeoutMilliseconds=100)

    with pytest.raises(ModelRoutingRuntimeConfigError) as raised:
        build_model_routing_rate_limiter(
            settings_for(
                model_routing_rate_limit_scope="deployment-shared",
                model_routing_rate_limit_sentinel_config_file=str(config_file),
                model_routing_rate_limit_operation_timeout_seconds=0.1,
            )
        )

    assert raised.value.code == "rate_limit_sentinel_config_invalid"


def test_sentinel_source_reports_missing_file_without_path_or_secret() -> None:
    with pytest.raises(ModelRoutingRuntimeConfigError) as raised:
        build_model_routing_rate_limiter(
            settings_for(
                model_routing_rate_limit_scope="deployment-shared",
                model_routing_rate_limit_sentinel_config_file="/missing/sentinel-secret.json",
            )
        )

    assert raised.value.code == "rate_limit_sentinel_config_unavailable"
    assert str(raised.value) == "rate_limit_sentinel_config_unavailable"


def test_sentinel_limiter_fails_closed_when_replica_acknowledgement_is_missing() -> None:
    limiter = RedisModelRoutingRateLimiter(
        FakeRedis(acknowledged_replicas=0),
        key_prefix="orchestra:test",
        required_replica_acks=1,
        replica_ack_timeout_milliseconds=50,
    )

    limiter.ping()

    with pytest.raises(ModelRoutingEnforcementError) as request_error:
        consume(limiter)
    assert request_error.value.code == "rate_limit_backend_unavailable"
    assert request_error.value.retry_after_seconds == 1


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
