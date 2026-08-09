from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
from redis import Redis
from redis.exceptions import ConnectionError as RedisConnectionError

import inference_engine.model_routing_runtime as model_routing_runtime
from inference_engine.model_routing_runtime import (
    MODEL_ROUTING_RATE_LIMIT_SCOPE_PROCESS,
    ModelRoutingEnforcementError,
    ModelRoutingRateLimiter,
    ModelRoutingRuntimeConfigError,
    RedisModelRoutingRateLimiter,
    build_model_routing_rate_limiter,
    observe_model_routing_usage,
    retain_model_routing_reservation,
    settle_model_routing_reservation,
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
        "model_routing_rate_limit_max_window_entries": 1_000,
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


def consume(limiter, *, limit: int = 2):
    return limiter.consume(
        digest="sha256:policy-secret",
        route_id="sensitive-route",
        org_id="org-sensitive",
        tenant="tenant-sensitive",
        limit=limit,
        policy_id="policy-1",
    )


def consume_budget(
    limiter,
    *,
    limit: int | None = 2,
    tokens: int = 900,
    tokens_limit: int | None = 1_000,
    cost_micros: int = 4_000,
    spend_limit: int | None = 50_000,
    budget_window_seconds: int | None = 3_600,
):
    return limiter.consume(
        digest="sha256:policy-secret",
        route_id="sensitive-route",
        org_id="org-sensitive",
        tenant="tenant-sensitive",
        limit=limit,
        policy_id="policy-1",
        tokens=tokens,
        max_tokens_per_minute=tokens_limit,
        cost_micros=cost_micros,
        max_cost_micros_per_window=spend_limit,
        budget_window_seconds=(budget_window_seconds if spend_limit is not None else None),
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
        FakeRedis(result=[0, 1_501, b"rate_limit_exceeded"]),
        key_prefix="orchestra:test",
    )

    with pytest.raises(ModelRoutingEnforcementError) as raised:
        consume(limiter)

    assert raised.value.code == "rate_limit_exceeded"
    assert raised.value.retry_after_seconds == 2


def test_shared_limiter_admits_every_dimension_in_one_round_trip() -> None:
    client = FakeRedis()
    limiter = RedisModelRoutingRateLimiter(client, key_prefix="orchestra:test")

    reservation = consume_budget(limiter)

    assert len(client.eval_calls) == 1
    call = client.eval_calls[0]
    assert call[1] == 5
    window_keys = call[2:7]
    assert window_keys[0].startswith("orchestra:test:rpm:")
    assert window_keys[1].startswith("orchestra:test:tpm:")
    assert window_keys[2] == f"{window_keys[1]}:amounts"
    assert window_keys[3].startswith("orchestra:test:spend:")
    assert window_keys[4] == f"{window_keys[3]}:amounts"
    for key in window_keys:
        assert len(key.split(":")[3]) == 64
    assert reservation is not None
    assert reservation.tokens is not None and reservation.tokens.amount == 900
    assert reservation.spend is not None and reservation.spend.amount == 4_000

    serialized = repr(call)
    for sensitive in ("policy-secret", "sensitive-route", "org-sensitive", "tenant-sensitive"):
        assert sensitive not in serialized


def test_shared_limiter_settles_only_the_amount_dimensions() -> None:
    client = FakeRedis()
    limiter = RedisModelRoutingRateLimiter(client, key_prefix="orchestra:test")
    reservation = consume_budget(limiter)
    assert reservation is not None

    observe_model_routing_usage(reservation, tokens=120, cost_micros=7)
    settle_model_routing_reservation(limiter, reservation)

    assert len(client.eval_calls) == 2
    settle = client.eval_calls[1]
    assert settle[1] == 4
    assert settle[2].startswith("orchestra:test:tpm:")
    assert settle[4].startswith("orchestra:test:spend:")
    assert settle[6] == reservation.nonce
    assert settle[7] == 2
    assert settle[8] == 120
    assert settle[9] == 60_000
    assert settle[10] == 7
    assert settle[11] == 3_600_000

    settle_model_routing_reservation(limiter, reservation)
    assert len(client.eval_calls) == 2


@pytest.mark.parametrize(
    ("denial", "code", "limit_requests", "limit_tokens"),
    [
        (b"rate_limit_exceeded", "rate_limit_exceeded", 2, None),
        (b"token_rate_limit_exceeded", "token_rate_limit_exceeded", None, 1_000),
        (b"budget_exceeded", "budget_exceeded", None, None),
    ],
)
def test_shared_limiter_reports_which_window_denied(
    denial: bytes,
    code: str,
    limit_requests: int | None,
    limit_tokens: int | None,
) -> None:
    limiter = RedisModelRoutingRateLimiter(
        FakeRedis(result=[0, 500, denial]),
        key_prefix="orchestra:test",
    )

    with pytest.raises(ModelRoutingEnforcementError) as raised:
        consume_budget(limiter)

    assert raised.value.code == code
    assert raised.value.limit_requests == limit_requests
    assert raised.value.limit_tokens == limit_tokens
    assert raised.value.retry_after_seconds == 1


def test_shared_limiter_fails_closed_on_the_new_dimensions_too() -> None:
    limiter = RedisModelRoutingRateLimiter(
        FakeRedis(error=RedisConnectionError("unavailable")),
        key_prefix="orchestra:test",
    )

    with pytest.raises(ModelRoutingEnforcementError) as raised:
        consume_budget(limiter, limit=None)

    assert raised.value.code == "rate_limit_backend_unavailable"
    assert raised.value.retry_after_seconds == 1


def test_shared_limiter_skips_the_store_when_no_limit_is_signed() -> None:
    client = FakeRedis()
    limiter = RedisModelRoutingRateLimiter(client, key_prefix="orchestra:test")

    assert consume_budget(limiter, limit=None, tokens_limit=None, spend_limit=None) is None
    assert client.eval_calls == []


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


@pytest.mark.skipif(not os.getenv("TEST_VALKEY_URL"), reason="TEST_VALKEY_URL is not configured")
def test_shared_token_and_spend_windows_are_atomic_across_clients() -> None:
    url = os.environ["TEST_VALKEY_URL"]
    prefix = f"orchestra:test:{uuid4().hex}"
    first = RedisModelRoutingRateLimiter(Redis.from_url(url), key_prefix=prefix)
    second = RedisModelRoutingRateLimiter(Redis.from_url(url), key_prefix=prefix)
    first.ping()
    second.ping()

    def attempt(index: int) -> str:
        try:
            reservation = consume_budget(
                first if index % 2 == 0 else second,
                limit=None,
                tokens=100,
                tokens_limit=1_000,
                cost_micros=250,
                spend_limit=2_500,
            )
            assert reservation is not None
            return "accepted"
        except ModelRoutingEnforcementError as exc:
            assert exc.code in {"token_rate_limit_exceeded", "budget_exceeded"}
            return "denied"

    try:
        with ThreadPoolExecutor(max_workers=10) as pool:
            results = list(pool.map(attempt, range(30)))
    finally:
        first.close()
        second.close()

    # 1_000 tokens at 100 reserved each is the binding window; the spend budget
    # would have allowed the same ten.
    assert results.count("accepted") == 10
    assert results.count("denied") == 20


@pytest.mark.skipif(not os.getenv("TEST_VALKEY_URL"), reason="TEST_VALKEY_URL is not configured")
def test_shared_settlement_returns_over_reserved_budget_to_the_window() -> None:
    url = os.environ["TEST_VALKEY_URL"]
    prefix = f"orchestra:test:{uuid4().hex}"
    limiter = RedisModelRoutingRateLimiter(Redis.from_url(url), key_prefix=prefix)
    limiter.ping()

    try:
        for _ in range(20):
            reservation = consume_budget(
                limiter,
                limit=None,
                tokens=100,
                tokens_limit=1_000,
                spend_limit=None,
            )
            assert reservation is not None
            observe_model_routing_usage(reservation, tokens=1, cost_micros=None)
            settle_model_routing_reservation(limiter, reservation)
    finally:
        limiter.close()


@pytest.mark.skipif(not os.getenv("TEST_VALKEY_URL"), reason="TEST_VALKEY_URL is not configured")
def test_shared_amount_window_refuses_to_retain_unbounded_reservations() -> None:
    url = os.environ["TEST_VALKEY_URL"]
    prefix = f"orchestra:test:{uuid4().hex}"
    limiter = RedisModelRoutingRateLimiter(
        Redis.from_url(url),
        key_prefix=prefix,
        max_window_entries=4,
    )
    limiter.ping()

    try:
        for _ in range(4):
            reservation = consume_budget(
                limiter,
                limit=None,
                tokens=10,
                tokens_limit=1_000_000,
                spend_limit=None,
            )
            assert reservation is not None
            observe_model_routing_usage(reservation, tokens=1, cost_micros=None)
            settle_model_routing_reservation(limiter, reservation)

        with pytest.raises(ModelRoutingEnforcementError) as raised:
            consume_budget(
                limiter,
                limit=None,
                tokens=10,
                tokens_limit=1_000_000,
                spend_limit=None,
            )
        assert raised.value.code == "rate_limit_state_capacity"
    finally:
        limiter.close()


def test_local_limiter_counts_every_dimension_against_its_bucket_ceiling() -> None:
    limiter = ModelRoutingRateLimiter(max_buckets=2, clock=lambda: 10.0)

    with pytest.raises(ModelRoutingEnforcementError) as raised:
        consume_budget(limiter)
    assert raised.value.code == "rate_limit_state_capacity"
    # The bucket ceiling and the entry ceiling deny under one code, so they
    # have to reach the operator through one counter.
    assert limiter.metrics_snapshot()["state_capacity_denials_total"] == 1

    roomy = ModelRoutingRateLimiter(max_buckets=3, clock=lambda: 10.0)
    assert consume_budget(roomy) is not None
    # One bucket per enforced dimension, which is what MAX_BUCKETS counts.
    assert len(roomy._buckets) == 3
    assert {key[0] for key in roomy._buckets} == {"rpm", "tpm", "spend"}
    with pytest.raises(ModelRoutingEnforcementError, match="rate_limit_state_capacity"):
        roomy.consume(
            digest="sha256:policy-secret",
            route_id="other-route",
            org_id="org-sensitive",
            tenant="tenant-sensitive",
            limit=2,
            policy_id="policy-1",
        )


def _window(limiter, name: str):
    return limiter._buckets[
        (name, "sha256:policy-secret", "sensitive-route", "org-sensitive", "tenant-sensitive")
    ]


def test_an_amount_window_refuses_to_retain_unbounded_reservations() -> None:
    limiter = ModelRoutingRateLimiter(max_window_entries=4, clock=lambda: 10.0)

    # Every one of these settles to a single token, so the running total never
    # approaches the limit and nothing else would ever stop them accumulating.
    for _ in range(4):
        reservation = consume_budget(
            limiter,
            limit=None,
            tokens=10,
            tokens_limit=1_000_000,
            spend_limit=None,
        )
        assert reservation is not None
        observe_model_routing_usage(reservation, tokens=1, cost_micros=None)
        settle_model_routing_reservation(limiter, reservation)

    window = _window(limiter, "tpm")
    assert window.total == 4
    assert len(window.amounts) == 4

    with pytest.raises(ModelRoutingEnforcementError) as raised:
        consume_budget(
            limiter,
            limit=None,
            tokens=10,
            tokens_limit=1_000_000,
            spend_limit=None,
        )
    assert raised.value.code == "rate_limit_state_capacity"


def test_a_reservation_settled_to_zero_leaves_no_trace_in_the_local_window() -> None:
    limiter = ModelRoutingRateLimiter(max_window_entries=4, clock=lambda: 10.0)

    for _ in range(8):
        reservation = consume_budget(
            limiter,
            limit=None,
            tokens=10,
            tokens_limit=1_000,
            spend_limit=None,
        )
        assert reservation is not None
        observe_model_routing_usage(reservation, tokens=0, cost_micros=None)
        settle_model_routing_reservation(limiter, reservation)

    window = _window(limiter, "tpm")
    assert window.total == 0
    assert window.amounts == {}
    # A released entry must stop dating the window, or the Retry-After on a
    # later denial is computed from requests that were fully refunded.
    assert window.oldest() is None
    assert len(window.order) == 0


def test_a_window_emptied_by_settlement_is_reclaimed_before_its_time() -> None:
    limiter = ModelRoutingRateLimiter(max_buckets=1, clock=lambda: 10.0)

    reservation = consume_budget(
        limiter,
        limit=None,
        tokens=10,
        tokens_limit=1_000,
        spend_limit=None,
    )
    assert reservation is not None
    observe_model_routing_usage(reservation, tokens=0, cost_micros=None)
    settle_model_routing_reservation(limiter, reservation)

    # The clock has not moved, so only the settlement can have freed the slot.
    other = limiter.consume(
        digest="sha256:policy-secret",
        route_id="other-route",
        org_id="org-sensitive",
        tenant="tenant-sensitive",
        limit=None,
        policy_id="policy-1",
        tokens=10,
        max_tokens_per_minute=1_000,
    )
    assert other is not None
    assert len(limiter._buckets) == 1


def test_a_retained_reservation_keeps_its_admission_reserve() -> None:
    limiter = ModelRoutingRateLimiter(clock=lambda: 10.0)
    reservation = consume_budget(limiter, limit=None)
    assert reservation is not None

    retain_model_routing_reservation(reservation)
    settle_model_routing_reservation(limiter, reservation)

    assert _window(limiter, "tpm").total == 900
    assert _window(limiter, "spend").total == 4_000


@pytest.fixture(
    params=[
        "process-replica",
        pytest.param(
            "deployment-shared",
            marks=pytest.mark.skipif(
                not os.getenv("TEST_VALKEY_URL"),
                reason="TEST_VALKEY_URL is not configured",
            ),
        ),
    ]
)
def limiter_at_a_five_entry_ceiling(request):
    if request.param == MODEL_ROUTING_RATE_LIMIT_SCOPE_PROCESS:
        limiter = ModelRoutingRateLimiter(max_window_entries=5, clock=lambda: 10.0)
    else:
        limiter = RedisModelRoutingRateLimiter(
            Redis.from_url(os.environ["TEST_VALKEY_URL"]),
            key_prefix=f"orchestra:test:{uuid4().hex}",
            max_window_entries=5,
        )
        limiter.ping()
    try:
        yield limiter
    finally:
        limiter.close()


def test_a_signed_request_limit_above_the_entry_ceiling_binds_in_either_scope(
    limiter_at_a_five_entry_ceiling,
) -> None:
    limiter = limiter_at_a_five_entry_ceiling

    # A request window has nothing to settle, so an admitted request carries no
    # reservation back: admission is the absence of a denial.
    for _ in range(10):
        assert consume(limiter, limit=10) is None

    with pytest.raises(ModelRoutingEnforcementError) as raised:
        consume(limiter, limit=10)

    assert raised.value.code == "rate_limit_exceeded"


def test_a_settling_window_hits_the_entry_ceiling_in_either_scope(
    limiter_at_a_five_entry_ceiling,
) -> None:
    limiter = limiter_at_a_five_entry_ceiling

    for _ in range(5):
        reservation = consume_budget(
            limiter,
            limit=None,
            tokens=10,
            tokens_limit=1_000_000,
            spend_limit=None,
        )
        assert reservation is not None
        observe_model_routing_usage(reservation, tokens=1, cost_micros=None)
        settle_model_routing_reservation(limiter, reservation)

    with pytest.raises(ModelRoutingEnforcementError) as raised:
        consume_budget(
            limiter,
            limit=None,
            tokens=10,
            tokens_limit=1_000_000,
            spend_limit=None,
        )

    assert raised.value.code == "rate_limit_state_capacity"
    assert limiter.metrics_snapshot() == {
        "state_capacity_denials_total": 1,
        "window_entries_peak": 5,
        "max_window_entries": 5,
    }


def test_an_exhausted_limit_outranks_a_later_entry_ceiling_in_either_scope(
    limiter_at_a_five_entry_ceiling,
) -> None:
    limiter = limiter_at_a_five_entry_ceiling

    for _ in range(5):
        assert (
            consume_budget(
                limiter,
                limit=5,
                tokens=10,
                tokens_limit=1_000_000,
                spend_limit=None,
            )
            is not None
        )

    with pytest.raises(ModelRoutingEnforcementError) as raised:
        consume_budget(
            limiter,
            limit=5,
            tokens=10,
            tokens_limit=1_000_000,
            spend_limit=None,
        )

    # The request window is exhausted and the token window is at its entry
    # ceiling at the same instant. Requests are the earlier dimension, so the
    # limit answers: a retryable 429, not a 503.
    assert raised.value.code == "rate_limit_exceeded"
    assert raised.value.retry_after_seconds == 60
    assert raised.value.limit_requests == 5


def test_an_entry_ceiling_outranks_a_later_exhausted_limit_in_either_scope(
    limiter_at_a_five_entry_ceiling,
) -> None:
    limiter = limiter_at_a_five_entry_ceiling

    for _ in range(5):
        assert (
            consume_budget(
                limiter,
                limit=None,
                tokens=10,
                tokens_limit=1_000_000,
                cost_micros=4_000,
                spend_limit=20_000,
            )
            is not None
        )

    with pytest.raises(ModelRoutingEnforcementError) as raised:
        consume_budget(
            limiter,
            limit=None,
            tokens=10,
            tokens_limit=1_000_000,
            cost_micros=4_000,
            spend_limit=20_000,
        )

    assert raised.value.code == "rate_limit_state_capacity"
    assert raised.value.retry_after_seconds is None
    assert raised.value.limit_requests is None


@pytest.fixture(
    params=[
        "process-replica",
        pytest.param(
            "deployment-shared",
            marks=pytest.mark.skipif(
                not os.getenv("TEST_VALKEY_URL"),
                reason="TEST_VALKEY_URL is not configured",
            ),
        ),
    ]
)
def limiter_on_a_wall_clock(request):
    if request.param == MODEL_ROUTING_RATE_LIMIT_SCOPE_PROCESS:
        limiter = ModelRoutingRateLimiter()
    else:
        limiter = RedisModelRoutingRateLimiter(
            Redis.from_url(os.environ["TEST_VALKEY_URL"]),
            key_prefix=f"orchestra:test:{uuid4().hex}",
        )
        limiter.ping()
    try:
        yield limiter
    finally:
        limiter.close()


def test_a_backlog_past_the_sweep_bound_still_admits_in_either_scope(
    limiter_on_a_wall_clock,
    monkeypatch,
) -> None:
    monkeypatch.setattr(model_routing_runtime, "MODEL_ROUTING_RATE_LIMIT_EXPIRY_SWEEP_LIMIT", 4)
    limiter = limiter_on_a_wall_clock
    backlog = model_routing_runtime.MODEL_ROUTING_RATE_LIMIT_EXPIRY_SWEEP_LIMIT + 2
    window = dict(limit=None, tokens_limit=None, spend_limit=1_000_000, budget_window_seconds=2)

    started = time.monotonic()
    for _ in range(backlog):
        assert consume_budget(limiter, cost_micros=50_000, **window) is not None

    # One admission halfway through keeps the shared window key from expiring
    # outright, so the backlog has to expire out of a window that still exists.
    time.sleep(max(0.0, 1.4 - (time.monotonic() - started)))
    assert consume_budget(limiter, cost_micros=50_000, **window) is not None
    time.sleep(0.9)

    # 50_000 micros are still live, so this request is exactly the whole budget
    # and nothing but an unreclaimed amount could turn it away.
    assert consume_budget(limiter, cost_micros=950_000, **window) is not None


@pytest.mark.skipif(
    not os.getenv("TEST_VALKEY_URL"),
    reason="TEST_VALKEY_URL is not configured",
)
def test_shared_expiry_sweep_is_bounded_and_amortised_across_admissions() -> None:
    sweep = model_routing_runtime.MODEL_ROUTING_RATE_LIMIT_EXPIRY_SWEEP_LIMIT
    backlog = sweep * 3
    client = Redis.from_url(os.environ["TEST_VALKEY_URL"])
    prefix = f"orchestra:test:{uuid4().hex}"
    limiter = RedisModelRoutingRateLimiter(client, key_prefix=prefix)
    admit = dict(limit=None, tokens=10, tokens_limit=1_000_000, spend_limit=None)
    try:
        limiter.ping()
        assert consume_budget(limiter, **admit) is not None
        window_key = next(
            key.decode() for key in client.keys(f"{prefix}:tpm:*") if not key.endswith(b":amounts")
        )
        amount_key = f"{window_key}:amounts"

        stale = {f"stale-{index}": 1 for index in range(backlog)}
        client.zadd(window_key, stale)
        client.hset(amount_key, mapping=stale)
        client.hincrby(amount_key, "#total", backlog)

        assert consume_budget(limiter, **admit) is not None
        assert client.zcard(window_key) == backlog + 2 - sweep
        assert int(client.hget(amount_key, "#total")) == backlog + 20 - sweep

        assert consume_budget(limiter, **admit) is not None
        assert client.zcard(window_key) == backlog + 3 - (2 * sweep)
    finally:
        leftover = client.keys(f"{prefix}:*")
        if leftover:
            client.delete(*leftover)
        limiter.close()


def test_shared_limiter_carries_the_window_entry_ceiling_into_the_script() -> None:
    client = FakeRedis()
    limiter = RedisModelRoutingRateLimiter(
        client,
        key_prefix="orchestra:test",
        max_window_entries=64,
    )

    assert consume_budget(limiter) is not None

    call = client.eval_calls[0]
    assert call[8] == 3
    assert call[9] == 64


def test_shared_limiter_fails_closed_on_a_window_at_its_entry_ceiling() -> None:
    limiter = RedisModelRoutingRateLimiter(
        FakeRedis(result=[0, 60_000, b"rate_limit_state_capacity"]),
        key_prefix="orchestra:test",
    )

    with pytest.raises(ModelRoutingEnforcementError) as raised:
        consume_budget(limiter)

    assert raised.value.code == "rate_limit_state_capacity"
    assert raised.value.retry_after_seconds is None
