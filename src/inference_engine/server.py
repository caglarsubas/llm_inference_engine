"""Fail-closed HTTP/TLS launcher for the inference-engine container."""

from __future__ import annotations

import json
import os
import ssl
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional

import uvicorn


class ServerConfigurationError(RuntimeError):
    """Stable listener configuration failure without certificate-path leakage."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code.replace("_", " "))


@dataclass(frozen=True)
class ServerTlsConfig:
    certificate_file: Path
    private_key_file: Path
    client_ca_file: Optional[Path] = None
    require_client_certificate: bool = False


@dataclass(frozen=True)
class ServerSettings:
    host: str
    port: int
    log_level: str
    tls: Optional[ServerTlsConfig] = None


def _strict_boolean(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized not in {"true", "false"}:
        raise ServerConfigurationError("server_tls_configuration_invalid")
    return normalized == "true"


def load_server_settings(environment: Optional[Mapping[str, str]] = None) -> ServerSettings:
    values = os.environ if environment is None else environment
    host = values.get("HOST", "127.0.0.1")
    if not host or host != host.strip() or len(host) > 255:
        raise ServerConfigurationError("server_listen_address_invalid")
    try:
        port = int(values.get("PORT", "8080"))
    except ValueError:
        raise ServerConfigurationError("server_listen_address_invalid") from None
    if not 1 <= port <= 65535:
        raise ServerConfigurationError("server_listen_address_invalid")
    log_level = values.get("LOG_LEVEL", "INFO")
    if log_level.upper() not in {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "TRACE"}:
        raise ServerConfigurationError("server_log_level_invalid")

    certificate = values.get("INFERENCE_ENGINE_SERVER_TLS_CERT_FILE", "")
    private_key = values.get("INFERENCE_ENGINE_SERVER_TLS_KEY_FILE", "")
    client_ca = values.get("INFERENCE_ENGINE_SERVER_TLS_CLIENT_CA_FILE", "")
    require_client = _strict_boolean(
        values.get("INFERENCE_ENGINE_SERVER_TLS_REQUIRE_CLIENT_CERTIFICATE", "false")
    )
    if bool(certificate) != bool(private_key):
        raise ServerConfigurationError("server_tls_configuration_invalid")
    if (client_ca or require_client) and not certificate:
        raise ServerConfigurationError("server_tls_configuration_invalid")
    if require_client and not client_ca:
        raise ServerConfigurationError("server_tls_client_ca_required")

    tls = None
    if certificate:
        tls = ServerTlsConfig(
            certificate_file=Path(certificate),
            private_key_file=Path(private_key),
            client_ca_file=Path(client_ca) if client_ca else None,
            require_client_certificate=require_client,
        )
    return ServerSettings(
        host=host,
        port=port,
        log_level=log_level,
        tls=tls,
    )


def build_server_ssl_context(config: ServerTlsConfig) -> ssl.SSLContext:
    try:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        context.options |= ssl.OP_NO_COMPRESSION
        context.load_cert_chain(
            certfile=str(config.certificate_file),
            keyfile=str(config.private_key_file),
        )
        if config.client_ca_file is not None:
            context.load_verify_locations(cafile=str(config.client_ca_file))
        context.verify_mode = (
            ssl.CERT_REQUIRED if config.require_client_certificate else ssl.CERT_NONE
        )
        return context
    except (OSError, ssl.SSLError, ValueError):
        raise ServerConfigurationError("server_tls_material_invalid") from None


def build_uvicorn_config(
    settings: ServerSettings,
    *,
    app="inference_engine.main:app",
) -> uvicorn.Config:
    tls = settings.tls
    ssl_context = build_server_ssl_context(tls) if tls is not None else None
    config = uvicorn.Config(
        app,
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
        ssl_certfile=str(tls.certificate_file) if tls is not None else None,
        ssl_keyfile=str(tls.private_key_file) if tls is not None else None,
        ssl_ca_certs=(
            str(tls.client_ca_file)
            if tls is not None and tls.client_ca_file is not None
            else None
        ),
        ssl_cert_reqs=(
            ssl.CERT_REQUIRED
            if tls is not None and tls.require_client_certificate
            else ssl.CERT_NONE
        ),
    )
    try:
        config.load()
    except (OSError, ssl.SSLError, ValueError):
        code = "server_tls_material_invalid" if tls is not None else "server_startup_invalid"
        raise ServerConfigurationError(code) from None
    if ssl_context is not None:
        config.ssl = ssl_context
    return config


def main() -> int:
    try:
        settings = load_server_settings()
        config = build_uvicorn_config(settings)
    except ServerConfigurationError as exc:
        print(
            json.dumps(
                {"type": "inference-engine.server", "status": "failed", "code": exc.code},
                separators=(",", ":"),
            ),
            file=sys.stderr,
        )
        return 2
    uvicorn.Server(config).run()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ServerConfigurationError",
    "ServerSettings",
    "ServerTlsConfig",
    "build_server_ssl_context",
    "build_uvicorn_config",
    "load_server_settings",
    "main",
]
