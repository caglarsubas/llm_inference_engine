"""Container listener TLS and mTLS contract tests."""

from __future__ import annotations

import ipaddress
import ssl
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

from inference_engine.server import (
    ServerConfigurationError,
    ServerSettings,
    ServerTlsConfig,
    build_server_ssl_context,
    build_uvicorn_config,
    load_server_settings,
    main,
)


async def _app(scope, receive, send):
    assert scope["type"] == "http"


def _write_key(path: Path, key) -> None:
    path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.TraditionalOpenSSL,
            serialization.NoEncryption(),
        )
    )


def _create_material(directory: Path):
    now = datetime.now(timezone.utc)
    ca_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    ca_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "engine-test-ca")])
    ca = (
        x509.CertificateBuilder()
        .subject_name(ca_name)
        .issuer_name(ca_name)
        .public_key(ca_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=1))
        .not_valid_after(now + timedelta(days=1))
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .sign(ca_key, hashes.SHA256())
    )
    ca_path = directory / "ca.crt"
    ca_path.write_bytes(ca.public_bytes(serialization.Encoding.PEM))

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    certificate = (
        x509.CertificateBuilder()
        .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "engine")]))
        .issuer_name(ca.subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=1))
        .not_valid_after(now + timedelta(days=1))
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(
            x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False
        )
        .add_extension(
            x509.SubjectAlternativeName(
                [
                    x509.DNSName("localhost"),
                    x509.IPAddress(ipaddress.ip_address("127.0.0.1")),
                ]
            ),
            critical=False,
        )
        .sign(ca_key, hashes.SHA256())
    )
    cert_path = directory / "server.crt"
    key_path = directory / "server.key"
    cert_path.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
    _write_key(key_path, key)
    return ca_path, cert_path, key_path


def test_default_listener_remains_plain_http():
    settings = load_server_settings({})
    assert settings == ServerSettings(host="127.0.0.1", port=8080, log_level="INFO")
    config = build_uvicorn_config(settings, app=_app)
    assert config.is_ssl is False
    assert config.ssl is None


def test_server_tls_context_has_explicit_minimum(tmp_path):
    ca, cert, key = _create_material(tmp_path)
    settings = load_server_settings(
        {
            "HOST": "0.0.0.0",
            "PORT": "8443",
            "INFERENCE_ENGINE_SERVER_TLS_CERT_FILE": str(cert),
            "INFERENCE_ENGINE_SERVER_TLS_KEY_FILE": str(key),
            "INFERENCE_ENGINE_SERVER_TLS_CLIENT_CA_FILE": str(ca),
        }
    )
    config = build_uvicorn_config(settings, app=_app)
    assert config.is_ssl is True
    assert isinstance(config.ssl, ssl.SSLContext)
    assert config.ssl.minimum_version == ssl.TLSVersion.TLSv1_2
    assert config.ssl.verify_mode == ssl.CERT_NONE


def test_mutual_tls_requires_and_activates_client_ca(tmp_path):
    ca, cert, key = _create_material(tmp_path)
    context = build_server_ssl_context(
        ServerTlsConfig(
            certificate_file=cert,
            private_key_file=key,
            client_ca_file=ca,
            require_client_certificate=True,
        )
    )
    assert context.verify_mode == ssl.CERT_REQUIRED

    with pytest.raises(ServerConfigurationError) as caught:
        load_server_settings(
            {
                "INFERENCE_ENGINE_SERVER_TLS_CERT_FILE": str(cert),
                "INFERENCE_ENGINE_SERVER_TLS_KEY_FILE": str(key),
                "INFERENCE_ENGINE_SERVER_TLS_REQUIRE_CLIENT_CERTIFICATE": "true",
            }
        )
    assert caught.value.code == "server_tls_client_ca_required"


@pytest.mark.parametrize(
    ("environment", "code"),
    [
        ({"PORT": "0"}, "server_listen_address_invalid"),
        ({"PORT": "many"}, "server_listen_address_invalid"),
        ({"HOST": " leading"}, "server_listen_address_invalid"),
        ({"LOG_LEVEL": "verbose"}, "server_log_level_invalid"),
        (
            {"INFERENCE_ENGINE_SERVER_TLS_CERT_FILE": "server.crt"},
            "server_tls_configuration_invalid",
        ),
        (
            {"INFERENCE_ENGINE_SERVER_TLS_REQUIRE_CLIENT_CERTIFICATE": "sometimes"},
            "server_tls_configuration_invalid",
        ),
    ],
)
def test_invalid_listener_configuration_has_stable_code(environment, code):
    with pytest.raises(ServerConfigurationError) as caught:
        load_server_settings(environment)
    assert caught.value.code == code


def test_invalid_certificate_material_has_stable_code(tmp_path):
    with pytest.raises(ServerConfigurationError) as caught:
        build_server_ssl_context(
            ServerTlsConfig(
                certificate_file=tmp_path / "missing.crt",
                private_key_file=tmp_path / "missing.key",
            )
        )
    assert caught.value.code == "server_tls_material_invalid"


def test_main_reports_configuration_failure_without_paths(monkeypatch, capsys):
    monkeypatch.setenv("PORT", "invalid")
    assert main() == 2
    error = capsys.readouterr().err
    assert error.strip() == (
        '{"type":"inference-engine.server","status":"failed",'
        '"code":"server_listen_address_invalid"}'
    )
