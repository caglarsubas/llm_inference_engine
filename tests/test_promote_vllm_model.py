from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "promote_vllm_model.py"


def _run(*args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
    )


def test_promote_vllm_model_appends_fakeshield_from_demanded_manifest(tmp_path: Path) -> None:
    live = tmp_path / ".vllm_models.json"
    demanded = ROOT / ".vllm_models.demanded.example.json"
    live.write_text(
        json.dumps(
            [
                {
                    "name": "minicpm-v-4.5-gguf-q4-k-m",
                    "tag": "dmr",
                    "endpoint": "http://127.0.0.1:12434/engines",
                    "model_id": "docker.io/local/minicpm-v-4.5-gguf:q4_k_m",
                }
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = _run(
        "fakeshield-22b:vllm",
        "--live-file",
        str(live),
        "--demanded-file",
        str(demanded),
        "--endpoint",
        "http://fakeshield.example:8000",
        "--strict-image-json-checked-at",
        "2026-06-20",
        cwd=tmp_path,
    )

    assert result.returncode == 0, result.stderr
    entries = json.loads(live.read_text(encoding="utf-8"))
    assert [f"{entry['name']}:{entry.get('tag', 'vllm')}" for entry in entries] == [
        "minicpm-v-4.5-gguf-q4-k-m:dmr",
        "fakeshield-22b:vllm",
    ]
    fakeshield = entries[1]
    assert fakeshield["endpoint"] == "http://fakeshield.example:8000"
    assert fakeshield["model_id"] == "zhipeixu/fakeshield-v1-22b"
    assert fakeshield["supports_strict_image_json"] is False
    assert fakeshield["strict_image_json_status"] == "pending_smoke"
    assert fakeshield["strict_image_json_checked_at"] == "2026-06-20"
    assert fakeshield["benchmark_only"] is True


def test_promote_vllm_model_replaces_existing_entry(tmp_path: Path) -> None:
    live = tmp_path / ".vllm_models.json"
    demanded = ROOT / ".vllm_models.demanded.example.json"
    live.write_text(
        json.dumps(
            [
                {
                    "name": "fakeshield-22b",
                    "tag": "vllm",
                    "endpoint": "http://old:8000",
                    "model_id": "zhipeixu/fakeshield-v1-22b",
                }
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = _run(
        "fakeshield-22b:vllm",
        "--live-file",
        str(live),
        "--demanded-file",
        str(demanded),
        "--endpoint",
        "http://new:8000",
        cwd=tmp_path,
    )

    assert result.returncode == 0, result.stderr
    entries = json.loads(live.read_text(encoding="utf-8"))
    assert len(entries) == 1
    assert entries[0]["endpoint"] == "http://new:8000"
    assert entries[0]["family"] == "FakeShield"


def test_promote_vllm_model_appends_sida13b_from_demanded_manifest(tmp_path: Path) -> None:
    live = tmp_path / ".vllm_models.json"
    demanded = ROOT / ".vllm_models.demanded.example.json"

    result = _run(
        "sida-13b:vllm",
        "--live-file",
        str(live),
        "--demanded-file",
        str(demanded),
        "--endpoint",
        "http://sida.example:8000",
        "--strict-image-json-detail",
        "Issue #46 live descriptor template",
        "--strict-image-json-checked-at",
        "2026-06-20",
        cwd=tmp_path,
    )

    assert result.returncode == 0, result.stderr
    entries = json.loads(live.read_text(encoding="utf-8"))
    assert len(entries) == 1
    sida = entries[0]
    assert sida["name"] == "sida-13b"
    assert sida["endpoint"] == "http://sida.example:8000"
    assert sida["model_id"] == "saberzl/SIDA-13B"
    assert sida["family"] == "SIDA"
    assert sida["supports_strict_image_json"] is False
    assert sida["strict_image_json_status"] == "pending_smoke"
    assert sida["strict_image_json_checked_at"] == "2026-06-20"
    assert "Issue #46" in sida["strict_image_json_detail"]
    assert sida["benchmark_only"] is True


def test_promote_vllm_model_can_require_upstream_probe(tmp_path: Path) -> None:
    live = tmp_path / ".vllm_models.json"
    demanded = ROOT / ".vllm_models.demanded.example.json"

    result = _run(
        "fakeshield-22b:vllm",
        "--live-file",
        str(live),
        "--demanded-file",
        str(demanded),
        "--endpoint",
        "http://127.0.0.1:9",
        "--require-upstream",
        "--upstream-timeout-seconds",
        "0.05",
        cwd=tmp_path,
    )

    assert result.returncode == 1
    assert "GET http://127.0.0.1:9/v1/models failed" in result.stderr
    assert not live.exists()
