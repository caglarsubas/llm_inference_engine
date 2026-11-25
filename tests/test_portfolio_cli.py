from __future__ import annotations

import json
from hashlib import sha256
from unittest.mock import Mock, call

import pytest

from inference_engine.portfolio import ModelPortfolio
from scripts import model_portfolio as cli


@pytest.fixture
def portfolio(monkeypatch):
    entries = [
        {
            "id": "org/Encoder",
            "family": "Encoder family",
            "revision": "a" * 40,
            "tasks": ["embedding"],
            "license": "Apache-2.0",
            "size": "0.6B",
            "inputs": ["text"],
            "outputs": ["embedding"],
            "integration": "ollama",
            "architecture": "dense",
            "release": "2026-01",
            "description": "Text embeddings",
        },
        {
            "id": "org/BPE",
            "family": "Tokenizer family",
            "revision": "b" * 40,
            "tasks": ["text-tokenization"],
            "license": "Apache-2.0",
            "artifact_files": ["tokenizer.model", "tokenizer.json"],
            "size": "128k vocabulary",
            "inputs": ["text"],
            "outputs": ["token IDs"],
            "integration": "standalone-asset",
            "architecture": "BPE",
            "release": "2026-01",
            "description": "Text tokenizer",
        },
    ]
    data = {"schema_version": 1, "reviewed_on": "2026-11-25", "entries": entries}
    monkeypatch.setattr(cli, "load_portfolio", lambda: ModelPortfolio.model_validate(data))
    return data


def test_list_and_filter_are_offline(portfolio, monkeypatch, capsys):
    download = Mock(side_effect=AssertionError("listing attempted a download"))
    monkeypatch.setattr(cli, "snapshot_download", download)
    assert cli.main([]) == 0
    output = capsys.readouterr().out
    assert "org/Encoder" in output and "org/BPE" in output
    assert "0.6B" in output and "128k vocabulary" in output

    assert cli.main(["--task", "text-tokenization", "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["reviewed_on"] == portfolio["reviewed_on"]
    assert [entry["id"] for entry in result["entries"]] == ["org/BPE"]
    assert result["entries"][0]["artifact_files"] == ["tokenizer.model", "tokenizer.json"]
    download.assert_not_called()


@pytest.mark.parametrize("idx", [0, 1])
def test_download_pins_one_checkpoint_and_honors_artifact_scope(
    idx, portfolio, monkeypatch, tmp_path, capsys,
):
    entry = portfolio["entries"][idx]
    download = Mock(return_value="/snapshots/example")
    monkeypatch.setattr(cli, "snapshot_download", download)
    args = ["download", entry["id"]]
    if idx == 1:
        args += ["--cache-dir", str(tmp_path)]

    assert cli.main(args) == 0
    assert capsys.readouterr().out.strip() == "/snapshots/example"
    download.assert_called_once_with(
        repo_id=entry["id"], revision=entry["revision"],
        allow_patterns=entry.get("artifact_files"), cache_dir=tmp_path if idx == 1 else None,
    )


@pytest.mark.parametrize("args", [
    ["download"], ["download", "org/encoder"], ["download", "org/Encoder", "org/BPE"],
    ["download", "org/BPE", "--task", "text-tokenization"], ["--task", "unlisted"],
])
def test_unknown_or_ambiguous_selection_causes_no_download(args, portfolio, monkeypatch):
    download = Mock()
    monkeypatch.setattr(cli, "snapshot_download", download)
    with pytest.raises(SystemExit) as error:
        cli.main(args)
    assert error.value.code == 2
    download.assert_not_called()


def test_rejects_empty_download_scope(portfolio, monkeypatch):
    entry = portfolio["entries"][1]
    entry["artifact_files"] = []
    download = Mock()
    monkeypatch.setattr(cli, "snapshot_download", download)
    with pytest.raises(SystemExit) as error:
        cli.main(["download", entry["id"]])
    assert error.value.code == 2
    download.assert_not_called()


def test_variant_selection_fetches_only_verified_files(portfolio, monkeypatch, tmp_path, capsys):
    repo_id = "unsloth/Encoder-GGUF"
    revision = "c" * 40
    payloads = {"q4.gguf": b"weights", "mmproj.gguf": b"projector"}
    files = [
        {"path": name, "role": role, "size_bytes": len(data), "sha256": sha256(data).hexdigest()}
        for (name, data), role in zip(payloads.items(), ["model", "projector"], strict=True)
    ]
    variant = {
        "repo_id": repo_id, "format": "gguf", "quantization": "Q4", "revision": revision,
        "license": "Apache-2.0", "runtime_note": "Use a vision runtime.", "files": files,
    }
    portfolio["entries"][0]["variants"] = [variant]
    for name, data in payloads.items():
        (tmp_path / name).write_bytes(data)
    fetch = Mock(side_effect=lambda **kw: str(tmp_path / kw["filename"]))
    monkeypatch.setattr(cli, "hf_hub_download", fetch)
    monkeypatch.setattr(cli, "snapshot_download", Mock(side_effect=AssertionError("full snapshot")))

    assert cli.main(["variants", "org/Encoder", "--task", "embedding", "--json"]) == 0
    row, = json.loads(capsys.readouterr().out)
    assert row["parent_id"] == "org/Encoder"
    assert row["size_bytes"] == sum(map(len, payloads.values()))
    for invalid in [["download", repo_id], ["download", repo_id, "--variant", "Q2"]]:
        with pytest.raises(SystemExit):
            cli.main(invalid)
    fetch.assert_not_called()

    assert cli.main(["download", repo_id, "--variant", "Q4", "--cache-dir", str(tmp_path)]) == 0
    assert fetch.call_args_list == [
        call(repo_id=repo_id, revision=revision, filename=name, cache_dir=tmp_path, force_download=False)
        for name in payloads
    ]
    assert capsys.readouterr().out.splitlines() == [str(tmp_path / name) for name in payloads]
    (tmp_path / "mmproj.gguf").write_bytes(b"corrupted")
    with pytest.raises(OSError, match="SHA-256 mismatch"):
        cli.main(["download", repo_id, "--variant", "Q4"])
    assert capsys.readouterr().out == ""
    assert fetch.call_args.kwargs["filename"] == "mmproj.gguf"
    assert fetch.call_args.kwargs["force_download"] is True

    def repair(**kwargs):
        path = tmp_path / kwargs["filename"]
        if kwargs["force_download"]:
            path.write_bytes(payloads[path.name])
        return str(path)

    fetch.side_effect = repair
    assert cli.main(["download", repo_id, "--variant", "Q4"]) == 0
    assert (tmp_path / "mmproj.gguf").read_bytes() == payloads["mmproj.gguf"]


def test_publisher_repo_can_distribute_both_native_and_gguf_builds(portfolio, monkeypatch):
    parent = portfolio["entries"][0]
    parent["artifact_files"] = ["weights.safetensors"]
    parent["variants"] = [{
        "repo_id": parent["id"], "revision": parent["revision"], "format": "gguf",
        "quantization": "Q4", "license": "Apache-2.0", "runtime_note": "GGUF runtime",
        "files": [{"path": "model.gguf", "role": "model", "sha256": "0" * 64, "size_bytes": 1}],
    }]
    snapshot = Mock(return_value="/cache/native")
    selected_build = Mock(return_value=[])
    monkeypatch.setattr(cli, "snapshot_download", snapshot)
    monkeypatch.setattr(cli, "download_variant", selected_build)
    assert cli.main(["download", parent["id"]]) == 0
    assert snapshot.call_args.kwargs["allow_patterns"] == ["weights.safetensors"]
    assert cli.main(["download", parent["id"], "--variant", "Q4"]) == 0
    assert selected_build.call_args.args[0].files[0].path == "model.gguf"
    snapshot.assert_called_once()
