from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_matrix_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "vlm_request_matrix.py"
    spec = importlib.util.spec_from_file_location("vlm_request_matrix", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_load_candidate_ids_keeps_image_candidates(tmp_path: Path) -> None:
    matrix = _load_matrix_module()
    candidate_file = tmp_path / "candidates.json"
    candidate_file.write_text(
        json.dumps(
            [
                {
                    "name": "qwen2.5-vl-7b-instruct",
                    "tag": "vllm",
                    "modality": "text+image->text",
                },
                {
                    "name": "plain-text-model",
                    "tag": "vllm",
                    "modality": "text->text",
                },
                {
                    "name": "molmo2-8b",
                    "tag": "vllm",
                    "family": "Molmo 2",
                },
            ]
        ),
        encoding="utf-8",
    )

    assert matrix._load_candidate_ids(candidate_file) == [  # noqa: SLF001
        "qwen2.5-vl-7b-instruct:vllm",
        "molmo2-8b:vllm",
    ]


def test_candidate_ids_adds_catalog_vlm_entries_after_files(tmp_path: Path) -> None:
    matrix = _load_matrix_module()
    candidate_file = tmp_path / "candidates.json"
    candidate_file.write_text(
        json.dumps(
            [
                {
                    "name": "qwen3-vl-8b-instruct",
                    "tag": "vllm",
                    "modality": "text+image->text",
                }
            ]
        ),
        encoding="utf-8",
    )
    catalog = {
        "data": [
            {
                "id": "qwen3-vl-8b-instruct:vllm",
                "modality": "text+image->text",
            },
            {
                "id": "qwen2.5-vl-72b-instruct:openrouter",
                "supports_images": True,
            },
        ],
        "unavailable": [
            {
                "id": "sida-13b:vllm",
                "family": "SIDA",
            }
        ],
    }

    ids = matrix._candidate_ids(  # noqa: SLF001
        explicit_models=[],
        candidate_files=[candidate_file],
        catalog_payload=catalog,
        include_catalog_candidates=True,
    )

    assert ids == [
        "qwen3-vl-8b-instruct:vllm",
        "qwen2.5-vl-72b-instruct:openrouter",
        "sida-13b:vllm",
    ]


def test_candidate_ids_skips_raw_mlx_weight_entries_without_image_metadata() -> None:
    matrix = _load_matrix_module()

    ids = matrix._candidate_ids(  # noqa: SLF001
        explicit_models=[],
        candidate_files=[],
        catalog_payload={
            "data": [
                {
                    "id": "Molmo-7B-D-0924-4bit:mlx",
                    "backend": "mlx",
                    "format": "mlx",
                },
                {
                    "id": "minicpm-v-4.5-gguf-q4-k-m:dmr",
                    "backend": "vllm",
                    "format": "vllm",
                },
            ],
            "unavailable": [],
        },
        include_catalog_candidates=True,
    )

    assert ids == ["minicpm-v-4.5-gguf-q4-k-m:dmr"]


def test_catalog_state_reports_downloaded_but_not_served() -> None:
    matrix = _load_matrix_module()
    lookup = matrix._catalog_lookup(  # noqa: SLF001
        {
            "data": [],
            "unavailable": [
                {
                    "id": "qwen3-vl-32b-instruct:vllm",
                    "availability_status": "downloaded_but_not_served",
                    "availability_detail": "local snapshot downloaded",
                    "download_status": "downloaded",
                }
            ],
        }
    )

    state = matrix._catalog_state("qwen3-vl-32b-instruct:vllm", lookup)  # noqa: SLF001

    assert state["catalog_state"] == "unavailable"
    assert state["availability_status"] == "downloaded_but_not_served"
    assert state["download_status"] == "downloaded"


def test_chat_payload_sends_image_and_json_mode() -> None:
    matrix = _load_matrix_module()

    payload = matrix._chat_payload(  # noqa: SLF001
        "internvl3.5-8b:vllm",
        "data:image/png;base64,abc",
        max_tokens=32,
    )

    assert payload["model"] == "internvl3.5-8b:vllm"
    assert payload["max_tokens"] == 32
    assert payload["response_format"] == {"type": "json_object"}
    content = payload["messages"][1]["content"]
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"] == "data:image/png;base64,abc"
