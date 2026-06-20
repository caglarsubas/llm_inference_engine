from __future__ import annotations

import base64
from pathlib import Path

from fastapi.testclient import TestClient

from scripts.serve_sida_openai import (
    SidaInput,
    _extract_sida_input,
    _materialize_image,
    create_app,
)


def test_extract_sida_input_from_multimodal_openai_message() -> None:
    result = _extract_sida_input(
        [
            {"role": "system", "content": "Return JSON."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Classify this vehicle image."},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,AAAA"},
                    },
                ],
            },
        ]
    )

    assert isinstance(result, SidaInput)
    assert result.prompt == "Return JSON.\nClassify this vehicle image."
    assert result.image_url == "data:image/png;base64,AAAA"


def test_materialize_image_decodes_data_url() -> None:
    raw = b"not really an image, but enough for routing"
    path = _materialize_image(
        "data:image/jpeg;base64," + base64.b64encode(raw).decode("ascii")
    )

    try:
        assert path.suffix == ".jpg"
        assert path.read_bytes() == raw
    finally:
        path.unlink()


class _FakeRuntime:
    served_model_name = "saberzl/SIDA-13B"

    def __init__(self) -> None:
        self.loaded = False
        self.calls: list[tuple[str, Path, int]] = []

    def load(self) -> None:
        self.loaded = True

    def generate(self, prompt: str, image_path: Path, *, max_tokens: int) -> str:
        self.calls.append((prompt, image_path, max_tokens))
        assert image_path.exists()
        return "[CLS] This image is classified as real."


def test_sida_worker_exposes_openai_models_and_chat() -> None:
    runtime = _FakeRuntime()
    with TestClient(create_app(runtime)) as client:
        models = client.get("/v1/models")
        assert models.status_code == 200
        assert models.json()["data"][0]["id"] == "saberzl/SIDA-13B"
        assert runtime.loaded is True

        image = "data:image/png;base64," + base64.b64encode(b"fake").decode("ascii")
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "saberzl/SIDA-13B",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Classify this."},
                            {"type": "image_url", "image_url": {"url": image}},
                        ],
                    }
                ],
                "max_tokens": 12,
            },
        )

        assert response.status_code == 200
        payload = response.json()
        assert payload["model"] == "saberzl/SIDA-13B"
        assert payload["choices"][0]["message"]["content"].startswith("[CLS]")
        assert runtime.calls[0][0] == "Classify this."
        assert runtime.calls[0][2] == 12


def test_sida_worker_rejects_wrong_model() -> None:
    with TestClient(create_app(_FakeRuntime())) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "other", "messages": [], "max_tokens": 1},
        )

    assert response.status_code == 404
