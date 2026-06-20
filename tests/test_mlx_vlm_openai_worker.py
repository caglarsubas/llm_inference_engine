from __future__ import annotations

import base64
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from scripts.serve_mlx_vlm_openai import (
    ImageResource,
    MLXVLMRuntime,
    VLMInput,
    create_app,
    extract_vlm_input,
    materialize_image,
)


def test_extract_vlm_input_from_multimodal_openai_message() -> None:
    result = extract_vlm_input(
        [
            {"role": "system", "content": "Return concise JSON."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe vehicle damage."},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,AAAA"},
                    },
                ],
            },
        ],
        worker_label="InternVL",
    )

    assert isinstance(result, VLMInput)
    assert result.prompt == "Return concise JSON.\nDescribe vehicle damage."
    assert result.image_url == "data:image/png;base64,AAAA"


def test_materialize_image_decodes_data_url() -> None:
    raw = b"tiny image stand-in"
    image = materialize_image(
        "data:image/webp;base64," + base64.b64encode(raw).decode("ascii")
    )

    try:
        assert isinstance(image, ImageResource)
        assert image.remove_after_use is True
        path = Path(image.locator)
        assert path.suffix == ".webp"
        assert path.read_bytes() == raw
    finally:
        Path(image.locator).unlink()


def test_materialize_image_allows_http_url() -> None:
    image = materialize_image("https://example.test/photo.jpg")

    assert image == ImageResource(locator="https://example.test/photo.jpg")


class _FakeRuntime:
    served_model_name = "OpenGVLab/InternVL3_5-8B"
    worker_label = "InternVL3.5 worker"

    def __init__(self) -> None:
        self.loaded = False
        self.calls: list[tuple[str, str, int, float | None]] = []

    def load(self) -> None:
        self.loaded = True

    def generate(
        self,
        prompt: str,
        image_locator: str,
        *,
        max_tokens: int,
        temperature: float | None,
    ) -> SimpleNamespace:
        self.calls.append((prompt, image_locator, max_tokens, temperature))
        assert Path(image_locator).exists()
        return SimpleNamespace(
            text='{"vehicle_visible": true}',
            prompt_tokens=7,
            generation_tokens=3,
            finish_reason="stop",
        )


def test_mlx_vlm_worker_exposes_openai_models_and_chat() -> None:
    runtime = _FakeRuntime()
    with TestClient(create_app(runtime)) as client:
        models = client.get("/v1/models")
        assert models.status_code == 200
        assert models.json()["data"][0]["id"] == "OpenGVLab/InternVL3_5-8B"
        assert runtime.loaded is True

        image = "data:image/png;base64," + base64.b64encode(b"fake").decode("ascii")
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "OpenGVLab/InternVL3_5-8B",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Return strict JSON."},
                            {"type": "image_url", "image_url": {"url": image}},
                        ],
                    }
                ],
                "max_tokens": 12,
                "temperature": 0,
            },
        )

        assert response.status_code == 200
        payload = response.json()
        assert payload["model"] == "OpenGVLab/InternVL3_5-8B"
        assert payload["choices"][0]["message"]["content"] == '{"vehicle_visible": true}'
        assert payload["usage"]["total_tokens"] == 10
        assert runtime.calls[0][0] == "Return strict JSON."
        assert runtime.calls[0][2] == 12
        assert runtime.calls[0][3] == 0


def test_mlx_vlm_runtime_accepts_local_or_hf_model_path() -> None:
    runtime = MLXVLMRuntime(
        model_path="/tmp/model",
        served_model_name="OpenGVLab/InternVL3_5-8B",
        worker_label="InternVL worker",
    )

    assert runtime.model_path == "/tmp/model"
    assert runtime.served_model_name == "OpenGVLab/InternVL3_5-8B"
    assert runtime.worker_label == "InternVL worker"


def test_mlx_vlm_worker_rejects_wrong_model() -> None:
    with TestClient(create_app(_FakeRuntime())) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "other", "messages": [], "max_tokens": 1},
        )

    assert response.status_code == 404
