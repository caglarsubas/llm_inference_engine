"""OpenRouter adapter.

OpenRouter exposes an OpenAI-compatible HTTP API, so the wire protocol is the
same as the vLLM adapter. The differences are operational: every request must
carry the OpenRouter bearer and every span/response should identify the key
source as ``openrouter-api-key``.
"""

from __future__ import annotations

from ..config import settings
from .vllm_adapter import VLLMAdapter


class OpenRouterAdapter(VLLMAdapter):
    backend_name = "openrouter"
    descriptor_format = "openrouter"
    request_key_source = "openrouter-api-key"

    def _headers(self) -> dict[str, str]:
        api_key = settings.openrouter_api_key.strip()
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is required for OpenRouter models")

        headers = {"Authorization": f"Bearer {api_key}"}
        referer = settings.openrouter_http_referer.strip()
        if referer:
            headers["HTTP-Referer"] = referer
        title = settings.openrouter_app_title.strip()
        if title:
            headers["X-Title"] = title
        return headers


__all__ = ["OpenRouterAdapter"]
