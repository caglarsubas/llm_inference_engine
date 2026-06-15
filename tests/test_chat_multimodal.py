from inference_engine.api.chat import _last_user_prompt
from inference_engine.schemas import ChatMessage, chat_content_text


def test_chat_content_text_extracts_text_parts_from_multimodal_content() -> None:
    content = [
        {"type": "text", "text": "Classify this vehicle photo."},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,abc"}},
        {"type": "text", "text": "Return strict JSON."},
    ]

    msg = ChatMessage(role="user", content=content)

    assert chat_content_text(msg.content) == "Classify this vehicle photo.\nReturn strict JSON."


def test_last_user_prompt_uses_text_parts_for_auto_eval_prompt() -> None:
    messages = [
        ChatMessage(role="system", content="You are a vehicle-photo judge."),
        ChatMessage(
            role="user",
            content=[
                {"type": "text", "text": "Find anomalies."},
                {"type": "image_url", "image_url": {"url": "https://example.test/photo.jpg"}},
            ],
        ),
    ]

    assert _last_user_prompt(messages) == "Find anomalies."
