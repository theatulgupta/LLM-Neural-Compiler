"""Groq preset of the generic OpenAI-compatible client. Prefer compiler.llm.provider."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from compiler.llm.provider import (
    ChatFn,
    HttpLlmClient,
    chat_completions,
    extract_json_object,
    get_provider,
    load_api_key,
    strip_invented_metrics,
)

GROQ_CHAT_URL = get_provider("groq").chat_url
DEFAULT_GROQ_MODEL = get_provider("groq").default_model
GROQ_ENV_PATH = Path.home() / ".config" / "nnc" / "groq.env"


def load_groq_api_key() -> str | None:
    return load_api_key(get_provider("groq"))


def groq_chat(
    messages: list[dict[str, str]],
    model: str,
    *,
    api_key: str,
    timeout: float = 30.0,
    sleep: Callable[[float], None] | None = None,
    urlopen: Callable[..., Any] | None = None,
) -> str:
    return chat_completions(
        messages,
        get_provider("groq"),
        model=model,
        api_key=api_key,
        timeout=timeout,
        sleep=sleep,
        urlopen=urlopen,
    )


class GroqLlmClient(HttpLlmClient):
    """Back-compat name. Same client as HttpLlmClient(provider='groq')."""

    def __init__(
        self,
        *,
        model: str | None = None,
        chat: ChatFn | None = None,
        api_key: str | None = None,
    ) -> None:
        super().__init__(provider="groq", model=model, chat=chat, api_key=api_key)
