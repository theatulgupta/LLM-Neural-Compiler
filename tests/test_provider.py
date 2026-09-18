from __future__ import annotations

import json
import urllib.error

from compiler.graph.graph_loader import load_graph
from compiler.graph.graph_summary import summarize_graph
from compiler.llm.llm_client import build_client
from compiler.llm.provider import HttpLlmClient, ProviderSpec, known_providers
from compiler.schema import SchemaError


def test_known_providers_include_groq_and_openai() -> None:
    names = known_providers()
    assert "groq" in names
    assert "openai" in names
    assert "ollama" in names
    assert "gemini" in names
    assert "openrouter" in names


def test_custom_provider_via_injected_chat(tiny_path) -> None:
    spec = ProviderSpec(
        name="lab",
        chat_url="https://example.invalid/v1/chat/completions",
        default_model="local-7b",
        api_key_env="LAB_KEY",
        auth_required=False,
    )

    def fake_chat(messages, model):
        assert model == "local-7b"
        return json.dumps({"strategy": "baseline", "rationale": "tiny graph", "source": "ignored"})

    client = HttpLlmClient(spec, chat=fake_chat, api_key="")
    rec = client.propose(summarize_graph(load_graph(tiny_path)))
    assert rec.source == "lab"
    assert rec.strategy == "baseline"


def test_chat_retries_on_custom_url() -> None:
    from compiler.llm.provider import chat_completions

    spec = ProviderSpec(
        name="lab",
        chat_url="https://example.invalid/v1/chat/completions",
        default_model="x",
        api_key_env="LAB_KEY",
        auth_required=False,
    )
    calls = {"n": 0}
    body = json.dumps({"choices": [{"message": {"content": '{"ok": true}'}}]})

    class _Resp:
        def read(self) -> bytes:
            return body.encode()

        def __enter__(self):
            return self

        def __exit__(self, *args) -> None:
            return None

    def fake_urlopen(request, timeout=30.0):
        calls["n"] += 1
        assert "example.invalid" in request.full_url
        if calls["n"] < 2:
            raise urllib.error.URLError("timeout")
        return _Resp()

    text = chat_completions(
        [{"role": "user", "content": "hi"}],
        spec,
        model="x",
        api_key=None,
        sleep=lambda _s: None,
        urlopen=fake_urlopen,
    )
    assert calls["n"] == 2
    assert "ok" in text


def test_unknown_provider_raises(monkeypatch) -> None:
    monkeypatch.setenv("NNC_LLM", "not-a-vendor")
    monkeypatch.delenv("NNC_LLM_BASE_URL", raising=False)
    try:
        build_client()
    except SchemaError as exc:
        assert "unknown LLM provider" in str(exc)
        return
    raise AssertionError("expected SchemaError")


def test_custom_v1_base_becomes_chat_url(monkeypatch) -> None:
    from compiler.llm.provider import spec_from_env

    monkeypatch.setenv("NNC_LLM", "custom")
    monkeypatch.setenv("NNC_LLM_BASE_URL", "https://lab.example/v1")
    monkeypatch.setenv("NNC_LLM_MODEL", "lab-7b")
    monkeypatch.setenv("NNC_LLM_JSON_OBJECT", "0")
    monkeypatch.setenv("NNC_LLM_AUTH_HEADER", "api-key")
    monkeypatch.setenv("NNC_LLM_AUTH_PREFIX", "")
    spec = spec_from_env()
    assert spec.chat_url == "https://lab.example/v1/chat/completions"
    assert spec.default_model == "lab-7b"
    assert spec.json_object is False
    assert spec.auth_header == "api-key"
    assert spec.auth_prefix == ""


def test_openai_model_override(monkeypatch) -> None:
    from compiler.llm.provider import spec_from_env

    monkeypatch.setenv("NNC_LLM", "openai")
    monkeypatch.setenv("NNC_LLM_MODEL", "gpt-4o")
    spec = spec_from_env()
    assert spec.name == "openai"
    assert spec.default_model == "gpt-4o"
    assert spec.chat_url.endswith("/v1/chat/completions")


def test_http_error_omits_vendor_body() -> None:
    from compiler.llm.provider import chat_completions

    spec = ProviderSpec(
        name="lab",
        chat_url="https://example.invalid/v1/chat/completions",
        default_model="x",
        api_key_env="LAB_KEY",
        auth_required=False,
    )

    class _Err(urllib.error.HTTPError):
        def __init__(self) -> None:
            super().__init__(url="http://x", code=429, msg="rate", hdrs=None, fp=None)

        def read(self) -> bytes:
            return b'{"error":{"message":"org_01shouldnotleak","code":"rate_limit_exceeded"}}'

    def fake_urlopen(request, timeout=30.0):
        raise _Err()

    try:
        chat_completions(
            [{"role": "user", "content": "hi"}],
            spec,
            model="x",
            api_key=None,
            sleep=lambda _s: None,
            urlopen=fake_urlopen,
        )
    except SchemaError as exc:
        text = str(exc)
        assert "429" in text
        assert "rate_limit_exceeded" in text
        assert "org_01" not in text
        assert "shouldnotleak" not in text
        return
    raise AssertionError("expected SchemaError")


def test_azure_style_auth_header() -> None:
    from compiler.llm.provider import chat_completions

    spec = ProviderSpec(
        name="azure",
        chat_url="https://example.invalid/openai/deployments/x/chat/completions",
        default_model="x",
        api_key_env="AZURE_OPENAI_KEY",
        auth_header="api-key",
        auth_prefix="",
        json_object=False,
    )
    captured: dict[str, str] = {}

    class _Resp:
        def read(self) -> bytes:
            return json.dumps({"choices": [{"message": {"content": '{"ok": true}'}}]}).encode()

        def __enter__(self):
            return self

        def __exit__(self, *args) -> None:
            return None

    def fake_urlopen(request, timeout=30.0):
        captured["api-key"] = request.headers.get("Api-key") or request.headers.get("api-key")
        captured["authorization"] = request.headers.get("Authorization") or ""
        return _Resp()

    text = chat_completions(
        [{"role": "user", "content": "hi"}],
        spec,
        model="x",
        api_key="secret-azure",
        urlopen=fake_urlopen,
    )
    assert captured["api-key"] == "secret-azure"
    assert captured["authorization"] == ""
    assert "ok" in text
