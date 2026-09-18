"""HTTP LLM providers. Groq is one named preset; any OpenAI-compatible API works.

Attach a new host with env only:

    NNC_LLM=custom
    NNC_LLM_BASE_URL=https://host/v1/chat/completions
    NNC_LLM_MODEL=the-model-id
    NNC_LLM_API_KEY_ENV=MY_KEY   # or NNC_LLM_API_KEY / ~/.config/nnc/llm.env

A built-in name (groq, openai, …) only fills URL/key-env defaults. Model is
always overridable with NNC_LLM_MODEL. Keys are never logged or written to JSON.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from compiler.graph.graph_summary import GraphSummary
from compiler.llm.llm_client import LlmProposal, proposal_from_dict
from compiler.llm.prompting import build_messages
from compiler.planner.plan import PRESETS
from compiler.schema import SchemaError

CONFIG_DIR = Path.home() / ".config" / "nnc"
ChatFn = Callable[[list[dict[str, str]], str], str]
UrlOpen = Callable[..., Any]
SleepFn = Callable[[float], None]

_METRIC_CLAIM = re.compile(
    r"\b(\d+(\.\d+)?\s*(fps|ips|ms|mhz|gflops)|throughput|p50|p95)\b",
    re.IGNORECASE,
)
_RETRY_HTTP = {408, 429, 500, 502, 503, 504}
_BACKOFF_S = (1.0, 2.0, 4.0)
_CF_UA = "Mozilla/5.0 (X11; Linux aarch64) AppleWebKit/537.36 Chrome/129.0.0.0 Safari/537.36"


def _http_error_message(code: int, detail: str) -> str:
    """Status (+ short error code). Never keep the vendor body; it can hold org ids."""

    hint = ""
    try:
        payload = json.loads(detail)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, dict):
        err = payload.get("error")
        if isinstance(err, dict):
            hint = str(err.get("code") or err.get("type") or "").strip()
        if not hint:
            hint = str(payload.get("code") or payload.get("type") or "").strip()
    if hint and len(hint) < 64:
        return f"LLM HTTP {code} ({hint})"
    return f"LLM HTTP {code}"


@dataclass(frozen=True, slots=True)
class ProviderSpec:
    """One OpenAI-compatible chat endpoint. Add a row; do not fork the client."""

    name: str
    chat_url: str
    default_model: str
    api_key_env: str
    extra_headers: tuple[tuple[str, str], ...] = ()
    json_object: bool = True
    auth_required: bool = True
    auth_header: str = "Authorization"
    auth_prefix: str = "Bearer "


_PROVIDERS: dict[str, ProviderSpec] = {}


def register_provider(spec: ProviderSpec) -> ProviderSpec:
    _PROVIDERS[spec.name] = spec
    return spec


def known_providers() -> tuple[str, ...]:
    return tuple(sorted(_PROVIDERS))


def get_provider(name: str) -> ProviderSpec:
    spec = _PROVIDERS.get(name)
    if spec is None:
        raise SchemaError(
            f"unknown LLM provider {name!r}; known={list(known_providers())} or use NNC_LLM=custom"
        )
    return spec


register_provider(
    ProviderSpec(
        name="groq",
        chat_url="https://api.groq.com/openai/v1/chat/completions",
        default_model="openai/gpt-oss-20b",
        api_key_env="GROQ_API_KEY",
        extra_headers=(("User-Agent", _CF_UA),),
    )
)
register_provider(
    ProviderSpec(
        name="openai",
        chat_url="https://api.openai.com/v1/chat/completions",
        default_model="gpt-4o-mini",
        api_key_env="OPENAI_API_KEY",
    )
)
register_provider(
    ProviderSpec(
        name="together",
        chat_url="https://api.together.xyz/v1/chat/completions",
        default_model="meta-llama/Llama-3.1-8B-Instruct-Turbo",
        api_key_env="TOGETHER_API_KEY",
    )
)
register_provider(
    ProviderSpec(
        name="fireworks",
        chat_url="https://api.fireworks.ai/inference/v1/chat/completions",
        default_model="accounts/fireworks/models/llama-v3p1-8b-instruct",
        api_key_env="FIREWORKS_API_KEY",
    )
)
register_provider(
    ProviderSpec(
        name="openrouter",
        chat_url="https://openrouter.ai/api/v1/chat/completions",
        default_model="openai/gpt-4o-mini",
        api_key_env="OPENROUTER_API_KEY",
    )
)
register_provider(
    ProviderSpec(
        name="ollama",
        chat_url="http://127.0.0.1:11434/v1/chat/completions",
        default_model="llama3.2",
        api_key_env="OLLAMA_API_KEY",
        auth_required=False,
    )
)
register_provider(
    ProviderSpec(
        name="gemini",
        chat_url="https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
        default_model="gemini-2.0-flash",
        api_key_env="GEMINI_API_KEY",
    )
)
register_provider(
    ProviderSpec(
        name="deepseek",
        chat_url="https://api.deepseek.com/chat/completions",
        default_model="deepseek-chat",
        api_key_env="DEEPSEEK_API_KEY",
    )
)
register_provider(
    ProviderSpec(
        name="mistral",
        chat_url="https://api.mistral.ai/v1/chat/completions",
        default_model="mistral-small-latest",
        api_key_env="MISTRAL_API_KEY",
    )
)
register_provider(
    ProviderSpec(
        name="xai",
        chat_url="https://api.x.ai/v1/chat/completions",
        default_model="grok-2-latest",
        api_key_env="XAI_API_KEY",
    )
)


def _normalize_chat_url(url: str) -> str:
    """Accept a host `/v1` base or a full `/chat/completions` URL."""

    cleaned = url.strip().rstrip("/")
    if not cleaned or cleaned.endswith("/chat/completions"):
        return cleaned
    if cleaned.endswith("/v1") or cleaned.endswith("/openai"):
        return cleaned + "/chat/completions"
    return cleaned


def _json_object_from_env(default: bool) -> bool:
    raw = os.environ.get("NNC_LLM_JSON_OBJECT", "").strip().lower()
    if raw in {"0", "false", "no", "off"}:
        return False
    if raw in {"1", "true", "yes", "on"}:
        return True
    return default


def _parse_env_file(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        value = value.strip().strip("'").strip('"')
        if key.strip() and value:
            out[key.strip()] = value
    return out


def _env_files(provider: str) -> list[Path]:
    return [CONFIG_DIR / "llm.env", CONFIG_DIR / f"{provider}.env"]


def load_api_key(spec: ProviderSpec) -> str | None:
    """Resolve a key from process env or ~/.config/nnc. Never logs the value."""

    for env_name in ("NNC_LLM_API_KEY", spec.api_key_env):
        existing = os.environ.get(env_name, "").strip()
        if existing:
            return existing
    for path in _env_files(spec.name):
        parsed = _parse_env_file(path)
        for env_name in ("NNC_LLM_API_KEY", spec.api_key_env, "API_KEY"):
            value = parsed.get(env_name, "").strip()
            if value:
                os.environ[spec.api_key_env] = value
                return value
    return None


def spec_from_env(name: str | None = None) -> ProviderSpec:
    """Built-in preset, or a custom URL from NNC_LLM_BASE_URL / NNC_LLM_MODEL."""

    raw = (name or os.environ.get("NNC_LLM_PROVIDER") or os.environ.get("NNC_LLM") or "").strip().lower()
    url = _normalize_chat_url(os.environ.get("NNC_LLM_BASE_URL", ""))
    model = os.environ.get("NNC_LLM_MODEL", "").strip()
    key_env = os.environ.get("NNC_LLM_API_KEY_ENV", "").strip()
    auth_header = os.environ.get("NNC_LLM_AUTH_HEADER", "").strip()
    auth_prefix = os.environ.get("NNC_LLM_AUTH_PREFIX")
    if raw in {"custom", "openai_compat", "http"} or (raw not in _PROVIDERS and url):
        if not url or not model:
            raise SchemaError(
                "custom LLM needs NNC_LLM_BASE_URL and NNC_LLM_MODEL "
                f"(known presets: {list(known_providers())})"
            )
        spec = ProviderSpec(
            name=raw or "custom",
            chat_url=url,
            default_model=model,
            api_key_env=key_env or "NNC_LLM_API_KEY",
            json_object=_json_object_from_env(True),
            auth_required=bool(key_env or os.environ.get("NNC_LLM_API_KEY")),
            auth_header=auth_header or "Authorization",
            auth_prefix="Bearer " if auth_prefix is None else auth_prefix,
        )
        return spec
    if raw not in _PROVIDERS:
        raise SchemaError(
            f"unknown LLM provider {raw!r}; known={list(known_providers())} "
            "or set NNC_LLM=custom with NNC_LLM_BASE_URL and NNC_LLM_MODEL"
        )
    spec = get_provider(raw)
    spec = replace(
        spec,
        chat_url=url or spec.chat_url,
        default_model=model or spec.default_model,
        api_key_env=key_env or spec.api_key_env,
        json_object=_json_object_from_env(spec.json_object),
        auth_header=auth_header or spec.auth_header,
        auth_prefix=spec.auth_prefix if auth_prefix is None else auth_prefix,
    )
    return spec


def extract_json_object(text: str) -> dict[str, Any]:
    blob = text.strip()
    if "```" in blob:
        parts = blob.split("```")
        for part in parts:
            piece = part.strip()
            if piece.startswith("json"):
                piece = piece[4:].strip()
            if "{" in piece:
                blob = piece
                break
    start = blob.find("{")
    end = blob.rfind("}")
    if start < 0 or end <= start:
        raise SchemaError("LLM reply did not contain a JSON object")
    try:
        payload = json.loads(blob[start : end + 1])
    except json.JSONDecodeError as exc:
        raise SchemaError(f"LLM reply was not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise SchemaError("LLM JSON must be an object")
    return payload


def strip_invented_metrics(rationale: str) -> tuple[str, bool]:
    if not _METRIC_CLAIM.search(rationale):
        return rationale, False
    cleaned = _METRIC_CLAIM.sub("[metric omitted]", rationale).strip()
    if not cleaned:
        cleaned = "allowlisted strategy only; invented metrics stripped"
    return cleaned, True


def chat_completions(
    messages: list[dict[str, str]],
    spec: ProviderSpec,
    *,
    model: str,
    api_key: str | None,
    timeout: float = 30.0,
    sleep: SleepFn | None = None,
    urlopen: UrlOpen | None = None,
) -> str:
    """POST /v1/chat/completions. Retries URLError/timeout/408/429/5xx three times."""

    sleeper = sleep or time.sleep
    opener = urlopen or urllib.request.urlopen
    body_obj: dict[str, Any] = {"model": model, "temperature": 0, "messages": messages}
    if spec.json_object:
        body_obj["response_format"] = {"type": "json_object"}
    body = json.dumps(body_obj).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    if api_key:
        headers[spec.auth_header] = f"{spec.auth_prefix}{api_key}" if spec.auth_prefix else api_key
    headers.update(spec.extra_headers)
    request = urllib.request.Request(spec.chat_url, data=body, method="POST", headers=headers)
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            with opener(request, timeout=timeout) as response:
                raw = response.read().decode("utf-8")
            parsed = json.loads(raw)
            try:
                return str(parsed["choices"][0]["message"]["content"])
            except (KeyError, IndexError, TypeError) as exc:
                raise SchemaError("LLM response missing choices[0].message.content") from exc
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:2000]
            last_error = SchemaError(_http_error_message(exc.code, detail))
            if exc.code not in _RETRY_HTTP or attempt == 2:
                raise last_error from exc
        except urllib.error.URLError as exc:
            last_error = SchemaError(f"LLM request failed: {exc}")
            if attempt == 2:
                raise last_error from exc
        except TimeoutError as exc:
            last_error = SchemaError(f"LLM request failed: {exc}")
            if attempt == 2:
                raise last_error from exc
        sleeper(_BACKOFF_S[attempt])
    raise last_error or SchemaError("LLM request failed")


class HttpLlmClient:
    """Schema-bound planner over any OpenAI-compatible chat API."""

    def __init__(
        self,
        spec: ProviderSpec | None = None,
        *,
        provider: str | None = None,
        model: str | None = None,
        chat: ChatFn | None = None,
        api_key: str | None = None,
    ) -> None:
        self.spec = spec or spec_from_env(provider)
        self.name = self.spec.name
        self.model = model or os.environ.get("NNC_LLM_MODEL", "").strip() or self.spec.default_model
        self._chat = chat
        self._api_key = api_key

    def propose(
        self,
        summary: GraphSummary,
        context: dict | None = None,
        feedback: list[dict[str, Any]] | None = None,
    ) -> LlmProposal:
        key = self._api_key if self._api_key is not None else load_api_key(self.spec)
        if self.spec.auth_required and not key and self._chat is None:
            raise SchemaError(f"{self.spec.api_key_env} is not set; refusing to invent a proposal")
        messages = build_messages(summary, context, feedback=feedback)
        if self._chat is not None:
            content = self._chat(messages, self.model)
        else:
            content = chat_completions(messages, self.spec, model=self.model, api_key=key)
        payload = extract_json_object(content)
        payload["source"] = self.name
        if payload.get("strategy") and payload["strategy"] not in PRESETS:
            raise SchemaError(f"strategy {payload['strategy']!r} is not allowlisted")
        rationale = str(payload.get("rationale", ""))
        cleaned, stripped = strip_invented_metrics(rationale)
        payload["rationale"] = cleaned
        if "plan_id" not in payload and "strategy" in payload:
            payload["plan_id"] = payload["strategy"]
        payload.pop("strategy", None)
        if "steps" not in payload:
            payload["steps"] = []
        if "options" not in payload:
            payload["options"] = {"ort_graph_opt": "disable"}
        if "confidence" not in payload:
            payload["confidence"] = 0.5
        proposal = proposal_from_dict(payload)
        if stripped:
            plan = dict(proposal.plan)
            plan["rationale"] = cleaned
            proposal = LlmProposal(
                strategy=proposal.strategy,
                rationale=cleaned,
                source=self.name,
                plan=plan,
            )
        return proposal
