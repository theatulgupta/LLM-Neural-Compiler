"""Groq OpenAI-compatible chat client. API keys never go into run JSON."""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

from compiler.graph.graph_summary import GraphSummary
from compiler.llm.context_builder import render_user_prompt
from compiler.llm.llm_client import LlmProposal, proposal_from_dict
from compiler.llm.prompts import SYSTEM_PROMPT, user_prompt
from compiler.schema_validate import SchemaError

GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_GROQ_MODEL = "openai/gpt-oss-20b"
GROQ_ENV_PATH = Path.home() / ".config" / "nnc" / "groq.env"
_METRIC_CLAIM = re.compile(
    r"\b(\d+(\.\d+)?\s*(fps|ips|ms|mhz|gflops)|throughput|p50|p95)\b",
    re.IGNORECASE,
)

ChatFn = Callable[[list[dict[str, str]], str], str]


def load_groq_api_key() -> str | None:
    """Load GROQ_API_KEY from the process env or ~/.config/nnc/groq.env. Never logs the value."""

    existing = os.environ.get("GROQ_API_KEY", "").strip()
    if existing:
        return existing
    if not GROQ_ENV_PATH.is_file():
        return None
    for line in GROQ_ENV_PATH.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("#") or not stripped:
            continue
        if stripped.startswith("GROQ_API_KEY="):
            value = stripped.split("=", 1)[1].strip().strip("'").strip('"')
            if value:
                os.environ["GROQ_API_KEY"] = value
                return value
    return None


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


def groq_chat(messages: list[dict[str, str]], model: str, *, api_key: str, timeout: float = 30.0) -> str:
    body = json.dumps(
        {
            "model": model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": messages,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        GROQ_CHAT_URL,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            # Cloudflare 1010 without a browser-like UA on this QEMU host.
            "User-Agent": "Mozilla/5.0 (X11; Linux aarch64) AppleWebKit/537.36 Chrome/129.0.0.0 Safari/537.36",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise SchemaError(f"Groq HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise SchemaError(f"Groq request failed: {exc}") from exc
    parsed = json.loads(raw)
    try:
        return str(parsed["choices"][0]["message"]["content"])
    except (KeyError, IndexError, TypeError) as exc:
        raise SchemaError("Groq response missing choices[0].message.content") from exc


class GroqLlmClient:
    name = "groq"

    def __init__(
        self,
        *,
        model: str | None = None,
        chat: ChatFn | None = None,
        api_key: str | None = None,
    ) -> None:
        self.model = model or os.environ.get("NNC_GROQ_MODEL", DEFAULT_GROQ_MODEL)
        self._chat = chat
        self._api_key = api_key

    def propose(self, summary: GraphSummary, context: dict | None = None) -> LlmProposal:
        key = self._api_key if self._api_key is not None else load_groq_api_key()
        if not key and self._chat is None:
            raise SchemaError("GROQ_API_KEY is not set; refusing to invent a proposal")
        prompt = render_user_prompt(context) if context is not None else user_prompt(summary)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        if self._chat is not None:
            content = self._chat(messages, self.model)
        else:
            assert key is not None
            content = groq_chat(messages, self.model, api_key=key)
        payload = extract_json_object(content)
        payload["source"] = "groq"
        from compiler.planner.plan import PRESETS

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
                source="groq",
                plan=plan,
            )
        return proposal
