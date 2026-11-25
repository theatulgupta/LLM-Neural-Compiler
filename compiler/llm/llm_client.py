"""LLM clients: deterministic heuristic, mock, unconstrained, structured API, and Outlines FSM.

Every proposal is validated against schemas/llm-proposal.schema.json and then
filtered through the strategy allowlist. Unknown names cannot leak through.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import re
import time
from typing import Any
import urllib.request

from compiler.graph.graph_summary import GraphSummary
from compiler.llm.prompts import SYSTEM_PROMPT, user_prompt
from compiler.schema_validate import SchemaError, validate_llm_proposal


@dataclass(frozen=True, slots=True)
class LlmProposal:
    strategy: str
    rationale: str
    source: str
    config: dict[str, Any] | None = None
    raw_reply: str | None = None
    decision_ms: float | None = None
    attempts: int = 1
    first_try_valid: bool = True

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "strategy": self.strategy,
            "rationale": self.rationale,
            "source": self.source,
        }
        if self.config is not None:
            out["config"] = self.config
        return out

    def telemetry_dict(self) -> dict[str, Any]:
        return asdict(self)


def proposal_from_dict(
    payload: dict[str, Any],
    *,
    raw_reply: str | None = None,
    decision_ms: float | None = None,
    attempts: int = 1,
    first_try_valid: bool = True,
) -> LlmProposal:
    checked = validate_llm_proposal({str(k): v for k, v in payload.items()})
    cfg = checked.get("config")
    return LlmProposal(
        strategy=str(checked["strategy"]),
        rationale=str(checked["rationale"]),
        source=str(checked["source"]),
        config=dict(cfg) if isinstance(cfg, dict) else None,
        raw_reply=raw_reply,
        decision_ms=decision_ms,
        attempts=attempts,
        first_try_valid=first_try_valid,
    )


def _extract_json_object(text: str) -> dict[str, Any]:
    """Parse JSON from raw text, tolerating markdown code fences for unconstrained baselines."""

    cleaned = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, flags=re.DOTALL)
    if fence:
        cleaned = fence.group(1)
    else:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end > start:
            cleaned = cleaned[start : end + 1]
    parsed = json.loads(cleaned)
    if not isinstance(parsed, dict):
        raise ValueError("LLM response did not decode to a JSON object")
    return parsed


def _openai_compatible_chat(
    *,
    base_url: str,
    api_key: str,
    model: str,
    system_msg: str,
    user_msg: str,
    temperature: float = 0.0,
    response_format: dict[str, Any] | None = None,
    timeout_s: float = 60.0,
) -> tuple[str, dict[str, Any]]:
    body: dict[str, Any] = {
        "model": model,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
    }
    if response_format is not None:
        body["response_format"] = response_format

    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "LLM-Neural-Compiler/1.0",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    content = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    return str(content), usage


class HeuristicLlmClient:
    """No network. Maps graph features onto the allowlist."""

    name = "heuristic"

    def propose(
        self,
        summary: GraphSummary,
        *,
        history_feedback: list[dict[str, Any]] | None = None,
    ) -> LlmProposal:
        _ = history_feedback
        if summary.node_count <= 8:
            payload = {
                "strategy": "baseline",
                "rationale": "Tiny graph; disable ORT opts first so the Flatten/Gemm contract is visible.",
                "source": self.name,
            }
        elif summary.op_counts.get("Conv", 0) >= 1:
            payload = {
                "strategy": "ort_extended",
                "rationale": "Convolutional graph benefits from extended ORT fusions on CPU.",
                "source": self.name,
            }
        else:
            payload = {
                "strategy": "ort_all",
                "rationale": "Generic graph; enable all ORT graph optimizations on CPU.",
                "source": self.name,
            }
        return proposal_from_dict(payload)


class MockLlmClient:
    """Deterministic advisor used in tests. Payload must match the proposal schema."""

    name = "mock"

    def __init__(self, proposal: dict[str, object] | None = None) -> None:
        self._proposal = proposal or {
            "strategy": "baseline",
            "rationale": "mock default: keep ORT graph opts off",
            "source": "mock",
        }

    def propose(
        self,
        summary: GraphSummary,
        *,
        history_feedback: list[dict[str, Any]] | None = None,
    ) -> LlmProposal:
        _ = summary, history_feedback
        return proposal_from_dict(dict(self._proposal))


class UnconstrainedLlmClient:
    """RQ1 Baseline: unconstrained text prompting + post-generation validation & retries."""

    name = "unconstrained_llm"

    def __init__(
        self,
        *,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        max_retries: int = 2,
    ) -> None:
        self.model = model or os.environ.get("NNC_LLM_MODEL", "llama-3.1-8b-instant")
        self.base_url = base_url or os.environ.get("NNC_LLM_BASE_URL", "https://api.groq.com/openai/v1")
        self.api_key = api_key or os.environ.get("NNC_LLM_API_KEY") or os.environ.get("GROQ_API_KEY") or "ollama"
        self.max_retries = max_retries

    def propose(
        self,
        summary: GraphSummary,
        *,
        history_feedback: list[dict[str, Any]] | None = None,
    ) -> LlmProposal:
        prompt = user_prompt(summary, history_feedback=history_feedback)
        t0 = time.perf_counter()
        last_err: Exception | None = None
        last_raw = ""
        for attempt in range(1, self.max_retries + 2):
            extra = "" if attempt == 1 else f"\nPrevious output failed validation: {last_err}. Return valid JSON only."
            raw, _ = _openai_compatible_chat(
                base_url=self.base_url,
                api_key=self.api_key,
                model=self.model,
                system_msg=SYSTEM_PROMPT,
                user_msg=prompt + extra,
                temperature=0.0,
                response_format=None,
            )
            last_raw = raw
            try:
                parsed = _extract_json_object(raw)
                parsed.setdefault("source", f"{self.name}:{self.model}")
                return proposal_from_dict(
                    parsed,
                    raw_reply=raw,
                    decision_ms=(time.perf_counter() - t0) * 1000.0,
                    attempts=attempt,
                    first_try_valid=(attempt == 1),
                )
            except Exception as exc:
                last_err = exc
        raise SchemaError(f"Unconstrained LLM failed after {self.max_retries + 1} attempts: {last_err} | raw={last_raw!r}")


class HostedStructuredLlmClient:
    """Hosted or Ollama OpenAI-compatible API with structured JSON mode + deterministic gate."""

    name = "structured_api"

    def __init__(
        self,
        *,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self.model = model or os.environ.get("NNC_LLM_MODEL", "qwen2.5-coder:7b")
        self.base_url = base_url or os.environ.get("NNC_LLM_BASE_URL", "http://127.0.0.1:11434/v1")
        self.api_key = api_key or os.environ.get("NNC_LLM_API_KEY") or os.environ.get("GROQ_API_KEY") or "ollama"

    def propose(
        self,
        summary: GraphSummary,
        *,
        history_feedback: list[dict[str, Any]] | None = None,
    ) -> LlmProposal:
        prompt = user_prompt(summary, history_feedback=history_feedback)
        t0 = time.perf_counter()
        raw, _ = _openai_compatible_chat(
            base_url=self.base_url,
            api_key=self.api_key,
            model=self.model,
            system_msg=SYSTEM_PROMPT,
            user_msg=prompt,
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        parsed = _extract_json_object(raw)
        parsed.setdefault("source", f"{self.name}:{self.model}")
        return proposal_from_dict(
            parsed,
            raw_reply=raw,
            decision_ms=(time.perf_counter() - t0) * 1000.0,
            attempts=1,
            first_try_valid=True,
        )


class ConstrainedOutlinesLlmClient:
    """Local FSM grammar-based logit-masked decoding using `outlines` + Pydantic."""

    name = "outlines_fsm"

    def __init__(
        self,
        *,
        model_name: str | None = None,
        backend: str = "transformers",
    ) -> None:
        self.model_name = model_name or os.environ.get("NNC_OUTLINES_MODEL", "Qwen/Qwen2.5-Coder-1.5B-Instruct")
        self.backend = backend
        self._generator: Any = None
        self._schema_cls: Any = None

    def _ensure_loaded(self) -> None:
        if self._generator is not None:
            return
        try:
            import outlines
            from pydantic import BaseModel, Field
            from typing import Literal
        except ImportError as exc:
            raise RuntimeError(
                "ConstrainedOutlinesLlmClient requires 'outlines' and 'pydantic' installed"
            ) from exc

        class MultiDimConfigModel(BaseModel):
            execution_provider: Literal["CPUExecutionProvider", "XNNPACKExecutionProvider"]
            graph_optimization_level: Literal["disable", "basic", "extended", "all"]
            precision: Literal["fp32", "int8_dynamic", "int8_static"]
            execution_mode: Literal["sequential", "parallel"]
            intra_op_num_threads: Literal[1, 2, 4, 8]
            inter_op_num_threads: Literal[1, 2]
            enable_mem_pattern: bool

        class ProposalModel(BaseModel):
            strategy: Literal["baseline", "ort_basic", "ort_extended", "ort_all"]
            config: MultiDimConfigModel
            rationale: str = Field(min_length=5)
            source: str = Field(min_length=1)

        if self.backend == "llamacpp":
            llm = outlines.models.llamacpp(self.model_name)
        else:
            llm = outlines.models.transformers(self.model_name)

        self._schema_cls = ProposalModel
        self._generator = outlines.generate.json(llm, ProposalModel)

    def propose(
        self,
        summary: GraphSummary,
        *,
        history_feedback: list[dict[str, Any]] | None = None,
    ) -> LlmProposal:
        self._ensure_loaded()
        prompt = f"{SYSTEM_PROMPT}\n\n{user_prompt(summary, history_feedback=history_feedback)}"
        t0 = time.perf_counter()
        result = self._generator(prompt)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        payload = result.model_dump()
        payload["source"] = f"{self.name}:{self.model_name}"
        return proposal_from_dict(
            payload,
            raw_reply=result.model_dump_json(),
            decision_ms=elapsed_ms,
            attempts=1,
            first_try_valid=True,
        )


def build_client() -> Any:
    """Factory: NNC_LLM_BACKEND=heuristic|mock|unconstrained|structured_api|outlines."""

    backend = os.environ.get("NNC_LLM_BACKEND", "heuristic").strip().lower()
    if backend in {"", "heuristic"}:
        return HeuristicLlmClient()
    if backend == "mock":
        raw = os.environ.get("NNC_LLM_PROPOSAL_JSON", "").strip()
        if not raw:
            return MockLlmClient()
        path = Path(raw)
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
        else:
            payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise SchemaError("NNC_LLM_PROPOSAL_JSON must be an object")
        return MockLlmClient(payload)
    if backend == "unconstrained":
        return UnconstrainedLlmClient()
    if backend == "structured_api":
        return HostedStructuredLlmClient()
    if backend == "outlines":
        return ConstrainedOutlinesLlmClient()
    raise SchemaError(
        f"unknown NNC_LLM_BACKEND {backend!r}; use heuristic, mock, unconstrained, structured_api, or outlines"
    )
