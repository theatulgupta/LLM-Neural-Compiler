"""LLM client with a deterministic heuristic fallback and a schema-bound mock.

Network models are optional and never required for tests or the CPU baseline.
Every proposal is validated against schemas/llm-proposal.schema.json and then
filtered through the strategy allowlist. Unknown names cannot leak through.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

from compiler.graph.graph_summary import GraphSummary
from compiler.schema_validate import SchemaError, validate_llm_proposal


@dataclass(frozen=True, slots=True)
class LlmProposal:
    strategy: str
    rationale: str
    source: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


class LlmClient(Protocol):
    """Any advisor: heuristic, mock, Groq, or a later provider. Same propose() contract."""

    name: str

    def propose(self, summary: GraphSummary) -> LlmProposal:
        ...


def proposal_from_dict(payload: dict[str, object]) -> LlmProposal:
    checked = validate_llm_proposal({str(k): v for k, v in payload.items()})
    return LlmProposal(
        strategy=str(checked["strategy"]),
        rationale=str(checked["rationale"]),
        source=str(checked["source"]),
    )


class HeuristicLlmClient:
    """No network. Maps graph features onto the allowlist."""

    name = "heuristic"

    def propose(self, summary: GraphSummary) -> LlmProposal:
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

    def propose(self, summary: GraphSummary) -> LlmProposal:
        _ = summary
        return proposal_from_dict(dict(self._proposal))


def build_client() -> LlmClient:
    """Factory: NNC_LLM_BACKEND=heuristic|mock|groq. Tests default to heuristic/mock.

    Groq is opt-in so pytest never hits the network even if ~/.config/nnc/groq.env exists.
    """

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
    if backend == "groq":
        from compiler.llm.groq_client import GroqLlmClient

        return GroqLlmClient()
    raise SchemaError(f"unknown NNC_LLM_BACKEND {backend!r}; use heuristic, mock, or groq")
