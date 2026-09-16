"""LLM client with a deterministic heuristic fallback.

Network models are optional and never required for tests or the CPU baseline.
If a client is configured, its proposal is still filtered through the allowlist.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from compiler.graph.graph_summary import GraphSummary


@dataclass(frozen=True, slots=True)
class LlmProposal:
    strategy: str
    rationale: str
    source: str


class HeuristicLlmClient:
    """No network. Maps graph features onto the allowlist."""

    name = "heuristic"

    def propose(self, summary: GraphSummary) -> LlmProposal:
        if summary.node_count <= 8:
            return LlmProposal(
                strategy="baseline",
                rationale="Tiny graph; disable ORT opts first so the Flatten/Gemm contract is visible.",
                source=self.name,
            )
        if summary.op_counts.get("Conv", 0) >= 1:
            return LlmProposal(
                strategy="ort_extended",
                rationale="Convolutional graph benefits from extended ORT fusions on CPU.",
                source=self.name,
            )
        return LlmProposal(
            strategy="ort_all",
            rationale="Generic graph; enable all ORT graph optimizations on CPU.",
            source=self.name,
        )


def build_client() -> HeuristicLlmClient:
    """Factory kept so a future HTTP client can be swapped in via env without API changes."""

    _ = os.environ.get("NNC_LLM_BACKEND", "heuristic")
    return HeuristicLlmClient()
