"""Turn a graph summary into an allowlisted Strategy (LLM cannot escape the set)."""

from __future__ import annotations

from dataclasses import dataclass

from compiler.graph.graph_summary import GraphSummary
from compiler.llm.llm_client import HeuristicLlmClient, LlmProposal, MockLlmClient, build_client
from compiler.strategies import ALLOWED_STRATEGY_NAMES, Strategy, get_strategy


@dataclass(frozen=True, slots=True)
class Recommendation:
    strategy: Strategy
    rationale: str
    source: str
    proposal: LlmProposal

    def to_dict(self) -> dict[str, str]:
        return {
            "strategy": self.strategy.name,
            "rationale": self.rationale,
            "source": self.source,
        }


def recommend_strategy(
    summary: GraphSummary,
    *,
    client: HeuristicLlmClient | MockLlmClient | None = None,
    override: str | None = None,
) -> Recommendation:
    if override is not None:
        strategy = get_strategy(override)
        proposal = LlmProposal(strategy=override, rationale="caller override", source="override")
        return Recommendation(strategy=strategy, rationale=proposal.rationale, source=proposal.source, proposal=proposal)

    active = client or build_client()
    proposal = active.propose(summary)
    strategy = get_strategy(proposal.strategy)
    if strategy.name not in ALLOWED_STRATEGY_NAMES:
        strategy = get_strategy("baseline")
    return Recommendation(
        strategy=strategy,
        rationale=proposal.rationale,
        source=proposal.source,
        proposal=proposal,
    )
