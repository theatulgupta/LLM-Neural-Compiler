"""Turn a graph summary into a verified Plan (LLM cannot escape the atom set)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from compiler.graph.graph_summary import GraphSummary
from compiler.hardware.profile import HardwareProfile, probe_hardware
from compiler.llm.llm_client import LlmClient, LlmProposal, build_client
from compiler.llm.session import PlanOutcome, propose_plan
from compiler.planner.plan import Strategy, get_strategy
from compiler.planner.verifier import VerifiedPlan


@dataclass(frozen=True, slots=True)
class Recommendation:
    strategy: Strategy
    rationale: str
    source: str
    proposal: LlmProposal
    verified: VerifiedPlan | None = None
    outcome: PlanOutcome | None = None

    def to_dict(self) -> dict[str, str]:
        return {
            "strategy": self.strategy.name,
            "rationale": self.rationale,
            "source": self.source,
        }


def recommend_plan(
    summary: GraphSummary,
    hardware: HardwareProfile | None = None,
    constraints: dict[str, Any] | None = None,
    client: LlmClient | None = None,
    history: list[dict[str, Any]] | None = None,
    backend_name: str = "ort_cpu",
) -> Recommendation:
    hw = hardware or probe_hardware()
    active = client or build_client()
    outcome = propose_plan(
        summary,
        hw,
        constraints,
        history,
        active,
        backend_name=backend_name,
    )
    plan = outcome.plan
    proposal = LlmProposal(
        strategy=plan.plan_id,
        rationale=plan.rationale,
        source=outcome.source,
        plan=plan.to_dict(),
    )
    return Recommendation(
        strategy=plan,
        rationale=plan.rationale,
        source=outcome.source,
        proposal=proposal,
        verified=outcome.verified,
        outcome=outcome,
    )


def recommend_strategy(
    summary: GraphSummary,
    *,
    client: LlmClient | None = None,
    override: str | None = None,
) -> Recommendation:
    if override is not None:
        strategy = get_strategy(override)
        proposal = LlmProposal(
            strategy=override,
            rationale="caller override",
            source="override",
            plan=strategy.to_dict(),
        )
        return Recommendation(
            strategy=strategy,
            rationale=proposal.rationale,
            source=proposal.source,
            proposal=proposal,
        )
    return recommend_plan(summary, client=client)
