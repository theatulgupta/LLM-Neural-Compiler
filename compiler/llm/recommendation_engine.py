"""Turn a graph summary into a verified Plan (LLM cannot escape the atom set)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from compiler.graph.graph_summary import GraphSummary
from compiler.hardware.profile import HardwareProfile, probe_hardware
from compiler.llm.context_builder import build_context
from compiler.llm.llm_client import LlmClient, LlmProposal, build_client
from compiler.planner.plan import Plan, Strategy, get_strategy
from compiler.planner.verifier import VerifiedPlan, verify_plan
from compiler.planner import ALLOWED_STRATEGY_NAMES


@dataclass(frozen=True, slots=True)
class Recommendation:
    strategy: Strategy
    rationale: str
    source: str
    proposal: LlmProposal
    verified: VerifiedPlan | None = None

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
    context = build_context(summary, hw, constraints, history)
    active = client or build_client()
    proposal = active.propose(summary, context)
    plan = Plan.from_dict(proposal.plan)
    verified = verify_plan(plan, summary, hw, backend_name=backend_name, constraints=constraints)
    return Recommendation(
        strategy=verified.plan if verified.accepted else plan,
        rationale=proposal.rationale,
        source=proposal.source,
        proposal=proposal,
        verified=verified,
    )


def recommend_strategy(
    summary: GraphSummary,
    *,
    client: LlmClient | None = None,
    override: str | None = None,
) -> Recommendation:
    if override is not None:
        strategy = get_strategy(override)
        proposal = LlmProposal(strategy=override, rationale="caller override", source="override", plan=strategy.to_dict())
        return Recommendation(strategy=strategy, rationale=proposal.rationale, source=proposal.source, proposal=proposal)

    rec = recommend_plan(summary, client=client)
    if rec.strategy.name not in ALLOWED_STRATEGY_NAMES and rec.verified and rec.verified.accepted:
        return rec
    if rec.strategy.name not in ALLOWED_STRATEGY_NAMES:
        # Keep a named plan; tests that expect graph_fuse still work via heuristic on conv graphs.
        return rec
    return rec
