"""Bounded LLM plan loop: schema check, verifier, retry with feedback, heuristic fallback.

The LLM is allowed to be wrong. This module never raises for a bad reply.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from compiler.graph.graph_summary import GraphSummary
from compiler.hardware.profile import HardwareProfile
from compiler.llm.context_builder import build_context
from compiler.llm.llm_client import HeuristicLlmClient, LlmClient
from compiler.planner.plan import Plan
from compiler.planner.verifier import VerifiedPlan, verify_plan
from compiler.schema import SchemaError

_TRANSPORT_PREFIXES = ("LLM HTTP ", "LLM request failed", "Groq HTTP ", "Groq request failed")


@dataclass(frozen=True, slots=True)
class PlanAttempt:
    n: int
    source: str
    plan_id: str | None
    outcome: str
    detail: str
    corrections: tuple[dict[str, str], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "source": self.source,
            "plan_id": self.plan_id,
            "outcome": self.outcome,
            "detail": self.detail,
            "corrections": [dict(item) for item in self.corrections],
        }


@dataclass(frozen=True, slots=True)
class PlanOutcome:
    plan: Plan
    accepted: bool
    source: str
    attempts: tuple[PlanAttempt, ...]
    fallback_reason: str | None = None
    verified: VerifiedPlan | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan": self.plan.to_dict(),
            "accepted": self.accepted,
            "source": self.source,
            "attempts": [item.to_dict() for item in self.attempts],
            "fallback_reason": self.fallback_reason,
            "verified": None
            if self.verified is None
            else {
                "accepted": self.verified.accepted,
                "rejections": list(self.verified.rejections),
                "numerics": self.verified.numerics,
            },
        }


def _classify_schema_error(exc: SchemaError) -> str:
    text = str(exc)
    if any(text.startswith(prefix) for prefix in _TRANSPORT_PREFIXES):
        return "transport_error"
    return "schema_error"


def _heuristic_plan(
    summary: GraphSummary,
    hardware: HardwareProfile,
    backend_name: str,
    constraints: dict[str, Any] | None,
) -> tuple[Plan, VerifiedPlan]:
    proposal = HeuristicLlmClient().propose(summary)
    plan = Plan.from_dict(proposal.plan)
    verified = verify_plan(plan, summary, hardware, backend_name=backend_name, constraints=constraints)
    return verified.plan if verified.accepted else plan, verified


def propose_plan(
    summary: GraphSummary,
    hardware: HardwareProfile,
    constraints: dict[str, Any] | None,
    history: list[dict[str, Any]] | None,
    client: LlmClient,
    *,
    backend_name: str = "ort_cpu",
    max_attempts: int = 3,
    feedback: list[dict[str, Any]] | None = None,
) -> PlanOutcome:
    """Ask `client` up to `max_attempts` times, then fall back to the heuristic."""

    context = build_context(summary, hardware, constraints, history)
    trail: list[PlanAttempt] = []
    notes: list[dict[str, Any]] = list(feedback or [])
    last_error = "no attempt"
    source_name = getattr(client, "name", "llm")

    for index in range(1, max(1, max_attempts) + 1):
        try:
            proposal = client.propose(summary, context, feedback=notes or None)
        except SchemaError as exc:
            kind = _classify_schema_error(exc)
            last_error = str(exc)
            trail.append(
                PlanAttempt(
                    n=index,
                    source=source_name,
                    plan_id=None,
                    outcome=kind,
                    detail=last_error,
                )
            )
            notes.append(trail[-1].to_dict())
            continue
        plan = Plan.from_dict(proposal.plan)
        verified = verify_plan(plan, summary, hardware, backend_name=backend_name, constraints=constraints)
        drops = tuple(dict(item) for item in verified.rejections if item.get("level") == "drop")
        if not verified.accepted:
            last_error = (
                "; ".join(f"{item.get('step')}: {item.get('reason')}" for item in verified.rejections)
                or "plan rejected"
            )
            trail.append(
                PlanAttempt(
                    n=index,
                    source=source_name,
                    plan_id=plan.plan_id,
                    outcome="rejected",
                    detail=last_error,
                    corrections=tuple(dict(item) for item in verified.rejections),
                )
            )
            notes.append(trail[-1].to_dict())
            continue
        trail.append(
            PlanAttempt(
                n=index,
                source=proposal.source,
                plan_id=verified.plan.plan_id,
                outcome="accepted",
                detail="ok",
                corrections=drops,
            )
        )
        return PlanOutcome(
            plan=verified.plan,
            accepted=True,
            source=proposal.source,
            attempts=tuple(trail),
            verified=verified,
        )

    plan, verified = _heuristic_plan(summary, hardware, backend_name, constraints)
    return PlanOutcome(
        plan=verified.plan if verified.accepted else plan,
        accepted=bool(verified.accepted),
        source="heuristic",
        attempts=tuple(trail),
        fallback_reason=last_error,
        verified=verified,
    )
