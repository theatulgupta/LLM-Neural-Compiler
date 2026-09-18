"""Candidate plans: presets, heuristic, LLM. The LLM row is never dropped."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from compiler.graph.graph_summary import GraphSummary
from compiler.hardware.profile import HardwareProfile
from compiler.llm.llm_client import HeuristicLlmClient, LlmClient
from compiler.llm.session import PlanOutcome
from compiler.planner.plan import PRESETS, Plan
from compiler.planner.verifier import VerifiedPlan, verify_plan


@dataclass(frozen=True, slots=True)
class Candidate:
    plan: Plan
    origin: str
    verified: VerifiedPlan | None = None
    same_as: str | None = None
    outcome: PlanOutcome | None = None


def json_key(plan: Plan) -> str:
    payload = plan.to_dict()
    return json.dumps({"steps": payload["steps"], "options": payload["options"]}, sort_keys=True)


def generate_candidates(
    summary: GraphSummary,
    hardware: HardwareProfile,
    constraints: dict[str, Any] | None = None,
    client: LlmClient | None = None,
    history: list[dict[str, Any]] | None = None,
    mode: str = "default",
    backend_name: str = "ort_cpu",
) -> list[Candidate]:
    from compiler.llm.recommendation_engine import recommend_plan

    names = ["baseline", "ort_default", "graph_fuse"]
    if mode == "all":
        names = list(PRESETS)
    out: list[Candidate] = []
    seen: dict[str, str] = {}
    for name in names:
        plan = PRESETS[name]
        verified = verify_plan(plan, summary, hardware, backend_name=backend_name, constraints=constraints)
        key = json_key(verified.plan)
        if key in seen:
            continue
        origin = f"preset:{name}"
        seen[key] = origin
        out.append(Candidate(plan=verified.plan, origin=origin, verified=verified))

    heur = HeuristicLlmClient().propose(summary)
    hplan = Plan.from_dict(heur.plan)
    hver = verify_plan(hplan, summary, hardware, backend_name=backend_name, constraints=constraints)
    hkey = json_key(hver.plan)
    h_same = seen.get(hkey)
    if hkey not in seen:
        seen[hkey] = "heuristic"
    out.append(Candidate(plan=hver.plan, origin="heuristic", verified=hver, same_as=h_same))

    rec = recommend_plan(
        summary, hardware, constraints, client=client, history=history, backend_name=backend_name
    )
    llm_plan = rec.strategy
    lkey = json_key(llm_plan)
    l_same = seen.get(lkey)
    if lkey not in seen:
        seen[lkey] = "llm"
    out.append(
        Candidate(
            plan=llm_plan,
            origin="llm",
            verified=rec.verified,
            same_as=l_same,
            outcome=rec.outcome,
        )
    )
    return out
