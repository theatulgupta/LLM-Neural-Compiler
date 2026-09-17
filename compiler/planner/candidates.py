"""Candidate plans: presets, heuristic, LLM."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from compiler.graph.graph_summary import GraphSummary
from compiler.hardware.profile import HardwareProfile
from compiler.llm.llm_client import HeuristicLlmClient, LlmClient
from compiler.planner.plan import PRESETS, Plan
from compiler.planner.verifier import VerifiedPlan, verify_plan


@dataclass(frozen=True, slots=True)
class Candidate:
    plan: Plan
    origin: str
    verified: VerifiedPlan | None = None


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
    seen: set[str] = set()
    for name in names:
        plan = PRESETS[name]
        verified = verify_plan(plan, summary, hardware, backend_name=backend_name, constraints=constraints)
        key = json_key(verified.plan)
        if key in seen:
            continue
        seen.add(key)
        out.append(Candidate(plan=verified.plan, origin=f"preset:{name}", verified=verified))

    heur = HeuristicLlmClient().propose(summary)
    hplan = Plan.from_dict(heur.plan)
    hver = verify_plan(hplan, summary, hardware, backend_name=backend_name, constraints=constraints)
    key = json_key(hver.plan)
    if key not in seen:
        seen.add(key)
        out.append(Candidate(plan=hver.plan, origin="heuristic", verified=hver))

    rec = recommend_plan(summary, hardware, constraints, client=client, history=history, backend_name=backend_name)
    if rec.verified and rec.verified.accepted:
        key = json_key(rec.verified.plan)
        if key not in seen:
            seen.add(key)
            out.append(Candidate(plan=rec.verified.plan, origin="llm", verified=rec.verified))
    return out


def json_key(plan: Plan) -> str:
    import json

    return json.dumps({"steps": plan.to_dict()["steps"], "options": plan.to_dict()["options"]}, sort_keys=True)
