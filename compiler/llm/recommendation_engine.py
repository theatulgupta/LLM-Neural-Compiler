"""Turn a graph summary into an allowlisted Strategy (LLM cannot escape the set)."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from compiler.graph.graph_summary import GraphSummary
from compiler.llm.llm_client import LlmProposal, build_client
from compiler.strategies import (
    ALLOWED_STRATEGY_NAMES,
    Strategy,
    get_strategy,
    strategy_from_config,
)


@dataclass(frozen=True, slots=True)
class Recommendation:
    strategy: Strategy
    rationale: str
    source: str
    proposal: LlmProposal
    fallback_triggered: bool = False
    rejection_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "strategy": self.strategy.name,
            "config": self.strategy.config_dict(),
            "rationale": self.rationale,
            "source": self.source,
            "fallback_triggered": self.fallback_triggered,
        }
        if self.rejection_reason is not None:
            out["rejection_reason"] = self.rejection_reason
        return out


def load_recent_history(results_dir: Path | None, *, model_sha256: str | None = None, limit: int = 5) -> list[dict[str, Any]]:
    if results_dir is None:
        return []
    history_path = results_dir / "history.jsonl"
    if not history_path.is_file():
        return []
    records: list[dict[str, Any]] = []
    for line in history_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if model_sha256 and rec.get("model", {}).get("sha256") != model_sha256:
            continue
        records.append(rec)
    return records[-limit:]


def recommend_strategy(
    summary: GraphSummary,
    *,
    client: Any = None,
    override: str | None = None,
    override_strategy: Strategy | None = None,
    history_feedback: list[dict[str, Any]] | None = None,
    fallback_on_error: bool = False,
) -> Recommendation:
    if override_strategy is not None:
        proposal = LlmProposal(
            strategy=override_strategy.name,
            config=override_strategy.config_dict(),
            rationale="caller strategy override",
            source="override",
        )
        return Recommendation(
            strategy=override_strategy,
            rationale=proposal.rationale,
            source=proposal.source,
            proposal=proposal,
        )

    if override is not None:
        strategy = get_strategy(override)
        proposal = LlmProposal(strategy=override, rationale="caller override", source="override")
        return Recommendation(strategy=strategy, rationale=proposal.rationale, source=proposal.source, proposal=proposal)

    active = client or build_client()
    try:
        proposal = active.propose(summary, history_feedback=history_feedback)
        if proposal.config is not None:
            strategy = strategy_from_config(proposal.strategy, proposal.config)
        else:
            strategy = get_strategy(proposal.strategy)
        if strategy.name not in ALLOWED_STRATEGY_NAMES:
            strategy = get_strategy("baseline")
        return Recommendation(
            strategy=strategy,
            rationale=proposal.rationale,
            source=proposal.source,
            proposal=proposal,
        )
    except Exception as exc:
        if not fallback_on_error:
            raise
        safe = get_strategy("baseline")
        fallback_proposal = LlmProposal(
            strategy=safe.name,
            config=safe.config_dict(),
            rationale=f"Deterministic fallback to baseline after gate rejection: {exc}",
            source=f"fallback({getattr(active, 'name', 'unknown')})",
            first_try_valid=False,
        )
        return Recommendation(
            strategy=safe,
            rationale=fallback_proposal.rationale,
            source=fallback_proposal.source,
            proposal=fallback_proposal,
            fallback_triggered=True,
            rejection_reason=f"{type(exc).__name__}: {exc}",
        )
