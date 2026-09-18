"""LLM planner: clients, prompt builder, recommendations."""

from __future__ import annotations

from compiler.llm.groq_client import GroqLlmClient
from compiler.llm.llm_client import HeuristicLlmClient, LlmClient, LlmProposal, MockLlmClient, build_client
from compiler.llm.prompting import build_messages, user_prompt
from compiler.llm.provider import HttpLlmClient, ProviderSpec, known_providers, register_provider
from compiler.llm.recommendation_engine import Recommendation, recommend_plan, recommend_strategy
from compiler.llm.session import PlanAttempt, PlanOutcome, propose_plan

__all__ = [
    "GroqLlmClient",
    "HeuristicLlmClient",
    "HttpLlmClient",
    "LlmClient",
    "LlmProposal",
    "MockLlmClient",
    "PlanAttempt",
    "PlanOutcome",
    "ProviderSpec",
    "Recommendation",
    "build_client",
    "build_messages",
    "known_providers",
    "propose_plan",
    "recommend_plan",
    "recommend_strategy",
    "register_provider",
    "user_prompt",
]
