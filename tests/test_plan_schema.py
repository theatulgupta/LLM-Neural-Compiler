from __future__ import annotations

from compiler.hardware.profile import probe_hardware
from compiler.planner.plan import Plan
from compiler.schema_validate import SchemaError, validate_llm_plan
import pytest


def test_valid_plan() -> None:
    payload = {
        "plan_id": "t",
        "steps": [{"atom": "constant_folding", "params": {}}],
        "options": {"ort_graph_opt": "disable"},
        "rationale": "fold",
        "confidence": 0.5,
        "source": "test",
    }
    validate_llm_plan(payload)
    Plan.from_dict(payload)


def test_invalid_atom() -> None:
    with pytest.raises(SchemaError):
        validate_llm_plan(
            {
                "plan_id": "t",
                "steps": [{"atom": "magic"}],
                "options": {},
                "rationale": "x",
                "confidence": 0.1,
                "source": "t",
            }
        )
