"""Validate compiler JSON against the checked-in schemas.

jsonschema is optional. The allowlist and required fields are enforced here
even when the extra package is missing, so tests stay honest on a bare venv.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from compiler.strategies import ALLOWED_STRATEGY_NAMES

SCHEMA_DIR = Path(__file__).resolve().parents[1] / "schemas"
LLM_PROPOSAL_SCHEMA_PATH = SCHEMA_DIR / "llm-proposal.schema.json"
RUN_RESULT_SCHEMA_PATH = SCHEMA_DIR / "run-result.schema.json"


class SchemaError(ValueError):
    """JSON does not match a compiler schema."""


def load_schema(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _try_jsonschema(instance: dict[str, Any], schema: dict[str, Any]) -> None:
    try:
        import jsonschema
    except ImportError:
        return
    jsonschema.validate(instance=instance, schema=schema)


def validate_llm_proposal(payload: dict[str, Any]) -> dict[str, Any]:
    schema = load_schema(LLM_PROPOSAL_SCHEMA_PATH)
    enum = tuple(schema["properties"]["strategy"]["enum"])
    if enum != ALLOWED_STRATEGY_NAMES:
        raise SchemaError(
            f"schema enum {list(enum)} does not match ALLOWED_STRATEGY_NAMES {list(ALLOWED_STRATEGY_NAMES)}"
        )
    required = schema.get("required", [])
    missing = [key for key in required if key not in payload]
    if missing:
        raise SchemaError(f"LLM proposal missing fields: {missing}")
    extra = [key for key in payload if key not in schema.get("properties", {})]
    if extra:
        raise SchemaError(f"LLM proposal has unknown fields: {extra}")
    strategy = payload.get("strategy")
    if strategy not in ALLOWED_STRATEGY_NAMES:
        raise SchemaError(
            f"strategy {strategy!r} is not allowlisted; allowed={list(ALLOWED_STRATEGY_NAMES)}"
        )
    rationale = payload.get("rationale")
    source = payload.get("source")
    if not isinstance(rationale, str) or not rationale.strip():
        raise SchemaError("rationale must be a non-empty string")
    if not isinstance(source, str) or not source.strip():
        raise SchemaError("source must be a non-empty string")
    _try_jsonschema(payload, schema)
    return payload


def validate_run_result(payload: dict[str, Any]) -> dict[str, Any]:
    schema = load_schema(RUN_RESULT_SCHEMA_PATH)
    required = schema.get("required", [])
    missing = [key for key in required if key not in payload]
    if missing:
        raise SchemaError(f"run result missing fields: {missing}")
    if payload.get("schema_version") != 1:
        raise SchemaError(f"unsupported schema_version {payload.get('schema_version')!r}")
    backend = payload.get("backend")
    allowed_backends = schema["properties"]["backend"].get("enum", ["ort_cpu", "tensorrt"])
    if backend not in allowed_backends:
        raise SchemaError(f"backend {backend!r} is not in {allowed_backends}")
    compile_block = payload.get("compile")
    if not isinstance(compile_block, dict) or "ok" not in compile_block:
        raise SchemaError("compile.ok is required")
    if not isinstance(compile_block["ok"], bool):
        raise SchemaError("compile.ok must be a boolean")
    if compile_block["ok"] is True:
        bench = payload.get("benchmark")
        skip = payload.get("skip")
        if skip:
            raise SchemaError("successful compile must not set skip")
        if not isinstance(bench, dict):
            raise SchemaError("successful compile requires a benchmark object")
        latency = bench.get("latency_ms") or {}
        mean = latency.get("mean")
        if not isinstance(mean, (int, float)) or mean <= 0:
            raise SchemaError("successful compile must record a positive mean latency; do not invent FPS")
    _try_jsonschema(payload, schema)
    return payload
