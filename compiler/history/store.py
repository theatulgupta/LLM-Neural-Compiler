"""Append-only JSONL history plus one JSON file per run."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from compiler.schema import validate_run_result
from compiler.utils.timeutil import utc_now_iso


def write_run_json(path: Path, record: dict[str, Any], *, validate: bool = True) -> Path:
    if validate:
        validate_run_result(record)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def append_history(history_path: Path, record: dict[str, Any]) -> None:
    history_path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, sort_keys=True)
    with history_path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def new_run_record(**kwargs: Any) -> dict[str, Any]:
    record = {"schema_version": 1, "created_at": utc_now_iso()}
    record.update(kwargs)
    return record


def read_history(history_path: Path) -> tuple[list[dict[str, Any]], int]:
    """Parsed rows and the count of non-empty lines that were not JSON."""

    if not history_path.is_file():
        return [], 0
    rows: list[dict[str, Any]] = []
    skipped = 0
    for line in history_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            skipped += 1
    return rows, skipped


def load_history(history_path: Path) -> list[dict[str, Any]]:
    rows, skipped = read_history(history_path)
    load_history.last_skipped = skipped  # type: ignore[attr-defined]
    return rows


def history_for_model(
    kind_or_sha: str, k: int = 5, *, history_path: Path | None = None
) -> list[dict[str, Any]]:
    from compiler.catalog import REPO_ROOT

    path = history_path or (REPO_ROOT / "experiments" / "results" / "history.jsonl")
    parsed, skipped = read_history(path)
    matched: list[dict[str, Any]] = []
    for row in reversed(parsed):
        model = row.get("model") or {}
        kind = str(model.get("kind") or "")
        sha = str(model.get("sha256") or "")
        if kind_or_sha not in {kind, sha}:
            continue
        bench = (row.get("benchmark") or {}).get("latency_ms") or {}
        matched.append(
            {
                "plan_id": (row.get("plan") or {}).get("plan_id")
                or (row.get("strategy") or {}).get("strategy"),
                "p50_ms": bench.get("p50"),
                "passed": (row.get("verification") or {}).get("passed"),
                "kind": kind,
                "origin": (row.get("plan") or {}).get("source") or (row.get("strategy") or {}).get("source"),
            }
        )
        if len(matched) >= k:
            break
    if skipped and matched:
        matched[-1] = {**matched[-1], "history_skipped": skipped}
    elif skipped:
        matched.append(
            {
                "plan_id": None,
                "p50_ms": None,
                "passed": None,
                "kind": "",
                "origin": None,
                "history_skipped": skipped,
            }
        )
    return matched
