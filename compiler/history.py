"""Append-only JSONL history plus one JSON file per run."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from compiler.schema_validate import validate_run_result
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
