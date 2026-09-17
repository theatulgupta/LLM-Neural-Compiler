"""Append-only compile history. Prompting reads this; it never invents metrics."""

from compiler.history.store import (
    append_history,
    history_for_model,
    load_history,
    new_run_record,
    write_run_json,
)

__all__ = [
    "append_history",
    "history_for_model",
    "load_history",
    "new_run_record",
    "write_run_json",
]
