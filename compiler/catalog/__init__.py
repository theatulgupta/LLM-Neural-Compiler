"""UAV model zoo (YAML) and repo-root path. Pipeline code does not switch on model names."""

from compiler.catalog.zoo import (
    REPO_ROOT,
    ZOO_PATH,
    ModelSpec,
    get_model,
    load_zoo,
    zoo_kinds,
    zoo_tasks,
)

__all__ = [
    "REPO_ROOT",
    "ZOO_PATH",
    "ModelSpec",
    "get_model",
    "load_zoo",
    "zoo_kinds",
    "zoo_tasks",
]
