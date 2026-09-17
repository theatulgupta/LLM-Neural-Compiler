"""Named runtime backends. New engines register here; pipeline does not switch on names."""

from __future__ import annotations

from nnc.backends.base import Backend

_BACKENDS: dict[str, type[Backend]] = {}


def register_backend(cls: type[Backend]) -> type[Backend]:
    name = getattr(cls, "name", "") or ""
    if not name:
        raise ValueError(f"{cls.__name__} must set a non-empty .name")
    _BACKENDS[name] = cls
    return cls


def unregister_backend(name: str) -> None:
    _BACKENDS.pop(name, None)


def known_backends() -> tuple[str, ...]:
    return tuple(sorted(_BACKENDS))


def get_backend(name: str) -> Backend:
    try:
        return _BACKENDS[name]()
    except KeyError as exc:
        raise ValueError(f"unknown backend {name!r}; known={list(known_backends())}") from exc
