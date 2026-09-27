"""Fail when a tracked file contains a key value. Documented prefixes are allowed."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_ASSIGN = re.compile(r"(?i)^(?!\s*#).*(?:GROQ_API_KEY|OPENAI_API_KEY|NNC_LLM_API_KEY)\s*=\s*\S+")
_TOKEN = re.compile(r"(?<![A-Za-z0-9])(?:gsk_|sk-)[A-Za-z0-9]{8,}")
_SKIP_DIRS = {".git", ".venv", "__pycache__", ".pixi", "build", "log", "dist"}
_SKIP_SUFFIX = {".onnx", ".png", ".jpg", ".npz", ".pyc"}


def _files(base: Path) -> list[Path]:
    if (base / ".git").is_dir():
        listed = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=base,
            check=False,
            capture_output=True,
        )
        if listed.returncode == 0:
            return [base / item.decode() for item in listed.stdout.split(b"\0") if item]
    found: list[Path] = []
    for path in base.rglob("*"):
        if path.is_file() and not any(part in _SKIP_DIRS for part in path.parts):
            found.append(path)
    return found


def find_secrets(root: Path | None = None) -> list[str]:
    """Return `path:line` hits. Empty assignments and the bare prefix do not count."""

    base = root or ROOT
    hits: list[str] = []
    for path in _files(base):
        if path.suffix in _SKIP_SUFFIX or path.name.endswith(".example") or path.name == "check_secrets.py":
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for number, line in enumerate(text.splitlines(), 1):
            if _ASSIGN.search(line) or _TOKEN.search(line):
                hits.append(f"{path.relative_to(base)}:{number}")
    return hits


def main() -> int:
    hits = find_secrets()
    if hits:
        print("secret-shaped text:\n" + "\n".join(hits), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
