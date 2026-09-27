from __future__ import annotations

from scripts.check_secrets import find_secrets


def test_repo_has_no_secret_values() -> None:
    assert find_secrets() == []


def test_secret_scan_flags_a_value(tmp_path) -> None:
    sample = tmp_path / "notes.py"
    sample.write_text('TOKEN = "' + "gsk_" + "abcdefghijklmnop" + '"\n', encoding="utf-8")
    hits = find_secrets(tmp_path)
    assert len(hits) == 1
    assert hits[0].startswith("notes.py:")
