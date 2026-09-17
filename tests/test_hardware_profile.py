from __future__ import annotations

from compiler.hardware.profile import probe_hardware


def test_hardware_profile_cpu_count() -> None:
    hw = probe_hardware()
    assert hw.cpu_count > 0
    assert hw.arch
    assert "CPUExecutionProvider" in hw.providers or hw.ort_version == "missing"
