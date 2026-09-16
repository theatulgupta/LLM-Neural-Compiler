#!/usr/bin/env python3
"""Probe PX4 SITL / ROS 2 / Gazebo. Never reports success unless processes or binaries actually work."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT))

from nnc.probe import probe_host, run_ros2  # noqa: E402


def _topics() -> dict:
    result = run_ros2(["topic", "list"])
    if not result.get("ok"):
        return {
            "ok": False,
            "returncode": result.get("returncode"),
            "skip": result.get("skip"),
            "error": result.get("error"),
            "stderr_head": result.get("stderr_head"),
            "topic_count": 0,
            "has_local_position": False,
            "topics_head": [],
        }
    topics = [line.strip() for line in (result.get("stdout_head") or "").splitlines() if line.strip()]
    return {
        "ok": True,
        "returncode": 0,
        "topic_count": len(topics),
        "has_local_position": any("vehicle_local_position" in t for t in topics),
        "stderr_head": result.get("stderr_head"),
        "topics_head": topics[:20],
    }


def main() -> int:
    facts = probe_host()
    facts["ros2_topics"] = _topics()
    px4 = facts["px4"]
    telemetry_ok = bool(facts["ros2_topics"].get("has_local_position"))
    sitl_ok = bool(px4.get("build_exists")) and telemetry_ok
    facts["verdict"] = {
        "sitl_running_with_telemetry": sitl_ok,
        "reason": (
            "vehicle_local_position topic is present"
            if sitl_ok
            else "PX4 SITL is not built or telemetry topics are not published on this host"
        ),
    }
    json.dump(facts, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    out = ROOT / "experiments" / "results" / "sitl_probe.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(facts, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {out}", file=sys.stderr)
    return 0 if facts["verdict"]["sitl_running_with_telemetry"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
