#!/usr/bin/env python3
"""Probe PX4 SITL / ROS 2 / Gazebo. Never reports success unless processes or binaries actually work."""

from __future__ import annotations

import json
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT))

from nnc.probe import probe_host, run_ros2, system_env  # noqa: E402


def _topics() -> dict:
    px4_setup = Path.home() / "ros2_px4_ws" / "install" / "setup.bash"
    result = run_ros2(["topic", "list"], extra_setup=px4_setup if px4_setup.is_file() else None)
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
        "local_position_topics": [t for t in topics if "vehicle_local_position" in t],
        "stderr_head": result.get("stderr_head"),
        "topics_head": topics[:40],
    }


def _record_position() -> dict:
    script = ROOT / "scripts" / "record_position.py"
    setup = Path("/opt/ros/jazzy/setup.bash")
    px4_setup = Path.home() / "ros2_px4_ws" / "install" / "setup.bash"
    ws_setup = ROOT / "ros2_ws" / "install" / "setup.bash"
    parts = [f"source {shlex.quote(str(setup))}"]
    if px4_setup.is_file():
        parts.append(f"source {shlex.quote(str(px4_setup))}")
    if ws_setup.is_file():
        parts.append(f"source {shlex.quote(str(ws_setup))}")
    out = ROOT / "experiments" / "results" / "sitl_position.json"
    parts.append(
        shlex.join(
            [
                "python3",
                str(script),
                "--seconds",
                "12",
                "--out",
                str(out),
            ]
        )
    )
    command = " && ".join(parts)
    try:
        completed = subprocess.run(
            ["bash", "-c", command],
            capture_output=True,
            text=True,
            timeout=25.0,
            check=False,
            env=system_env(),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"ok": False, "error": str(exc)}
    payload: dict
    try:
        payload = json.loads(completed.stdout) if completed.stdout.strip().startswith("{") else {}
    except json.JSONDecodeError:
        payload = {}
    payload["returncode"] = completed.returncode
    payload["stderr_head"] = (completed.stderr or "")[:2000]
    if not payload.get("ok"):
        payload.setdefault("stdout_head", (completed.stdout or "")[:2000])
    return payload


def main() -> int:
    facts = probe_host()
    facts["ros2_topics"] = _topics()
    px4 = facts["px4"]
    position = None
    if facts["ros2_topics"].get("has_local_position"):
        position = _record_position()
        facts["position_sample"] = position
    else:
        facts["position_sample"] = {
            "ok": False,
            "reason": "vehicle_local_position is not on the ROS graph",
        }

    telemetry_ok = bool(facts["ros2_topics"].get("has_local_position"))
    xyz_changed = bool(position and position.get("ok") and position.get("changed_xyz"))
    sitl_ok = bool(px4.get("px4_bin") or px4.get("build_exists")) and xyz_changed
    if sitl_ok:
        reason = f"x,y,z changed on {position.get('topic')}"
    elif telemetry_ok and position and position.get("ok") and not xyz_changed:
        reason = position.get("reason") or "topic present but x,y,z did not change"
    elif telemetry_ok:
        reason = position.get("reason") if position else "topic listed but no samples"
    else:
        reason = "PX4 SITL is not built or telemetry topics are not published on this host"
    facts["verdict"] = {
        "sitl_running_with_telemetry": sitl_ok,
        "xyz_changed": xyz_changed,
        "reason": reason,
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
