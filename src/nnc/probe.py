"""Host/stack facts used in baseline JSON. Reports only what is actually present."""

from __future__ import annotations

import os
import platform
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any

from nnc.backends.tensorrt import nvidia_probe

REPO_ROOT = Path(__file__).resolve().parents[2]


def system_env() -> dict[str, str]:
    """Strip the compiler venv so ROS 2 uses /opt/ros Python packages."""

    env = os.environ.copy()
    env.pop("PYTHONHOME", None)
    venv = env.get("VIRTUAL_ENV")
    if venv:
        path = [part for part in env.get("PATH", "").split(":") if part and not part.startswith(venv)]
        env["PATH"] = ":".join(path)
        env.pop("VIRTUAL_ENV", None)
    pythonpath = [
        part
        for part in env.get("PYTHONPATH", "").split(":")
        if part and "LLM-Neural-Compiler" not in part and ".venv" not in part
    ]
    env["PYTHONPATH"] = ":".join(pythonpath)
    return env


def _finished(argv: list[str], completed: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    stdout = (completed.stdout or "").strip()
    stderr = (completed.stderr or "").strip()
    return {
        "cmd": argv,
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "stdout_head": stdout[:4000],
        "stderr_head": stderr[:2000],
    }


def _run(argv: list[str], *, timeout: float = 8.0) -> dict[str, Any]:
    executable = shutil.which(argv[0]) if os.path.sep not in argv[0] else argv[0]
    if executable is None and os.path.sep not in argv[0]:
        return {"cmd": argv, "ok": False, "skip": f"{argv[0]} not on PATH"}
    try:
        completed = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            env=system_env(),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"cmd": argv, "ok": False, "error": str(exc)}
    return _finished(argv, completed)


def run_ros2(args: list[str], *, timeout: float = 8.0, extra_setup: Path | None = None) -> dict[str, Any]:
    distro = os.environ.get("ROS_DISTRO") or "jazzy"
    setup = Path(f"/opt/ros/{distro}/setup.bash")
    argv = ["ros2", *args]
    if not setup.is_file():
        return {"cmd": argv, "ok": False, "skip": f"{setup} not found"}
    prefixes = [f"source {shlex.quote(str(setup))}"]
    if extra_setup and extra_setup.is_file():
        prefixes.append(f"source {shlex.quote(str(extra_setup))}")
    command = " && ".join([*prefixes, shlex.join(argv)])
    try:
        completed = subprocess.run(
            ["bash", "-c", command],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            env=system_env(),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"cmd": argv, "ok": False, "error": str(exc)}
    return _finished(argv, completed)


def probe_host() -> dict[str, Any]:
    gpu_ok, gpu_reason = nvidia_probe()
    px4_home = Path.home() / "PX4-Autopilot"
    px4_script_legacy = REPO_ROOT / "third_party" / "PX4-Autopilot"
    px4_build = px4_home / "build" / "px4_sitl_default"
    px4_bin = px4_build / "bin" / "px4"
    agent = Path.home() / "px4_ros_uxrce_dds_ws" / "install" / "microxrcedds_agent" / "bin" / "MicroXRCEAgent"
    px4_msgs = Path.home() / "ros2_px4_ws" / "install" / "px4_msgs" / "share" / "px4_msgs" / "package.sh"
    ros_ws = REPO_ROOT / "ros2_ws" / "install" / "setup.bash"
    ros_distro = os.environ.get("ROS_DISTRO") or "jazzy"
    return {
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "release": platform.release(),
        },
        "nvidia": {"present": gpu_ok, "detail": gpu_reason},
        "ros2": {
            "distro": ros_distro,
            "ros2_bin": shutil.which("ros2"),
            "probe": run_ros2(["--help"]),
            "px4_msgs_setup": str(Path.home() / "ros2_px4_ws" / "install" / "setup.bash"),
            "px4_msgs_present": px4_msgs.is_file()
            or (Path.home() / "ros2_px4_ws" / "install" / "px4_msgs").is_dir(),
            "llm_uav_core_setup": str(ros_ws),
            "llm_uav_core_built": ros_ws.is_file(),
        },
        "gazebo": {
            "gz_bin": shutil.which("gz"),
            "probe": _run(["gz", "sim", "--version"])
            if shutil.which("gz")
            else {"ok": False, "skip": "gz not on PATH"},
        },
        "px4": {
            "tree": str(px4_home),
            "tree_exists": px4_home.is_dir(),
            "build_dir": str(px4_build),
            "build_exists": px4_build.is_dir(),
            "px4_bin": str(px4_bin) if px4_bin.is_file() else None,
            "legacy_third_party_tree": str(px4_script_legacy),
            "legacy_third_party_exists": px4_script_legacy.is_dir(),
        },
        "uxrce_agent": {
            "bin": str(agent),
            "present": agent.is_file(),
        },
    }
