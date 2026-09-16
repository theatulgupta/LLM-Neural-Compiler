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


def run_ros2(args: list[str], *, timeout: float = 8.0) -> dict[str, Any]:
    distro = os.environ.get("ROS_DISTRO") or "jazzy"
    setup = Path(f"/opt/ros/{distro}/setup.bash")
    argv = ["ros2", *args]
    if not setup.is_file():
        return {"cmd": argv, "ok": False, "skip": f"{setup} not found"}
    command = f"source {shlex.quote(str(setup))} && {shlex.join(argv)}"
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
    px4_script = Path.home() / "LLM-Neural-Compiler" / "third_party" / "PX4-Autopilot"
    px4_build = px4_home / "build"
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
            "build_exists": px4_build.is_dir(),
            "script_path_exists": px4_script.is_dir(),
            "px4_bin": shutil.which("px4"),
        },
    }
