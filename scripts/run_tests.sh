#!/usr/bin/env bash
# ROS 2 Jazzy registers pytest plugins (launch_testing) via setuptools
# entry points. Disable autoload so compiler tests do not import rclpy/yaml.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
ruff check compiler src scripts tests ros2_ws/src
exec python -m pytest "$@"
