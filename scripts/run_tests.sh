#!/usr/bin/env bash
# ROS 2 Jazzy registers pytest plugins (launch_testing) via setuptools
# entry points. Disable autoload so compiler tests do not import rclpy/yaml.
set -euo pipefail
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
exec python -m pytest "$@"
