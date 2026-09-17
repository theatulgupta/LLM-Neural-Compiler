#!/usr/bin/env bash
# Bridge the Gazebo mono_cam image topic onto ROS 2 /nnc/camera/image_raw.
set -euo pipefail
set +u
source /opt/ros/jazzy/setup.bash
set -u

GZ_TOPIC=""
for _ in $(seq 1 60); do
  GZ_TOPIC="$(gz topic -l 2>/dev/null | grep -m1 '/sensor/camera/image$' || true)"
  if [[ -n "$GZ_TOPIC" ]]; then
    break
  fi
  sleep 1
done
if [[ -z "$GZ_TOPIC" ]]; then
  echo "no Gazebo camera image topic after 60s" >&2
  exit 1
fi
echo "bridging $GZ_TOPIC -> /nnc/camera/image_raw"
exec ros2 run ros_gz_bridge parameter_bridge \
  "${GZ_TOPIC}@sensor_msgs/msg/Image[gz.msgs.Image" \
  --ros-args -r "${GZ_TOPIC}:=/nnc/camera/image_raw"
