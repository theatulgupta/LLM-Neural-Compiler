#!/usr/bin/env bash
# Bridge Gazebo camera image + camera_info onto /nnc/camera/*.
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
INFO_TOPIC="${GZ_TOPIC%/image}/camera_info"
echo "bridging $GZ_TOPIC -> /nnc/camera/image_raw"
echo "bridging $INFO_TOPIC -> /nnc/camera/camera_info"
exec ros2 run ros_gz_bridge parameter_bridge \
  "${GZ_TOPIC}@sensor_msgs/msg/Image[gz.msgs.Image" \
  "${INFO_TOPIC}@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo" \
  --ros-args \
  -r "${GZ_TOPIC}:=/nnc/camera/image_raw" \
  -r "${INFO_TOPIC}:=/nnc/camera/camera_info"
