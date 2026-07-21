#!/bin/bash

source /opt/ros/jazzy/setup.bash

cd ~/LLM-Guided-UAV-System/ros2_ws

colcon build --symlink-install

source install/setup.bash
