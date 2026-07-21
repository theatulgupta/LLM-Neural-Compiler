#!/bin/bash

source /opt/ros/jazzy/setup.bash

cd ~/thesis_uav/ros2_ws

colcon build --symlink-install

source install/setup.bash
