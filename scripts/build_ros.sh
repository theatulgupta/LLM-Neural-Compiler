#!/bin/bash

source /opt/ros/jazzy/setup.bash

cd ~/LLM-Neural-Compiler/ros2_ws

colcon build --symlink-install

source install/setup.bash
