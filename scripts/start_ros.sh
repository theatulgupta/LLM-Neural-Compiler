#!/bin/bash

source /opt/ros/jazzy/setup.bash
source ~/thesis_uav/ros2_ws/install/setup.bash

ros2 run llm_uav_core telemetry_node
