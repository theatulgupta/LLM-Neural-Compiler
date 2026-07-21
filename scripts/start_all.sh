#!/bin/bash

gnome-terminal -- bash -c "~/thesis_uav/scripts/start_agent.sh; exec bash"

sleep 2

gnome-terminal -- bash -c "~/thesis_uav/scripts/start_px4.sh; exec bash"

sleep 8

gnome-terminal -- bash -c "~/thesis_uav/scripts/start_ros.sh; exec bash"
