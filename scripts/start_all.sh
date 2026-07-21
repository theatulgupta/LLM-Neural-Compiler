#!/bin/bash

gnome-terminal -- bash -c "~/LLM-Guided-UAV-System/scripts/start_agent.sh; exec bash"

sleep 2

gnome-terminal -- bash -c "~/LLM-Guided-UAV-System/scripts/start_px4.sh; exec bash"

sleep 8

gnome-terminal -- bash -c "~/LLM-Guided-UAV-System/scripts/start_ros.sh; exec bash"
