#!/bin/bash

echo "start ptpd for time-synchronisation"
sudo ptpd -M -i eth0 -C &   #& -> runs in background
PTPD_PID=$!

echo "start ros2 launchfile"
source ~/ws_livox/install/setup.bash
ros2 launch code_pkg all_sensors.launch.py

echo "Exit PTPD..."
sudo kill $PTPD_PID
