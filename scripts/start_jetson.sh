#!/bin/bash
# Jetson 端启动脚本（ROS2 + 本地 S1）
# 用法: bash scripts/start_jetson.sh "Go to the red chair" 192.168.1.100 /path/to/navdp.ckpt

INSTRUCTION="${1:-Go to the red chair}"
S2_HOST="${2:-192.168.1.100}"
S1_CHECKPOINT="${3:-}"

source /opt/ros/humble/setup.bash
conda activate navdp

if [ -n "$S1_CHECKPOINT" ]; then
    python -m lingnav.robot.ros_client \
        --instruction "$INSTRUCTION" \
        --s2_host "$S2_HOST" \
        --local_s1 \
        --s1_checkpoint "$S1_CHECKPOINT" \
        --s1_half
else
    python -m lingnav.robot.ros_client \
        --instruction "$INSTRUCTION" \
        --s2_host "$S2_HOST"
fi
