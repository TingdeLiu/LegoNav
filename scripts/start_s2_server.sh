#!/bin/bash
# S2 服务器启动脚本（GPU 机器）
# 用法: bash scripts/start_s2_server.sh /path/to/Qwen3-VL-8B-Instruct

MODEL_PATH="${1:-Qwen/Qwen3-VL-7B-Instruct}"

conda activate qwen3vl
python -m legonav.server.s2_server \
    --model_path "$MODEL_PATH" \
    --port 8890
