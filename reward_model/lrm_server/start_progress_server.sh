#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

LRM_PROGRESS_MODEL_PATH="${LRM_PROGRESS_MODEL_PATH:-/data_jiang/wl/DCCP/LRM/model/progress}"
LRM_BASE_MODEL_PATH="${LRM_BASE_MODEL_PATH:-Qwen/Qwen3-VL-8B-Instruct}"
LRM_OFFICIAL_SERVER_PY="${LRM_OFFICIAL_SERVER_PY:-/home/wl/DCCP/Large-Reward-Models/vlm_reward/vlm_reward_server.py}"

LRM_HOST="${LRM_HOST:-127.0.0.1}"
LRM_PROGRESS_PORT="${LRM_PROGRESS_PORT:-8002}"
LRM_PROGRESS_GPU_ID="${LRM_PROGRESS_GPU_ID:-1}"
LRM_PROGRESS_BACKEND="${LRM_PROGRESS_BACKEND:-reward}"

python reward_model/lrm_server/dccp_lrm_server.py \
  --mode progress \
  --model_path "${LRM_PROGRESS_MODEL_PATH}" \
  --base_model_path "${LRM_BASE_MODEL_PATH}" \
  --official_server_py "${LRM_OFFICIAL_SERVER_PY}" \
  --gpu_id "${LRM_PROGRESS_GPU_ID}" \
  --host "${LRM_HOST}" \
  --port "${LRM_PROGRESS_PORT}" \
  --progress_backend "${LRM_PROGRESS_BACKEND}"