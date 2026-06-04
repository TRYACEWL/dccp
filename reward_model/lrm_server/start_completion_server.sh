#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

LRM_COMPLETION_MODEL_PATH="${LRM_COMPLETION_MODEL_PATH:-checkpoint_files/lrm/completion}"
LRM_BASE_MODEL_PATH="${LRM_BASE_MODEL_PATH:-Qwen/Qwen3-VL-8B-Instruct}"
LRM_OFFICIAL_SERVER_PY="${LRM_OFFICIAL_SERVER_PY:-external/Large-Reward-Models/vlm_reward/vlm_reward_server.py}"

LRM_HOST="${LRM_HOST:-127.0.0.1}"
LRM_COMPLETION_PORT="${LRM_COMPLETION_PORT:-8001}"
LRM_COMPLETION_GPU_ID="${LRM_COMPLETION_GPU_ID:-0}"

python reward_model/lrm_server/dccp_lrm_server.py \
  --mode completion \
  --model_path "${LRM_COMPLETION_MODEL_PATH}" \
  --base_model_path "${LRM_BASE_MODEL_PATH}" \
  --official_server_py "${LRM_OFFICIAL_SERVER_PY}" \
  --gpu_id "${LRM_COMPLETION_GPU_ID}" \
  --host "${LRM_HOST}" \
  --port "${LRM_COMPLETION_PORT}"