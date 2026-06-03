#!/usr/bin/env bash
set -euo pipefail

# 这个脚本是多头 reward model 的独立训练入口。
# 它不会改动 PPO 主训练循环，而是直接在 reward_model/ 目录下训练 dual-head checkpoint。
#
# 训练出来的 checkpoint 包含：
# 1. traj_head：完整 rollout 成功判断
# 2. loc_head：near-failure / recovery ranking 用的局部 progress 头
#
# 用法示例：
#   TRAIN_PATTERN="/path/to/train/**/*.tar" \
#   VAL_PATTERN="/path/to/val/**/*.tar" \
#   EXPERT_PATTERN="/path/to/expert/**/*.tar" \
#   INIT_REWARD_CHECKPOINT="/path/to/original_single_head_reward.pth" \
#   bash reward_model/train_multi_resolution_videomae.sh

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
TRAIN_PATTERN="${TRAIN_PATTERN:-}"
VAL_PATTERN="${VAL_PATTERN:-}"
EXPERT_PATTERN="${EXPERT_PATTERN:-}"
INIT_REWARD_CHECKPOINT="${INIT_REWARD_CHECKPOINT:-}"
CKPT_DIR="${CKPT_DIR:-ckpts_multi_resolution_reward}"

MODEL_NAME="${MODEL_NAME:-MCG-NJU/videomae-base}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-True}"

IMG_SIZE="${IMG_SIZE:-224}"
WINDOW="${WINDOW:-8}"
STRIDE_TRAIN="${STRIDE_TRAIN:-8}"
STRIDE_VAL="${STRIDE_VAL:-1}"

BATCH_SIZE="${BATCH_SIZE:-4}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PERSISTENT_WORKERS="${PERSISTENT_WORKERS:-True}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"

LR="${LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
MAX_STEPS="${MAX_STEPS:-200000}"
EVAL_STEPS="${EVAL_STEPS:-1000}"
SEED="${SEED:-42}"

LAMBDA_ANCHOR="${LAMBDA_ANCHOR:-1.0}"
LAMBDA_RANK="${LAMBDA_RANK:-0.5}"
LAMBDA_FAIL="${LAMBDA_FAIL:-0.5}"
LAMBDA_SMOOTH="${LAMBDA_SMOOTH:-0.0}"
FAILED_TAIL_K="${FAILED_TAIL_K:-2}"
RANK_NUM_PAIRS="${RANK_NUM_PAIRS:-2}"
PROGRESS_CURVE_BINS="${PROGRESS_CURVE_BINS:-5}"

MIX_WEIGHT_ROLLOUT="${MIX_WEIGHT_ROLLOUT:-0.7}"
MIX_WEIGHT_EXPERT="${MIX_WEIGHT_EXPERT:-0.3}"
USE_RESAMPLE_TRAIN="${USE_RESAMPLE_TRAIN:-True}"
DROP_LAST="${DROP_LAST:-True}"

if [[ -z "${TRAIN_PATTERN}" || -z "${VAL_PATTERN}" ]]; then
  echo "必须显式提供 TRAIN_PATTERN 和 VAL_PATTERN。" >&2
  echo "示例：" >&2
  echo "TRAIN_PATTERN='/path/to/train/**/*.tar' VAL_PATTERN='/path/to/val/**/*.tar' bash reward_model/train_multi_resolution_videomae.sh" >&2
  exit 1
fi

torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" reward_model/train_multi_resolution_videomae.py \
  --train-pattern "${TRAIN_PATTERN}" \
  --val-pattern "${VAL_PATTERN}" \
  --expert-pattern "${EXPERT_PATTERN}" \
  --mix-weight-rollout "${MIX_WEIGHT_ROLLOUT}" \
  --mix-weight-expert "${MIX_WEIGHT_EXPERT}" \
  --img-size "${IMG_SIZE}" \
  --window "${WINDOW}" \
  --stride-train "${STRIDE_TRAIN}" \
  --stride-val "${STRIDE_VAL}" \
  --batch-size "${BATCH_SIZE}" \
  --val-batch-size "${VAL_BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --persistent-workers "${PERSISTENT_WORKERS}" \
  --prefetch-factor "${PREFETCH_FACTOR}" \
  --lr "${LR}" \
  --weight-decay "${WEIGHT_DECAY}" \
  --max-steps "${MAX_STEPS}" \
  --eval-steps "${EVAL_STEPS}" \
  --ckpt-dir "${CKPT_DIR}" \
  --seed "${SEED}" \
  --model-name "${MODEL_NAME}" \
  --local-files-only "${LOCAL_FILES_ONLY}" \
  --use-resample-train "${USE_RESAMPLE_TRAIN}" \
  --drop-last "${DROP_LAST}" \
  --init-reward-checkpoint "${INIT_REWARD_CHECKPOINT}" \
  --lambda-anchor "${LAMBDA_ANCHOR}" \
  --lambda-rank "${LAMBDA_RANK}" \
  --lambda-fail "${LAMBDA_FAIL}" \
  --lambda-smooth "${LAMBDA_SMOOTH}" \
  --failed-tail-k "${FAILED_TAIL_K}" \
  --rank-num-pairs "${RANK_NUM_PAIRS}" \
  --progress-curve-bins "${PROGRESS_CURVE_BINS}"
