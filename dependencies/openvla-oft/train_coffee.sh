#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

export TF_CPP_MIN_LOG_LEVEL=2
export WANDB_API_KEY="${WANDB_API_KEY:-}"    # TODO: use your own WANDB_API_KEY

cd "$SCRIPT_DIR"

torchrun --standalone --nnodes 1 --nproc-per-node 4 "$SCRIPT_DIR"/vla-scripts/finetune.py \
 --vla_path openvla/openvla-7b \
 --data_root_dir "$REPO_ROOT"/data_files/sft_data/tensorflow_datasets \
 --dataset_name coffee_d0_300_demos \
 --run_root_dir "$REPO_ROOT"/checkpoint_files/openvla-oft/coffee_d0_300_demos \
 --use_l1_regression False \
 --use_diffusion False \
 --use_film False \
 --num_images_in_input 1 \
 --use_proprio False \
 --batch_size 8 \
 --learning_rate 5e-4 \
 --num_steps_before_decay 100000 \
 --max_steps 150000 \
 --save_freq 10000 \
 --save_latest_checkpoint_only False \
 --image_aug True \
 --lora_rank 32 \
 --wandb_entity "2978516418-neu" \
 --wandb_project "openvla-oft" \
 --run_id_note coffee_d0_300_demos
