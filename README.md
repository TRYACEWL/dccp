<h1 align="center"></h1>

<p align="center">
  A research codebase for recovery-aware VLA training, world-model rollouts, and multi-resolution reward modeling
</p>

## 📖 Overview

This repository extends a Vision-Language-Action training stack with three tightly connected components:

1. A recovery-aware policy optimization branch that mines near-failure states and searches for corrective actions.
2. A multi-resolution reward pipeline that separates trajectory-level success evaluation from local progress estimation.
3. A complete training workflow linking world-model rollouts, reward scoring, policy updates, and offline analysis tools.

The repository is organized as an experimental platform rather than a paper-only release. The focus is on runnable training entry points, configurable recovery logic, reward-model extensions, and a code structure that is easy to continue developing.

## ✨ Highlights

- Recovery-aware training that automatically identifies near-failure states from imagined trajectories and builds corrective supervision targets.
- Multi-resolution reward modeling with a shared VideoMAE backbone and separate heads for trajectory success and local progress.
- World-model rollouts built on top of the OpenSora path for imagined video generation and downstream reward evaluation.
- Offline analysis utilities for near-failure inspection, local score debugging, and recovery search export.
- Support for both single-node and multi-node execution through local scripts and Ray cluster launch helpers.

## ⚒️ Repository Structure

```text
.
├── docs/                     # design notes and implementation records
├── examples/
│   ├── mimicgen/             # policy training, recovery training, evaluation, offline analysis
│   └── opensora/             # world-model training scripts
├── reward_model/             # reward-model training and inference entry points
├── verl/                     # trainer, rollout, reward, and recovery implementations
├── dependencies/             # third-party source dependencies
├── install.sh                # environment setup script
├── launch_head.sh            # Ray head-node launcher
├── launch_worker.sh          # Ray worker-node launcher
├── download_hf.py            # checkpoint and data download helper
└── upload_hf.py              # checkpoint and data upload helper
```

Key code entry points:

- `verl/trainer/main_ppo.py`: main policy-training entry point.
- `verl/workers/rollout/robwm_rollout.py`: imagined rollout pipeline and recovery-branch integration.
- `verl/utils/recovery_mining.py`: near-failure mining logic.
- `verl/utils/recovery_search.py`: candidate recovery-action search.
- `verl/utils/reward_scorer.py`: unified trajectory/local reward scoring utilities.
- `reward_model/train_multi_resolution_videomae.py`: standalone dual-head reward training entry point.

## ⚒️ Getting Started

### Install the environment

We recommend using:

- `python=3.11.x`
- `torch=2.5.1`
- Linux + CUDA

Run the following commands to prepare the environment:

```bash
pip install -r requirements.txt
bash install.sh
```

`install.sh` continues by installing local dependencies under `dependencies/` and cloning the required external robotics repositories such as `robosuite`, `robomimic`, and `mimicgen`.

### Prepare datasets and checkpoints

To download the default checkpoints and data package, run:

```bash
python download_hf.py
```

You can override the default Hugging Face repository through an environment variable:

```bash
HF_REPO_ID=your-org/your-repo python download_hf.py
```

The repository typically depends on the following assets:

- initial policy checkpoints
- reward-model checkpoints
- world-model checkpoints
- dataset statistics files
- rollout and reward training data

Because the full asset set is large, it is usually better to download only the pieces required for your current experiment and adjust path variables in the scripts to match your local setup.

## 🚀 Running the Experiments

### Policy training and recovery training

Policy-related scripts live under `examples/mimicgen/<task>/`. The current task directories include:

- `coffee`
- `square`
- `stack_three`
- `three_piece_assembly`

Each task directory contains training and evaluation scripts. The `coffee` directory additionally includes the recovery-enabled training path and near-failure analysis utilities. Before launching experiments, the most important variables to verify are:

- `NUM_NODES`
- `NUM_GPUS_PER_NODE`
- `SFT_MODEL_PATH`
- `REWARD_MODEL_PATH`
- `DATASET_STATISTICS_PATH`
- `WANDB_API_KEY`

Notes:

- Some script filenames still retain legacy naming for compatibility.
- For actual usage, rely on the script functionality and directory location rather than the historical filename alone.

### Multi-node training

We use Ray to manage multi-node execution. The cluster launch scripts are located in the repository root:

```bash
bash launch_head.sh
bash launch_worker.sh
```

Before launching, update the following fields in the scripts:

- `MASTER_ADDR`
- `RAY_PORT`
- `NUM_GPUS_PER_NODE`

### World-model training

World-model training scripts are located under `examples/opensora/`, organized by task and rollout budget. Before running them, check:

- `GPUS_PER_NODE`
- `NNODES`
- `MASTER_ADDR`
- `node_rank`

### Reward-model training

The standalone training entry point for the multi-resolution reward model is:

```bash
bash reward_model/train_multi_resolution_videomae.sh
```

This script expects explicit training and validation shard patterns:

```bash
TRAIN_PATTERN="/path/to/train/**/*.tar" \
VAL_PATTERN="/path/to/val/**/*.tar" \
EXPERT_PATTERN="/path/to/expert/**/*.tar" \
INIT_REWARD_CHECKPOINT="/path/to/init_reward.pth" \
bash reward_model/train_multi_resolution_videomae.sh
```

Key inputs include:

- `TRAIN_PATTERN`
- `VAL_PATTERN`
- `EXPERT_PATTERN`
- `INIT_REWARD_CHECKPOINT`
- `CKPT_DIR`

### Evaluation and offline analysis

Evaluation scripts are provided inside each task directory. The current offline near-failure analysis script is located under `examples/mimicgen/coffee/`. These tools are useful for:

- checking whether local reward scores behave as expected
- inspecting mined near-failure states
- exporting recovery-search videos and metadata

## 🔧 Configuration Notes

The most commonly adjusted configuration groups in the current training stack are:

- recovery switches and parameters: `use_recovery_branch`, `recovery.*`
- multi-resolution reward switches and parameters: `use_multi_resolution_reward`, `reward.*`, `reward_loc.*`
- resource settings for actor, rollout, and reference components
- reward-model threshold, image size, and batch-size settings

The main default configuration lives in:

- `verl/trainer/config/ppo_trainer.yaml`

## 📚 Documentation

If you want to continue developing this repository, a good reading order is:

1. the recovery-branch implementation note in `docs/`
2. `docs/reward_model_change.md`
3. `verl/workers/rollout/robwm_rollout.py`
4. `verl/utils/recovery_mining.py`
5. `verl/utils/recovery_search.py`
6. `reward_model/train_multi_resolution_videomae.py`

The first document in `docs/` is still named with a legacy filename, but its content describes the current recovery-aware implementation path in this repository.

## 🙏 Acknowledgement

This repository builds on and adapts ideas and implementations from the following open-source projects:

- [Open-Sora](https://github.com/hpcaitech/Open-Sora)
- [openvla-oft](https://github.com/moojink/openvla-oft)
- [VideoMAE](https://github.com/MCG-NJU/VideoMAE)
- [verl](https://github.com/volcengine/verl)
- [mimicgen](https://github.com/NVlabs/mimicgen)
