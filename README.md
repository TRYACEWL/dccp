<h1 align="center"></h1>

<p align="center">
  A research codebase for DCCP, world-model rollouts, LRM-based scoring, and decision-level counterfactual VLA post-training
</p>

## 📖 Overview

This repository implements DCCP on top of the WMPO training stack. It keeps the world-model rollout and policy-optimization workflow from WMPO, but replaces the recovery-aware branch with decision-level counterfactual comparison.

DCCP is organized around three tightly connected components:

1. An LRM-based scoring pipeline that separates trajectory-level completion evaluation from local progress estimation.
2. A decision-sensitive rollout branch that mines high-impact policy decision states from imagined trajectories.
3. A counterfactual preference-construction workflow that builds winner-loser action-token preferences for local DPO-style optimization.

The repository is organized as a research codebase for paper implementation and follow-up development. The focus is on a clean DCCP implementation, configurable LRM scoring, stable `pref_*` tensor interfaces, and compatibility with the existing WMPO world-model rollout workflow.

## ✨ Highlights

- DCCP rollout-side preference construction from imagined trajectories.
- LRM completion scoring for full trajectory-level task completion supervision.
- LRM progress scoring for nominal suffixes and short counterfactual branches.
- Decision-sensitive state mining based on local progress curvature and action-token entropy.
- Counterfactual first-action branch construction under the same visual-language state.
- High-margin winner-loser action-token preference construction.
- DPO-style preference loss interface through `pref_*` batch fields.
- LRM server adapter with `/completion` and `/progress` endpoints.
- Support for single-node and multi-node execution through the inherited Ray launch workflow.

## ⚒️ Repository Structure

```text
DCCP/
├── configs/                  # world-model and OpenSora configuration files
├── dependencies/             # third-party source dependencies
├── external/                 # optional external repositories, such as Large-Reward-Models
├── checkpoint_files/         # downloaded checkpoints and dataset files, not tracked by git
├── reward_model/             # reward/scoring backend entry points
│   └── lrm_server/           # DCCP LRM server adapter
├── tools/                    # optional debugging and smoke-test utilities
├── verl/                     # trainer, rollout, actor, and DCCP implementations
├── install.sh                # environment setup script
├── launch_head.sh            # Ray head-node launcher
├── launch_worker.sh          # Ray worker-node launcher
├── requirements.txt          # Python dependencies
└── README.md                 # project documentation
```

Key code entry points:

- `verl/trainer/main_ppo.py`: main policy-training entry point.
- `verl/workers/rollout/robwm_rollout.py`: world-model rollout pipeline and DCCP branch integration.
- `verl/workers/actor/dp_rob.py`: actor update and DCCP preference-loss computation.
- `verl/trainer/ppo/ray_trainer.py`: rollout collection, metric logging, and actor-update orchestration.
- `verl/trainer/config/ppo_trainer.yaml`: default PPO / WMPO / DCCP configuration.
- `verl/utils/dccp_schema.py`: canonical DCCP `pref_*` tensor-field names.
- `verl/utils/dccp_scorer.py`: LRM completion/progress scoring wrapper.
- `verl/utils/dccp_mining.py`: decision-sensitive state mining.
- `verl/utils/dccp_branching.py`: counterfactual branch construction.
- `verl/utils/dccp_preferences.py`: high-margin winner-loser preference construction.
- `verl/utils/dccp_world_model_rollout.py`: rollout-side DCCP assembler.
- `reward_model/lrm_server/dccp_lrm_server.py`: LRM HTTP server adapter for DCCP.

## ⚒️ Getting Started

### Install the WMPO / DCCP training environment

We recommend using:

- `python=3.11.x`
- `torch=2.5.1`
- Linux + CUDA

Run the following commands in the repository root:

```bash
cd DCCP
pip install -r requirements.txt
bash install.sh
```

`install.sh` installs local dependencies under `dependencies/` and prepares external robotics dependencies such as `robosuite`, `robomimic`, and `mimicgen`.

The DCCP training process should run in the WMPO environment, for example:

```bash
conda activate wmpo
```

### Install the LRM server environment

LRM completion/progress scoring is served by a separate HTTP server. We recommend using a separate environment, for example:

```bash
conda activate vlm_reward
```

The LRM server adapter in this repository reuses the official Large-Reward-Models implementation. Clone the LRM source code under `external/`:

```bash
cd DCCP
mkdir -p external

git clone https://github.com/physical-superintelligence-lab/Large-Reward-Models.git \
  external/Large-Reward-Models
```

The upstream LRM repository is:

```text
https://github.com/physical-superintelligence-lab/Large-Reward-Models/tree/main
```

By default, the DCCP LRM adapter expects the official server implementation at:

```text
external/Large-Reward-Models/vlm_reward/vlm_reward_server.py
```

If you place the external repository elsewhere, set `LRM_OFFICIAL_SERVER_PY` when starting the LRM server.

### Prepare datasets and checkpoints

DCCP uses two groups of assets:

1. WMPO-side policy, world-model, dataset, and first-frame assets.
2. LRM completion/progress checkpoints.

#### WMPO assets

WMPO checkpoints and data are released at:

```text
https://huggingface.co/fangqi/WMPO
```

You can use the inherited helper:

```bash
cd DCCP
python download_hf.py
```

The helper downloads the WMPO `checkpoint_files/**` and `data_files/**` assets into the repository root.

You can also download them explicitly with Hugging Face CLI:

```bash
cd DCCP

huggingface-cli download fangqi/WMPO \
  --repo-type model \
  --local-dir . \
  --local-dir-use-symlinks False \
  --include "checkpoint_files/**" "data_files/**"
```

The expected organization is:

```text
DCCP/
├── checkpoint_files/
│   ├── SFT_models/
│   ├── WMPO_models/
│   ├── world_models/
│   └── reward_models/
└── data_files/
```

DCCP does not use the legacy `reward_models/` path for its main scoring, but the rest of the WMPO assets are still used by the inherited training and rollout workflow.

#### LRM completion/progress checkpoints

The LRM checkpoints are released at:

```text
https://huggingface.co/USC-PSI-Lab/LRM-models
```

This model repository contains three subfolders:

```text
contrastive/
progress/
completion/
```

For DCCP, the required subfolders are:

```text
completion/
progress/
```

Download them into `checkpoint_files/lrm/`:

```bash
cd DCCP
mkdir -p checkpoint_files/lrm

huggingface-cli download USC-PSI-Lab/LRM-models \
  --repo-type model \
  --local-dir checkpoint_files/lrm \
  --local-dir-use-symlinks False \
  --include "completion/**" "progress/**"
```

Expected organization:

```text
DCCP/
└── checkpoint_files/
    └── lrm/
        ├── completion/
        └── progress/
```

The `completion/` checkpoint is used for:

```text
S_comp(Γ_traj(τ), instruction) -> {0, 1}
```

The `progress/` checkpoint is used for:

```text
S_prog(Γ_loc(ρ), instruction) -> [0, 1]
```

The LRM model is fine-tuned from Qwen3-VL-8B-Instruct. Transformers may download the base model automatically. If you want to pre-download it, use:

```text
https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct
```

Example command:

```bash
cd DCCP
mkdir -p checkpoint_files/base_models

huggingface-cli download Qwen/Qwen3-VL-8B-Instruct \
  --repo-type model \
  --local-dir checkpoint_files/base_models/Qwen3-VL-8B-Instruct \
  --local-dir-use-symlinks False
```

If the base model is stored locally, pass it through:

```bash
LRM_BASE_MODEL_PATH=checkpoint_files/base_models/Qwen3-VL-8B-Instruct
```

Because all checkpoints are large, `checkpoint_files/` and `data_files/` should not be committed to git.

## 🚀 Running the Experiments

### Start the LRM completion server

In terminal 1:

```bash
cd DCCP
conda activate vlm_reward

LRM_COMPLETION_MODEL_PATH=checkpoint_files/lrm/completion \
LRM_COMPLETION_GPU_ID=0 \
LRM_OFFICIAL_SERVER_PY=external/Large-Reward-Models/vlm_reward/vlm_reward_server.py \
bash reward_model/lrm_server/start_completion_server.sh
```

Default endpoint:

```text
http://127.0.0.1:8001/completion
```

This endpoint implements the DCCP completion interface:

```text
S_comp(Γ_traj(τ), instruction) -> {0, 1}
```

### Start the LRM progress server

In terminal 2:

```bash
cd DCCP
conda activate vlm_reward

LRM_PROGRESS_MODEL_PATH=checkpoint_files/lrm/progress \
LRM_PROGRESS_GPU_ID=0 \
LRM_PROGRESS_BACKEND=reward \
LRM_OFFICIAL_SERVER_PY=external/Large-Reward-Models/vlm_reward/vlm_reward_server.py \
bash reward_model/lrm_server/start_progress_server.sh
```

Default endpoint:

```text
http://127.0.0.1:8002/progress
```

This endpoint implements the DCCP progress interface:

```text
S_prog(Γ_loc(ρ), instruction) -> [0, 1]
```

`LRM_PROGRESS_BACKEND=reward` uses the absolute-progress text-generation backend. Robometer reward-head inference is optional and is not required by DCCP.

If the LRM server and training process are not on the same machine, set:

```bash
LRM_HOST=0.0.0.0
```

and replace `127.0.0.1` in the training configuration with the server machine's internal IP address.

### Test LRM endpoints

Completion endpoint:

```bash
python - <<'PY'
import requests

url = "http://127.0.0.1:8001/completion"
r = requests.get(url, timeout=5)
print("GET", r.status_code, r.text[:200])
PY
```

Progress endpoint:

```bash
python - <<'PY'
import requests

url = "http://127.0.0.1:8002/progress"
r = requests.get(url, timeout=5)
print("GET", r.status_code, r.text[:200])
PY
```

Expected endpoint outputs should include fields such as:

- completion: `score`, `completion_score`, `complete`, `success`
- progress: `score`, `progress_score`, `progress`, `success`

### Run DCCP policy training

DCCP does not require a new standalone training script. It is enabled by applying command-line overrides to the original WMPO training command.

For smoke testing, use small DCCP settings:

```bash
use_dccp_branch=true \
reward_model.enable=false \
scorer.completion_endpoint=http://127.0.0.1:8001/completion \
scorer.progress_endpoint=http://127.0.0.1:8002/progress \
dccp.horizon_H=1 \
dccp.state_budget_per_traj=1 \
dccp.num_candidates=2 \
dccp.max_pairs_per_rollout=1 \
dccp.max_pairs_per_batch=8 \
dccp.require_entropy=false \
trainer.total_epochs=1 \
trainer.val_before_train=false \
trainer.val_only=false \
trainer.save_freq=-1 \
trainer.test_freq=-1
```

For paper-style DCCP training, use the default DCCP settings:

```bash
use_dccp_branch=true \
reward_model.enable=false \
scorer.completion_endpoint=http://127.0.0.1:8001/completion \
scorer.progress_endpoint=http://127.0.0.1:8002/progress \
dccp.horizon_H=3 \
dccp.state_budget_per_traj=2 \
dccp.num_candidates=8 \
dccp.max_pairs_per_rollout=4 \
dccp.max_pairs_per_batch=64 \
dccp.require_entropy=true \
dccp.beta_dpo=0.1 \
dccp.lambda_pref=0.3 \
dccp.lambda_pref_warmup_steps=1000
```

`reward_model.enable=false` is required because DCCP uses LRM completion/progress endpoints instead of the legacy reward-model path.

### Multi-node training

The inherited Ray launch scripts are located in the repository root:

```bash
bash launch_head.sh
bash launch_worker.sh
```

Before launching, update:

- `MASTER_ADDR`
- `RAY_PORT`
- `NUM_GPUS_PER_NODE`

If LRM servers run on a different machine, replace `127.0.0.1` in the endpoints with the server machine's internal IP address.

### World-model training

World-model training is inherited from the WMPO / OpenSora workflow. Configuration files are located under:

```text
configs/
```

Before running world-model experiments, check:

- `GPUS_PER_NODE`
- `NNODES`
- `MASTER_ADDR`
- `node_rank`
- world-model checkpoint paths
- task-specific rollout configuration paths

### Evaluation and offline analysis

DCCP rollout-side metrics are logged during training. Useful expected logs include:

```text
[dccp] rollout completion summary
[dccp] preference batch valid_pairs=...
DCCP branch generated ... preference pairs
dccp/valid_pairs_rollout
dccp/margin_mean
loss/dccp_pref
dccp/valid_pairs_actor
dccp/lambda_pref
```

If `pref_valid` is always zero, possible causes include:

- LRM progress scores are too flat
- `margin_pos` / `margin_neg` are too large
- world-model rollout quality is insufficient
- decision-sensitive states do not produce high-margin branches
- completion/progress endpoints are unavailable or slow

## 🔧 Configuration Notes

The main default configuration lives in:

```text
verl/trainer/config/ppo_trainer.yaml
```

The most commonly adjusted DCCP configuration groups are:

- `use_dccp_branch`: enables rollout-side DCCP preference construction.
- `scorer.*`: completion/progress endpoint settings.
- `lrm_input.*`: keyframe extraction and LRM input construction.
- `dccp.horizon_H`: short branch imagination horizon.
- `dccp.state_budget_per_traj`: number of selected decision-sensitive states per trajectory.
- `dccp.num_candidates`: number of first-action candidates per selected state.
- `dccp.margin_pos` and `dccp.margin_neg`: high-margin filtering thresholds.
- `dccp.max_pairs_per_rollout`: number of preference pairs stored per rollout.
- `dccp.beta_dpo`: DPO-style preference-loss inverse temperature.
- `dccp.lambda_pref`: target preference-loss coefficient.
- `dccp.lambda_pref_warmup_steps`: warmup schedule for preference loss.

The canonical `pref_*` fields are:

```text
pref_input_ids
pref_attention_mask
pref_pixel_values
pref_winner_responses
pref_loser_responses
pref_response_mask
pref_weight
pref_delta_ref
pref_margin
pref_valid
```

Expected tensor layout:

```text
pref_valid:              [B, P]
pref_input_ids:          [B, P, L]
pref_attention_mask:     [B, P, L]
pref_pixel_values:       [B, P, ...]
pref_winner_responses:   [B, P, A]
pref_loser_responses:    [B, P, A]
pref_response_mask:      [B, P, A]
pref_weight:             [B, P]
pref_delta_ref:          [B, P]
pref_margin:             [B, P]
```

Where:

- `B`: rollout batch size
- `P`: `dccp.max_pairs_per_rollout`
- `L`: prompt length
- `A`: action-token response length

## 📚 Documentation

A recommended reading order is:

1. `verl/utils/dccp_schema.py`
2. `verl/utils/dccp_scorer.py`
3. `verl/utils/dccp_mining.py`
4. `verl/utils/dccp_branching.py`
5. `verl/utils/dccp_preferences.py`
6. `verl/utils/dccp_world_model_rollout.py`
7. `verl/workers/rollout/robwm_rollout.py`
8. `verl/trainer/ppo/ray_trainer.py`
9. `verl/workers/actor/dp_rob.py`
10. `reward_model/lrm_server/dccp_lrm_server.py`

For implementation-change details and A/B handoff information, see the accompanying code-change document.

## 🙏 Acknowledgement

This repository builds on the WMPO training framework and adapts several open-source components for DCCP research. We thank the authors and maintainers of:

- WMPO: https://github.com/WM-PO/WMPO
- Large-Reward-Models: https://github.com/physical-superintelligence-lab/Large-Reward-Models/tree/main
- Open-Sora
- openvla-oft
- verl
- robosuite
- robomimic
- mimicgen

DCCP removes the recovery-zero near-failure recovery branch and replaces it with LRM-based completion/progress scoring, decision-sensitive state mining, counterfactual branch comparison, and high-margin winner-loser preference construction.
