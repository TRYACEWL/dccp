# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Rollout with huggingface models.
TODO: refactor this class. Currently, it will hang when using FSDP HybridShard. We should actually create a single GPU model.
Then, get full state_dict and bind the state_dict to the single GPU model. Then, use the single GPU model to perform generation.
"""
import os
import imageio
import contextlib
import time
import json
import copy
import torch
import torch.distributed
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
from tensordict import TensorDict
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn.utils.rnn import pad_sequence

from verl import DataProto
from verl.utils.torch_functional import get_eos_mask
import verl.utils.torch_functional as verl_F
from .base import BaseRollout

from transformers import GenerationConfig, AutoProcessor
import tensorflow as tf
import numpy as np
from PIL import Image
from verl import DataProto

try:
    from libero.libero import benchmark
    from verl.utils.libero_utils import get_libero_env, get_libero_dummy_action, get_image_resize_size, get_libero_image, get_libero_wrist_image, quat2axisangle, normalize_gripper_action, invert_gripper_action, save_rollout_video
except:
    print("please install libero")

try:
    from robomimic.config import config_factory
    import robomimic.utils.file_utils as FileUtils
    import robomimic.utils.env_utils as EnvUtils
    import robomimic.utils.obs_utils as ObsUtils
    import mimicgen.envs.robosuite  # noqa: F401
except:
    print("please install robomimic")

from codetiming import Timer
from collections import deque
import random

import multiprocessing
import gc
from multiprocessing import Process, Queue
import multiprocessing as mp
mp.set_start_method("spawn", force=True)

from collections import defaultdict

from verl.utils.libero_utils import resize_image

from verl.utils.dccp_action_logprob import (
    build_action_vocab_mask,
    compute_action_entropy_from_logits,
    compute_reference_gap_from_logits,
)
from verl.utils.dccp_branching import DCCPActionCandidate, DCCPBranchingConfig
from verl.utils.dccp_mining import DCCPMiningConfig
from verl.utils.dccp_preferences import DCCPPreferenceConfig
from verl.utils.dccp_schema import PREF_KEYS
from verl.utils.dccp_scorer import DCCPScorer
from verl.utils.dccp_world_model_rollout import (
    DCCPNominalRollout,
    DCCPNominalStep,
    DCCPRolloutConfig,
    DCCPWorldModelRolloutAssembler,
)

__all__ = ['RobHFRollout']

OPENVLA_V01_SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)

def crop_and_resize(image, crop_scale, batch_size):
    """
    Center-crops an image to have area `crop_scale` * (original image area), and then resizes back
    to original size. We use the same logic seen in the `dlimp` RLDS datasets wrapper to avoid
    distribution shift at test time.

    Args:
        image: TF Tensor of shape (batch_size, H, W, C) or (H, W, C) and datatype tf.float32 with
               values between [0,1].
        crop_scale: The area of the center crop with respect to the original image.
        batch_size: Batch size.
    """
    # Convert from 3D Tensor (H, W, C) to 4D Tensor (batch_size, H, W, C)
    assert image.shape.ndims == 3 or image.shape.ndims == 4
    expanded_dims = False
    if image.shape.ndims == 3:
        image = tf.expand_dims(image, axis=0)
        expanded_dims = True

    # Get height and width of crop
    new_heights = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))
    new_widths = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))

    # Get bounding box representing crop
    height_offsets = (1 - new_heights) / 2
    width_offsets = (1 - new_widths) / 2
    bounding_boxes = tf.stack(
        [
            height_offsets,
            width_offsets,
            height_offsets + new_heights,
            width_offsets + new_widths,
        ],
        axis=1,
    )

    # Crop and then resize back up
    image = tf.image.crop_and_resize(image, bounding_boxes, tf.range(batch_size), (224, 224))

    # Convert back to 3D Tensor (H, W, C)
    if expanded_dims:
        image = image[0]

    return image

def center_crop_image(image):
    batch_size = 1
    crop_scale = 0.9

    # Convert to TF Tensor and record original data type (should be tf.uint8)
    image = tf.convert_to_tensor(np.array(image))
    orig_dtype = image.dtype

    # Convert to data type tf.float32 and values between [0,1]
    image = tf.image.convert_image_dtype(image, tf.float32)

    # Crop and then resize back to original size
    image = crop_and_resize(image, crop_scale, batch_size)

    # Convert back to original data type
    image = tf.clip_by_value(image, 0, 1)
    image = tf.image.convert_image_dtype(image, orig_dtype, saturate=True)

    # Convert back to PIL Image
    image = Image.fromarray(image.numpy())
    image = image.convert("RGB")
    return image

def _create_env(cfg):
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=cfg.train.data)
    shape_meta = FileUtils.get_shape_metadata_from_dataset(
        dataset_path=cfg.train.data,
        all_obs_keys=cfg.all_obs_keys,
        verbose=False,
    )
    if cfg.experiment.env is not None:
        env_meta["env_name"] = cfg.experiment.env
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        env_name=env_meta["env_name"],
        render=False,
        render_offscreen=True,
        use_image_obs=shape_meta["use_images"],
        use_depth_obs=shape_meta["use_depths"],
    )
    return EnvUtils.wrap_env_from_config(env, config=cfg)

def sample_state(cfg_dict, n_samples):
    cfg = config_factory(cfg_dict["algo_name"])
    with cfg.values_unlocked():
        cfg.update(cfg_dict)
    cfg.lock()
    ObsUtils.initialize_obs_utils_with_config(cfg)
    env = _create_env(cfg)  # 确保这里不再读数据集文件
    state_list = []
    for i in range(n_samples):
        env.reset()
        state = env.get_state()
        state_list.append(state)
    env.env.close()
    return state_list



class RobWMHFRollout(BaseRollout):

    def __init__(self, module: nn.Module, world_model_mapping, config):
        super().__init__()
        self.config = config
        self.module = module
        self.world_model_mapping = world_model_mapping
        self.processor = AutoProcessor.from_pretrained(config.pretrained_checkpoint, trust_remote_code=True)
        self.vla_preprocess()

        self.task = self.config.unnorm_key.split('_d0')[0]
        if "aloha" in self.task:
            self.task = "aloha"
        if self.task == "square":
            self.task_description = "Insert the square into the stick"
        elif self.task == "aloha":
            self.task_description = "Insert the square into the stick"
        else:
            self.task_description = self.task
            assert self.task in ["coffee", "stack_three", "three_piece_assembly"]
        if self.task != "aloha":
            env_config = f"./data_files/core_train_configs/bc_rnn_image_ds_{self.task}_D0_seed_101.json"
            ext_cfg = json.load(open(env_config, "r"))
            self.ext_cfg = ext_cfg
        
        if self.task == "coffee":
            self.max_steps = 256
        elif self.task == "stack_three":
            self.max_steps = 320
        elif self.task == "three_piece_assembly":
            self.max_steps = 384
        elif self.task == "square":
            self.max_steps = 184
        elif self.task == "aloha":
            self.max_steps = 224
        else:
            assert False

        self.vae = world_model_mapping["vae"]
        self.world_model = world_model_mapping["model"]
        self.scheduler = world_model_mapping["scheduler"]
        self.model_args = world_model_mapping["model_args"]
        self.dccp_scorer = self._build_dccp_scorer()
        self.dccp_rollout_config = self._build_dccp_rollout_config()
        self.queue_len = 4
        self.device = self.vae.device
        self.dtype = self.vae.dtype
        self.latent_size = self.vae.get_latent_size(input_size = [12, 256, 256])
        
        self.dccp_assembler = DCCPWorldModelRolloutAssembler(
            scorer=self.dccp_scorer,
            config=self.dccp_rollout_config,
            sample_alternative_actions_fn=self._sample_dccp_alternative_actions,
            rollout_counterfactual_branch_fn=self._rollout_dccp_counterfactual_branch,
            reference_gap_fn=self._compute_dccp_reference_gap,
        )
        
        
    
    @torch.no_grad()
    def predict_success(self, videos, batch_size=128):
        """使用 LRM completion 对完整 imagined trajectories 打分"""
        total_frames = videos.shape[1]
        instructions = [self.task_description for _ in range(len(videos))]
        metadata = [
            {
                "score_type": "trajectory_completion",
                "trajectory_index": int(index),
            }
            for index in range(len(videos))
        ]

        complete_list = self.dccp_scorer.score_trajectory_completion_batch(
            rollout_videos=[videos[index] for index in range(len(videos))],
            instructions=instructions,
            batch_metadata=metadata,
        )

        complete = torch.as_tensor(complete_list, dtype=torch.bool)
        finish_step = torch.full(
            (len(complete_list),),
            fill_value=int(total_frames - 1),
            dtype=torch.int64,
        )

        return {
            "complete": complete,
            "finish_step": finish_step,
        }

    def _cfg_get(self, cfg, key, default=None):
        """兼容 dict、OmegaConf 和普通对象的配置读取"""
        if cfg is None:
            return default
        if isinstance(cfg, dict):
            return cfg.get(key, default)
        if hasattr(cfg, "get"):
            try:
                return cfg.get(key, default)
            except Exception:
                pass
        return getattr(cfg, key, default)

    def _cfg_to_plain_dict(self, cfg):
        """将配置节点转换为普通 dict"""
        if cfg is None:
            return {}
        if isinstance(cfg, dict):
            return dict(cfg)
        try:
            from omegaconf import OmegaConf
            if OmegaConf.is_config(cfg):
                return OmegaConf.to_container(cfg, resolve=True)
        except Exception:
            pass
        if hasattr(cfg, "__dict__"):
            return {
                key: value
                for key, value in vars(cfg).items()
                if not key.startswith("_")
            }
        return {}

    def _build_dccp_scorer(self):
        """构造 DCCP completion/progress scorer"""
        scorer_cfg = self._cfg_to_plain_dict(self._cfg_get(self.config, "scorer", {}))
        lrm_input_cfg = self._cfg_to_plain_dict(self._cfg_get(self.config, "lrm_input", {}))

        return DCCPScorer.from_config_dict(
            {
                "scorer": scorer_cfg,
                "lrm_input": lrm_input_cfg,
            }
        )

    def _build_dccp_rollout_config(self):
        """构造 DCCP rollout-side preference construction 配置"""
        dccp_cfg = self._cfg_get(self.config, "dccp", {})
        dccp_cfg = self._cfg_to_plain_dict(dccp_cfg)

        mining_config = DCCPMiningConfig(
            horizon_H=int(dccp_cfg.get("horizon_H", 3)),
            frames_per_action=int(self.config.action_chunks_len),
            state_budget_per_traj=int(dccp_cfg.get("state_budget_per_traj", 2)),
            nms_gap=int(dccp_cfg.get("nms_gap", 2)),
            lambda_curvature=float(dccp_cfg.get("lambda_curvature", 1.0)),
            lambda_entropy=float(dccp_cfg.get("lambda_entropy", 1.0)),
            require_entropy=bool(dccp_cfg.get("require_entropy", False)),
        )

        branching_config = DCCPBranchingConfig(
            horizon_H=int(dccp_cfg.get("horizon_H", 3)),
            frames_per_action=int(self.config.action_chunks_len),
            num_candidates=int(dccp_cfg.get("num_candidates", 8)),
            max_branches_per_state=dccp_cfg.get("max_branches_per_state", None),
        )

        preference_config = DCCPPreferenceConfig(
            margin_pos=float(dccp_cfg.get("margin_pos", 0.10)),
            margin_neg=float(dccp_cfg.get("margin_neg", 0.10)),
            max_pairs_per_state=dccp_cfg.get("max_pairs_per_state", None),
            max_pairs_per_batch=int(dccp_cfg.get("max_pairs_per_batch", 64)),
        )

        max_pairs_per_rollout = int(
            dccp_cfg.get(
                "max_pairs_per_rollout",
                dccp_cfg.get("max_pairs_per_traj", 4),
            )
        )

        return DCCPRolloutConfig(
            mining=mining_config,
            branching=branching_config,
            preference=preference_config,
            max_pairs_per_rollout=max_pairs_per_rollout,
            max_pairs_per_batch=int(dccp_cfg.get("max_pairs_per_batch", 64)),
            require_completion_score=bool(dccp_cfg.get("require_completion_score", True)),
        )

    def vla_preprocess(self):
        if self.config.vla in ["openvla","openvla-oft"]:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:  
                    tf.config.experimental.set_memory_growth(gpu, True)
        
        if self.config.vla in ["openvla-oft"]:
            if  self.config.unnorm_key not in self.module.norm_stats and f"{self.config.unnorm_key}_no_noops" in self.module.norm_stats:
                self.config.unnorm_key = f"{self.config.unnorm_key}_no_noops"
            assert self.config.unnorm_key in self.module.norm_stats, f"Action un-norm key {self.config.unnorm_key} not found in VLA `norm_stats`!"

    def generate_wm_sequences(self, prompts):
        # breakpoint()
        batch_size = prompts.batch.batch_size[0]
        if prompts.meta_info.get('n_samples') is None:
            micro_batch_size = self.config.val_micro_batch_size if self.config.val_micro_batch_size is not None else 1
        else:
            micro_batch_size = self.config.get('micro_batch_size', batch_size)
        
        num_chunks = max(batch_size // micro_batch_size, 1)
        batch_prompts = prompts.chunk(chunks=num_chunks)
        output = [self._generate_wm_minibatch(p) for p in batch_prompts]
        output = DataProto.concat(output)
        return output
    
    def _prepare_data(self, image_paths, repeat):
        """
        一个私有的生成器方法，用于加载、重复和批处理初始数据。
        现在将一次性返回所有数据，而不使用 batch_size。
        """
        batch_buffer = []
        
        for image_path in image_paths:
            task_name = os.path.basename(image_path).split('.')[0]
            task_description = self.task_description
            init_frame_np = imageio.v2.imread(image_path)
            # init_frame_np = resize_image(init_frame_np, (224, 224))
            init_frame_tensor = torch.from_numpy(init_frame_np).permute(2, 0, 1).float() / 255.0 * 2 - 1
            init_frame_tensor = init_frame_tensor.to(self.device).to(self.dtype)

            for i in range(repeat):
                video_name = f"{task_name}_repeat_{i}"
                batch_buffer.append((init_frame_tensor.clone(), init_frame_np.copy(), task_description, video_name))
        
        init_tensors, init_numpys, descs, names = zip(*batch_buffer)
        return torch.stack(init_tensors), list(init_numpys), list(descs), list(names)

    
    @torch.no_grad()
    def run_wm_inference(self, image_paths, max_steps, repeat=1):
        """
        使用小批量（mini-batch）运行视频生成推理，并返回生成的视频数据。

        Returns:
            list: 一个字典列表。每个字典包含两个键:
                  'name' (str): 视频的名称。
                  'video' (np.ndarray): 视频的Numpy数组，形状为 (T, H, W, C)。
        """
        self.world_model.eval()
        vla_history = []
        latent_chunk = self.config.action_chunks_len        
        init_frames_tensors, init_frames_numpys, task_descriptions, video_names = self._prepare_data(image_paths, repeat)

        current_batch_size = init_frames_tensors.shape[0]
        init_frames_for_vae = init_frames_tensors.unsqueeze(2)
        with torch.no_grad():
            latents = self.vae.encode(init_frames_for_vae)
        image_history_tensor = latents.repeat(1, 1, self.queue_len, 1, 1)
        predicted_videos = [[np.expand_dims(frame, axis=0)] for frame in init_frames_numpys]
        current_frames_np = init_frames_numpys
        frame_num = 1
        while frame_num <= max_steps:
            current_inputs = [{'full_image': resize_image(frame, (224, 224))} for frame in current_frames_np]
    
            vla_input = self.process_input(current_inputs, task_descriptions)
            vla_output = self._generate_one_step(vla_input)
            actions = vla_output["action"]
            # breakpoint()
            step_data = {
                "responses": vla_output["responses"],
                "input_ids": vla_output["input_ids"],
                "attention_mask": vla_output["attention_mask"],
                "pixel_values": vla_output["pixel_values"],
                "action": actions,
                "normalized_actions": vla_output["normalized_actions"],
                "step": frame_num - 1,
            }

            if "action_token_entropy" in vla_output:
                step_data["action_token_entropy"] = vla_output["action_token_entropy"]

            if "action_token_logits" in vla_output:
                step_data["action_token_logits"] = vla_output["action_token_logits"]
            vla_history.append(step_data)
            
            actions = torch.from_numpy(vla_output['normalized_actions'])
            y = actions.to(self.device).to(self.dtype).reshape(current_batch_size, latent_chunk, -1)
            
            latent_size = self.latent_size

            z = torch.randn(current_batch_size, self.vae.out_channels, latent_chunk, *latent_size[1:], device=self.device, dtype=self.dtype)
            
            # z_combined 的形状: (B, C, T_history + chunk, H, W)
            z_combined = torch.concat([image_history_tensor, z], dim=2)
            
            masks = torch.zeros(current_batch_size, image_history_tensor.shape[2] + latent_chunk, device=self.device, dtype=torch.long)
            masks[:, -latent_chunk:] = 1
            samples = self.scheduler.sample(self.world_model, z=z_combined, y=y, device=self.device, additional_args=self.model_args, progress=False, mask=masks)
            
            # pred_latents 的形状: (B, C_latent, chunk, H_latent, W_latent)
            pred_latents = samples[:, :, -latent_chunk:].to(self.dtype)

            image_history_tensor = pred_latents.clone()[:, :, -self.queue_len:]
            
            # [修正 3] 移除解码前的 permute
            # pred_latents 已经是正确的 (B, C, T, H, W) 格式，可直接输入 vae.decode
            decoded_images = self.vae.decode(pred_latents)
            
            # decoded_images 的输出形状: (B, chunk, C, H, W)
            # permute 以便转换为Numpy: (B, chunk, H, W, C)
            pred_imgs_np = ((decoded_images.to(torch.float32).cpu().permute(0, 2, 3, 4, 1).numpy() * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)

            new_current_frames = []
            for i in range(current_batch_size):
                # pred_imgs_np[i] 的形状: (chunk, H, W, C)
                predicted_videos[i].append(pred_imgs_np[i])
                # 更新当前帧为8帧里的最后一帧
                new_current_frames.append(pred_imgs_np[i, -1])
            current_frames_np = new_current_frames
            frame_num += latent_chunk
            print(f"Batch processing frame_num: {frame_num}")
            # 
            # --- 结果处理与保存 (此部分有修改) ---
            # os.makedirs('./debug/wm', exist_ok=True)
            # for i in range(current_batch_size):
            #     final_video_frames = np.concatenate(predicted_videos[i], axis=0)
                
            #     result_item = {
            #         'name': batch_video_names[i],
            #         'video': final_video_frames
            #     }
            #     all_results.append(result_item)
                
            #     imageio.mimwrite(f"./debug/wm/{batch_video_names[i]}.mp4", final_video_frames, fps=30)
            
            # print(f"Finished processing batch. Saved videos: {batch_video_names}")
        predicted_videos = [np.concatenate(predicted_videos[i], axis=0) for i in range(current_batch_size)]
        predicted_videos = np.array(predicted_videos)
        # import pickle
        # debug = {
        #     "vla_history": vla_history,
        #     "predicted_videos": predicted_videos
        # }
        # local_rank = dist.get_rank() % 8
        # os.makedirs('./debug/pickle', exist_ok=True)
        # with open(f'./debug/pickle/debug_{local_rank}.pkl', 'wb') as f:
        #     pickle.dump(debug, f)
        return vla_history, predicted_videos

    def _candidate_to_world_model_action(self, candidate):
        """抽取 world model 使用的 normalized action"""
        if isinstance(candidate, DCCPActionCandidate):
            actions = candidate.action_for_world_model
        elif isinstance(candidate, dict) and "normalized_actions" in candidate:
            actions = candidate["normalized_actions"]
        else:
            actions = candidate

        if isinstance(actions, torch.Tensor):
            actions = actions.detach().float().cpu().numpy()

        actions = np.asarray(actions, dtype=np.float32)

        if actions.ndim == 2:
            actions = actions[None]

        return actions

    @torch.no_grad()
    def _sample_dccp_alternative_actions(self, nominal_rollout, nominal_step, num_alternatives):
        """从 selected state 采样 alternative first actions"""
        dccp_cfg = self._cfg_to_plain_dict(self._cfg_get(self.config, "dccp", {}))

        candidate_temperature = float(dccp_cfg.get("candidate_temperature", self.config.temperature))
        candidate_top_p = float(dccp_cfg.get("candidate_top_p", self.config.get("top_p", 1.0)))
        candidate_top_k = int(dccp_cfg.get("candidate_top_k", self.config.get("top_k", 0)))

        state_frame = nominal_step.state_context["frame"]
        vla_input = self.process_input(
            [{"full_image": resize_image(state_frame, (224, 224))}],
            [nominal_rollout.instruction],
        )

        vla_input["do_sample"] = True
        vla_input["temperature"] = candidate_temperature
        vla_input["top_p"] = candidate_top_p
        vla_input["top_k"] = candidate_top_k

        alternative_responses = []
        alternative_actions = []
        alternative_logprobs = []
        alternative_metadata = []

        for candidate_index in range(max(int(num_alternatives), 0)):
            candidate = self._generate_one_step(vla_input)

            alternative_responses.append(candidate["responses"].detach().clone().squeeze(0))
            alternative_actions.append(candidate["normalized_actions"])
            alternative_logprobs.append(None)
            alternative_metadata.append(
                {
                    "candidate_index": int(candidate_index + 1),
                    "temperature": float(candidate_temperature),
                    "top_p": float(candidate_top_p),
                    "top_k": int(candidate_top_k),
                }
            )

        return alternative_responses, alternative_actions, alternative_logprobs, alternative_metadata

    @torch.no_grad()
    def _rollout_dccp_counterfactual_branch(self, state_context, first_candidate, horizon_H, metadata=None):
        """固定 first action 并生成 counterfactual branch"""
        self.world_model.eval()

        latent_chunk = self.config.action_chunks_len
        horizon_H = max(int(horizon_H), 1)

        start_frame = state_context["frame"]
        instruction = state_context["instruction"]

        start_frame_tensor = torch.from_numpy(start_frame).permute(2, 0, 1).float() / 255.0 * 2 - 1
        start_frame_tensor = start_frame_tensor.to(self.device).to(self.dtype)

        init_frame_for_vae = start_frame_tensor.unsqueeze(0).unsqueeze(2)

        image_history_tensor = self.vae.encode(init_frame_for_vae).repeat(1, 1, self.queue_len, 1, 1)

        current_frame = start_frame
        predicted_video = [np.expand_dims(start_frame, axis=0)]

        for local_step in range(horizon_H):
            if local_step == 0:
                actions_np = self._candidate_to_world_model_action(first_candidate)
            else:
                vla_input = self.process_input(
                    [{"full_image": resize_image(current_frame, (224, 224))}],
                    [instruction],
                )
                vla_output = self._generate_one_step(vla_input)
                actions_np = self._candidate_to_world_model_action(vla_output)

            y = torch.from_numpy(actions_np).to(self.device).to(self.dtype).reshape(1, latent_chunk, -1)

            latent_size = self.latent_size
            z = torch.randn(
                1,
                self.vae.out_channels,
                latent_chunk,
                *latent_size[1:],
                device=self.device,
                dtype=self.dtype,
            )

            z_combined = torch.concat([image_history_tensor, z], dim=2)

            masks = torch.zeros(
                1,
                image_history_tensor.shape[2] + latent_chunk,
                device=self.device,
                dtype=torch.long,
            )
            masks[:, -latent_chunk:] = 1

            samples = self.scheduler.sample(
                self.world_model,
                z=z_combined,
                y=y,
                device=self.device,
                additional_args=self.model_args,
                progress=False,
                mask=masks,
            )

            pred_latents = samples[:, :, -latent_chunk:].to(self.dtype)
            image_history_tensor = pred_latents.clone()[:, :, -self.queue_len:]

            decoded_images = self.vae.decode(pred_latents)

            pred_imgs_np = (
                (decoded_images.to(torch.float32).cpu().permute(0, 2, 3, 4, 1).numpy() * 0.5 + 0.5) * 255
            ).clip(0, 255).astype(np.uint8)

            predicted_video.append(pred_imgs_np[0])
            current_frame = pred_imgs_np[0, -1]

        return np.concatenate(predicted_video, axis=0)

    def _make_response_mask(self, response_tokens):
        """构造 action response mask"""
        return torch.ones_like(response_tokens, dtype=torch.bool, device=response_tokens.device)

    def _get_action_vocab_mask(self, device):
        """构造 action-token vocabulary mask"""
        if hasattr(self.module, "vocab_size") and hasattr(self.module, "bin_centers"):
            vocab_size = int(self.module.vocab_size)
            num_bins = int(self.module.bin_centers.shape[0])
            action_token_begin = max(vocab_size - num_bins, 0)
            action_token_end = vocab_size

            return build_action_vocab_mask(
                vocab_size=vocab_size,
                device=device,
                action_token_begin=action_token_begin,
                action_token_end=action_token_end,
            )

        return None

    def _should_compute_dccp_action_entropy(self, prompts=None):
        """判断当前是否需要计算 DCCP action-token entropy"""
        if prompts is not None and "return_action_token_entropy" in prompts:
            return bool(prompts["return_action_token_entropy"])

        return bool(self.config.get("use_dccp_branch", False))

    def _extract_logits_from_policy_output(self, output):
        """从 policy forward 输出中提取 logits"""
        if hasattr(output, "logits"):
            return output.logits

        if isinstance(output, dict) and "logits" in output:
            return output["logits"]

        if isinstance(output, (tuple, list)) and len(output) > 0:
            return output[0]

        raise RuntimeError("Cannot extract logits from policy output")

    @torch.no_grad()
    def _teacher_force_action_logits(self, input_ids, attention_mask, pixel_values, response_tokens):
        """使用 teacher forcing 计算 action response positions 的 logits"""
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must be 2D, got {tuple(input_ids.shape)}")
        if attention_mask.ndim != 2:
            raise ValueError(f"attention_mask must be 2D, got {tuple(attention_mask.shape)}")
        if response_tokens.ndim != 2:
            raise ValueError(f"response_tokens must be 2D, got {tuple(response_tokens.shape)}")

        response_attention_mask = torch.ones_like(
            response_tokens,
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )

        full_input_ids = torch.cat(
            [input_ids, response_tokens.to(device=input_ids.device)],
            dim=1,
        )

        full_attention_mask = torch.cat(
            [attention_mask, response_attention_mask.to(device=attention_mask.device)],
            dim=1,
        )

        prompt_length = int(input_ids.shape[1])
        response_length = int(response_tokens.shape[1])

        param_ctx = contextlib.nullcontext()

        if isinstance(self.module, FSDP):
            param_ctx = FSDP.summon_full_params(self.module, writeback=False, recurse=False)

        with param_ctx:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = self.module(
                    input_ids=full_input_ids,
                    pixel_values=pixel_values,
                    attention_mask=full_attention_mask,
                )

        logits = self._extract_logits_from_policy_output(output)

        if logits.ndim != 3:
            raise ValueError(f"policy logits must be 3D [B, L, V], got {tuple(logits.shape)}")

        start = prompt_length - 1
        end = start + response_length

        if start < 0 or end > logits.shape[1]:
            raise ValueError(
                f"invalid action logit slice [{start}, {end}) for logits length {logits.shape[1]}"
            )

        return logits[:, start:end, :]

    @torch.no_grad()
    def _compute_dccp_action_entropy_from_response(
        self,
        input_ids,
        attention_mask,
        pixel_values,
        response_tokens,
    ):
        """根据已生成 response tokens 计算 DCCP action-token entropy"""
        action_logits = self._teacher_force_action_logits(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            response_tokens=response_tokens,
        )

        response_mask = torch.ones_like(
            response_tokens,
            dtype=torch.bool,
            device=response_tokens.device,
        )

        action_vocab_mask = self._get_action_vocab_mask(device=action_logits.device)

        entropy_result = compute_action_entropy_from_logits(
            logits=action_logits,
            response_mask=response_mask.to(device=action_logits.device),
            action_vocab_mask=action_vocab_mask,
        )

        return entropy_result.sequence_entropies.detach()

    def _compute_dccp_entropy_for_step(self, step_data, traj_idx):
        """读取或计算当前 decision step 的 action-token entropy"""
        if "action_token_entropy" in step_data:
            entropy = step_data["action_token_entropy"][traj_idx]
            if isinstance(entropy, torch.Tensor):
                return float(entropy.detach().cpu().item())
            return float(entropy)

        if "action_token_logits" not in step_data:
            return 0.0

        logits = step_data["action_token_logits"][traj_idx]
        response_tokens = step_data["responses"][traj_idx]
        response_mask = self._make_response_mask(response_tokens)
        action_vocab_mask = self._get_action_vocab_mask(device=logits.device)

        entropy_result = compute_action_entropy_from_logits(
            logits=logits.unsqueeze(0),
            response_mask=response_mask.unsqueeze(0),
            action_vocab_mask=action_vocab_mask,
        )

        return float(entropy_result.sequence_entropies[0].detach().cpu().item())

    def _compute_dccp_reference_gap(self, context, winner_candidate, loser_candidate):
        """计算 Δ_ref = logπ_ref(a_w|x) - logπ_ref(a_l|x)"""
        if "pref_delta_ref" in context:
            return float(context["pref_delta_ref"])

        input_ids = context[PREF_KEYS.input_ids].unsqueeze(0)
        attention_mask = context[PREF_KEYS.attention_mask].unsqueeze(0)
        pixel_values = context[PREF_KEYS.pixel_values].unsqueeze(0)

        winner_responses = winner_candidate.response_tokens.unsqueeze(0)
        loser_responses = loser_candidate.response_tokens.unsqueeze(0)

        response_mask = context[PREF_KEYS.response_mask].unsqueeze(0)

        if hasattr(self.module, "compute_dccp_reference_gap"):
            return float(
                self.module.compute_dccp_reference_gap(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    winner_responses=winner_responses,
                    loser_responses=loser_responses,
                    response_mask=response_mask,
                    unnorm_key=self.config.unnorm_key,
                )
            )

        winner_logits = self._teacher_force_action_logits(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            response_tokens=winner_responses,
        )

        loser_logits = self._teacher_force_action_logits(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            response_tokens=loser_responses,
        )

        action_vocab_mask = self._get_action_vocab_mask(device=winner_logits.device)

        reference_gap_result = compute_reference_gap_from_logits(
            winner_logits=winner_logits,
            loser_logits=loser_logits,
            winner_tokens=winner_responses.to(device=winner_logits.device),
            loser_tokens=loser_responses.to(device=loser_logits.device),
            winner_response_mask=response_mask.to(device=winner_logits.device),
            loser_response_mask=response_mask.to(device=loser_logits.device),
            action_vocab_mask=action_vocab_mask,
        )

        return float(reference_gap_result.delta_ref.detach().cpu().reshape(-1)[0].item())

    def _build_dccp_nominal_rollout(self, vla_history, videos, traj_idx, rollout_id):
        """将一条 imagined rollout 转换成 DCCPNominalRollout"""
        steps = []

        for step_data in vla_history:
            state_index = int(step_data["step"])
            frame_index = min(state_index, videos.shape[1] - 1)

            response_tokens = step_data["responses"][traj_idx].detach().clone()
            response_mask = self._make_response_mask(response_tokens)

            context = {
                PREF_KEYS.input_ids: step_data["input_ids"][traj_idx].detach().clone(),
                PREF_KEYS.attention_mask: step_data["attention_mask"][traj_idx].detach().clone(),
                PREF_KEYS.pixel_values: step_data["pixel_values"][traj_idx].detach().clone(),
                PREF_KEYS.response_mask: response_mask.detach().clone(),
            }

            state_context = {
                "frame": videos[traj_idx, frame_index],
                "instruction": self.task_description,
                "state_index": int(state_index),
                "frame_index": int(frame_index),
            }

            entropy_score = self._compute_dccp_entropy_for_step(step_data, traj_idx)

            step = DCCPNominalStep(
                state_index=state_index,
                context=context,
                state_context=state_context,
                response_tokens=response_tokens,
                action_for_world_model=step_data["normalized_actions"][traj_idx],
                response_mask=response_mask,
                action_token_logits=step_data.get("action_token_logits", None),
                entropy_score=float(entropy_score),
                generation_logprob=None,
                metadata={
                    "rollout_id": str(rollout_id),
                    "traj_idx": int(traj_idx),
                    "frame_index": int(frame_index),
                },
            )

            steps.append(step)

        return DCCPNominalRollout(
            rollout_id=str(rollout_id),
            instruction=self.task_description,
            video=videos[traj_idx],
            steps=steps,
            metadata={
                "traj_idx": int(traj_idx),
                "task": str(self.task),
            },
        )

    def _build_dccp_pref_batch(self, vla_history, videos, batch_size, global_steps=0, task_records=None):
        """构造 DCCP preference batch"""
        if not self.config.get("use_dccp_branch", False):
            return {}

        if len(vla_history) == 0:
            return {}

        first_step = vla_history[0]
        first_response = first_step["responses"][0].detach().clone()
        padding_response_mask = self._make_response_mask(first_response)

        padding_context = {
            PREF_KEYS.input_ids: torch.zeros_like(first_step["input_ids"][0]),
            PREF_KEYS.attention_mask: torch.zeros_like(first_step["attention_mask"][0]),
            PREF_KEYS.pixel_values: torch.zeros_like(first_step["pixel_values"][0]),
            PREF_KEYS.response_mask: torch.zeros_like(padding_response_mask),
        }

        nominal_rollouts = [
            self._build_dccp_nominal_rollout(
                vla_history=vla_history,
                videos=videos,
                traj_idx=traj_idx,
                rollout_id=f"global_{int(global_steps)}_traj_{int(traj_idx)}",
            )
            for traj_idx in range(batch_size)
        ]

        pref_batch, rollout_results, dccp_metrics = self.dccp_assembler.build_preference_batch(
            nominal_rollouts=nominal_rollouts,
            padding_context=padding_context,
            padding_response_tokens=first_response,
            padding_response_mask=padding_response_mask,
        )

        valid_pairs = int(pref_batch[PREF_KEYS.valid].sum().item())
        margin_mean = float(pref_batch[PREF_KEYS.margin][pref_batch[PREF_KEYS.valid]].mean().item()) if valid_pairs > 0 else 0.0

        print(
            "[dccp] preference batch "
            f"valid_pairs={valid_pairs} "
            f"max_pairs_per_rollout={int(self.dccp_rollout_config.max_pairs_per_rollout)} "
            f"margin_mean={margin_mean:.6f}",
            flush=True,
        )

        return pref_batch

    def _generate_wm_minibatch(self, prompts):        
        self.module.eval()
        meta_info = prompts.meta_info
        n_samples = meta_info.get('n_samples', 1)
        state_ids = prompts.batch['state_id'].cpu().reshape(-1).tolist()
        return_rollouts = meta_info.get('return_rollouts', False)
        max_steps = self.max_steps
        batch_size = len(state_ids) * n_samples
        if self.task == "square":
            image_paths = [f"./data_files/first_images/{self.task}/{state_id}.png" for state_id in state_ids]
        elif "aloha" in self.task:
            image_paths = [f"./data_files/first_images/{self.task}/{state_id}.png" for state_id in state_ids]
        elif self.task in ["coffee", "stack_three", "three_piece_assembly"]:
            image_paths = [f"./data_files/first_images/{self.task}/{state_id}.png" for state_id in state_ids]
        
        import time
        start_time = time.time()
        vla_history, videos = self.run_wm_inference(image_paths, max_steps, repeat=n_samples) 
        end_time = time.time()
        print(f"Generate video time cost: {end_time-start_time}")
        # import pickle
        # local_rank = dist.get_rank() % 8
        # with open(f'./debug/pickle/debug_{local_rank}.pkl', 'rb') as f:
        #     debug = pickle.load(f)
        # vla_history = debug['vla_history']
        # videos = debug['predicted_videos']
        import time
        start = time.time()
        task_records = self.predict_success(videos, batch_size=512)
        end = time.time()
        print(f"Predict success time: {end-start}")
        complete_flags = task_records["complete"].detach().cpu().numpy().astype(bool)
        finish_steps = task_records["finish_step"].detach().cpu().numpy().astype(np.int64)
        success_count = int(complete_flags.sum())
        failure_count = int(len(complete_flags) - success_count)
        print(
            "[dccp] rollout completion summary "
            f"batch_size={len(complete_flags)} success_count={success_count} "
            f"failure_count={failure_count}",
            flush=True,
        )

        for traj_idx, (complete_flag, finish_step) in enumerate(zip(complete_flags.tolist(), finish_steps.tolist())):
            print(
                "[dccp] rollout completion status "
                f"traj={traj_idx + 1}/{len(complete_flags)} complete={int(bool(complete_flag))} "
                f"finish_step={int(finish_step)}",
                flush=True,
            )
        
        batch = {
                'responses': [],
                'input_ids': [],  # here input_ids become the whole sentences
                'attention_mask': [],
                'pixel_values': [],
            }
        for k in ["responses", "input_ids", "attention_mask", "pixel_values"]:
            for h in vla_history:
                batch[k].append(h[k])
        
        for k,v in batch.items():
            batch[k] = torch.stack(v,dim=1) 
  
        batch["complete"] = task_records["complete"].to(dtype=torch.bool, device=self.device)
        batch["finish_step"] = task_records["finish_step"].to(dtype=torch.int64, device=self.device)
        batch['state_id'] = prompts.batch['state_id'].repeat_interleave(n_samples, dim=0)
        if self.config.get("use_dccp_branch", False):
            start_time = time.time()

            dccp_pref_batch = self._build_dccp_pref_batch(
                vla_history=vla_history,
                videos=videos,
                batch_size=batch_size,
                global_steps=meta_info.get("global_steps", 0),
                task_records=task_records,
            )

            if dccp_pref_batch:
                batch.update(dccp_pref_batch)
                valid_pairs = int(dccp_pref_batch[PREF_KEYS.valid].sum().item())
            else:
                valid_pairs = 0

            print(
                f"DCCP branch generated {valid_pairs} preference pairs in {time.time() - start_time:.2f} seconds",
                flush=True,
            )
        print(f"return_rollouts: {return_rollouts}")
        if return_rollouts:
            batch["action"] = []
            for h in vla_history:
                batch['action'].append(h['action'])
            batch['action'] = torch.tensor(batch['action'], dtype=torch.float32)
            batch['action'] = batch['action'].permute(1, 0, 2, 3).reshape(batch_size, -1, batch['action'].shape[-1])

            start_time = time.time()
            H, W, C = videos[0][0].shape 
            T = max_steps + 1
            placeholder = torch.empty((T, H, W, C), dtype=torch.uint8)
            videos_as_tensors = [torch.from_numpy(np.array(v, dtype=np.uint8)) for v in videos]
            # 同样使用 pad_sequence
            padded_with_placeholder = rnn_utils.pad_sequence(
                videos_as_tensors + [placeholder],  # 临时加入占位符
                batch_first=True,
                padding_value=0
            )
            padded_videos = padded_with_placeholder[:-1]
            batch["video"] = padded_videos
            end_time = time.time()
            print(f"Optimized padding time: {end_time - start_time} seconds")
        else:
            del videos

        output_batch = TensorDict(
            batch,
            batch_size=batch_size)

        # import pickle
        # local_rank = dist.get_rank() % 8
        # os.makedirs('./debug/output_batch', exist_ok=True)
        # with open(f'./debug/output_batch/output_batch_{local_rank}.pkl', 'wb') as f:
        #     pickle.dump(output_batch, f)
        # local_rank = dist.get_rank() % 8
        # with open(f'./debug/output_batch/output_batch_{local_rank}.pkl', 'rb') as f:
        #     output_batch = pickle.load(f)
        return DataProto(batch=output_batch)

    def generate_sequences(self, prompts):
        batch_size = prompts.batch.batch_size[0]
        
        if prompts.meta_info.get('n_samples') is None:
            micro_batch_size = self.config.val_micro_batch_size if self.config.val_micro_batch_size is not None else 1
        else:
            micro_batch_size = self.config.get('micro_batch_size', batch_size)
        
        num_chunks = max(batch_size // micro_batch_size, 1)
        batch_prompts = prompts.chunk(chunks=num_chunks)
        output = [self._generate_minibatch(p) for p in batch_prompts]
        output = DataProto.concat(output)
        return output
    
    def process_input(self,inputs:list, task_descriptions:list):
        
        batchdata = {"input_ids":[],"attention_mask":[],"pixel_values":[]}  
        
        for i in range(len(inputs)):
            input = inputs[i]
            task_description = task_descriptions[i]
           
            image = Image.fromarray(input["full_image"]).convert("RGB")
            if self.config.center_crop:
                image = center_crop_image(image)
            prompt = f"In: What action should the robot take to {task_description.lower()}?\nOut:"
            batch_feature  = self.processor(prompt, image)
            
            if "wrist_image" in input.keys():
                wrist_image = Image.fromarray(input["wrist_image"]).convert("RGB")
                if self.config.center_crop:
                    wrist_image = center_crop_image(wrist_image)
                wrist_batch_feature = self.processor(prompt, wrist_image)
                primary_pixel_values = batch_feature["pixel_values"]
                batch_feature["pixel_values"] = torch.cat([primary_pixel_values] + [wrist_batch_feature["pixel_values"]], dim=1)
                
            input_ids = batch_feature["input_ids"]
            attention_mask = batch_feature["attention_mask"]
            pixel_values = batch_feature["pixel_values"]
            
            if not torch.all(input_ids[:, -1] == 29871):
                input_ids = torch.cat(
                    (input_ids, torch.unsqueeze(torch.Tensor([29871]).long(), dim=0).to(input_ids.device)), dim=1
                )
                if self.config.vla in ["openvla-oft"]:
                    attention_mask = torch.cat(
                        (attention_mask, torch.unsqueeze(torch.Tensor([True]).bool(), dim=0).to(attention_mask.device)), dim=1
                    )
            
            batchdata["input_ids"].append(input_ids)    
            batchdata["attention_mask"].append(attention_mask)    
            batchdata["pixel_values"].append(pixel_values)    
        
        
        device = torch.device('cuda') 
        
        if self.config.vla in ["openvla-oft"]:
            batchdata["input_ids"] = [x.transpose(0, 1) for x in batchdata["input_ids"]]
            batchdata["attention_mask"] = [x.transpose(0, 1) for x in batchdata["attention_mask"]]
            batchdata["input_ids"] = pad_sequence(batchdata["input_ids"], batch_first=True, padding_value=self.processor.tokenizer.pad_token_id).squeeze(-1).to(device)
            batchdata["attention_mask"] = pad_sequence(batchdata["attention_mask"], batch_first=True, padding_value=0).squeeze(-1).to(device)
            
            padding_mask = batchdata["input_ids"].ne(self.processor.tokenizer.pad_token_id)
            assert  torch.all(padding_mask==batchdata["attention_mask"].ne(0))
            padding_mask = ~padding_mask
            padding_mask = padding_mask.int() 
            sorted_indices = torch.argsort(padding_mask, dim=1, descending=True, stable=True)
            batchdata["input_ids"] = torch.gather(batchdata["input_ids"], 1, sorted_indices)
            batchdata["attention_mask"] = torch.gather(batchdata["attention_mask"], 1, sorted_indices)
            
            
            batchdata["pixel_values"] = torch.cat(batchdata["pixel_values"] , dim=0).to(device)
            assert torch.all(batchdata["attention_mask"].ne(0) == batchdata["input_ids"].ne(self.processor.tokenizer.pad_token_id))
        else:
            for key in ["input_ids", "attention_mask", "pixel_values"]:
                batchdata[key] = torch.cat(batchdata[key], dim=0).to(device)

        return batchdata
   
    def _generate_minibatch(self, prompts):
        self.module.eval()
        meta_info = prompts.meta_info
        n_samples = meta_info.get('n_samples', 1)
        states = np.array(prompts.batch['states'].cpu())
        models = prompts.non_tensor_batch['model']
        state_list = [{"states": state, "model": model} for state, model in zip(states, models)]
        return_rollouts = meta_info.get('return_rollouts', False)
        max_steps = self.max_steps
        batch_size = prompts.batch.batch_size[0] * n_samples
        is_valid = meta_info.get('n_samples') is None
        global_steps = meta_info.get('global_steps', 0) if is_valid else 0
        is_valid = True

        # --- 初始化多个环境 ---
        envs = []
        inputs = []
        task_descriptions = []
        task_records = []
        valid_video = [[] for _ in range(batch_size)]

        for idx in range(batch_size):
            state = state_list[int(idx / n_samples)]
            cfg = config_factory(self.ext_cfg["algo_name"])
            with cfg.values_unlocked():
                cfg.update(self.ext_cfg)
            cfg.lock()
            ObsUtils.initialize_obs_utils_with_config(cfg)
            env = _create_env(cfg)

            if state:
                env.reset_to(state)
            else:
                env.reset()

            # 预跑 num_steps_wait
            t = 0
            valid_images = []
            obs = None
            while t < self.config.num_steps_wait:
                obs, _, _, _ = env.step(np.zeros(7))
                obs["agentview_image"] = (obs["agentview_image"]*255).astype(np.uint8).transpose(1,2,0)
                t += 1
            if is_valid:
                valid_images.append(obs["agentview_image"])

            envs.append(env)
            task_descriptions.append(self.task_description)
            inputs.append(self._obs_to_input(obs))
            task_records.append({
                "active": True,
                "complete": False,
                "finish_step": 0
            })
            if is_valid:
                valid_video[idx].extend(valid_images)

        # --- 主循环 ---
        vla_history = []
        step = 0
        while step < max_steps:
            print(f"Step = {step}")
            active_indices = [i for i, r in enumerate(task_records) if r['active']]

            current_inputs = inputs
            current_task_descriptions = task_descriptions
            vla_input = self.process_input(current_inputs, current_task_descriptions)
            vla_input.update(meta_info)
            vla_output = self._generate_one_step(vla_input)
            actions = vla_output["action"]

            step_data = {
                "responses": vla_output["responses"],
                "input_ids": vla_output["input_ids"],
                "attention_mask": vla_output["attention_mask"],
                "pixel_values": vla_output["pixel_values"],
                "action": actions,
                "step": step
            }
            vla_history.append(step_data)

            new_inputs = inputs.copy()
            for idx in active_indices:
                env = envs[idx]
                step_images = []

                for a in actions[idx]:
                    obs, reward, done, info = env.step(a.tolist())
                    obs["agentview_image"] = (obs["agentview_image"]*255).astype(np.uint8).transpose(1,2,0)
                    if is_valid:
                        step_images.append(obs["agentview_image"])

                    task_records[idx]['finish_step'] += 1
                    if reward > 0.0 or task_records[idx]['finish_step'] >= max_steps:
                        task_records[idx]['active'] = False
                        task_records[idx]['complete'] = reward > 0.0
                        break

                new_inputs[idx] = self._obs_to_input(obs)
                if is_valid:
                    valid_video[idx].extend(step_images)

            inputs = new_inputs
            step += self.config.action_chunks_len

        # --- 清理环境 ---
        for env in envs:
            env.env.close()
        import gc
        gc.collect()
        torch.cuda.empty_cache()        
        self.module.train()
        
        batch = {
                'responses': [],
                'input_ids': [],  # here input_ids become the whole sentences
                'attention_mask': [],
                'pixel_values': [],
            }
        for k in ["responses", "input_ids", "attention_mask", "pixel_values"]:
            for h in vla_history:
                batch[k].append(h[k])
        
        for k,v in batch.items():
            batch[k] = torch.stack(v,dim=1) 
  
        batch["complete"] = []
        batch["finish_step"] = []

        if return_rollouts:
            batch["action"] = []
            for h in vla_history:
                batch['action'].append(h['action'])
            batch['action'] = torch.tensor(batch['action'], dtype=torch.float32)
            batch['action'] = batch['action'].permute(1, 0, 2, 3).reshape(batch_size, -1, batch['action'].shape[-1])

            start_time = time.time()
            H, W, C = valid_video[0][0].shape 
            T = max_steps + 1
            placeholder = torch.empty((T, H, W, C), dtype=torch.uint8)
            videos_as_tensors = [torch.from_numpy(np.array(v, dtype=np.uint8)) for v in valid_video]
            # 同样使用 pad_sequence
            padded_with_placeholder = rnn_utils.pad_sequence(
                videos_as_tensors + [placeholder],  # 临时加入占位符
                batch_first=True,
                padding_value=0
            )
            padded_videos = padded_with_placeholder[:-1]
            batch["video"] = padded_videos
            end_time = time.time()
            print(f"Optimized padding time: {end_time - start_time} seconds")

        # batch['video'] = valid_video
        for k in task_records:
            batch["complete"].append(k["complete"])
            batch["finish_step"].append(k["finish_step"])
        
        batch["complete"] = torch.tensor(batch["complete"], dtype=torch.bool, device=batch['responses'].device)
        batch["finish_step"] = torch.tensor(batch["finish_step"], dtype=torch.int64, device=batch['responses'].device)
        # f()
        output_batch = TensorDict(
            batch,
            batch_size=batch_size)
        # TODO
        
        return DataProto(batch=output_batch)
    
    @torch.no_grad()
    def _generate_one_step(self, prompts: dict):
        if self.config.vla == "openvla-oft":
            idx = prompts['input_ids']  # (bs, prompt_length)
            attention_mask = prompts['attention_mask']  # left-padded attention_mask
            pixel_values = prompts["pixel_values"]
            raw_prompt_input_ids = idx
            raw_prompt_attention_mask = attention_mask
        
            param_ctx = contextlib.nullcontext()

            # make sampling args can be overriden by inputs
            do_sample = prompts.get('do_sample', self.config.do_sample)
        

            temperature = prompts.get('temperature', self.config.temperature)

            #generation_config = GenerationConfig(temperature=temperature, top_p=top_p, top_k=top_k)

            if isinstance(self.module, FSDP):
                # recurse need to set to False according to https://github.com/pytorch/pytorch/issues/100069
                param_ctx = FSDP.summon_full_params(self.module, writeback=False, recurse=False)
            
            with param_ctx:
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    actions, response, normalized_actions = self.module.generate_action_verl(
                        input_ids=idx,
                        pixel_values=pixel_values,
                        attention_mask=attention_mask,
                        padding_idx = self.processor.tokenizer.pad_token_id,
                        do_sample=do_sample,
                        unnorm_key=self.config.unnorm_key,
                        temperature=temperature, )
            
            
            assert self.processor.tokenizer.pad_token_id is not None

            assert idx.ndim == 2
            action_token_entropy = None
            if self._should_compute_dccp_action_entropy(prompts):
                action_token_entropy = self._compute_dccp_action_entropy_from_response(
                    input_ids=idx,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    response_tokens=response,
                )
            
            idx = verl_F.pad_sequence_to_length(idx,max_seq_len=self.config.max_prompt_length,pad_token_id=self.processor.tokenizer.pad_token_id,left_pad=True)
            
            assert attention_mask.ndim == 2
            attention_mask = verl_F.pad_sequence_to_length(attention_mask,max_seq_len=self.config.max_prompt_length,pad_token_id=0,left_pad=True)
            
            
            assert idx.device.type == 'cuda'
            assert response.device.type == 'cuda'
            #assert seq.device.type == 'cuda'
            assert attention_mask.device.type == 'cuda'
            assert pixel_values.device.type == 'cuda'
            batch = {
                "responses": response,
                "input_ids": idx,
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "action": actions,
                "normalized_actions": normalized_actions,
            }

            if action_token_entropy is not None:
                batch["action_token_entropy"] = action_token_entropy

            return batch
        
        elif self.config.vla == "openvla": 
            idx = prompts['input_ids']  # (bs, prompt_length)
            attention_mask = prompts['attention_mask']  # left-padded attention_mask
            pixel_values = prompts["pixel_values"]

            raw_prompt_input_ids = idx
            raw_prompt_attention_mask = attention_mask
            
            # used to construct attention_mask
            eos_token_id = prompts['eos_token_id']
            pad_token_id = prompts['pad_token_id']

            batch_size = idx.size(0)
            prompt_length = idx.size(1)
            #self.module.eval()
            param_ctx = contextlib.nullcontext()

            do_sample = prompts.get('do_sample', self.config.do_sample)
            response_length =  self.module.get_action_dim(self.config.unnorm_key)
            top_p = prompts.get('top_p', self.config.get('top_p', 1.0))
            top_k = prompts.get('top_k', self.config.get('top_k', 0))
            if top_k is None:
                top_k = 0
            top_k = max(0, top_k)  # to be compatible with vllm

            temperature = prompts.get('temperature', self.config.temperature)
            generation_config = GenerationConfig(temperature=temperature, top_p=top_p, top_k=top_k)

            if isinstance(self.module, FSDP):
                # recurse need to set to False according to https://github.com/pytorch/pytorch/issues/100069
                param_ctx = FSDP.summon_full_params(self.module, writeback=False, recurse=False)
            
            with param_ctx:
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    
                    output = self.module.generate(
                        input_ids=idx,
                        pixel_values=pixel_values,
                        attention_mask=attention_mask,
                        do_sample=do_sample,
                        max_new_tokens=response_length,
                        # max_length=max_length,
                        eos_token_id=eos_token_id,
                        pad_token_id=pad_token_id,
                        generation_config=generation_config,
                        # renormalize_logits=True,
                        output_scores=False,  # this is potentially very large
                        return_dict_in_generate=True,
                        use_cache=True)
                    
           
            seq = output.sequences
            sequence_length = prompt_length + response_length
            delta_length = sequence_length - seq.shape[1]
            
            assert delta_length == 0
            assert seq.shape[1] == sequence_length

            prompt = seq[:, :prompt_length]  # (bs, prompt_length)
            response = seq[:, prompt_length:]  # (bs, response_length)

            action_token_entropy = None

            if self._should_compute_dccp_action_entropy(prompts):
                action_token_entropy = self._compute_dccp_action_entropy_from_response(
                    input_ids=raw_prompt_input_ids,
                    attention_mask=raw_prompt_attention_mask,
                    pixel_values=pixel_values,
                    response_tokens=response,
                )
            
            response_length = response.size(1)
            #delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
            #delta_position_id = delta_position_id.unsqueeze(0).repeat(batch_size, 1)
            #response_position_ids = position_ids[:, -1:] + delta_position_id
            #position_ids = torch.cat([position_ids, response_position_ids], dim=-1)

            response_attention_mask = get_eos_mask(response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype)
            attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

            # Extract predicted action tokens and translate into (normalized) continuous actions
            predicted_action_token_ids = response.detach().cpu().numpy()
            discretized_actions = self.module.vocab_size - predicted_action_token_ids
            discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=self.module.bin_centers.shape[0] - 1)
            normalized_actions = self.module.bin_centers[discretized_actions]

            # Unnormalize actions
            action_norm_stats = self.module.get_action_stats(self.config.unnorm_key)
            mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
            action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
            actions = np.where(
                mask,
                0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
                normalized_actions,
            )
            
            actions = np.expand_dims(actions, axis=1)
            
            assert self.processor.tokenizer.pad_token_id is not None
            assert prompt.ndim == 2
            prompt = verl_F.pad_sequence_to_length(prompt,max_seq_len=self.config.max_prompt_length,pad_token_id=self.processor.tokenizer.pad_token_id,left_pad=True)
            assert seq.ndim == 2
            seq = verl_F.pad_sequence_to_length(seq,max_seq_len=self.config.max_prompt_length,pad_token_id=self.processor.tokenizer.pad_token_id,left_pad=True)
            assert attention_mask.ndim == 2
            attention_mask = verl_F.pad_sequence_to_length(attention_mask,max_seq_len=self.config.max_prompt_length,pad_token_id=0,left_pad=True)
            
            batch = {
                "prompts": prompt,
                "responses": response,
                "input_ids": seq,
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "action": actions,
                "normalized_actions": normalized_actions,
            }

            if action_token_entropy is not None:
                batch["action_token_entropy"] = action_token_entropy
            
            return batch
                    
    def _obs_to_input(self, obs):
        
        if self.config.num_images_in_input > 1:
            return {
                "full_image": get_libero_image(obs, 224),
                "wrist_image": get_libero_wrist_image(obs, 224),
                "state": np.concatenate([
                    obs["robot0_eef_pos"],
                    quat2axisangle(obs["robot0_eef_quat"]),
                    obs["robot0_gripper_qpos"]
                ])
            }
        else:
            return {
                "full_image": obs['agentview_image'], # get_libero_image(obs, 224),
                # "state": np.concatenate([
                #     obs["robot0_eef_pos"],
                #     quat2axisangle(obs["robot0_eef_quat"]),
                #     obs["robot0_gripper_qpos"]
                # ])
            }
