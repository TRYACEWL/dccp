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
from verl.utils.recovery_mining import compute_local_success_scores, select_near_failure_states
from verl.utils.recovery_search import evaluate_recoverability
from verl.utils.recovery_targets import build_recovery_target
from verl.utils.reward_scorer import RewardScorer
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
        self.rm_model = world_model_mapping["rm_model"]
        self.rm_threshold = world_model_mapping["rm_threshold"]
        self.rm_feature_extractor = world_model_mapping["feature_extractor"]
        self.use_multi_resolution_reward = bool(world_model_mapping.get("use_multi_resolution_reward", False))
        self.queue_len = 4
        self.device = self.vae.device
        self.dtype = self.vae.dtype
        self.latent_size = self.vae.get_latent_size(input_size = [12, 256, 256])
        # Recovery 分支只复用已有 reward model，不更新 reward model 参数。
        # RewardScorer 提供 clip/trajectory 两种粒度打分接口，供 near-failure mining 和 search 共用。
        self.recovery_scorer = RewardScorer(
            self.rm_model,
            self.rm_feature_extractor,
            threshold=self.rm_threshold,
            device=self.device,
            batch_size=int(getattr(self.config.wm.reward, "batch_size", 128)),
            clip_len=int(getattr(self.config.wm.reward, "clip_len", 8)),
        )
        
        
    
    @torch.no_grad()
    def predict_success(self, videos, batch_size = 128):
        self.rm_model.eval()
        # videos B T H W C
        total_frames = videos.shape[1]
        window_size = int(getattr(self.recovery_scorer, "clip_len", 8))
        stride = 1
        min_steps = 100
        results = []
        # start_time = time.time()

        for video_idx, video in enumerate(videos):
            clips = []
            for end in range(total_frames, min_steps + window_size - 1, -stride):
                clip = video[end - window_size:end]
                clips.append((clip, end - window_size, end))
            clips = clips[::-1]
            clip_batches = [clips[i:i+batch_size] for i in range(0, len(clips), batch_size)]
            
            finish_step = total_frames - 1
            complete = 0
            for batch_idx, batch in enumerate(clip_batches):
                # current_time = time.time()
                # elapsed_time = current_time - start_time
                # print(f"Rank {dist.get_rank()}: Elapsed time: {elapsed_time:.2f} seconds : video {video_idx}/{len(videos)} batch {batch_idx}/{len(clip_batches)}")
                ranges = [(c[1], c[2]) for c in batch]
                clip_videos = [c[0] for c in batch]
                probs = self.recovery_scorer.score_traj_clip_segments(clip_videos)
                preds = [1 if float(p) >= self.rm_threshold else 0 for p in probs]
                
                for (start, end), prob, pred in zip(ranges, probs, preds):
                    if pred == 1 and end - 1 < finish_step:
                        finish_step = end - 1
                        complete = 1
                        break
            results.append({"complete": complete, 'finish_step': finish_step})

        complete = torch.from_numpy(np.array([r['complete'] for r in results]))
        finish_step = torch.from_numpy(np.array([r['finish_step'] for r in results]))

        return {
            "complete": complete,
            "finish_step": finish_step
        }
    
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
                "step": frame_num-1
            }
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

    def _empty_recovery_batch(self, vla_history, batch_size):
        """创建固定形状的 recovery 张量容器。

        即使某条轨迹没有挖到有效 near-failure 状态，也返回 rec_valid=False 的占位张量。
        这样 DataProto/TensorDict 在分布式 concat、split 时形状稳定，actor 端只需要看 rec_valid。
        """
        recovery_cfg = self.config.get("recovery", {})
        max_states = max(int(recovery_cfg.get("max_states_per_traj", 1)), 1)
        first_step = vla_history[0]
        rec_batch = {
            # rec_* 的前两维是 [trajectory, selected_state_slot]。
            # 后续维度复用原 stepwise vla_history 的 input/action token 形状。
            "rec_input_ids": torch.zeros(
                (batch_size, max_states) + tuple(first_step["input_ids"].shape[1:]),
                dtype=first_step["input_ids"].dtype,
                device=first_step["input_ids"].device,
            ),
            "rec_attention_mask": torch.zeros(
                (batch_size, max_states) + tuple(first_step["attention_mask"].shape[1:]),
                dtype=first_step["attention_mask"].dtype,
                device=first_step["attention_mask"].device,
            ),
            "rec_pixel_values": torch.zeros(
                (batch_size, max_states) + tuple(first_step["pixel_values"].shape[1:]),
                dtype=first_step["pixel_values"].dtype,
                device=first_step["pixel_values"].device,
            ),
            "rec_responses": torch.zeros(
                (batch_size, max_states) + tuple(first_step["responses"].shape[1:]),
                dtype=first_step["responses"].dtype,
                device=first_step["responses"].device,
            ),
            "rec_confidence": torch.zeros((batch_size, max_states), dtype=torch.float32, device=self.device),
            "rec_gain": torch.zeros((batch_size, max_states), dtype=torch.float32, device=self.device),
            "rec_baseline_score": torch.zeros((batch_size, max_states), dtype=torch.float32, device=self.device),
            "rec_best_score": torch.zeros((batch_size, max_states), dtype=torch.float32, device=self.device),
            "rec_num_candidates": torch.zeros((batch_size, max_states), dtype=torch.float32, device=self.device),
            # actor 端根据 rec_valid 过滤真正参与 L_rec 的样本。
            "rec_valid": torch.zeros((batch_size, max_states), dtype=torch.bool, device=self.device),
        }
        return rec_batch

    def _candidate_to_normalized_actions(self, candidate):
        """抽取 world model 使用的 normalized action。

        OpenVLA/OFT 同时返回 token response 和连续/归一化动作：
        - token response 用来做 recovery loss 的 log_prob 监督；
        - normalized action 用来喂给 world model 做局部短视界 rollout。
        """
        actions = candidate["normalized_actions"]
        if isinstance(actions, torch.Tensor):
            actions = actions.detach().float().cpu().numpy()
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim == 2:
            actions = actions[None]
        return actions

    @torch.no_grad()
    def _sample_recovery_candidates(self, start_frame, num_candidates):
        """从 near-failure 边界状态重新采样候选第一步动作。

        这里不重跑整条 trajectory，只把边界帧重新编码成当前策略输入，
        连续采样 num_candidates 次，得到候选第一步 token/action。
        注意：candidate_* 采样参数只作用于 recovery 的“第一步候选动作”，
        不改变原始 WMPO rollout，也不改变候选动作之后用于评估 recoverability 的后续策略。
        """
        recovery_cfg = self.config.get("recovery", {})
        candidate_temperature = float(recovery_cfg.get("candidate_temperature", self.config.temperature))
        candidate_top_p = float(recovery_cfg.get("candidate_top_p", self.config.get("top_p", 1.0)))
        candidate_top_k = int(recovery_cfg.get("candidate_top_k", self.config.get("top_k", 0)))
        vla_input = self.process_input([{"full_image": resize_image(start_frame, (224, 224))}], [self.task_description])
        vla_input["do_sample"] = True
        # recovery search 的目的就是在 near-failure 边界附近主动找“不同的第一步动作”。
        # 因此这里单独使用更高温度，避免候选动作都退化成普通 rollout 中几乎相同的输出。
        vla_input["temperature"] = candidate_temperature
        vla_input["top_p"] = candidate_top_p
        vla_input["top_k"] = candidate_top_k
        candidates = []
        for _ in range(max(int(num_candidates), 1)):
            candidate = self._generate_one_step(vla_input)
            candidates.append(candidate)
        return candidates

    @torch.no_grad()
    def _rollout_recovery_candidate(self, start_frame, _lang, first_candidate, horizon_hr):
        """固定候选第一步动作，并从该边界状态做局部 WM rollout。

        第 0 个 local_step 使用传入的 first_candidate；
        后续 local_step 重新调用当前策略，以保持“只搜索第一步恢复动作”的定义。
        返回值是短视频片段，由 RewardScorer 转成 recoverability 分数。
        """
        self.world_model.eval()
        latent_chunk = self.config.action_chunks_len
        horizon_hr = max(int(horizon_hr), 1)
        start_frame_tensor = torch.from_numpy(start_frame).permute(2, 0, 1).float() / 255.0 * 2 - 1
        start_frame_tensor = start_frame_tensor.to(self.device).to(self.dtype)
        init_frame_for_vae = start_frame_tensor.unsqueeze(0).unsqueeze(2)
        image_history_tensor = self.vae.encode(init_frame_for_vae).repeat(1, 1, self.queue_len, 1, 1)
        current_frame = start_frame
        predicted_video = [np.expand_dims(start_frame, axis=0)]

        for local_step in range(horizon_hr):
            if local_step == 0:
                # 第一步必须固定为当前候选动作，这是 recovery search 的核心约束。
                actions_np = self._candidate_to_normalized_actions(first_candidate)
            else:
                # 后续动作仍由当前策略采样，避免把局部 search 扩展成指数级动作树。
                vla_input = self.process_input(
                    [{"full_image": resize_image(current_frame, (224, 224))}],
                    [self.task_description],
                )
                vla_output = self._generate_one_step(vla_input)
                actions_np = self._candidate_to_normalized_actions(vla_output)

            y = torch.from_numpy(actions_np).to(self.device).to(self.dtype).reshape(1, latent_chunk, -1)
            latent_size = self.latent_size
            z = torch.randn(1, self.vae.out_channels, latent_chunk, *latent_size[1:], device=self.device, dtype=self.dtype)
            z_combined = torch.concat([image_history_tensor, z], dim=2)
            masks = torch.zeros(1, image_history_tensor.shape[2] + latent_chunk, device=self.device, dtype=torch.long)
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
            # 解码成 uint8 视频帧，保持与原 WM rollout / reward model 输入格式一致。
            pred_imgs_np = (
                (decoded_images.to(torch.float32).cpu().permute(0, 2, 3, 4, 1).numpy() * 0.5 + 0.5) * 255
            ).clip(0, 255).astype(np.uint8)
            predicted_video.append(pred_imgs_np[0])
            current_frame = pred_imgs_np[0, -1]

        return np.concatenate(predicted_video, axis=0)

    def _video_to_uint8(self, video):
        """把任意视频数组规范成 imageio 可以直接写入的 uint8 THWC。

        Recovery 导出只用于 debug/可视化，因此这里不参与训练图，也不改变原始张量。
        原始 WM rollout 和局部 recovery rollout 正常都是 uint8；这个函数主要兜底
        处理 float [0,1] 或单帧 HWC 输入，避免保存视频时因为 dtype/shape 崩掉。
        """
        array = np.asarray(video)
        if array.ndim == 3:
            array = array[None]
        if array.dtype != np.uint8:
            max_value = float(array.max()) if array.size > 0 else 0.0
            if max_value <= 1.0:
                array = array * 255.0
            array = np.clip(array, 0, 255).astype(np.uint8)
        return array

    def _save_recovery_video_bundle(
        self,
        export_dir,
        global_steps,
        traj_idx,
        slot,
        step_idx,
        frame_idx,
        original_prefix,
        recovery_local,
        original_full,
        metadata,
        fps,
        export_full_original,
        status_tag="valid_target",
    ):
        """保存一次 recovery search 对应的可视化视频。

        输出文件含义：
        - 00_original_until_near_failure.mp4：原 imagined rollout 从起点到 near-failure 边界；
        - 01_recovery_local_best_candidate.mp4：从边界帧开始，固定当前 best candidate 后的局部 WM 续写；
        - 02_recovery_process_combined.mp4：前两者拼接，展示“先走到险境，再尝试 recovery 动作纠偏”；
        - 03_original_full_imagined_rollout.mp4：可选，原始 imagined rollout 全程，便于对照失败趋势；
        - metadata.json：候选分数、baseline、gain、导出状态等检索信息。
        """
        rank = 0
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
        step_dir = os.path.join(
            str(export_dir),
            f"global_step_{int(global_steps)}",
            f"rank_{rank}",
            (
                f"traj_{int(traj_idx):04d}_slot_{int(slot):02d}_frame_{int(frame_idx):04d}"
                f"_{str(status_tag)}"
            ),
        )
        os.makedirs(step_dir, exist_ok=True)

        original_prefix = self._video_to_uint8(original_prefix)
        recovery_local = self._video_to_uint8(recovery_local)
        original_full = self._video_to_uint8(original_full)
        if len(recovery_local) > 1:
            combined = np.concatenate([original_prefix, recovery_local[1:]], axis=0)
        else:
            combined = original_prefix

        imageio.mimwrite(os.path.join(step_dir, "00_original_until_near_failure.mp4"), original_prefix, fps=fps)
        imageio.mimwrite(os.path.join(step_dir, "01_recovery_local_best_candidate.mp4"), recovery_local, fps=fps)
        imageio.mimwrite(os.path.join(step_dir, "02_recovery_process_combined.mp4"), combined, fps=fps)
        if export_full_original:
            imageio.mimwrite(os.path.join(step_dir, "03_original_full_imagined_rollout.mp4"), original_full, fps=fps)

        metadata = dict(metadata)
        metadata.update(
            {
                "global_steps": int(global_steps),
                "rank": int(rank),
                "traj_idx": int(traj_idx),
                "slot": int(slot),
                "step_idx": int(step_idx),
                "frame_idx": int(frame_idx),
                "status_tag": str(status_tag),
                "fps": int(fps),
                "original_prefix_frames": int(len(original_prefix)),
                "recovery_local_frames": int(len(recovery_local)),
                "combined_frames": int(len(combined)),
            }
        )
        with open(os.path.join(step_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2, sort_keys=True)
        print(f"[recovery] exported recovery video bundle: {step_dir}", flush=True)

    def _save_near_failure_video_bundle(
        self,
        export_dir,
        global_steps,
        traj_idx,
        slot,
        step_idx,
        frame_idx,
        original_prefix,
        original_suffix,
        original_full,
        metadata,
        fps,
        export_full_original,
        status_tag="near_failure_state",
    ):
        """保存 near-failure 状态本身的原 imagined rollout 视频。

        这个导出不依赖 recovery search 是否执行成功，主要用于排查：
        1. near-failure mining 是否挑到了“真的快出错”的边界；
        2. 被 state budget 跳过的 near-failure 后面到底发生了什么；
        3. search 没形成 target 时，原轨迹本身的失败模式是什么。
        """
        rank = 0
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
        step_dir = os.path.join(
            str(export_dir),
            f"global_step_{int(global_steps)}",
            f"rank_{rank}",
            (
                f"traj_{int(traj_idx):04d}_slot_{int(slot):02d}_frame_{int(frame_idx):04d}"
                f"_{str(status_tag)}"
            ),
        )
        os.makedirs(step_dir, exist_ok=True)

        original_prefix = self._video_to_uint8(original_prefix)
        original_suffix = self._video_to_uint8(original_suffix)
        original_full = self._video_to_uint8(original_full)

        imageio.mimwrite(os.path.join(step_dir, "00_original_until_near_failure.mp4"), original_prefix, fps=fps)
        imageio.mimwrite(os.path.join(step_dir, "01_original_from_near_failure.mp4"), original_suffix, fps=fps)
        if export_full_original:
            imageio.mimwrite(os.path.join(step_dir, "02_original_full_imagined_rollout.mp4"), original_full, fps=fps)

        metadata = dict(metadata)
        metadata.update(
            {
                "global_steps": int(global_steps),
                "rank": int(rank),
                "traj_idx": int(traj_idx),
                "slot": int(slot),
                "step_idx": int(step_idx),
                "frame_idx": int(frame_idx),
                "status_tag": str(status_tag),
                "fps": int(fps),
                "original_prefix_frames": int(len(original_prefix)),
                "original_suffix_frames": int(len(original_suffix)),
                "original_full_frames": int(len(original_full)),
            }
        )
        with open(os.path.join(step_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2, sort_keys=True)
        print(f"[recovery] exported near-failure video bundle: {step_dir}", flush=True)

    def _save_failed_rollout_video_bundle(
        self,
        export_dir,
        global_steps,
        traj_idx,
        original_full,
        metadata,
        fps,
        status_tag="failed_rollout_no_near_failure",
    ):
        """保存失败 rollout 本身的 imagined 视频。

        用于 complete=False 但 near-failure mining 没选出任何状态的情况，帮助排查：
        1. 失败轨迹是否整体都很差，导致 current_score 从未达到 alpha；
        2. 分数是否缓慢退化，而不是满足 min_gap/beta 的骤降条件；
        3. reward model 对这条失败轨迹的局部分数分布是否异常。
        """
        rank = 0
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
        step_dir = os.path.join(
            str(export_dir),
            f"global_step_{int(global_steps)}",
            f"rank_{rank}",
            f"traj_{int(traj_idx):04d}_{str(status_tag)}",
        )
        os.makedirs(step_dir, exist_ok=True)

        original_full = self._video_to_uint8(original_full)
        imageio.mimwrite(os.path.join(step_dir, "00_failed_full_imagined_rollout.mp4"), original_full, fps=fps)

        metadata = dict(metadata)
        metadata.update(
            {
                "global_steps": int(global_steps),
                "rank": int(rank),
                "traj_idx": int(traj_idx),
                "status_tag": str(status_tag),
                "fps": int(fps),
                "original_full_frames": int(len(original_full)),
            }
        )
        with open(os.path.join(step_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2, sort_keys=True)
        print(f"[recovery] exported failed-rollout video bundle: {step_dir}", flush=True)

    @torch.no_grad()
    def _build_recovery_batch(self, vla_history, videos, batch_size, global_steps=0, task_records=None):
        """从完整 imagined rollout 中挖掘并构造 recovery supervision batch。

        数据流：
        1. 对已有 imagined video 的每个 policy 决策边界计算局部成功分数；
        2. 用分数下降趋势筛 near-failure step；
        3. 在该边界状态采样候选第一步动作；
        4. 固定候选第一步，在 world model 里做短视界局部 rollout；
        5. 将有正增益的 best token action 写入 rec_* 张量。
        """
        recovery_cfg = self.config.get("recovery", {})
        if not self.config.get("use_recovery_branch", False):
            print("[recovery] disabled, skip recovery batch construction")
            return {}
        if len(vla_history) == 0:
            print("[recovery] empty vla_history, skip recovery batch construction")
            return {}
        rec_batch = self._empty_recovery_batch(vla_history, batch_size)

        step_frame_indices = [int(h["step"]) for h in vla_history]
        horizon_h = int(recovery_cfg.get("horizon_h", 16))
        alpha = float(recovery_cfg.get("alpha", 0.4))
        beta = float(recovery_cfg.get("beta", 0.1))
        min_gap = int(recovery_cfg.get("min_gap", 1))
        max_states = rec_batch["rec_valid"].shape[1]
        num_candidates = int(recovery_cfg.get("num_candidates", 4))
        candidate_temperature = float(recovery_cfg.get("candidate_temperature", self.config.temperature))
        candidate_top_p = float(recovery_cfg.get("candidate_top_p", self.config.get("top_p", 1.0)))
        candidate_top_k = int(recovery_cfg.get("candidate_top_k", self.config.get("top_k", 0)))
        recovery_horizon = int(recovery_cfg.get("recovery_horizon", 3))
        num_mc_samples = int(recovery_cfg.get("num_mc_samples", 2))
        eps_gain = float(recovery_cfg.get("eps_gain", 0.01))
        eps_recoverable = float(recovery_cfg.get("eps_recoverable", 0.0))
        target_baseline_mode = str(recovery_cfg.get("target_baseline_mode", "future_min_gap"))
        tau_gain = float(recovery_cfg.get("tau_gain", 0.1))
        top_b = int(recovery_cfg.get("top_b", 1))
        # local recovery search 的计算量约为：
        # searched_states * num_candidates * num_mc_samples * recovery_horizon。
        # 由于当前短视界 WM rollout 是 batch=1 串行执行，这个保护阈值可以避免
        # 一个 rollout batch 里 near-failure 过多时长时间没有训练反馈。
        default_max_total_states = batch_size * max_states
        max_total_states = int(recovery_cfg.get("max_total_states_per_batch", default_max_total_states))
        max_total_states = max(max_total_states, 0)
        search_log_every = max(int(recovery_cfg.get("search_log_every", 1)), 1)
        def _as_bool(value):
            """兼容 Hydra CLI 里传入的 True/False 字符串，避免 bool("False") 误判为 True。"""
            if isinstance(value, str):
                return value.lower() in ("1", "true", "yes", "y", "on")
            return bool(value)
        # 视频导出只用于观察 recovery search 的纠错过程，默认关闭。
        # 打开后仅保存“成功形成 recovery target”的 best candidate，避免把所有候选/MC 都落盘。
        export_videos = _as_bool(recovery_cfg.get("export_videos", False))
        export_dir = recovery_cfg.get("export_dir", "./tmp_files/recovery_videos")
        export_fps = int(recovery_cfg.get("export_fps", 10))
        export_full_original = _as_bool(recovery_cfg.get("export_full_original", True))
        export_max_videos_per_batch = int(recovery_cfg.get("export_max_videos_per_batch", 2))
        export_failed_search_videos = _as_bool(recovery_cfg.get("export_failed_search_videos", False))
        export_all_near_failure_videos = _as_bool(recovery_cfg.get("export_all_near_failure_videos", False))
        export_failed_rollout_videos = _as_bool(recovery_cfg.get("export_failed_rollout_videos", False))
        only_failed_rollouts = _as_bool(recovery_cfg.get("only_failed_rollouts", False))
        detailed_logging = _as_bool(recovery_cfg.get("detailed_logging", False))
        export_max_videos_per_batch = max(export_max_videos_per_batch, 0)
        exported_video_count = 0
        mined_state_count = 0
        searched_state_count = 0
        sampled_candidate_count = 0
        valid_target_count = 0
        skipped_no_gain = 0
        skipped_unrecoverable = 0
        skipped_state_budget = 0
        skipped_completed_rollouts = 0
        debug_examples = []

        complete_flags = None
        finish_steps = None
        if isinstance(task_records, dict):
            if "complete" in task_records:
                complete_flags = task_records["complete"].detach().cpu().numpy().astype(bool)
            if "finish_step" in task_records:
                finish_steps = task_records["finish_step"].detach().cpu().numpy().astype(np.int64)

        print(
            "[recovery] start "
            f"batch_size={batch_size} decision_steps={len(vla_history)} "
            f"horizon_h={horizon_h} alpha={alpha} beta={beta} min_gap={min_gap} "
            f"num_candidates={num_candidates} candidate_temperature={candidate_temperature} "
            f"candidate_top_p={candidate_top_p} candidate_top_k={candidate_top_k} "
            f"recovery_horizon={recovery_horizon} "
            f"num_mc_samples={num_mc_samples} max_states={max_states} "
            f"target_baseline_mode={target_baseline_mode} "
            f"max_total_states={max_total_states} search_log_every={search_log_every} "
            f"export_videos={export_videos} export_failed_search_videos={export_failed_search_videos} "
            f"export_all_near_failure_videos={export_all_near_failure_videos} "
            f"export_failed_rollout_videos={export_failed_rollout_videos} "
            f"only_failed_rollouts={only_failed_rollouts} detailed_logging={detailed_logging} "
            f"export_dir={export_dir}",
            flush=True,
        )

        def _select_target_baseline_score(local_scores, step_idx, frame_idx, traj_idx):
            """选择 recovery target 的 nominal baseline。

            多尺度 reward 版本中，baseline 改为：
            - 从同一个 near-failure 边界沿原 imagined rollout 继续走一个同长度短分支；
            - 用和 candidate search 完全同口径的 `R_loc` 打分。

            关闭 multi-resolution reward 时，仍保留旧的 future/current baseline 逻辑。
            """
            if self.use_multi_resolution_reward:
                local_horizon_frames = 1 + int(recovery_horizon) * int(self.config.action_chunks_len)
                nominal_video = videos[traj_idx, frame_idx : min(videos.shape[1], frame_idx + local_horizon_frames)]
                return float(self.recovery_scorer.score_loc_full_trajectory(nominal_video))

            scores = np.asarray(local_scores, dtype=np.float32)
            if len(scores) == 0:
                return 0.0
            step_idx = max(0, min(int(step_idx), len(scores) - 1))
            if target_baseline_mode == "future_min_gap":
                future_idx = min(step_idx + max(int(min_gap), 1), len(scores) - 1)
                return float(scores[future_idx])
            if target_baseline_mode == "future_min":
                future_scores = scores[step_idx + 1:]
                return float(np.min(future_scores)) if len(future_scores) > 0 else float(scores[step_idx])
            if target_baseline_mode != "current":
                print(
                    "[recovery] unknown target_baseline_mode, fallback to current "
                    f"mode={target_baseline_mode}",
                    flush=True,
                )
            return float(scores[step_idx])

        for traj_idx in range(batch_size):
            traj_timer = time.time()
            traj_complete = bool(complete_flags[traj_idx]) if complete_flags is not None else False
            traj_finish_step = int(finish_steps[traj_idx]) if finish_steps is not None else int(videos.shape[1] - 1)
            eligible_for_recovery = (not only_failed_rollouts) or (not traj_complete)
            if detailed_logging:
                print(
                    "[recovery] traj status "
                    f"traj={traj_idx + 1}/{batch_size} complete={int(traj_complete)} "
                    f"finish_step={traj_finish_step} eligible_for_recovery={int(eligible_for_recovery)}",
                    flush=True,
                )
            if not eligible_for_recovery:
                skipped_completed_rollouts += 1
                print(
                    "[recovery] traj skip "
                    f"traj={traj_idx + 1}/{batch_size} reason=complete_rollout "
                    f"only_failed_rollouts={only_failed_rollouts} finish_step={traj_finish_step}",
                    flush=True,
                )
                continue
            # vla_history 是按 policy step 存的，step_frame_indices 把 policy step 映射回视频帧。
            local_scores = compute_local_success_scores(
                {"video": videos[traj_idx]},
                self.recovery_scorer,
                h=horizon_h,
                step_indices=step_frame_indices,
            )
            state_indices = select_near_failure_states(
                local_scores,
                alpha=alpha,
                beta=beta,
                min_gap=min_gap,
                max_states=max_states,
            )
            mined_state_count += len(state_indices)
            if len(local_scores) > 0:
                score_min = float(np.min(local_scores))
                score_max = float(np.max(local_scores))
                score_last = float(local_scores[-1])
            else:
                score_min = score_max = score_last = 0.0
            print(
                "[recovery] mining "
                f"traj={traj_idx + 1}/{batch_size} selected={len(state_indices)} "
                f"complete={int(traj_complete)} finish_step={traj_finish_step} "
                f"score_min={score_min:.4f} score_max={score_max:.4f} score_last={score_last:.4f} "
                f"elapsed={time.time() - traj_timer:.2f}s",
                flush=True,
            )
            if detailed_logging and len(local_scores) > 1:
                candidate_debug = []
                for candidate_step_idx in range(max(int(min_gap), 1), len(local_scores)):
                    previous_step_idx = max(candidate_step_idx - max(int(min_gap), 1), 0)
                    current_score = float(local_scores[candidate_step_idx])
                    previous_score = float(local_scores[previous_step_idx])
                    score_drop = max(0.0, previous_score - current_score)
                    candidate_debug.append(
                        (
                            score_drop,
                            candidate_step_idx,
                            previous_step_idx,
                            current_score,
                            previous_score,
                        )
                    )
                for score_drop, candidate_step_idx, previous_step_idx, current_score, previous_score in sorted(
                    candidate_debug, key=lambda item: item[0], reverse=True
                )[:3]:
                    candidate_frame_idx = min(step_frame_indices[candidate_step_idx], videos.shape[1] - 1)
                    previous_frame_idx = min(step_frame_indices[previous_step_idx], videos.shape[1] - 1)
                    qualifies = int(alpha < current_score < beta and score_drop > 0.0)
                    print(
                        "[recovery] mining candidate "
                        f"traj={traj_idx + 1}/{batch_size} step_idx={candidate_step_idx} "
                        f"frame_idx={candidate_frame_idx} previous_step_idx={previous_step_idx} "
                        f"previous_frame_idx={previous_frame_idx} previous_score={previous_score:.4f} "
                        f"current_score={current_score:.4f} "
                        f"drop={score_drop:.4f} qualifies={qualifies}",
                        flush=True,
                    )
            if (
                export_videos
                and export_failed_rollout_videos
                and (not traj_complete)
                and len(state_indices) == 0
            ):
                top_drop_candidates = []
                if len(local_scores) > 1:
                    for candidate_step_idx in range(max(int(min_gap), 1), len(local_scores)):
                        previous_step_idx = max(candidate_step_idx - max(int(min_gap), 1), 0)
                        current_score = float(local_scores[candidate_step_idx])
                        previous_score = float(local_scores[previous_step_idx])
                        score_drop = max(0.0, previous_score - current_score)
                        top_drop_candidates.append(
                            {
                                "step_idx": int(candidate_step_idx),
                                "frame_idx": int(min(step_frame_indices[candidate_step_idx], videos.shape[1] - 1)),
                                "previous_step_idx": int(previous_step_idx),
                                "previous_frame_idx": int(min(step_frame_indices[previous_step_idx], videos.shape[1] - 1)),
                                "previous_score": float(previous_score),
                                "current_score": float(current_score),
                                "drop": float(score_drop),
                                "qualifies": bool(alpha < current_score < beta and score_drop > 0.0),
                            }
                        )
                self._save_failed_rollout_video_bundle(
                    export_dir=export_dir,
                    global_steps=global_steps,
                    traj_idx=traj_idx,
                    original_full=videos[traj_idx],
                    metadata={
                        "complete": bool(traj_complete),
                        "finish_step": int(traj_finish_step),
                        "score_min": float(score_min),
                        "score_max": float(score_max),
                        "score_last": float(score_last),
                        "alpha": float(alpha),
                        "beta": float(beta),
                        "min_gap": int(min_gap),
                        "max_states_per_traj": int(max_states),
                        "reason": "failed_rollout_but_no_near_failure_selected",
                        "top_drop_candidates": top_drop_candidates[:5],
                    },
                    fps=export_fps,
                )
            for slot, step_idx in enumerate(state_indices[:max_states]):
                frame_idx = min(step_frame_indices[step_idx], videos.shape[1] - 1)
                current_score = float(local_scores[step_idx]) if len(local_scores) > 0 else 0.0
                previous_step_idx = max(step_idx - max(int(min_gap), 1), 0) if len(local_scores) > 0 else 0
                previous_score = float(local_scores[previous_step_idx]) if len(local_scores) > 0 else 0.0
                score_drop = max(0.0, previous_score - current_score)
                previous_frame_idx = min(step_frame_indices[previous_step_idx], videos.shape[1] - 1) if len(step_frame_indices) > 0 else frame_idx
                frames_remaining = int(videos.shape[1] - frame_idx - 1)
                print(
                    "[recovery] near-failure selected "
                    f"traj={traj_idx + 1}/{batch_size} slot={slot} step_idx={int(step_idx)} "
                    f"frame_idx={int(frame_idx)} previous_step_idx={int(previous_step_idx)} "
                    f"previous_frame_idx={int(previous_frame_idx)} previous_score={previous_score:.4f} "
                    f"current_score={current_score:.4f} drop={score_drop:.4f} "
                    f"complete={int(traj_complete)} finish_step={traj_finish_step} "
                    f"frames_remaining={frames_remaining}",
                    flush=True,
                )
                if export_videos and export_all_near_failure_videos:
                    self._save_near_failure_video_bundle(
                        export_dir=export_dir,
                        global_steps=global_steps,
                        traj_idx=traj_idx,
                        slot=slot,
                        step_idx=step_idx,
                        frame_idx=frame_idx,
                        original_prefix=videos[traj_idx, : frame_idx + 1],
                        original_suffix=videos[traj_idx, frame_idx:],
                        original_full=videos[traj_idx],
                        metadata={
                            "current_score": current_score,
                            "previous_step_idx": int(previous_step_idx),
                            "previous_frame_idx": int(previous_frame_idx),
                            "previous_score": previous_score,
                            "score_drop": score_drop,
                            "complete": bool(traj_complete),
                            "finish_step": int(traj_finish_step),
                            "frames_remaining": int(frames_remaining),
                            "target_baseline_mode": str(target_baseline_mode),
                            "near_failure_export_reason": "selected_by_mining",
                        },
                        fps=export_fps,
                        export_full_original=export_full_original,
                        status_tag="near_failure_selected",
                    )
                if searched_state_count >= max_total_states:
                    skipped_state_budget += 1
                    if export_videos and export_all_near_failure_videos:
                        self._save_near_failure_video_bundle(
                            export_dir=export_dir,
                            global_steps=global_steps,
                            traj_idx=traj_idx,
                            slot=slot,
                            step_idx=step_idx,
                            frame_idx=frame_idx,
                            original_prefix=videos[traj_idx, : frame_idx + 1],
                            original_suffix=videos[traj_idx, frame_idx:],
                            original_full=videos[traj_idx],
                            metadata={
                                "current_score": current_score,
                                "previous_step_idx": int(previous_step_idx),
                                "previous_frame_idx": int(previous_frame_idx),
                                "previous_score": previous_score,
                                "score_drop": score_drop,
                                "complete": bool(traj_complete),
                                "finish_step": int(traj_finish_step),
                                "frames_remaining": int(frames_remaining),
                                "target_baseline_mode": str(target_baseline_mode),
                                "near_failure_export_reason": "skipped_by_state_budget",
                                "max_total_states_per_batch": int(max_total_states),
                            },
                            fps=export_fps,
                            export_full_original=export_full_original,
                            status_tag="near_failure_skipped_state_budget",
                        )
                    print(
                        "[recovery] state skip "
                        f"traj={traj_idx} slot={slot} step_idx={int(step_idx)} "
                        f"reason=max_total_states_per_batch({max_total_states})",
                        flush=True,
                    )
                    continue

                start_frame = videos[traj_idx, frame_idx]
                searched_state_count += 1
                state_timer = time.time()
                print(
                    "[recovery] state start "
                    f"traj={traj_idx} slot={slot} searched={searched_state_count}/{max_total_states} "
                    f"step_idx={int(step_idx)} frame_idx={int(frame_idx)} "
                    f"current_score={current_score:.4f}",
                    flush=True,
                )
                # 只从边界状态采样候选第一步动作，不回到 trajectory 起点。
                candidates = self._sample_recovery_candidates(start_frame, num_candidates)
                sampled_candidate_count += len(candidates)
                candidate_actions = [self._candidate_to_normalized_actions(candidate) for candidate in candidates]
                # 当前 actor loss 需要 token response，因此 target 退化为 best candidate response。
                candidate_responses = [candidate["responses"].detach().clone().squeeze(0) for candidate in candidates]
                # 这里记录候选动作多样性，便于判断 recovery search 是否真的在探索。
                # unique_response_count 看 token action 是否重复；action_std_mean 看归一化连续动作的平均离散程度。
                response_keys = [
                    tuple(candidate["responses"].detach().cpu().reshape(-1).tolist())
                    for candidate in candidates
                ]
                unique_response_count = len(set(response_keys))
                if len(candidate_actions) > 1:
                    action_stack = np.stack([np.asarray(action).reshape(-1) for action in candidate_actions], axis=0)
                    action_std_mean = float(np.std(action_stack, axis=0).mean())
                else:
                    action_std_mean = 0.0
                print(
                    "[recovery] candidates sampled "
                    f"traj={traj_idx} slot={slot} count={len(candidates)} "
                    f"unique_responses={unique_response_count} action_std_mean={action_std_mean:.6f} "
                    f"elapsed={time.time() - state_timer:.2f}s",
                    flush=True,
                )

                total_search_rollouts = len(candidates) * max(int(num_mc_samples), 1)
                export_candidate_videos = {}

                def _log_search_progress(candidate_idx, mc_idx, phase, score):
                    # 这里的 unit 是“候选动作的一次 MC 评估”，不是 world model 内部的每个 denoise step。
                    # 每个 unit 还会继续 rollout recovery_horizon 个 action chunk，所以单个 unit 本身也可能较慢。
                    unit = candidate_idx * max(int(num_mc_samples), 1) + mc_idx + 1
                    should_log = phase == "start" and (unit == 1 or unit % search_log_every == 0)
                    should_log = should_log or (phase == "done" and (unit % search_log_every == 0 or unit == total_search_rollouts))
                    if not should_log:
                        return
                    score_text = "" if score is None else f" score={float(score):.4f}"
                    print(
                        "[recovery] search "
                        f"traj={traj_idx} slot={slot} unit={unit}/{total_search_rollouts} "
                        f"candidate={candidate_idx + 1}/{len(candidates)} mc={mc_idx + 1}/{max(int(num_mc_samples), 1)} "
                        f"phase={phase}{score_text} elapsed={time.time() - state_timer:.2f}s",
                        flush=True,
                    )

                def _capture_recovery_video(candidate_idx, mc_idx, video, score):
                    """缓存每个候选动作得分最高的一段局部 WM 视频，供最终 best target 导出。

                    这里不马上写盘，因为 target 构造可能会因 no_positive_gain / below_eps_recoverable
                    被跳过。等确认该 near-failure state 真的贡献了监督样本，再只保存 best candidate。
                    """
                    if not export_videos or exported_video_count >= export_max_videos_per_batch:
                        return
                    score = float(score)
                    cached = export_candidate_videos.get(candidate_idx)
                    if cached is None or score > cached["score"]:
                        export_candidate_videos[candidate_idx] = {
                            "score": score,
                            "mc_idx": int(mc_idx),
                            "video": np.asarray(video).copy(),
                        }

                candidate_scores = evaluate_recoverability(
                    self.world_model,
                    self.module,
                    self.recovery_scorer,
                    start_frame,
                    self.task_description,
                    candidates,
                    horizon_hr=recovery_horizon,
                    num_mc_samples=num_mc_samples,
                    rollout_fn=self._rollout_recovery_candidate,
                    progress_callback=_log_search_progress,
                    artifact_callback=_capture_recovery_video,
                )
                target_baseline_score = _select_target_baseline_score(local_scores, step_idx, frame_idx, traj_idx)
                print(
                    "[recovery] search done "
                    f"traj={traj_idx} slot={slot} scores={np.round(candidate_scores, 4).tolist()} "
                    f"current_score={float(local_scores[step_idx]):.4f} "
                    f"nominal_baseline={target_baseline_score:.4f} "
                    f"elapsed={time.time() - state_timer:.2f}s",
                    flush=True,
                )

                def _export_candidate_debug_video(status_tag, export_reason, best_index, gain=None, confidence=None):
                    nonlocal exported_video_count
                    if not export_videos or exported_video_count >= export_max_videos_per_batch:
                        return
                    cached_video = export_candidate_videos.get(best_index)
                    if cached_video is None:
                        # 极端情况下如果没有缓存到视频，补跑一次当前 best candidate 的局部 WM rollout。
                        # 这个分支只在 debug 导出开启时触发，不影响默认训练开销。
                        recovery_local_video = self._rollout_recovery_candidate(
                            start_frame,
                            self.task_description,
                            candidates[best_index],
                            recovery_horizon,
                        )
                        cached_score = float(candidate_scores[best_index])
                        cached_mc_idx = -1
                    else:
                        recovery_local_video = cached_video["video"]
                        cached_score = float(cached_video["score"])
                        cached_mc_idx = int(cached_video["mc_idx"])
                    metadata = {
                        "candidate_scores": [float(x) for x in np.asarray(candidate_scores).tolist()],
                        "cached_candidate_score": cached_score,
                        "cached_mc_idx": cached_mc_idx,
                        "current_score": float(local_scores[step_idx]),
                        "target_baseline_score": float(target_baseline_score),
                        "previous_step_idx": int(previous_step_idx),
                        "previous_frame_idx": int(previous_frame_idx),
                        "previous_score": float(previous_score),
                        "score_drop": float(score_drop),
                        "complete": bool(traj_complete),
                        "finish_step": int(traj_finish_step),
                        "frames_remaining": int(frames_remaining),
                        "target_baseline_mode": str(target_baseline_mode),
                        "best_index": int(best_index),
                        "best_score": float(candidate_scores[best_index]),
                        "gain": None if gain is None else float(gain),
                        "confidence": None if confidence is None else float(confidence),
                        "num_candidates": int(len(candidates)),
                        "num_mc_samples": int(num_mc_samples),
                        "recovery_horizon": int(recovery_horizon),
                        "export_reason": str(export_reason),
                    }
                    self._save_recovery_video_bundle(
                        export_dir=export_dir,
                        global_steps=global_steps,
                        traj_idx=traj_idx,
                        slot=slot,
                        step_idx=step_idx,
                        frame_idx=frame_idx,
                        original_prefix=videos[traj_idx, : frame_idx + 1],
                        recovery_local=recovery_local_video,
                        original_full=videos[traj_idx],
                        metadata=metadata,
                        fps=export_fps,
                        export_full_original=export_full_original,
                        status_tag=status_tag,
                    )
                    exported_video_count += 1

                target = build_recovery_target(
                    candidate_actions,
                    candidate_scores,
                    baseline_score=target_baseline_score,
                    eps_gain=eps_gain,
                    tau_gain=tau_gain,
                    top_b=top_b,
                    allow_weighted_average=False,
                    candidate_responses=candidate_responses,
                )
                # 没有正增益，或 best 分数仍低于可恢复阈值时，不写入有效监督样本。
                if target is None:
                    skipped_no_gain += 1
                    if export_failed_search_videos and len(candidate_scores) > 0:
                        best_index = int(np.argmax(candidate_scores))
                        _export_candidate_debug_video(
                            status_tag="failed_no_positive_gain",
                            export_reason="no_positive_gain",
                            best_index=best_index,
                            gain=float(candidate_scores[best_index] - target_baseline_score),
                            confidence=0.0,
                        )
                    print(
                        "[recovery] target skip "
                        f"traj={traj_idx} slot={slot} reason=no_positive_gain "
                        f"best={float(np.max(candidate_scores)) if len(candidate_scores) else 0.0:.4f} "
                        f"nominal_baseline={target_baseline_score:.4f} "
                        f"current_score={float(local_scores[step_idx]):.4f}",
                        flush=True,
                    )
                    continue
                if candidate_scores[target["best_index"]] < eps_recoverable:
                    skipped_unrecoverable += 1
                    if export_failed_search_videos:
                        _export_candidate_debug_video(
                            status_tag="failed_below_eps_recoverable",
                            export_reason="below_eps_recoverable",
                            best_index=int(target["best_index"]),
                            gain=float(target["gain"]),
                            confidence=float(target["confidence"]),
                        )
                    print(
                        "[recovery] target skip "
                        f"traj={traj_idx} slot={slot} reason=below_eps_recoverable "
                        f"best={float(candidate_scores[target['best_index']]):.4f} eps={eps_recoverable:.4f}",
                        flush=True,
                    )
                    continue

                # x_t/lang 在当前实现中由 input_ids/attention_mask/pixel_values 共同表示；
                # a_star 是 rec_responses，c_t 是 rec_confidence。
                rec_batch["rec_input_ids"][traj_idx, slot].copy_(vla_history[step_idx]["input_ids"][traj_idx])
                rec_batch["rec_attention_mask"][traj_idx, slot].copy_(vla_history[step_idx]["attention_mask"][traj_idx])
                rec_batch["rec_pixel_values"][traj_idx, slot].copy_(vla_history[step_idx]["pixel_values"][traj_idx])
                rec_batch["rec_responses"][traj_idx, slot].copy_(target["a_star"].to(rec_batch["rec_responses"].device))
                rec_batch["rec_confidence"][traj_idx, slot] = target["confidence"]
                rec_batch["rec_gain"][traj_idx, slot] = target["gain"]
                rec_batch["rec_baseline_score"][traj_idx, slot] = target_baseline_score
                rec_batch["rec_best_score"][traj_idx, slot] = float(candidate_scores[target["best_index"]])
                rec_batch["rec_num_candidates"][traj_idx, slot] = float(len(candidates))
                rec_batch["rec_valid"][traj_idx, slot] = True
                valid_target_count += 1
                _export_candidate_debug_video(
                    status_tag="valid_target",
                    export_reason="valid_target",
                    best_index=int(target["best_index"]),
                    gain=float(target["gain"]),
                    confidence=float(target["confidence"]),
                )
                if len(debug_examples) < 5:
                    debug_examples.append(
                        {
                            "traj": traj_idx,
                            "step_idx": int(step_idx),
                            "frame_idx": int(frame_idx),
                            "baseline": float(target_baseline_score),
                            "best_score": float(candidate_scores[target["best_index"]]),
                            "gain": float(target["gain"]),
                            "confidence": float(target["confidence"]),
                        }
                    )

        print(
            "[recovery] summary "
            f"mined_states={mined_state_count} searched_states={searched_state_count} "
            f"skipped_state_budget={skipped_state_budget} sampled_candidates={sampled_candidate_count} "
            f"skipped_completed_rollouts={skipped_completed_rollouts} "
            f"valid_targets={valid_target_count} skipped_no_gain={skipped_no_gain} "
            f"skipped_unrecoverable={skipped_unrecoverable}",
            flush=True,
        )
        for example in debug_examples:
            print(
                "[recovery] example "
                f"traj={example['traj']} step_idx={example['step_idx']} frame_idx={example['frame_idx']} "
                f"baseline={example['baseline']:.4f} best={example['best_score']:.4f} "
                f"gain={example['gain']:.4f} confidence={example['confidence']:.4f}",
                flush=True,
            )

        return rec_batch

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
            "[recovery] rollout success summary "
            f"batch_size={len(complete_flags)} success_count={success_count} "
            f"failure_count={failure_count} rm_threshold={self.rm_threshold:.4f}",
            flush=True,
        )
        for traj_idx, (complete_flag, finish_step) in enumerate(zip(complete_flags.tolist(), finish_steps.tolist())):
            print(
                "[recovery] rollout status "
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
        # 默认关闭。只有显式 use_recovery_branch=True 的 WM imagined rollout 才会附加 rec_* 字段。
        if self.config.get("use_recovery_branch", False):
            start_time = time.time()
            recovery_batch = self._build_recovery_batch(
                vla_history,
                videos,
                batch_size,
                global_steps=meta_info.get("global_steps", 0),
                task_records=task_records,
            )
            if recovery_batch:
                batch.update(recovery_batch)
                valid_targets = int(recovery_batch["rec_valid"].sum().item())
            else:
                valid_targets = 0
            print(f"Recovery branch generated {valid_targets} targets in {time.time() - start_time:.2f} seconds")
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
            idx = verl_F.pad_sequence_to_length(idx,max_seq_len=self.config.max_prompt_length,pad_token_id=self.processor.tokenizer.pad_token_id,left_pad=True)
            
            assert attention_mask.ndim == 2
            attention_mask = verl_F.pad_sequence_to_length(attention_mask,max_seq_len=self.config.max_prompt_length,pad_token_id=0,left_pad=True)
            
            
            assert idx.device.type == 'cuda'
            assert response.device.type == 'cuda'
            #assert seq.device.type == 'cuda'
            assert attention_mask.device.type == 'cuda'
            assert pixel_values.device.type == 'cuda'
            batch ={
                    'responses': response,
                    'input_ids': idx,
                    'attention_mask': attention_mask,
                    "pixel_values":pixel_values,
                    "action":actions,
                    "normalized_actions": normalized_actions
                }

            return batch
        
        elif self.config.vla == "openvla": 
            idx = prompts['input_ids']  # (bs, prompt_length)
            attention_mask = prompts['attention_mask']  # left-padded attention_mask
            pixel_values = prompts["pixel_values"]
            
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
            
            batch ={
                    'prompts': prompt,
                    'responses': response,
                    'input_ids': seq,
                    'attention_mask': attention_mask,
                    "pixel_values":pixel_values,
                    "action":actions,
                    #'position_ids': position_ids
                }
            
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
