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
Single Process Actor
"""

import itertools
from typing import Iterable, Tuple

import torch
from torch import nn
import torch.nn.functional as F
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import torch.distributed as dist


from verl import DataProto
from verl.trainer.ppo import core_algos
from verl.workers.actor import BasePPOActor
from verl.utils.dccp_schema import PREF_KEYS
from verl.utils.py_functional import append_to_dict
from verl.utils.torch_functional import logprobs_from_logits, log_probs_from_logits_all_rmpad
from verl.utils.seqlen_balancing import rearrange_micro_batches, get_reverse_idx
import verl.utils.torch_functional as verl_F
from codetiming import Timer
from flash_attn.bert_padding import pad_input, unpad_input, rearrange, index_first_axis

__all__ = ['RobDataParallelPPOActor']

DCCP_PREF_ALIASES = {
    "input_ids": (PREF_KEYS.input_ids, "pref_context_input_ids"),
    "attention_mask": (PREF_KEYS.attention_mask, "pref_context_attention_mask"),
    "pixel_values": (PREF_KEYS.pixel_values, "pref_multi_modal_inputs", "multi_modal_inputs"),
    "winner_responses": (PREF_KEYS.winner_responses, "pref_aw_responses", "pref_w_responses"),
    "loser_responses": (PREF_KEYS.loser_responses, "pref_al_responses", "pref_l_responses"),
    "response_mask": (PREF_KEYS.response_mask, "pref_aw_response_mask", "pref_response_masks"),
    "weight": (PREF_KEYS.weight,),
    "delta_ref": (PREF_KEYS.delta_ref,),
    "margin": (PREF_KEYS.margin,),
    "valid": (PREF_KEYS.valid,),
    "winner_input_ids": ("pref_aw_input_ids", "pref_winner_input_ids"),
    "loser_input_ids": ("pref_al_input_ids", "pref_loser_input_ids"),
    "winner_attention_mask": ("pref_aw_attention_mask", "pref_winner_attention_mask"),
    "loser_attention_mask": ("pref_al_attention_mask", "pref_loser_attention_mask"),
}

DCCP_OPTIONAL_KEYS = (
    PREF_KEYS.nominal_score,
    PREF_KEYS.alternative_score,
    PREF_KEYS.entropy,
    PREF_KEYS.curvature,
    PREF_KEYS.state_index,
    PREF_KEYS.candidate_index,
)



class RobDataParallelPPOActor(BasePPOActor):

    def __init__(
        self,
        config,
        actor_module: nn.Module,
        actor_optimizer: torch.optim.Optimizer = None,
    ):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        self.use_remove_padding = self.config.get('use_remove_padding', False)
        print(f'Actor use_remove_padding={self.use_remove_padding}')
        print(f'PRM use dynamic bsz={self.config.get("use_dynamic_bsz", False)}')
        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = False #self.ulysses_sequence_parallel_size > 1
        self.compute_entropy_from_logits = torch.compile(verl_F.entropy_from_logits, dynamic=True)
       
    def process_tensor(self, tensor, pad_id):
        mask = tensor != pad_id
        if not torch.all(mask == mask[0:1], dim=1).all():
            raise ValueError("Padding error!")
        base_mask = mask[0]
        valid_len = base_mask.sum().item()
        return tensor[:, base_mask], valid_len
    
    def generate_traj_mask(self, end_step, traj_len):
        """
        Args:
            end_step: (batch_size,), 
            traj_len: 
        Returns:
            mask: (batch_size, traj_len),
        """
        steps = torch.arange(traj_len, device=end_step.device)  # (traj_len,)
        steps_expanded = steps.unsqueeze(0).expand(end_step.size(0), -1)
        mask = steps_expanded < end_step.unsqueeze(1)  # (batch_size, traj_len)
        return mask
    
    def apply_mask_with_grad_control(self, log_probs, entropy, mask):
        """
        Args:
            log_probs: (batch_size, traj_len, ...)
            entropy:   (batch_size, traj_len, ...)
            mask:      (batch_size, traj_len)
        Returns:
            log_probs_masked: 
            entropy_masked:   
        """
        mask_expanded = mask.unsqueeze(-1)  

        log_probs_masked = torch.where(
            mask_expanded,
            log_probs,
            torch.zeros_like(log_probs, requires_grad=False)  
        )

        entropy_masked = torch.where(
            mask_expanded,
            entropy,
            torch.zeros_like(entropy, requires_grad=False)   
        )

        return log_probs_masked, entropy_masked

    def _forward_micro_batch(self, micro_batch, temperature) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        micro_batch:
        
        Returns: 
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        
        batch_size = micro_batch['responses'].size(0)
        traj_len = micro_batch['responses'].size(1)
        tot_pad_len = micro_batch['input_ids'].size(2)
        
        assert all(micro_batch[key].size(0) == batch_size for key in ['responses', 'input_ids', 'attention_mask', 'pixel_values'])
        assert all(micro_batch[key].size(1) == traj_len for key in ['responses', 'input_ids', 'attention_mask', 'pixel_values'])
        assert all(micro_batch[key].size(2) == tot_pad_len for key in [ 'input_ids', 'attention_mask'])
        
            
        response_length = micro_batch['responses'].size(-1) # 7*8
        
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            input_ids = micro_batch['input_ids']
            attention_mask = micro_batch['attention_mask']
            pixel_values = micro_batch["pixel_values"]
            responses = micro_batch["responses"]
            
            input_ids = input_ids.reshape((batch_size * traj_len,) + input_ids.shape[2:])
            attention_mask = attention_mask.reshape((batch_size * traj_len,) + attention_mask.shape[2:])
            pixel_values = pixel_values.reshape((batch_size * traj_len,) + pixel_values.shape[2:])
            responses = responses.reshape((batch_size * traj_len,) + responses.shape[2:])
            
            input_ids_unpad, _ = self.process_tensor(input_ids, self.pad_token_id)
            attention_mask_unpad, _ = self.process_tensor(attention_mask, 0)
            
            if self.config.vla == "openvla-oft":
                # breakpoint()
                logits = self.actor_module(input_ids=input_ids_unpad,
                                        attention_mask=attention_mask_unpad,
                                        pixel_values=pixel_values,
                                        )  # prevent model thinks we are generating
                
                assert self.actor_module.vocab_size == 32000
                start_index = self.actor_module.vocab_size - 256 
                logits = logits[..., -256-64:-64]  # Shape: [batch_size, seq_len, 256]
                responses = responses - start_index
                #assert (0<=responses<=255).all()
            
                logits = logits.div(temperature) 
                
                log_probs = logprobs_from_logits(logits, responses)
                entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
            
                assert len(log_probs.shape)==2 and len(entropy.shape)==2 
                log_probs = log_probs.reshape((batch_size, traj_len*8,7) )
                entropy = entropy.reshape((batch_size, traj_len*8,7) )

                mask = self.generate_traj_mask(micro_batch['finish_step'], traj_len*8)
                log_probs, entropy = self.apply_mask_with_grad_control(log_probs, entropy, mask)
                
                log_probs = log_probs.reshape((batch_size, traj_len*response_length))
                entropy = entropy.reshape((batch_size, traj_len*response_length)) 
                
            elif self.config.vla == "openvla":
                output = self.actor_module(input_ids=input_ids_unpad,
                                    attention_mask=attention_mask_unpad,
                                    pixel_values=pixel_values,
                                    use_cache=False)  # prevent model thinks we are generating
                logits = output.logits
                
                logits = logits[:, -response_length - 1:-1]  # (bsz, response_length)
                logits = logits.div(temperature) 
                
                log_probs = logprobs_from_logits(logits, responses)
                entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                #ADD
                
                log_probs = log_probs.reshape((batch_size, traj_len,) + log_probs.shape[1:])
                entropy = entropy.reshape((batch_size, traj_len,) + entropy.shape[1:])

                
                mask = self.generate_traj_mask(micro_batch['finish_step'], traj_len)
                log_probs, entropy = self.apply_mask_with_grad_control(log_probs, entropy, mask)
                
                log_probs = log_probs.reshape((batch_size, traj_len*response_length))
                entropy = entropy.reshape((batch_size, traj_len*response_length))
                
                

            return entropy, log_probs
    
    def _forward_micro_batch_update(self, input_ids, attention_mask, pixel_values, responses, temperature) -> Tuple[torch.Tensor, torch.Tensor]:
       
        
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            if self.config.vla == "openvla-oft":
                
                input_ids_unpad, _ = self.process_tensor(input_ids, self.pad_token_id)
                attention_mask_unpad, _ = self.process_tensor(attention_mask, 0)

                
                logits = self.actor_module(input_ids=input_ids_unpad,
                                                attention_mask=attention_mask_unpad,
                                                pixel_values=pixel_values,
                                                )  
                
                assert logits.requires_grad 
                
                assert self.actor_module.vocab_size == 32000
                start_index = self.actor_module.vocab_size - 256 
                logits = logits[..., -256-64:-64]  # Shape: [batch_size, seq_len, 256]
                responses = responses - start_index
                
                logits = logits.div(temperature) 
                
                log_probs = logprobs_from_logits(logits, responses)
                entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                
                log_probs = log_probs.reshape((1, -1))
                entropy = entropy.reshape((1, -1))
                
                return entropy, log_probs
            
            elif self.config.vla == "openvla":
                response_length = responses.size(-1)
                input_ids_unpad, _ = self.process_tensor(input_ids, self.pad_token_id)
                attention_mask_unpad, _ = self.process_tensor(attention_mask, 0)
                output = self.actor_module(input_ids=input_ids_unpad,
                                        attention_mask=attention_mask_unpad,
                                        pixel_values=pixel_values,
                                        use_cache=False)  # prevent model thinks we are generating
                logits = output.logits
                #
                
                logits = logits[:, -response_length - 1:-1]  # (bsz, response_length)
                logits = logits.div(temperature) 
                
                log_probs = logprobs_from_logits(logits, responses)
                entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                
                
                log_probs = log_probs.reshape((1, -1))
                entropy = entropy.reshape((1, -1))

                return entropy, log_probs
                

    def _forward_micro_batch_entropy(self, micro_batch, temperature) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = micro_batch['responses'].size(0)
        traj_len = micro_batch['responses'].size(1)
        tot_pad_len = micro_batch['input_ids'].size(2)
 
        assert all(micro_batch[key].size(0) == batch_size for key in ['responses', 'input_ids', 'attention_mask', 'pixel_values'])
        assert all(micro_batch[key].size(1) == traj_len for key in ['responses', 'input_ids', 'attention_mask', 'pixel_values'])
        assert all(micro_batch[key].size(2) == tot_pad_len for key in [ 'input_ids', 'attention_mask'])
            
        response_length = micro_batch['responses'].size(-1)
        #assert response_length == 7*8
        
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            input_ids = micro_batch['input_ids']
            #batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch['attention_mask']
            pixel_values = micro_batch["pixel_values"]
            
            input_ids = input_ids.reshape((batch_size * traj_len,) + input_ids.shape[2:])
            attention_mask = attention_mask.reshape((batch_size * traj_len,) + attention_mask.shape[2:])
            pixel_values = pixel_values.reshape((batch_size * traj_len,) + pixel_values.shape[2:])
            
            
            input_ids_unpad, _ = self.process_tensor(input_ids, self.pad_token_id)
            attention_mask_unpad, _ = self.process_tensor(attention_mask, 0)

            if  self.config.vla == "openvla-oft":
            
                logits = self.actor_module(input_ids=input_ids_unpad,
                                                attention_mask=attention_mask_unpad,
                                                pixel_values=pixel_values,
                                                ) 
            
                assert self.actor_module.vocab_size == 32000
                start_index = self.actor_module.vocab_size - 256 
                logits = logits[..., -256-64:-64]  # Shape: [batch_size, seq_len, 256]
            
                logits = logits.div(temperature) 
            
                entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)

                assert len(entropy.shape)==2 
                entropy = entropy.reshape((batch_size, traj_len*8,7) )
                mask = self.generate_traj_mask(micro_batch['finish_step'], traj_len*8)
                _, entropy = self.apply_mask_with_grad_control(entropy, entropy, mask)
                entropy = entropy.reshape((batch_size, traj_len*response_length))
                return entropy
            
            elif self.config.vla == "openvla":
                output = self.actor_module(input_ids=input_ids_unpad,
                                        attention_mask=attention_mask_unpad,
                                        pixel_values=pixel_values,
                                        use_cache=False)  # prevent model thinks we are generating
                logits = output.logits
                #
                
                
                logits = logits[:, -response_length - 1:-1]  # (bsz, response_length)
                logits = logits.div(temperature) 
                
                entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                #ADD

                entropy = entropy.reshape((batch_size, traj_len,) + entropy.shape[1:])
                mask = self.generate_traj_mask(micro_batch['finish_step'], traj_len)
                _, entropy = self.apply_mask_with_grad_control(entropy, entropy, mask)
                entropy = entropy.reshape((batch_size, traj_len*response_length))
                return entropy


    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        self.actor_optimizer.step()
        return grad_norm

    def compute_log_prob(self, data: DataProto) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # breakpoint()
        self.actor_module.eval()

        micro_batch_size = data.meta_info['micro_batch_size'] #256
        temperature = data.meta_info['temperature']  # temperature must be in the data.meta_info to avoid slient error # 1
        use_dynamic_bsz = data.meta_info['use_dynamic_bsz'] #trues
        self.pad_token_id = data.meta_info['pad_token_id']
        
        select_keys = ['responses', 'input_ids', 'attention_mask', 'pixel_values',"finish_step"]
        batch = data.select(batch_keys=select_keys).batch

        if use_dynamic_bsz:
            # split using dynamic bsz
            max_token_len = data.meta_info['max_token_len'] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        log_probs_lst = []
        # import time
        # start_time = time.time()
        for batch_idx, micro_batch in  enumerate(micro_batches):
            # current_time = time.time()
            # elapsed_time = current_time - start_time
            # print(f'Rank: {dist.get_rank()} elapsed time: {elapsed_time:.2f} seconds Batch {batch_idx}/{len(micro_batches)}')
            with torch.no_grad():
                _, log_probs = self._forward_micro_batch(micro_batch, temperature=temperature)
            log_probs_lst.append(log_probs)
        log_probs = torch.concat(log_probs_lst, dim=0)

        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            log_probs = log_probs[revert_indices]

        return log_probs

    def _forward_dccp_sequence_logprob(
        self,
        input_ids,
        attention_mask,
        pixel_values,
        responses,
        response_mask,
        temperature,
    ):
        """计算一组 action responses 的 sequence logprob"""
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            input_ids_unpad, _ = self.process_tensor(input_ids, self.pad_token_id)
            attention_mask_unpad, _ = self.process_tensor(attention_mask, 0)

            if self.config.vla == "openvla-oft":
                logits = self.actor_module(
                    input_ids=input_ids_unpad,
                    attention_mask=attention_mask_unpad,
                    pixel_values=pixel_values,
                )

                assert self.actor_module.vocab_size == 32000
                start_index = self.actor_module.vocab_size - 256
                logits = logits[..., -256 - 64:-64]
                target_responses = responses - start_index

            elif self.config.vla == "openvla":
                output = self.actor_module(
                    input_ids=input_ids_unpad,
                    attention_mask=attention_mask_unpad,
                    pixel_values=pixel_values,
                    use_cache=False,
                )
                logits = output.logits
                response_length = responses.size(-1)
                logits = logits[:, -response_length - 1:-1]
                target_responses = responses

            else:
                raise NotImplementedError(f"Unsupported VLA type for DCCP preference loss: {self.config.vla}")

            logits = logits.div(temperature)
            token_logprobs = logprobs_from_logits(logits, target_responses)
            token_logprobs = token_logprobs * response_mask.float()
            return token_logprobs.reshape(token_logprobs.shape[0], -1).sum(dim=-1)

    def _get_dccp_lambda_pref(self, global_steps):
        """计算 DCCP preference loss 的当前权重"""
        dccp_cfg = self.config.get("dccp", {}) or {}
        lambda_pref = float(dccp_cfg.get("lambda_pref", 0.3))
        warmup_steps = int(dccp_cfg.get("lambda_pref_warmup_steps", 0))

        if warmup_steps <= 0:
            return lambda_pref

        progress = min(max(float(global_steps) / float(warmup_steps), 0.0), 1.0)
        return lambda_pref * progress

    def _flatten_pref_tensor(self, tensor, valid_ndim):
        """将 pref tensor 的样本维展开，保留每个样本内部维度"""
        return tensor.reshape((-1,) + tuple(tensor.shape[valid_ndim:]))

    def _get_dccp_cfg_value(self, key, default=None):
        dccp_cfg = self.config.get("dccp", {}) or {}
        return dccp_cfg.get(key, default)

    def _get_pref_tensor(self, data, aliases, default=None):
        for key in aliases:
            if key in data.keys():
                return data[key]
        return default

    def _extract_pref_pixel_values(self, value):
        if value is None:
            return None
        if torch.is_tensor(value):
            return value
        if hasattr(value, "keys") and "pixel_values" in value.keys():
            return value["pixel_values"]
        if isinstance(value, dict):
            return value.get("pixel_values", None)
        return None

    def _zero_dccp_loss(self, data):
        for value in data.values():
            if torch.is_tensor(value):
                return value.new_zeros((), dtype=torch.float32, requires_grad=True)
        return torch.zeros((), dtype=torch.float32, requires_grad=True)

    def _select_valid_pref_tensor(self, tensor, flat_valid, valid_ndim):
        if tensor is None:
            return None
        return self._flatten_pref_tensor(tensor, valid_ndim)[flat_valid]

    def _select_valid_pref_scalar(self, tensor, flat_valid, valid_ndim, default_value, device):
        if tensor is None:
            return torch.full((int(flat_valid.sum().item()),), default_value, dtype=torch.float32, device=device)
        flat_tensor = tensor.reshape(-1) if tensor.ndim == valid_ndim else self._flatten_pref_tensor(tensor, valid_ndim).reshape(flat_valid.numel(), -1)[:, 0]
        return flat_tensor[flat_valid].float()

    def _count_dccp_valid_pairs(self, data):
        pref_valid = self._get_pref_tensor(data, DCCP_PREF_ALIASES["valid"])
        if pref_valid is None:
            return 0
        return int(pref_valid.bool().sum().item())

    @torch.no_grad()
    def compute_dccp_reference_gap(self, data, temperature):
        """Compute frozen reference Δ_ref for packed DCCP preference pairs."""
        pref_valid = self._get_pref_tensor(data, DCCP_PREF_ALIASES["valid"])
        if pref_valid is None:
            return None

        pref_valid = pref_valid.bool()
        valid_ndim = pref_valid.ndim
        flat_valid = pref_valid.reshape(-1)
        output = torch.zeros_like(pref_valid, dtype=torch.float32)

        if not bool(flat_valid.any().item()):
            return output

        pref_input_ids_all = self._get_pref_tensor(data, DCCP_PREF_ALIASES["input_ids"])
        pref_attention_mask_all = self._get_pref_tensor(data, DCCP_PREF_ALIASES["attention_mask"])
        pref_pixel_values_all = self._extract_pref_pixel_values(
            self._get_pref_tensor(data, DCCP_PREF_ALIASES["pixel_values"])
        )
        winner_responses_all = self._get_pref_tensor(data, DCCP_PREF_ALIASES["winner_responses"])
        loser_responses_all = self._get_pref_tensor(data, DCCP_PREF_ALIASES["loser_responses"])
        response_mask_all = self._get_pref_tensor(data, DCCP_PREF_ALIASES["response_mask"])

        pref_input_ids = self._select_valid_pref_tensor(pref_input_ids_all, flat_valid, valid_ndim)
        pref_attention_mask = self._select_valid_pref_tensor(pref_attention_mask_all, flat_valid, valid_ndim)
        pref_pixel_values = self._select_valid_pref_tensor(pref_pixel_values_all, flat_valid, valid_ndim)
        winner_responses = self._select_valid_pref_tensor(winner_responses_all, flat_valid, valid_ndim)
        loser_responses = self._select_valid_pref_tensor(loser_responses_all, flat_valid, valid_ndim)
        response_mask = self._select_valid_pref_tensor(response_mask_all, flat_valid, valid_ndim)

        if response_mask is None and winner_responses is not None:
            response_mask = torch.ones_like(winner_responses, dtype=torch.bool)

        required = (pref_input_ids, pref_attention_mask, pref_pixel_values, winner_responses, loser_responses, response_mask)
        if any(value is None for value in required):
            raise RuntimeError("Cannot compute DCCP reference gap: missing packed pref_* tensors")

        winner_logprob = self._forward_dccp_sequence_logprob(
            input_ids=pref_input_ids,
            attention_mask=pref_attention_mask,
            pixel_values=pref_pixel_values,
            responses=winner_responses,
            response_mask=response_mask,
            temperature=temperature,
        )
        loser_logprob = self._forward_dccp_sequence_logprob(
            input_ids=pref_input_ids,
            attention_mask=pref_attention_mask,
            pixel_values=pref_pixel_values,
            responses=loser_responses,
            response_mask=response_mask,
            temperature=temperature,
        )

        flat_output = output.reshape(-1)
        flat_output[flat_valid] = (winner_logprob - loser_logprob).detach().to(flat_output.device)
        return output

    def _compute_dccp_pref_loss(self, data, temperature, global_steps=0, loss_normalizer=None):
        """计算 DCCP 论文中的 local DPO-style preference objective，不是 recovery BC loss。"""
        if not self.config.get("use_dccp_branch", False):
            return None, {}
        if not bool(self._get_dccp_cfg_value("enable_loss", True)):
            return self._zero_dccp_loss(data), {
                "dccp/valid_pairs": 0.0,
                "dccp/valid_pairs_actor": 0.0,
                "dccp/missing_pref_fields": 0.0,
                "dccp/use_ref_gap": float(bool(self._get_dccp_cfg_value("use_ref_gap", True))),
                "dccp/lambda_pref": float(self._get_dccp_lambda_pref(global_steps)),
                "loss/dccp_pref": 0.0,
            }

        pref_valid = self._get_pref_tensor(data, DCCP_PREF_ALIASES["valid"])
        if pref_valid is None:
            metrics = {
                "dccp/valid_pairs": 0.0,
                "dccp/valid_pairs_actor": 0.0,
                "dccp/pref_weight_mean": 0.0,
                "dccp/pref_weight_mean_actor": 0.0,
                "dccp/margin_mean": 0.0,
                "dccp/margin_mean_actor": 0.0,
                "dccp/delta_ref_mean": 0.0,
                "dccp/delta_ref_mean_actor": 0.0,
                "dccp/delta_theta_mean": 0.0,
                "dccp/delta_theta_mean_actor": 0.0,
                "dccp/missing_pref_fields": 1.0,
                "dccp/use_ref_gap": float(bool(self._get_dccp_cfg_value("use_ref_gap", True))),
                "dccp/lambda_pref": float(self._get_dccp_lambda_pref(global_steps)),
                "loss/dccp_pref": 0.0,
            }
            return self._zero_dccp_loss(data), metrics

        pref_valid = pref_valid.bool()
        valid_ndim = pref_valid.ndim
        flat_valid = pref_valid.reshape(-1)
        num_valid = int(flat_valid.sum().item())

        global_valid = torch.tensor([num_valid], dtype=torch.long, device=flat_valid.device)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(global_valid, op=dist.ReduceOp.MAX)
        max_valid_across_ranks = int(global_valid.item())

        metrics = {
            "dccp/valid_pairs": float(num_valid),
            "dccp/valid_pairs_actor": float(num_valid),
            "dccp/pref_weight_mean": 0.0,
            "dccp/pref_weight_mean_actor": 0.0,
            "dccp/margin_mean": 0.0,
            "dccp/margin_mean_actor": 0.0,
            "dccp/delta_ref_mean": 0.0,
            "dccp/delta_ref_mean_actor": 0.0,
            "dccp/delta_theta_mean": 0.0,
            "dccp/delta_theta_mean_actor": 0.0,
            "dccp/logp_winner_mean": 0.0,
            "dccp/logp_loser_mean": 0.0,
            "dccp/missing_pref_fields": 0.0,
            "dccp/use_ref_gap": float(bool(self._get_dccp_cfg_value("use_ref_gap", True))),
            "dccp/lambda_pref": float(self._get_dccp_lambda_pref(global_steps)),
            "loss/dccp_pref": 0.0,
        }

        if max_valid_across_ranks == 0:
            return self._zero_dccp_loss(data), metrics

        pref_input_ids_all = self._get_pref_tensor(data, DCCP_PREF_ALIASES["input_ids"])
        pref_attention_mask_all = self._get_pref_tensor(data, DCCP_PREF_ALIASES["attention_mask"])
        pref_pixel_values_all = self._extract_pref_pixel_values(
            self._get_pref_tensor(data, DCCP_PREF_ALIASES["pixel_values"])
        )
        winner_responses_all = self._get_pref_tensor(data, DCCP_PREF_ALIASES["winner_responses"])
        loser_responses_all = self._get_pref_tensor(data, DCCP_PREF_ALIASES["loser_responses"])
        response_mask_all = self._get_pref_tensor(data, DCCP_PREF_ALIASES["response_mask"])
        winner_input_ids_all = self._get_pref_tensor(data, DCCP_PREF_ALIASES["winner_input_ids"])
        loser_input_ids_all = self._get_pref_tensor(data, DCCP_PREF_ALIASES["loser_input_ids"])
        winner_attention_mask_all = self._get_pref_tensor(data, DCCP_PREF_ALIASES["winner_attention_mask"])
        loser_attention_mask_all = self._get_pref_tensor(data, DCCP_PREF_ALIASES["loser_attention_mask"])

        if num_valid > 0:
            pref_input_ids = self._select_valid_pref_tensor(pref_input_ids_all, flat_valid, valid_ndim)
            pref_attention_mask = self._select_valid_pref_tensor(pref_attention_mask_all, flat_valid, valid_ndim)
            pref_pixel_values = self._select_valid_pref_tensor(pref_pixel_values_all, flat_valid, valid_ndim)
            winner_responses = self._select_valid_pref_tensor(winner_responses_all, flat_valid, valid_ndim)
            loser_responses = self._select_valid_pref_tensor(loser_responses_all, flat_valid, valid_ndim)
            response_mask = self._select_valid_pref_tensor(response_mask_all, flat_valid, valid_ndim)
            winner_input_ids = self._select_valid_pref_tensor(winner_input_ids_all, flat_valid, valid_ndim)
            loser_input_ids = self._select_valid_pref_tensor(loser_input_ids_all, flat_valid, valid_ndim)
            winner_attention_mask = self._select_valid_pref_tensor(winner_attention_mask_all, flat_valid, valid_ndim)
            loser_attention_mask = self._select_valid_pref_tensor(loser_attention_mask_all, flat_valid, valid_ndim)

            if pref_input_ids is None:
                pref_input_ids = winner_input_ids
            if pref_attention_mask is None:
                pref_attention_mask = winner_attention_mask
            if winner_responses is None and winner_input_ids is not None and response_mask is not None:
                winner_responses = winner_input_ids[..., -response_mask.shape[-1]:]
            if loser_responses is None and loser_input_ids is not None and response_mask is not None:
                loser_responses = loser_input_ids[..., -response_mask.shape[-1]:]
            if response_mask is None and winner_responses is not None:
                response_mask = torch.ones_like(winner_responses, dtype=torch.bool)

            missing_pref_fields = [
                name
                for name, value in (
                    ("input_ids", pref_input_ids),
                    ("attention_mask", pref_attention_mask),
                    ("pixel_values", pref_pixel_values),
                    ("winner_responses", winner_responses),
                    ("loser_responses", loser_responses),
                )
                if value is None
            ]
            if missing_pref_fields:
                metrics["dccp/missing_pref_fields"] = float(len(missing_pref_fields))
                return self._zero_dccp_loss(data), metrics

            weight_tensor = self._get_pref_tensor(data, DCCP_PREF_ALIASES["weight"])
            margin_tensor = self._get_pref_tensor(data, DCCP_PREF_ALIASES["margin"])
            delta_ref_tensor = self._get_pref_tensor(data, DCCP_PREF_ALIASES["delta_ref"])
            pref_margin = self._select_valid_pref_scalar(margin_tensor, flat_valid, valid_ndim, 0.0, flat_valid.device)
            pref_weight = self._select_valid_pref_scalar(weight_tensor, flat_valid, valid_ndim, 1.0, flat_valid.device)
            if weight_tensor is None and margin_tensor is not None:
                pref_weight = pref_margin.abs()
            pref_delta_ref = self._select_valid_pref_scalar(delta_ref_tensor, flat_valid, valid_ndim, 0.0, flat_valid.device)
        else:
            pref_input_ids = data["input_ids"].reshape((-1,) + tuple(data["input_ids"].shape[2:]))[:1]
            pref_attention_mask = data["attention_mask"].reshape((-1,) + tuple(data["attention_mask"].shape[2:]))[:1]
            pref_pixel_values = data["pixel_values"].reshape((-1,) + tuple(data["pixel_values"].shape[2:]))[:1]
            winner_responses = data["responses"].reshape((-1,) + tuple(data["responses"].shape[2:]))[:1]
            loser_responses = winner_responses.clone()
            response_mask = torch.ones_like(winner_responses, dtype=torch.bool, device=winner_responses.device)
            winner_input_ids = None
            loser_input_ids = None
            winner_attention_mask = None
            loser_attention_mask = None

            pref_weight = torch.zeros((1,), dtype=torch.float32, device=flat_valid.device)
            pref_delta_ref = torch.zeros((1,), dtype=torch.float32, device=flat_valid.device)
            pref_margin = torch.zeros((1,), dtype=torch.float32, device=flat_valid.device)

        pref_weight = pref_weight.detach()
        pref_margin = pref_margin.detach()
        pref_delta_ref = pref_delta_ref.detach()
        if not bool(self._get_dccp_cfg_value("use_ref_gap", True)):
            pref_delta_ref = torch.zeros_like(pref_delta_ref)

        winner_logprob = self._forward_dccp_sequence_logprob(
            input_ids=winner_input_ids if winner_input_ids is not None else pref_input_ids,
            attention_mask=winner_attention_mask if winner_attention_mask is not None else pref_attention_mask,
            pixel_values=pref_pixel_values,
            responses=winner_responses,
            response_mask=response_mask,
            temperature=temperature,
        )

        loser_logprob = self._forward_dccp_sequence_logprob(
            input_ids=loser_input_ids if loser_input_ids is not None else pref_input_ids,
            attention_mask=loser_attention_mask if loser_attention_mask is not None else pref_attention_mask,
            pixel_values=pref_pixel_values,
            responses=loser_responses,
            response_mask=response_mask,
            temperature=temperature,
        )

        delta_theta = winner_logprob - loser_logprob
        beta_dpo = float(self._get_dccp_cfg_value("beta", self._get_dccp_cfg_value("beta_dpo", 0.1)))
        logits = beta_dpo * (delta_theta - pref_delta_ref.to(delta_theta.device))

        raw_loss = -F.logsigmoid(logits)
        weight = pref_weight.to(raw_loss.device).float()
        # Paper objective: -E_b [w * log sigmoid(beta * ((logp_w-logp_l)-Delta_ref))].
        # Normalize by the number of valid preference pairs, not by sum(weight).
        if loss_normalizer is None:
            normalizer = max(num_valid, 1)
        else:
            normalizer = max(int(loss_normalizer), 1)
        dccp_pref_loss = (weight * raw_loss).sum() / float(normalizer)

        if num_valid > 0:
            metrics.update(
                {
                    "dccp/pref_weight_mean": pref_weight.mean().detach().item(),
                    "dccp/pref_weight_mean_actor": pref_weight.mean().detach().item(),
                    "dccp/margin_mean": pref_margin.mean().detach().item(),
                    "dccp/margin_mean_actor": pref_margin.mean().detach().item(),
                    "dccp/delta_ref_mean": pref_delta_ref.mean().detach().item(),
                    "dccp/delta_ref_mean_actor": pref_delta_ref.mean().detach().item(),
                    "dccp/delta_theta_mean": delta_theta.detach().mean().item(),
                    "dccp/delta_theta_mean_actor": delta_theta.detach().mean().item(),
                    "dccp/logp_winner_mean": winner_logprob.detach().mean().item(),
                    "dccp/logp_loser_mean": loser_logprob.detach().mean().item(),
                    "loss/dccp_pref": dccp_pref_loss.detach().item(),
                }
            )

        return dccp_pref_loss, metrics

    def update_policy(self, data: DataProto):
        self.actor_module.train()

        assert self.config.ppo_mini_batch_size % self.config.ppo_micro_batch_size == 0
        self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size
        temperature = data.meta_info['temperature']  # temperature must be in the data.meta_info to avoid slient error
        self.pad_token_id = data.meta_info.get('pad_token_id', getattr(self, 'pad_token_id', None))
        global_steps = int(data.meta_info.get('global_steps', 0))

        select_keys = ['responses', 'input_ids', 'attention_mask', 'pixel_values', 'old_log_probs', 'advantages', "finish_step"]

        if self.config.get("use_dccp_branch", False):
            dccp_keys = sorted(set(itertools.chain.from_iterable(DCCP_PREF_ALIASES.values())) | set(DCCP_OPTIONAL_KEYS))
            select_keys = select_keys + [key for key in dccp_keys if key in data.batch.keys()]
        batch = data.select(batch_keys=select_keys).batch
        assert self.config.ppo_micro_batch_size == 1

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        dataloader = batch.split(self.config.ppo_mini_batch_size)
        metrics = {}
        for batch_idx, data in enumerate(dataloader):
            # split batch into micro_batches
            mini_batch = data
            dccp_pair_normalizer = 0
            if self.config.get("use_dccp_branch", False):
                dccp_pair_normalizer = self._count_dccp_valid_pairs(mini_batch)
            if self.config.use_dynamic_bsz:
                max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
            else:
                # split batch into micro_batches
                micro_batches = mini_batch.split(self.config.ppo_micro_batch_size)

            self.actor_optimizer.zero_grad()

            for test_idx, data in enumerate(micro_batches):
                data = data.cuda()  # actor device is cpu when using offload
                responses = data['responses']
                
                response_length = responses.size(1) *  responses.size(2)
                finish_step = data['finish_step'] * self.config.action_token_len
                steps = torch.arange(response_length, device=data['responses'].device)  # (traj_len,)
                steps_expanded = steps.unsqueeze(0).expand(data['responses'].size(0), -1)
                response_mask = steps_expanded < finish_step.unsqueeze(1)  # (batch_size, traj_len)
                
                response_mask_sum = response_mask.sum(axis=None)

                old_log_prob = data['old_log_probs']
                advantages = data['advantages']
                
                #clip_ratio = self.config.clip_ratio
                clip_ratio_high = self.config.clip_ratio_high
                clip_ratio_low = self.config.clip_ratio_low
                entropy_coeff = self.config.entropy_coeff

                batch_size = data['responses'].size(0)
                traj_len = data['responses'].size(1)
                tot_pad_len = data['input_ids'].size(2)
                
                
                input_ids = data['input_ids']
                attention_mask = data['attention_mask']
                pixel_values = data["pixel_values"]
                responses = data["responses"]
                
                
                input_ids = input_ids.reshape((batch_size * traj_len,) + input_ids.shape[2:])
                attention_mask = attention_mask.reshape((batch_size * traj_len,) + attention_mask.shape[2:])
                pixel_values = pixel_values.reshape((batch_size * traj_len,) + pixel_values.shape[2:])
                responses = responses.reshape((batch_size * traj_len,) + responses.shape[2:])
                
                loss_info = {
                    #'actor/entropy_loss': entropy_loss.detach().item(),
                    'actor/pg_loss':0,
                    'actor/pg_clipfrac': 0,
                    'actor/ppo_kl': 0,
                }
                
                assert traj_len % self.config.traj_mini_batch_size ==0
                traj_split_num = int(traj_len/self.config.traj_mini_batch_size)
                
                
    

                for i in range(0, traj_len, int(traj_len/traj_split_num)):
                    entropy, log_prob = self._forward_micro_batch_update(input_ids=input_ids[i:i+int(traj_len/traj_split_num)], attention_mask=attention_mask[i:i+int(traj_len/traj_split_num)], pixel_values=pixel_values[i:i+int(traj_len/traj_split_num)], responses=responses[i:i+int(traj_len/traj_split_num)], temperature=temperature)
                    
                    slice_id = i*self.config.action_token_len*self.config.action_chunks_len
                    next_slice_id = (i+int(traj_len/traj_split_num))*self.config.action_token_len*self.config.action_chunks_len
                    old_log_prob_tmp = old_log_prob[:, slice_id: next_slice_id]
                    advantages_tmp = advantages[:, slice_id: next_slice_id]
                    response_mask_tmp = response_mask[:, slice_id: next_slice_id]
                        
                    pg_loss, pg_clipfrac, ppo_kl = core_algos.compute_policy_loss(old_log_prob=old_log_prob_tmp,
                                                                            log_prob=log_prob,
                                                                            advantages=advantages_tmp,
                                                                            eos_mask=response_mask_tmp,
                                                                            clip_ratio_high=clip_ratio_high,
                                                                            clip_ratio_low=clip_ratio_low)
                    
                    response_mask_tmp_sum = response_mask_tmp.sum(axis=None)
                    pg_loss = pg_loss* response_mask_tmp_sum
                    pg_clipfrac = pg_clipfrac* response_mask_tmp_sum / response_mask_sum
                    ppo_kl = ppo_kl* response_mask_tmp_sum / response_mask_sum
                    
                    policy_loss = pg_loss / response_mask_sum
                    
                    loss = policy_loss / self.gradient_accumulation
                    
                    loss.backward()
                    
                    loss_info['actor/pg_loss'] =  loss_info['actor/pg_loss'] + policy_loss.detach().item()
                    loss_info['actor/pg_clipfrac'] = loss_info['actor/pg_clipfrac'] + pg_clipfrac.detach().item()
                    loss_info['actor/ppo_kl'] = loss_info['actor/ppo_kl'] +  ppo_kl.detach().item()

                dccp_pref_loss, dccp_info = self._compute_dccp_pref_loss(
                    data=data,
                    temperature=temperature,
                    global_steps=global_steps,
                    loss_normalizer=dccp_pair_normalizer,
                )
                if dccp_info:
                    for key, value in dccp_info.items():
                        loss_info[key] = loss_info.get(key, 0) + value
                if dccp_pref_loss is not None:
                    lambda_pref = self._get_dccp_lambda_pref(global_steps)
                    # This extra backward is algebraically the same objective as
                    # original_actor_loss + lambda_pref * L_pref, while preserving
                    # the existing trajectory-split PPO/GRPO backward path.
                    dccp_backward_scale = 1.0 if dccp_pair_normalizer > 0 else 1.0 / self.gradient_accumulation
                    (lambda_pref * dccp_pref_loss * dccp_backward_scale).backward()
                    loss_info["loss/total"] = loss_info.get("actor/pg_loss", 0) + lambda_pref * dccp_pref_loss.detach().item()

                append_to_dict(metrics, loss_info)
               
            grad_norm = self._optimizer_step()
            data = {'actor/grad_norm': grad_norm.detach().item()}
            append_to_dict(metrics, data)
            torch.cuda.empty_cache()
        self.actor_optimizer.zero_grad()
        torch.cuda.synchronize()
        torch.distributed.barrier()
        torch.cuda.empty_cache()
        return metrics

    
    def compute_entropy(self, bacth_data: DataProto):
        
        if bacth_data.meta_info['train_mode'] ==True:
            self.actor_module.train()
            print("train mode")
        else:
            self.actor_module.eval()
            print("eval mode")

        assert self.config.ppo_mini_batch_size % self.config.ppo_micro_batch_size == 0
        self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size
        temperature = bacth_data.meta_info['temperature']  # temperature must be in the data.meta_info to avoid slient error

        select_keys = ['responses', 'input_ids', 'attention_mask', 'pixel_values', "finish_step"]
        batch = bacth_data.select(batch_keys=select_keys).batch

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        dataloader = batch.split(self.config.ppo_mini_batch_size)
        print("dataloader_length:", len(dataloader))
        
        metrics = {}
        for batch_idx, data in enumerate(dataloader):
            # split batch into micro_batches
            mini_batch = data
            if self.config.use_dynamic_bsz:
                max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
            else:
                # split batch into micro_batches
                micro_batches = mini_batch.split(self.config.ppo_micro_batch_size)

            for data in micro_batches:
                data = data.cuda()  # actor device is cpu when using offload
                responses = data['responses']
                response_length = responses.size(1) *  responses.size(2)
                finish_step = data['finish_step'] * self.config.action_token_len
                steps = torch.arange(response_length, device=data['responses'].device)  # (traj_len,)
                steps_expanded = steps.unsqueeze(0).expand(data['responses'].size(0), -1)
                response_mask = steps_expanded < finish_step.unsqueeze(1)  # (batch_size, traj_len)
                

                with torch.no_grad():
                    entropy = self._forward_micro_batch_entropy(micro_batch=data, temperature=temperature)
                    entropy_loss = verl_F.masked_mean(entropy, response_mask)

                if bacth_data.meta_info['is_filtered'] and bacth_data.meta_info['train_mode']:
                    data = {
                        'actor_after/entropy_loss_train': entropy_loss.detach().item(),
                    }
                    append_to_dict(metrics, data)
                elif bacth_data.meta_info['is_filtered'] and not bacth_data.meta_info['train_mode']:
                    data = {
                        'actor_after/entropy_loss_eval': entropy_loss.detach().item(),
                    }
                    append_to_dict(metrics, data)
                elif not bacth_data.meta_info['is_filtered'] and bacth_data.meta_info['train_mode']:
                    data = {
                        'actor_before/entropy_loss_train': entropy_loss.detach().item(),
                    }
                    append_to_dict(metrics, data)
                elif not bacth_data.meta_info['is_filtered'] and not bacth_data.meta_info['train_mode']:
                    data = {
                        'actor_before/entropy_loss_eval': entropy_loss.detach().item(),
                    }
                    append_to_dict(metrics, data)
                        
                
        torch.cuda.synchronize()
        torch.distributed.barrier()
        torch.cuda.empty_cache()
        return metrics
