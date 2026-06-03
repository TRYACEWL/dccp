from __future__ import annotations

from types import SimpleNamespace
from typing import Optional

import torch
import torch.nn as nn


class MultiResolutionVideoRewardModel(nn.Module):
    """共享 VideoMAE encoder 的双头奖励模型。

    设计目标：
    1. `traj_head` 保持原有 terminal-success / clip-success 二分类语义；
    2. `loc_head` 输出局部 progress / viability 的单标量 logit；
    3. 尽量复用现有 VideoMAE 分类器的 encoder、fc_norm 和初始化权重；
    4. 兼容从旧单头 `VideoMAEForVideoClassification` checkpoint 加载。
    """

    def __init__(
        self,
        videomae: nn.Module,
        fc_norm: Optional[nn.Module],
        traj_head: nn.Module,
        loc_head: Optional[nn.Module] = None,
        use_multi_resolution_reward: bool = True,
    ) -> None:
        super().__init__()
        self.videomae = videomae
        self.fc_norm = fc_norm
        self.traj_head = traj_head
        self.use_multi_resolution_reward = bool(use_multi_resolution_reward)

        if loc_head is None:
            if not isinstance(traj_head, nn.Linear):
                raise TypeError("Default loc_head initialization requires traj_head to be nn.Linear.")
            loc_head = nn.Linear(traj_head.in_features, 1)
            with torch.no_grad():
                # 用旧成功头的“成功类”参数初始化局部 progress 头，保证冷启动更平滑。
                success_row = 1 if traj_head.out_features > 1 else 0
                loc_head.weight.copy_(traj_head.weight[success_row : success_row + 1])
                if traj_head.bias is not None:
                    loc_head.bias.copy_(traj_head.bias[success_row : success_row + 1])
        self.loc_head = loc_head

    @classmethod
    def from_videomae_classifier(cls, base_model, use_multi_resolution_reward: bool = True):
        return cls(
            videomae=base_model.videomae,
            fc_norm=base_model.fc_norm,
            traj_head=base_model.classifier,
            loc_head=None,
            use_multi_resolution_reward=use_multi_resolution_reward,
        )

    def encode_features(
        self,
        pixel_values: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> torch.Tensor:
        outputs = self.videomae(
            pixel_values,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        if self.fc_norm is not None:
            return self.fc_norm(sequence_output.mean(1))
        return sequence_output[:, 0]

    def score_traj_logits(self, pixel_values: torch.Tensor, **kwargs) -> torch.Tensor:
        pooled = self.encode_features(pixel_values, **kwargs)
        return self.traj_head(pooled)

    def score_loc_logits(self, pixel_values: torch.Tensor, **kwargs) -> torch.Tensor:
        pooled = self.encode_features(pixel_values, **kwargs)
        return self.loc_head(pooled).squeeze(-1)

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        head: str = "traj",
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: bool = True,
        **kwargs,
    ):
        pooled = self.encode_features(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        traj_logits = self.traj_head(pooled)
        loc_logits = self.loc_head(pooled).squeeze(-1)

        if head == "traj" or not self.use_multi_resolution_reward:
            logits = traj_logits
        elif head == "loc":
            logits = loc_logits
        elif head == "both":
            logits = traj_logits
        else:
            raise ValueError(f"Unsupported reward head: {head}")

        if not return_dict:
            return logits, traj_logits, loc_logits, pooled
        return SimpleNamespace(
            logits=logits,
            traj_logits=traj_logits,
            loc_logits=loc_logits,
            pooled_output=pooled,
        )

    def load_reward_state_dict(self, state_dict, strict: bool = False):
        """兼容旧单头 VideoMAE checkpoint。"""
        mapped = dict(state_dict)

        if "classifier.weight" in mapped and "traj_head.weight" not in mapped:
            mapped["traj_head.weight"] = mapped["classifier.weight"]
        if "classifier.bias" in mapped and "traj_head.bias" not in mapped:
            mapped["traj_head.bias"] = mapped["classifier.bias"]

        if "loc_head.weight" not in mapped:
            if "classifier.weight" in mapped:
                success_row = 1 if mapped["classifier.weight"].shape[0] > 1 else 0
                mapped["loc_head.weight"] = mapped["classifier.weight"][success_row : success_row + 1].clone()
            elif "traj_head.weight" in mapped:
                success_row = 1 if mapped["traj_head.weight"].shape[0] > 1 else 0
                mapped["loc_head.weight"] = mapped["traj_head.weight"][success_row : success_row + 1].clone()
        if "loc_head.bias" not in mapped:
            if "classifier.bias" in mapped:
                success_row = 1 if mapped["classifier.bias"].shape[0] > 1 else 0
                mapped["loc_head.bias"] = mapped["classifier.bias"][success_row : success_row + 1].clone()
            elif "traj_head.bias" in mapped:
                success_row = 1 if mapped["traj_head.bias"].shape[0] > 1 else 0
                mapped["loc_head.bias"] = mapped["traj_head.bias"][success_row : success_row + 1].clone()

        mapped.pop("classifier.weight", None)
        mapped.pop("classifier.bias", None)
        return self.load_state_dict(mapped, strict=strict)

    @staticmethod
    def is_dual_head_checkpoint(state_dict) -> bool:
        return "loc_head.weight" in state_dict and "loc_head.bias" in state_dict
