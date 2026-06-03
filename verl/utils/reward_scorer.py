"""奖励模型打分封装。

这个模块现在同时支持：
1. `R_traj`：完整 imagined rollout / terminal-success 判定；
2. `R_loc`：near-failure mining 与 short recovery branch ranking 的局部 progress 打分。

命名约定：
- `score_traj_*` 一律对应 `R_traj`
- `score_loc_*` 一律对应 `R_loc`
- `score_full_trajectory` 只保留给 `R_traj` 的兼容入口，不再默认走 `R_loc`
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import torch


class RewardScorer:
    """共享 encoder 的 reward scorer。

    - `score_traj_*`：返回轨迹成功概率，默认用于 complete / finish_step 判断；
    - `score_loc_*`：返回局部 progress 概率，默认用于 near-failure mining 和 recovery ranking。
    """

    def __init__(
        self,
        model,
        feature_extractor,
        threshold: float = 0.5,
        device: Optional[torch.device] = None,
        batch_size: int = 128,
        clip_len: int = 8,
    ) -> None:
        self.model = model
        self.feature_extractor = feature_extractor
        self.threshold = threshold
        self.device = device
        self.batch_size = batch_size
        self.clip_len = clip_len
        self.use_multi_resolution_reward = bool(getattr(model, "use_multi_resolution_reward", False))

    def _device(self):
        # 优先使用显式传入的 device；否则从模型参数推断，避免在 rollout worker 中写死 cuda id。
        if self.device is not None:
            return self.device
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _pad_clip(self, clip: np.ndarray) -> np.ndarray:
        """把任意长度局部片段整理成 reward model 期望的固定长度。

        recovery mining/search 可能从 near-failure 边界截到不足 8 帧的短片段。
        这里用最后一帧补齐，保持输入稳定，同时避免把 padding 逻辑散落在调用方。
        """
        if len(clip) == 0:
            raise ValueError("Cannot score an empty clip segment.")
        if len(clip) >= self.clip_len:
            return clip[: self.clip_len]
        pad = np.repeat(clip[-1: ], self.clip_len - len(clip), axis=0)
        return np.concatenate([clip, pad], axis=0)

    def _extract_probs(self, logits: torch.Tensor, head: str) -> torch.Tensor:
        if logits.ndim == 1:
            return torch.sigmoid(logits)
        if logits.shape[-1] == 1:
            return torch.sigmoid(logits.squeeze(-1))
        if head == "loc" and logits.shape[-1] >= 1:
            return torch.sigmoid(logits[..., -1])
        return torch.sigmoid(logits)[:, 1]

    @torch.no_grad()
    def _score_clip_segments(self, clips: Iterable[np.ndarray], head: str) -> np.ndarray:
        clips = [self._pad_clip(np.asarray(clip)) for clip in clips]
        if not clips:
            return np.zeros((0,), dtype=np.float32)

        self.model.eval()
        scores = []
        device = self._device()
        for start in range(0, len(clips), self.batch_size):
            # feature_extractor 接收的是 list[list[frame]]，保持与原 predict_success 里的调用一致。
            batch = clips[start : start + self.batch_size]
            clip_imgs = [[img for img in clip] for clip in batch]
            inputs = self.feature_extractor(clip_imgs, return_tensors="pt")["pixel_values"].to(device)
            if self.use_multi_resolution_reward:
                logits = self.model(pixel_values=inputs, head=head).logits
            else:
                logits = self.model(pixel_values=inputs).logits
            probs = self._extract_probs(logits, head=head)
            scores.append(probs.detach().float().cpu())
        return torch.cat(scores, dim=0).numpy()

    @torch.no_grad()
    def score_traj_clip_segments(self, clips: Iterable[np.ndarray]) -> np.ndarray:
        return self._score_clip_segments(clips, head="traj")

    @torch.no_grad()
    def score_loc_clip_segments(self, clips: Iterable[np.ndarray]) -> np.ndarray:
        if not self.use_multi_resolution_reward:
            return self._score_clip_segments(clips, head="traj")
        return self._score_clip_segments(clips, head="loc")

    @torch.no_grad()
    def score_clip_segment(self, clip: np.ndarray) -> float:
        clips = self.score_clip_segments([clip])
        return float(clips[0])

    @torch.no_grad()
    def score_clip_segments(self, clips: Iterable[np.ndarray]) -> np.ndarray:
        return self.score_loc_clip_segments(clips)

    @torch.no_grad()
    def _score_full_trajectory(
        self,
        video: np.ndarray,
        head: str,
        window_size: Optional[int] = None,
        stride: int = 1,
        aggregate: str = "max",
    ) -> float:
        video = np.asarray(video)
        if len(video) == 0:
            return 0.0
        window_size = window_size or self.clip_len
        clips = []
        if len(video) <= window_size:
            clips.append(video)
        else:
            for start in range(0, len(video) - window_size + 1, max(stride, 1)):
                clips.append(video[start : start + window_size])
        scores = self._score_clip_segments(clips, head=head)
        if len(scores) == 0:
            return 0.0
        if aggregate == "last":
            return float(scores[-1])
        if aggregate == "mean":
            return float(scores.mean())
        return float(scores.max())

    @torch.no_grad()
    def score_traj_full_trajectory(self, video: np.ndarray, window_size: Optional[int] = None, stride: int = 1) -> float:
        return self._score_full_trajectory(video, head="traj", window_size=window_size, stride=stride, aggregate="max")

    @torch.no_grad()
    def score_loc_full_trajectory(self, video: np.ndarray, window_size: Optional[int] = None, stride: int = 1) -> float:
        if not self.use_multi_resolution_reward:
            return self._score_full_trajectory(video, head="traj", window_size=window_size, stride=stride, aggregate="max")
        return self._score_full_trajectory(video, head="loc", window_size=window_size, stride=stride, aggregate="last")

    @torch.no_grad()
    def score_full_trajectory(self, video: np.ndarray, window_size: Optional[int] = None, stride: int = 1) -> float:
        return self.score_traj_full_trajectory(video, window_size=window_size, stride=stride)
