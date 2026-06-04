"""
DCCP scorer.

本文件封装completion scoring 和 progress scoring。

completion scoring:
    对完整 imagined trajectory 调用 S_comp，用于轨迹级 GRPO 的二值完成奖励。

progress scoring:
    对 short suffix 或 counterfactual branch 调用 S_prog，
    用于 decision-sensitive state mining 和 high-margin preference construction。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from verl.utils.dccp_frame_extractor import extract_completion_frames, extract_progress_frames
from verl.utils.dccp_lrm_client import DCCPLRMClient


@dataclass
class DCCPScorerConfig:
    completion_num_keyframes: int = 8
    progress_num_keyframes: int = 4
    image_size: int = 224


class DCCPScorer:
    """DCCP completion/progress scorer."""

    def __init__(self, client: DCCPLRMClient, config: Optional[DCCPScorerConfig] = None):
        self.client = client
        self.config = config or DCCPScorerConfig()

        self.last_completion_latency_sec = 0.0
        self.last_progress_latency_sec = 0.0

    @classmethod
    def from_config_dict(cls, cfg: dict[str, Any]) -> "DCCPScorer":
        scorer_cfg = cfg.get("scorer", {})
        lrm_input_cfg = cfg.get("lrm_input", {})

        client = DCCPLRMClient.from_config_dict(scorer_cfg)

        config = DCCPScorerConfig(
            completion_num_keyframes=int(lrm_input_cfg.get("completion_num_keyframes", 8)),
            progress_num_keyframes=int(lrm_input_cfg.get("progress_num_keyframes", 4)),
            image_size=int(lrm_input_cfg.get("image_size", 224)),
        )

        return cls(client=client, config=config)

    def score_trajectory_completion(
        self,
        rollout_video: Any,
        instruction: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> int:
        """对完整 imagined trajectory 计算 completion label"""
        frames = extract_completion_frames(
            rollout_video,
            num_keyframes=self.config.completion_num_keyframes,
            image_size=self.config.image_size,
        )

        start_time = time.time()
        score = self.client.score_completion(
            frames=frames,
            instruction=instruction,
            metadata=metadata,
        )
        self.last_completion_latency_sec = time.time() - start_time

        return int(score)

    def score_local_progress(
        self,
        branch_or_suffix_video: Any,
        instruction: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> float:
        """对 short suffix 或 counterfactual branch 计算 progress score"""
        frames = extract_progress_frames(
            branch_or_suffix_video,
            num_keyframes=self.config.progress_num_keyframes,
            image_size=self.config.image_size,
        )

        start_time = time.time()
        score = self.client.score_progress(
            frames=frames,
            instruction=instruction,
            metadata=metadata,
        )
        self.last_progress_latency_sec = time.time() - start_time

        return float(np.clip(score, 0.0, 1.0))

    def score_trajectory_completion_batch(
        self,
        rollout_videos: list[Any],
        instructions: list[str],
        batch_metadata: Optional[list[dict[str, Any]]] = None,
    ) -> list[int]:
        """批量计算 completion labels"""
        frames_batch = [
            extract_completion_frames(
                video,
                num_keyframes=self.config.completion_num_keyframes,
                image_size=self.config.image_size,
            )
            for video in rollout_videos
        ]

        start_time = time.time()
        scores = self.client.score_completion_batch(
            batch_frames=frames_batch,
            batch_instructions=instructions,
            batch_metadata=batch_metadata,
        )
        self.last_completion_latency_sec = time.time() - start_time

        return [int(score) for score in scores]

    def score_local_progress_batch(
        self,
        branch_or_suffix_videos: list[Any],
        instructions: list[str],
        batch_metadata: Optional[list[dict[str, Any]]] = None,
    ) -> np.ndarray:
        """批量计算 progress scores"""
        frames_batch = [
            extract_progress_frames(
                video,
                num_keyframes=self.config.progress_num_keyframes,
                image_size=self.config.image_size,
            )
            for video in branch_or_suffix_videos
        ]

        start_time = time.time()
        scores = self.client.score_progress_batch(
            batch_frames=frames_batch,
            batch_instructions=instructions,
            batch_metadata=batch_metadata,
        )
        self.last_progress_latency_sec = time.time() - start_time

        return np.clip(np.asarray(scores, dtype=np.float32), 0.0, 1.0)

    def metrics(self) -> dict[str, float]:
        """返回 LRM scoring 相关日志指标"""
        return {
            "dccp/lrm_completion_latency": float(self.last_completion_latency_sec),
            "dccp/lrm_progress_latency": float(self.last_progress_latency_sec),
            "dccp/lrm_cache_hit_rate": float(self.client.cache_hit_rate()),
        }