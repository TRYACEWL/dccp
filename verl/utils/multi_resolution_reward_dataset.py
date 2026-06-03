from __future__ import annotations

import io
import json
import random
from typing import Iterable, List, Tuple

import numpy as np
import webdataset as wds
from PIL import Image
from torch.utils.data import IterableDataset
from transformers import VideoMAEFeatureExtractor


class ProgressWindowDataset(IterableDataset):
    """为多尺度 reward 训练采样固定长度 clip。

    输出字段分三类：
    1. `traj_clip` / `traj_label`: 保持原 terminal-success 训练路径；
    2. `rank_*`: 成功轨迹上的时间排序样本；
    3. `fail_tail_*`: 失败轨迹尾段负样本；
    4. `smooth_*`: 可选平滑正则相邻片段。
    """

    def __init__(
        self,
        shards: List[str],
        clip_len: int = 8,
        stride: int = 8,
        img_size: int = 224,
        rank_num_pairs: int = 2,
        failed_tail_k: int = 2,
        progress_curve_bins: int = 5,
        mode: str = "train",
        use_resample: bool = False,
    ) -> None:
        super().__init__()
        assert mode == "train", "ProgressWindowDataset is only used for reward training."
        self.clip_len = int(clip_len)
        self.stride = int(stride)
        self.rank_num_pairs = max(int(rank_num_pairs), 1)
        self.failed_tail_k = max(int(failed_tail_k), 1)
        self.progress_curve_bins = max(int(progress_curve_bins), 2)
        self.fe = VideoMAEFeatureExtractor(size=img_size)
        self.pipeline = wds.DataPipeline(
            wds.SimpleShardList(shards) if not use_resample else wds.ResampledShards(shards, seed=42),
            wds.split_by_node,
            wds.split_by_worker,
            wds.tarfile_to_samples(handler=wds.warn_and_continue),
            wds.to_tuple("video.npy", "meta.json"),
            self._samples,
            wds.shuffle(500, initial=500),
        )

    def __iter__(self):
        return iter(self.pipeline)

    def _tensorize(self, clip: np.ndarray):
        frames = [Image.fromarray(f.astype(np.uint8)) for f in clip]
        return self.fe(frames, return_tensors="pt")["pixel_values"][0]

    def _safe_clip(self, video: np.ndarray, end: int) -> np.ndarray:
        end = max(int(end), self.clip_len)
        end = min(end, len(video))
        start = max(0, end - self.clip_len)
        clip = video[start:end]
        if len(clip) < self.clip_len:
            pad = np.repeat(clip[-1:], self.clip_len - len(clip), axis=0)
            clip = np.concatenate([clip, pad], axis=0)
        return clip

    def _valid_ends(self, horizon_end: int) -> List[int]:
        horizon_end = max(int(horizon_end), self.clip_len)
        return list(range(self.clip_len, horizon_end + 1, self.stride))

    def _sample_rank_pairs(self, ends: List[int]) -> Tuple[List[Tuple[int, int]], int]:
        if len(ends) < 2:
            fallback = ends[-1] if ends else self.clip_len
            return [(fallback, fallback)] * self.rank_num_pairs, 0
        pairs = []
        for _ in range(self.rank_num_pairs):
            i, j = sorted(random.sample(range(len(ends)), 2))
            pairs.append((ends[i], ends[j]))
        return pairs, 1

    def _sample_fail_tail(self, ends: List[int]) -> Tuple[List[int], int]:
        if not ends:
            return [self.clip_len] * self.failed_tail_k, 0
        tail = ends[-self.failed_tail_k :]
        if len(tail) < self.failed_tail_k:
            tail = [tail[0]] * (self.failed_tail_k - len(tail)) + tail
        return tail, 1

    def _sample_smooth_pair(self, ends: List[int]) -> Tuple[Tuple[int, int], int]:
        if len(ends) < 2:
            fallback = ends[-1] if ends else self.clip_len
            return (fallback, fallback), 0
        idx = random.randint(0, len(ends) - 2)
        return (ends[idx], ends[idx + 1]), 1

    def _sample_progress_curve(self, ends: List[int], complete: bool, terminal_end: int):
        if (not complete) or len(ends) == 0:
            return [terminal_end] * self.progress_curve_bins, 0
        sample_indices = np.linspace(0, len(ends) - 1, num=self.progress_curve_bins).astype(np.int64)
        return [ends[int(idx)] for idx in sample_indices.tolist()], 1

    def _samples(self, stream: Iterable[Tuple[bytes, bytes]]):
        for v_bytes, m_bytes in stream:
            video = np.load(io.BytesIO(v_bytes))
            meta = json.loads(m_bytes.decode())
            complete = bool(meta.get("complete", True))
            finish_step = min(max(int(meta.get("finish_step", len(video))), self.clip_len), len(video))
            terminal_end = finish_step
            terminal_clip = self._safe_clip(video, terminal_end)
            ends = self._valid_ends(terminal_end)

            rank_pairs, rank_valid = self._sample_rank_pairs(ends) if complete else ([(terminal_end, terminal_end)] * self.rank_num_pairs, 0)
            fail_tail_ends, fail_valid = self._sample_fail_tail(ends) if not complete else ([terminal_end] * self.failed_tail_k, 0)
            smooth_pair, smooth_valid = self._sample_smooth_pair(ends)
            progress_curve_ends, progress_curve_valid = self._sample_progress_curve(ends, complete, terminal_end)

            yield {
                "traj_clip": self._tensorize(terminal_clip),
                "traj_label": int(complete),
                "anchor_clip": self._tensorize(terminal_clip),
                "anchor_label": float(complete),
                "rank_early_clips": np.stack([self._tensorize(self._safe_clip(video, early)).numpy() for early, _ in rank_pairs], axis=0),
                "rank_late_clips": np.stack([self._tensorize(self._safe_clip(video, late)).numpy() for _, late in rank_pairs], axis=0),
                "rank_valid": int(rank_valid),
                "fail_tail_clips": np.stack([self._tensorize(self._safe_clip(video, end)).numpy() for end in fail_tail_ends], axis=0),
                "fail_valid": int(fail_valid),
                "smooth_prev_clip": self._tensorize(self._safe_clip(video, smooth_pair[0])),
                "smooth_next_clip": self._tensorize(self._safe_clip(video, smooth_pair[1])),
                "smooth_valid": int(smooth_valid),
                "progress_curve_clips": np.stack([self._tensorize(self._safe_clip(video, end)).numpy() for end in progress_curve_ends], axis=0),
                "progress_curve_valid": int(progress_curve_valid),
            }


class MixedProgressWindowDataset(IterableDataset):
    def __init__(
        self,
        shards_pattern1: str,
        shards_pattern2: str,
        weights: Tuple[float, float] = (0.7, 0.3),
        clip_len: int = 8,
        stride: int = 8,
        img_size: int = 224,
        rank_num_pairs: int = 2,
        failed_tail_k: int = 2,
        progress_curve_bins: int = 5,
    ):
        super().__init__()
        ds1 = ProgressWindowDataset(
            shards_pattern1,
            clip_len=clip_len,
            stride=stride,
            img_size=img_size,
            rank_num_pairs=rank_num_pairs,
            failed_tail_k=failed_tail_k,
            progress_curve_bins=progress_curve_bins,
            use_resample=False,
        )
        ds2 = ProgressWindowDataset(
            shards_pattern2,
            clip_len=clip_len,
            stride=stride,
            img_size=img_size,
            rank_num_pairs=rank_num_pairs,
            failed_tail_k=failed_tail_k,
            progress_curve_bins=progress_curve_bins,
            use_resample=True,
        )
        self.datasets = [ds1, ds2]
        total = float(sum(weights))
        if total <= 0:
            raise ValueError("weights must sum to a positive value")
        self.weights = [float(w) / total for w in weights]

    def _choose_dataset_idx(self) -> int:
        r = random.random()
        cum = 0.0
        for idx, weight in enumerate(self.weights):
            cum += weight
            if r <= cum:
                return idx
        return len(self.weights) - 1

    def __iter__(self):
        iterators = [iter(ds) for ds in self.datasets]
        while True:
            idx = self._choose_dataset_idx()
            try:
                yield next(iterators[idx])
            except StopIteration:
                iterators[idx] = iter(self.datasets[idx])
                yield next(iterators[idx])
