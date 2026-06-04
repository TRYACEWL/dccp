"""
DCCP frame extraction utilities.

本文件实现 Γ_traj 和 Γ_loc。

Γ_traj 用于完整 imagined trajectory 的 completion scoring。
Γ_loc 用于 short suffix 或 counterfactual branch 的 progress scoring。

所有输入最终会被转换成 uint8 RGB 图像帧。
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

import numpy as np
from PIL import Image


def sample_even_indices(num_frames: int, num_keyframes: int) -> list[int]:
    """均匀抽取关键帧索引。视频较短时允许重复索引，以保持 LRM 输入长度稳定"""
    num_frames = int(num_frames)
    num_keyframes = int(num_keyframes)

    if num_frames <= 0:
        return []
    if num_keyframes <= 0:
        raise ValueError(f"num_keyframes must be positive, got {num_keyframes}.")
    if num_keyframes == 1:
        return [num_frames - 1]

    return np.linspace(0, num_frames - 1, num=num_keyframes).round().astype(np.int64).tolist()


def to_uint8_rgb_frame(frame: Any, image_size: Optional[int] = None) -> np.ndarray:
    """将单帧图像转换为 uint8 RGB 格式，输出形状为 [H, W, 3]"""
    if isinstance(frame, Image.Image):
        image = frame.convert("RGB")
    else:
        array = np.asarray(frame)

        if array.ndim != 3:
            raise ValueError(f"Expected frame shape [H, W, C] or [C, H, W], got {array.shape}.")

        # 兼容 [C, H, W] 输入。
        if array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
            array = np.transpose(array, (1, 2, 0))

        if array.shape[-1] == 1:
            array = np.repeat(array, 3, axis=-1)
        elif array.shape[-1] == 4:
            array = array[..., :3]
        elif array.shape[-1] != 3:
            raise ValueError(f"Expected 1, 3, or 4 channels, got frame shape {array.shape}.")

        if array.dtype != np.uint8:
            array = array.astype(np.float32)
            max_value = float(array.max()) if array.size > 0 else 0.0
            if max_value <= 1.0:
                array = array * 255.0
            array = np.clip(array, 0, 255).astype(np.uint8)

        image = Image.fromarray(array).convert("RGB")

    if image_size is not None:
        image_size = int(image_size)
        image = image.resize((image_size, image_size), Image.BICUBIC)

    return np.asarray(image, dtype=np.uint8)


def normalize_video_array(video: Any) -> np.ndarray:
    """将视频统一为 [T, H, W, C] 格式"""
    if isinstance(video, (list, tuple)):
        if len(video) == 0:
            return np.zeros((0,), dtype=np.uint8)
        frames = [to_uint8_rgb_frame(frame) for frame in video]
        return np.stack(frames, axis=0)

    array = np.asarray(video)

    if array.ndim != 4:
        raise ValueError(f"Expected video shape [T, H, W, C] or [T, C, H, W], got {array.shape}.")

    # 兼容 [T, C, H, W] 输入。
    if array.shape[1] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
        array = np.transpose(array, (0, 2, 3, 1))

    return array


def extract_keyframes(
    video: Any,
    num_keyframes: int,
    image_size: Optional[int] = None,
    indices: Optional[Iterable[int]] = None,
) -> list[np.ndarray]:
    """从视频中抽取关键帧"""
    video_array = normalize_video_array(video)

    if len(video_array) == 0:
        return []

    if indices is None:
        selected_indices = sample_even_indices(len(video_array), num_keyframes)
    else:
        selected_indices = [
            max(0, min(int(index), len(video_array) - 1))
            for index in indices
        ]

    return [
        to_uint8_rgb_frame(video_array[index], image_size=image_size)
        for index in selected_indices
    ]


def extract_completion_frames(
    rollout_video: Any,
    num_keyframes: int = 8,
    image_size: Optional[int] = 224,
) -> list[np.ndarray]:
    """Γ_traj：从完整 imagined trajectory 中抽帧，用于 completion scoring"""
    return extract_keyframes(
        rollout_video,
        num_keyframes=num_keyframes,
        image_size=image_size,
    )


def extract_progress_frames(
    branch_or_suffix_video: Any,
    num_keyframes: int = 4,
    image_size: Optional[int] = 224,
) -> list[np.ndarray]:
    """Γ_loc：从 short suffix 或 counterfactual branch 中抽帧，用于 progress scoring"""
    return extract_keyframes(
        branch_or_suffix_video,
        num_keyframes=num_keyframes,
        image_size=image_size,
    )


def slice_video_segment(video: Any, start: int, horizon: int) -> np.ndarray:
    """截取局部视频片段 [start, start + horizon)，后续用于 nominal suffix 或 branch scoring"""
    video_array = normalize_video_array(video)

    start = int(start)
    horizon = int(horizon)

    if horizon <= 0:
        raise ValueError(f"horizon must be positive, got {horizon}.")
    if start < 0 or start >= len(video_array):
        raise IndexError(f"start index {start} is out of range for video length {len(video_array)}.")

    end = min(start + horizon, len(video_array))
    return video_array[start:end]