"""near-failure 状态挖掘工具。

这个文件只负责从 imagined rollout 的 stepwise 视频/分数中找出“还没彻底失败，
但成功前景已经明显下降”的局部边界状态。它不依赖具体 policy/world model，
因此可以独立测试和复用。
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np


def compute_local_success_scores(
    rollout,
    reward_scorer,
    h: int,
    step_indices: Optional[Iterable[int]] = None,
) -> np.ndarray:
    """计算每个候选边界状态的局部成功分数。

    Args:
        rollout: T,H,W,C 视频数组，或包含 ``video`` 字段的 dict。
        reward_scorer: 提供 `score_loc_*` 的局部 reward 打分器。
        h: 局部视界长度，单位是视频帧。
        step_indices: policy 决策点对应的视频帧索引；不传时默认每帧都打分。

    Returns:
        一维 float 数组，长度等于 step_indices，元素是局部成功概率。
    """
    video = rollout["video"] if isinstance(rollout, dict) else rollout
    video = np.asarray(video)
    if video.ndim < 4 or len(video) == 0:
        return np.zeros((0,), dtype=np.float32)

    # 将 step index 裁剪到视频范围内，避免 rollout 长度不足时越界。
    h = max(int(h), 1)
    if step_indices is None:
        step_indices = range(len(video))
    step_indices = [max(0, min(int(idx), len(video) - 1)) for idx in step_indices]

    clips = []
    for idx in step_indices:
        end = min(idx + h, len(video))
        clips.append(video[idx:end])

    # 如果局部视界不超过 reward model 的输入长度，可以一次性批量打分。
    # 若 h 更长，则退化为 score_full_trajectory，让 scorer 内部用滑窗聚合。
    clip_len = int(getattr(reward_scorer, "clip_len", h))
    if hasattr(reward_scorer, "score_loc_clip_segments") and h <= clip_len:
        return np.asarray(reward_scorer.score_loc_clip_segments(clips), dtype=np.float32)
    if hasattr(reward_scorer, "score_loc_full_trajectory"):
        return np.asarray([reward_scorer.score_loc_full_trajectory(clip) for clip in clips], dtype=np.float32)
    if hasattr(reward_scorer, "score_clip_segments") and h <= clip_len:
        return np.asarray(reward_scorer.score_clip_segments(clips), dtype=np.float32)
    if hasattr(reward_scorer, "score_clip_segment"):
        return np.asarray([reward_scorer.score_clip_segment(clip) for clip in clips], dtype=np.float32)
    raise AttributeError("reward_scorer must provide score_loc_* methods for near-failure mining.")


def select_near_failure_states(
    local_success_scores,
    alpha: float,
    beta: float,
    min_gap: int,
    max_states: int = 1,
) -> list[int]:
    """根据局部分数区间和退化趋势选择 near-failure 状态。

    统一后的语义：
    - `alpha < s_t < beta`：当前局部 progress 处于“还可恢复、但已不稳”的中间区间；
    - `drop = max(0, s_{t-min_gap} - s_t)`：只保留相对前一个边界确实发生退化的位置；
    - `min_gap` 在当前实现里只参与 drop 计算窗口，不承担候选去重职责；
    - 按 `drop` 排序，优先返回退化最明显的状态。
    """
    scores = np.asarray(local_success_scores, dtype=np.float32)
    if scores.ndim != 1 or len(scores) <= 1:
        return []

    min_gap = max(int(min_gap), 1)
    max_states = max(int(max_states), 0)
    alpha = float(alpha)
    beta = float(beta)
    if alpha >= beta:
        raise ValueError(f"Expected alpha < beta for near-failure interval scoring, got alpha={alpha}, beta={beta}.")
    candidates = []
    for idx in range(min_gap, len(scores)):
        current = float(scores[idx])
        previous = float(scores[idx - min_gap])
        drop = max(0.0, previous - current)
        if alpha < current < beta and drop > 0.0:
            candidates.append((drop, idx))

    candidates.sort(key=lambda item: item[0], reverse=True)
    return [idx for _, idx in candidates[:max_states]]
