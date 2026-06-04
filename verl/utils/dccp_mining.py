"""
DCCP decision-sensitive state mining

本文件实现决策敏感状态挖掘

核心公式：
    q_t = S_prog(Γ_loc(τ_t:t+H), instruction)
    c_t = |q_{t+1} - 2q_t + q_{t-1}|
    h_t = average action-token entropy
    d_t = λ_c * normalized(c_t) + λ_h * normalized(h_t)

该模块只负责选择 decision-sensitive states
后续 counterfactual branch construction 和 preference construction 由其他 DCCP 模块完成
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

try:
    import torch
except ImportError:
    torch = None

from verl.utils.dccp_frame_extractor import normalize_video_array
from verl.utils.dccp_scorer import DCCPScorer
from verl.utils.dccp_action_logprob import compute_action_entropy_from_logits

@dataclass
class DCCPMiningConfig:
    """Hyperparameters for DCCP decision-sensitive state mining"""

    horizon_H: int = 3
    frames_per_action: int = 1
    state_budget_per_traj: int = 2
    nms_gap: int = 2

    lambda_curvature: float = 1.0
    lambda_entropy: float = 1.0

    eps: float = 1e-8
    require_entropy: bool = True

@dataclass
class DCCPSelectedState:
    """One selected decision-sensitive state"""

    index: int
    decision_score: float
    progress_score: float
    curvature_score: float
    entropy_score: float


@dataclass
class DCCPMiningResult:
    """Full mining result for one nominal imagined rollout"""

    progress_scores: np.ndarray
    curvature_scores: np.ndarray
    entropy_scores: np.ndarray
    decision_scores: np.ndarray
    valid_mask: np.ndarray
    selected_states: list[DCCPSelectedState]

    @property
    def selected_indices(self) -> list[int]:
        return [state.index for state in self.selected_states]


def compute_nominal_progress_scores(
    rollout_video: Any,
    instruction: str,
    scorer: DCCPScorer,
    horizon_H: int,
    frames_per_action: int = 1,
    decision_indices: Optional[list[int]] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """计算 nominal rollout 中 decision steps 对应的 progress scores"""
    video_array = normalize_video_array(rollout_video)
    horizon_H = int(horizon_H)
    frames_per_action = int(frames_per_action)

    if horizon_H <= 0:
        raise ValueError(f"horizon_H must be positive, got {horizon_H}")
    if frames_per_action <= 0:
        raise ValueError(f"frames_per_action must be positive, got {frames_per_action}")

    num_frames = len(video_array)
    suffix_horizon_frames = 1 + horizon_H * frames_per_action

    if decision_indices is None:
        candidate_starts = list(range(max(num_frames - suffix_horizon_frames + 1, 0)))
    else:
        candidate_starts = [
            int(index)
            for index in decision_indices
            if 0 <= int(index) < num_frames
        ]

    valid_starts = [
        start
        for start in candidate_starts
        if start + suffix_horizon_frames <= num_frames
    ]

    if len(valid_starts) == 0:
        return (
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.float32),
        )

    suffix_videos = []
    instructions = []
    batch_metadata = []

    for start in valid_starts:
        suffix_videos.append(video_array[start : start + suffix_horizon_frames])
        instructions.append(instruction)

        item_metadata = dict(metadata or {})
        item_metadata.update(
            {
                "score_type": "nominal_suffix_progress",
                "suffix_start": int(start),
                "horizon_H": int(horizon_H),
                "frames_per_action": int(frames_per_action),
                "suffix_horizon_frames": int(suffix_horizon_frames),
            }
        )
        batch_metadata.append(item_metadata)

    scores = scorer.score_local_progress_batch(
        branch_or_suffix_videos=suffix_videos,
        instructions=instructions,
        batch_metadata=batch_metadata,
    )

    return (
        np.asarray(valid_starts, dtype=np.int64),
        np.asarray(scores, dtype=np.float32),
    )


def compute_progress_curvature(progress_scores: Any) -> tuple[np.ndarray, np.ndarray]:
    """根据 q_t 计算 c_t = |q_{t+1} - 2q_t + q_{t-1}|"""
    q = _as_numpy_1d(progress_scores, name="progress_scores")

    curvature = np.zeros_like(q, dtype=np.float32)
    valid_mask = np.zeros_like(q, dtype=bool)

    if len(q) < 3:
        return curvature, valid_mask

    curvature[1:-1] = np.abs(q[2:] - 2.0 * q[1:-1] + q[:-2])
    valid_mask[1:-1] = True

    return curvature, valid_mask



def select_decision_sensitive_states(
    progress_scores: Any,
    entropy_scores: Optional[Any],
    config: DCCPMiningConfig,
) -> DCCPMiningResult:
    """根据 progress curvature 和 action-token entropy 选择 decision-sensitive states"""
    progress = _as_numpy_1d(progress_scores, name="progress_scores")
    curvature, valid_mask = compute_progress_curvature(progress)

    if entropy_scores is None:
        if config.require_entropy and float(config.lambda_entropy) != 0.0:
            raise ValueError("entropy_scores is required when lambda_entropy is non-zero")
        entropy = np.zeros_like(progress, dtype=np.float32)
    else:
        entropy = _as_numpy_1d(entropy_scores, name="entropy_scores")
        if len(entropy) < len(progress):
            raise ValueError(
                f"entropy_scores must have length at least {len(progress)}, got {len(entropy)}"
            )
        entropy = entropy[: len(progress)].astype(np.float32)

    curvature_norm = normalize_scores(curvature, valid_mask=valid_mask, eps=float(config.eps))
    entropy_norm = normalize_scores(entropy, valid_mask=valid_mask, eps=float(config.eps))

    decision_scores = (
        float(config.lambda_curvature) * curvature_norm
        + float(config.lambda_entropy) * entropy_norm
    ).astype(np.float32)

    local_maxima_mask = find_local_maxima(decision_scores, valid_mask=valid_mask)

    selected_indices = temporal_nms(
        scores=decision_scores,
        candidate_mask=local_maxima_mask,
        top_k=int(config.state_budget_per_traj),
        nms_gap=int(config.nms_gap),
    )

    selected_states = [
        DCCPSelectedState(
            index=int(index),
            decision_score=float(decision_scores[index]),
            progress_score=float(progress[index]),
            curvature_score=float(curvature[index]),
            entropy_score=float(entropy[index]),
        )
        for index in selected_indices
    ]

    return DCCPMiningResult(
        progress_scores=progress.astype(np.float32),
        curvature_scores=curvature.astype(np.float32),
        entropy_scores=entropy.astype(np.float32),
        decision_scores=decision_scores.astype(np.float32),
        valid_mask=valid_mask.astype(bool),
        selected_states=selected_states,
    )


def mine_decision_sensitive_states(
    rollout_video: Any,
    instruction: str,
    scorer: DCCPScorer,
    entropy_scores: Optional[Any],
    config: DCCPMiningConfig,
    decision_indices: Optional[list[int]] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> DCCPMiningResult:
    """从一条 nominal imagined rollout 中挖掘 decision-sensitive states"""
    valid_starts, progress_scores = compute_nominal_progress_scores(
        rollout_video=rollout_video,
        instruction=instruction,
        scorer=scorer,
        horizon_H=int(config.horizon_H),
        frames_per_action=int(config.frames_per_action),
        decision_indices=decision_indices,
        metadata=metadata,
    )

    result = select_decision_sensitive_states(
        progress_scores=progress_scores,
        entropy_scores=_gather_entropy_for_decision_indices(
            entropy_scores=entropy_scores,
            decision_indices=valid_starts,
        ),
        config=config,
    )

    remapped_states = []

    for state in result.selected_states:
        local_index = int(state.index)
        original_index = int(valid_starts[local_index])

        remapped_states.append(
            DCCPSelectedState(
                index=original_index,
                decision_score=float(state.decision_score),
                progress_score=float(state.progress_score),
                curvature_score=float(state.curvature_score),
                entropy_score=float(state.entropy_score),
            )
        )

    return DCCPMiningResult(
        progress_scores=result.progress_scores,
        curvature_scores=result.curvature_scores,
        entropy_scores=result.entropy_scores,
        decision_scores=result.decision_scores,
        valid_mask=result.valid_mask,
        selected_states=remapped_states,
    )


def normalize_scores(values: Any, valid_mask: Any, eps: float = 1e-8) -> np.ndarray:
    """在同一条 rollout 内对有效位置做 min-max normalization"""
    values_array = _as_numpy_1d(values, name="values").astype(np.float32)
    mask_array = np.asarray(valid_mask, dtype=bool).reshape(-1)

    if len(values_array) != len(mask_array):
        raise ValueError(
            f"values and valid_mask must have the same length, got {len(values_array)} and {len(mask_array)}"
        )

    normalized = np.zeros_like(values_array, dtype=np.float32)

    if not np.any(mask_array):
        return normalized

    valid_values = values_array[mask_array]
    min_value = float(np.min(valid_values))
    max_value = float(np.max(valid_values))

    if max_value - min_value <= float(eps):
        return normalized

    normalized[mask_array] = (values_array[mask_array] - min_value) / (max_value - min_value)
    return normalized.astype(np.float32)


def find_local_maxima(scores: Any, valid_mask: Any) -> np.ndarray:
    """选择 decision score 的 local maxima"""
    score_array = _as_numpy_1d(scores, name="scores")
    mask_array = np.asarray(valid_mask, dtype=bool).reshape(-1)

    if len(score_array) != len(mask_array):
        raise ValueError(
            f"scores and valid_mask must have the same length, got {len(score_array)} and {len(mask_array)}"
        )

    local_maxima = np.zeros_like(mask_array, dtype=bool)

    if len(score_array) < 3:
        return local_maxima

    for index in range(1, len(score_array) - 1):
        if not mask_array[index]:
            continue

        left_score = score_array[index - 1]
        current_score = score_array[index]
        right_score = score_array[index + 1]

        if current_score >= left_score and current_score >= right_score:
            local_maxima[index] = True

    if not np.any(local_maxima):
        local_maxima = mask_array.copy()

    return local_maxima


def temporal_nms(
    scores: Any,
    candidate_mask: Any,
    top_k: int,
    nms_gap: int,
) -> list[int]:
    """对候选状态做 temporal non-maximum suppression"""
    score_array = _as_numpy_1d(scores, name="scores")
    mask_array = np.asarray(candidate_mask, dtype=bool).reshape(-1)

    if len(score_array) != len(mask_array):
        raise ValueError(
            f"scores and candidate_mask must have the same length, got {len(score_array)} and {len(mask_array)}"
        )

    top_k = int(top_k)
    nms_gap = int(nms_gap)

    if top_k <= 0:
        return []
    if nms_gap < 0:
        raise ValueError(f"nms_gap must be non-negative, got {nms_gap}")

    candidate_indices = np.where(mask_array & np.isfinite(score_array))[0].tolist()
    candidate_indices = sorted(candidate_indices, key=lambda index: (-float(score_array[index]), int(index)))

    selected: list[int] = []

    for index in candidate_indices:
        if len(selected) >= top_k:
            break

        too_close = any(abs(int(index) - int(chosen)) < nms_gap for chosen in selected)
        if too_close:
            continue

        selected.append(int(index))

    selected = sorted(selected)
    return selected


def summarize_mining_result(result: DCCPMiningResult) -> dict[str, float]:
    """汇总 mining 结果，用于 rollout 侧日志"""
    valid_mask = result.valid_mask.astype(bool)

    if np.any(valid_mask):
        progress_mean = float(np.mean(result.progress_scores[valid_mask]))
        curvature_mean = float(np.mean(result.curvature_scores[valid_mask]))
        entropy_mean = float(np.mean(result.entropy_scores[valid_mask]))
    else:
        progress_mean = 0.0
        curvature_mean = 0.0
        entropy_mean = 0.0

    return {
        "dccp/selected_states_mean": float(len(result.selected_states)),
        "dccp/progress_mean": progress_mean,
        "dccp/curvature_mean": curvature_mean,
        "dccp/entropy_mean": entropy_mean,
    }


def _as_numpy_1d(values: Any, name: str) -> np.ndarray:
    if torch is not None and isinstance(values, torch.Tensor):
        array = values.detach().cpu().numpy()
    else:
        array = np.asarray(values)

    array = np.asarray(array, dtype=np.float32).reshape(-1)

    if np.any(~np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")

    return array

def compute_action_token_entropy_from_logits(
    action_token_logits,
    response_mask,
    action_vocab_mask=None,
):
    result = compute_action_entropy_from_logits(
        logits=action_token_logits,
        response_mask=response_mask,
        action_vocab_mask=action_vocab_mask,
    )
    return result.sequence_entropies.detach().cpu().numpy().astype(np.float32)

def _gather_entropy_for_decision_indices(
    entropy_scores: Optional[Any],
    decision_indices: np.ndarray,
) -> Optional[np.ndarray]:
    """根据 decision step 的原始帧索引收集 entropy"""
    if entropy_scores is None:
        return None

    entropy_array = _as_numpy_1d(entropy_scores, name="entropy_scores")

    gathered = []

    for index in decision_indices:
        index = int(index)
        if 0 <= index < len(entropy_array):
            gathered.append(float(entropy_array[index]))
        else:
            gathered.append(0.0)

    return np.asarray(gathered, dtype=np.float32)