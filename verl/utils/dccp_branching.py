"""
DCCP counterfactual branch construction

本文件实现反事实分支构造

核心思想：
    对同一个 selected state x_t，共享相同的历史前缀、视觉上下文和语言指令
    候选动作集合必须包含 nominal action 和若干 sampled alternative actions
    nominal branch 使用原 imagined rollout 中缓存的 short suffix
    alternative branch 只替换当前 first action，后续仍由 policy 和 world model 继续生成

本模块只负责 branch construction 和 branch progress scoring
winner-loser preference construction 由 dccp_preferences.py 完成
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

from verl.utils.dccp_frame_extractor import slice_video_segment
from verl.utils.dccp_scorer import DCCPScorer


@dataclass
class DCCPBranchingConfig:
    """Hyperparameters for DCCP counterfactual branching"""

    horizon_H: int = 3
    frames_per_action: int = 1
    num_candidates: int = 8
    max_branches_per_state: Optional[int] = None


@dataclass
class DCCPActionCandidate:
    """One first-action candidate at a selected state"""

    candidate_index: int
    is_nominal: bool

    response_tokens: Any
    action_for_world_model: Any

    generation_logprob: Optional[float] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DCCPBranch:
    """One generated branch for a first-action candidate"""

    state_index: int
    candidate: DCCPActionCandidate
    video: Any
    progress_score: Optional[float] = None


@dataclass
class DCCPBranchSet:
    """Nominal branch and counterfactual branches for one selected state"""

    state_index: int
    nominal_branch: DCCPBranch
    alternative_branches: list[DCCPBranch]

    @property
    def all_branches(self) -> list[DCCPBranch]:
        return [self.nominal_branch] + list(self.alternative_branches)

    @property
    def num_candidates(self) -> int:
        return 1 + len(self.alternative_branches)


@dataclass
class DCCPScoredBranchSet:
    """Progress scores for nominal and alternative branches"""

    state_index: int

    nominal_candidate: DCCPActionCandidate
    nominal_video: Any
    nominal_score: float

    alternative_candidates: list[DCCPActionCandidate]
    alternative_videos: list[Any]
    alternative_scores: np.ndarray

    def margins_against_nominal(self) -> np.ndarray:
        return self.alternative_scores.astype(np.float32) - float(self.nominal_score)


CounterfactualRolloutFn = Callable[
    [Any, DCCPActionCandidate, int, Optional[dict[str, Any]]],
    Any,
]


def build_candidate_action_set(
    nominal_response_tokens: Any,
    nominal_action_for_world_model: Any,
    alternative_response_tokens: list[Any],
    alternative_actions_for_world_model: list[Any],
    config: DCCPBranchingConfig,
    nominal_generation_logprob: Optional[float] = None,
    alternative_generation_logprobs: Optional[list[Optional[float]]] = None,
    nominal_metadata: Optional[dict[str, Any]] = None,
    alternative_metadata: Optional[list[dict[str, Any]]] = None,
) -> list[DCCPActionCandidate]:
    """构造 first-action candidate set，其中 candidate 0 必须是 nominal action"""
    if int(config.num_candidates) <= 1:
        raise ValueError(f"num_candidates must be greater than 1, got {config.num_candidates}")

    if len(alternative_response_tokens) != len(alternative_actions_for_world_model):
        raise ValueError(
            f"alternative_response_tokens and alternative_actions_for_world_model must have the same length, "
            f"got {len(alternative_response_tokens)} and {len(alternative_actions_for_world_model)}"
        )

    max_alternatives = int(config.num_candidates) - 1

    if config.max_branches_per_state is not None:
        max_alternatives = min(max_alternatives, int(config.max_branches_per_state))

    selected_alt_responses = list(alternative_response_tokens[:max_alternatives])
    selected_alt_actions = list(alternative_actions_for_world_model[:max_alternatives])

    if alternative_generation_logprobs is None:
        selected_alt_logprobs = [None for _ in selected_alt_responses]
    else:
        selected_alt_logprobs = list(alternative_generation_logprobs[: len(selected_alt_responses)])

    if alternative_metadata is None:
        selected_alt_metadata = [{} for _ in selected_alt_responses]
    else:
        selected_alt_metadata = list(alternative_metadata[: len(selected_alt_responses)])

    candidates = [
        DCCPActionCandidate(
            candidate_index=0,
            is_nominal=True,
            response_tokens=nominal_response_tokens,
            action_for_world_model=nominal_action_for_world_model,
            generation_logprob=nominal_generation_logprob,
            metadata=dict(nominal_metadata or {}),
        )
    ]

    for local_index, (response_tokens, action_for_world_model, logprob, metadata) in enumerate(
        zip(
            selected_alt_responses,
            selected_alt_actions,
            selected_alt_logprobs,
            selected_alt_metadata,
        ),
        start=1,
    ):
        candidates.append(
            DCCPActionCandidate(
                candidate_index=int(local_index),
                is_nominal=False,
                response_tokens=response_tokens,
                action_for_world_model=action_for_world_model,
                generation_logprob=logprob,
                metadata=dict(metadata or {}),
            )
        )

    return candidates


def extract_cached_nominal_branch(
    rollout_video: Any,
    state_index: int,
    horizon_H: int,
    frames_per_action: int = 1,
) -> Any:
    """从 nominal imagined rollout 中截取 cached nominal branch"""
    horizon_frames = 1 + int(horizon_H) * int(frames_per_action)

    return slice_video_segment(
        video=rollout_video,
        start=int(state_index),
        horizon=int(horizon_frames),
    )


def build_counterfactual_branch_set(
    state_index: int,
    state_context: Any,
    rollout_video: Any,
    candidates: list[DCCPActionCandidate],
    rollout_branch_fn: CounterfactualRolloutFn,
    config: DCCPBranchingConfig,
    rollout_metadata: Optional[dict[str, Any]] = None,
) -> DCCPBranchSet:
    """为一个 selected state 构造 nominal branch 和 alternative branches"""
    if len(candidates) == 0:
        raise ValueError("candidates must not be empty")

    nominal_candidate = candidates[0]

    if not nominal_candidate.is_nominal or nominal_candidate.candidate_index != 0:
        raise ValueError("the first candidate must be the nominal action with candidate_index = 0")

    nominal_video = extract_cached_nominal_branch(
        rollout_video=rollout_video,
        state_index=int(state_index),
        horizon_H=int(config.horizon_H),
        frames_per_action=int(config.frames_per_action),
    )

    nominal_branch = DCCPBranch(
        state_index=int(state_index),
        candidate=nominal_candidate,
        video=nominal_video,
        progress_score=None,
    )

    alternative_branches: list[DCCPBranch] = []

    for candidate in candidates[1:]:
        if candidate.is_nominal:
            raise ValueError("only the first candidate can be nominal")

        branch_metadata = dict(rollout_metadata or {})
        branch_metadata.update(
            {
                "state_index": int(state_index),
                "candidate_index": int(candidate.candidate_index),
                "horizon_H": int(config.horizon_H),
            }
        )

        branch_video = rollout_branch_fn(
            state_context,
            candidate,
            int(config.horizon_H),
            branch_metadata,
        )

        alternative_branches.append(
            DCCPBranch(
                state_index=int(state_index),
                candidate=candidate,
                video=branch_video,
                progress_score=None,
            )
        )

    return DCCPBranchSet(
        state_index=int(state_index),
        nominal_branch=nominal_branch,
        alternative_branches=alternative_branches,
    )


def score_branch_set(
    branch_set: DCCPBranchSet,
    instruction: str,
    scorer: DCCPScorer,
    metadata: Optional[dict[str, Any]] = None,
) -> DCCPScoredBranchSet:
    """对 nominal branch 和 alternative branches 计算 progress scores"""
    nominal_metadata = dict(metadata or {})
    nominal_metadata.update(
        {
            "score_type": "nominal_branch_progress",
            "state_index": int(branch_set.state_index),
            "candidate_index": 0,
        }
    )

    nominal_score = scorer.score_local_progress(
        branch_or_suffix_video=branch_set.nominal_branch.video,
        instruction=instruction,
        metadata=nominal_metadata,
    )

    alternative_branches = list(branch_set.alternative_branches)

    if len(alternative_branches) == 0:
        return DCCPScoredBranchSet(
            state_index=int(branch_set.state_index),
            nominal_candidate=branch_set.nominal_branch.candidate,
            nominal_video=branch_set.nominal_branch.video,
            nominal_score=float(nominal_score),
            alternative_candidates=[],
            alternative_videos=[],
            alternative_scores=np.zeros((0,), dtype=np.float32),
        )

    alternative_videos = [branch.video for branch in alternative_branches]
    alternative_instructions = [instruction for _ in alternative_branches]
    alternative_metadata = []

    for branch in alternative_branches:
        item_metadata = dict(metadata or {})
        item_metadata.update(
            {
                "score_type": "counterfactual_branch_progress",
                "state_index": int(branch.state_index),
                "candidate_index": int(branch.candidate.candidate_index),
            }
        )
        alternative_metadata.append(item_metadata)

    alternative_scores = scorer.score_local_progress_batch(
        branch_or_suffix_videos=alternative_videos,
        instructions=alternative_instructions,
        batch_metadata=alternative_metadata,
    )

    alternative_scores = np.asarray(alternative_scores, dtype=np.float32)

    for branch, score in zip(alternative_branches, alternative_scores):
        branch.progress_score = float(score)

    branch_set.nominal_branch.progress_score = float(nominal_score)

    return DCCPScoredBranchSet(
        state_index=int(branch_set.state_index),
        nominal_candidate=branch_set.nominal_branch.candidate,
        nominal_video=branch_set.nominal_branch.video,
        nominal_score=float(nominal_score),
        alternative_candidates=[branch.candidate for branch in alternative_branches],
        alternative_videos=alternative_videos,
        alternative_scores=alternative_scores,
    )


def build_and_score_counterfactual_branches(
    state_index: int,
    state_context: Any,
    rollout_video: Any,
    candidates: list[DCCPActionCandidate],
    rollout_branch_fn: CounterfactualRolloutFn,
    instruction: str,
    scorer: DCCPScorer,
    config: DCCPBranchingConfig,
    rollout_metadata: Optional[dict[str, Any]] = None,
    score_metadata: Optional[dict[str, Any]] = None,
) -> DCCPScoredBranchSet:
    """构造并评分一个 selected state 的 counterfactual branches"""
    branch_set = build_counterfactual_branch_set(
        state_index=state_index,
        state_context=state_context,
        rollout_video=rollout_video,
        candidates=candidates,
        rollout_branch_fn=rollout_branch_fn,
        config=config,
        rollout_metadata=rollout_metadata,
    )

    return score_branch_set(
        branch_set=branch_set,
        instruction=instruction,
        scorer=scorer,
        metadata=score_metadata,
    )


def summarize_scored_branch_set(scored_branch_set: DCCPScoredBranchSet) -> dict[str, float]:
    """汇总 branch scores，用于 rollout 侧日志"""
    margins = scored_branch_set.margins_against_nominal()

    if len(margins) == 0:
        return {
            "dccp/branch_nominal_score": float(scored_branch_set.nominal_score),
            "dccp/branch_alternative_score_mean": 0.0,
            "dccp/branch_margin_mean": 0.0,
            "dccp/branch_margin_abs_mean": 0.0,
        }

    return {
        "dccp/branch_nominal_score": float(scored_branch_set.nominal_score),
        "dccp/branch_alternative_score_mean": float(np.mean(scored_branch_set.alternative_scores)),
        "dccp/branch_margin_mean": float(np.mean(margins)),
        "dccp/branch_margin_abs_mean": float(np.mean(np.abs(margins))),
    }


def validate_candidate_set(candidates: list[DCCPActionCandidate]) -> None:
    """检查 candidate set 是否满足 DCCP 的 nominal-first 约束"""
    if len(candidates) == 0:
        raise ValueError("candidate set must not be empty")

    if candidates[0].candidate_index != 0 or not candidates[0].is_nominal:
        raise ValueError("candidate 0 must be the nominal action")

    seen_indices = set()

    for candidate in candidates:
        if candidate.candidate_index in seen_indices:
            raise ValueError(f"duplicated candidate_index: {candidate.candidate_index}")
        seen_indices.add(candidate.candidate_index)

    for candidate in candidates[1:]:
        if candidate.is_nominal:
            raise ValueError("only candidate 0 can be nominal")