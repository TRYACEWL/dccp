"""
DCCP high-margin preference construction

本文件实现high-margin winner-loser preference construction

核心公式：
    u_t^(k) = S_prog(Γ_loc(ρ_t^(k)), instruction)
    m_t^(k) = u_t^(k) - u_t^(0)

偏好规则：
    if m_t^(k) > δ+:
        alternative action wins nominal action

    if m_t^(k) < -δ-:
        nominal action wins alternative action

    otherwise:
        discard this comparison

每个保留的偏好样本对应：
    b = (x, a_w, a_l, w, Δ_ref)

其中：
    a_w 是 winner action
    a_l 是 loser action
    w = |m_t^(k)|
    Δ_ref = logπ_ref(a_w | x) - logπ_ref(a_l | x)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

try:
    import torch
except ImportError:
    torch = None

from verl.utils.dccp_branching import DCCPActionCandidate, DCCPScoredBranchSet
from verl.utils.dccp_schema import PREF_KEYS


@dataclass
class DCCPPreferenceConfig:
    """Hyperparameters for DCCP preference construction"""

    margin_pos: float = 0.05
    margin_neg: float = 0.05
    max_pairs_per_state: Optional[int] = None
    max_pairs_per_batch: int = 64


@dataclass
class DCCPPreferencePair:
    """One DCCP winner-loser preference pair"""

    state_index: int
    candidate_index: int

    winner_candidate: DCCPActionCandidate
    loser_candidate: DCCPActionCandidate

    winner_responses: Any
    loser_responses: Any

    weight: float
    delta_ref: float
    margin: float

    nominal_score: float
    alternative_score: float

    entropy_score: float = 0.0
    curvature_score: float = 0.0

    context: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DCCPPreferenceBuildResult:
    """Preference construction result for one selected state"""

    state_index: int
    pairs: list[DCCPPreferencePair]

    @property
    def valid_pair_count(self) -> int:
        return len(self.pairs)


ReferenceGapFn = Callable[
    [dict[str, Any], DCCPActionCandidate, DCCPActionCandidate],
    float,
]


def build_high_margin_preferences(
    scored_branch_set: DCCPScoredBranchSet,
    context: dict[str, Any],
    reference_gap_fn: ReferenceGapFn,
    config: DCCPPreferenceConfig,
    entropy_score: float = 0.0,
    curvature_score: float = 0.0,
    metadata: Optional[dict[str, Any]] = None,
) -> DCCPPreferenceBuildResult:
    """根据 branch progress margin 构造 high-margin winner-loser preference pairs"""
    if reference_gap_fn is None:
        raise ValueError("reference_gap_fn is required to compute pref_delta_ref")

    margin_pos = float(config.margin_pos)
    margin_neg = float(config.margin_neg)

    if margin_pos < 0.0 or margin_neg < 0.0:
        raise ValueError(f"margin thresholds must be non-negative, got {margin_pos} and {margin_neg}")

    nominal_candidate = scored_branch_set.nominal_candidate
    nominal_score = float(scored_branch_set.nominal_score)
    margins = scored_branch_set.margins_against_nominal()

    pairs: list[DCCPPreferencePair] = []

    for local_index, margin in enumerate(margins):
        alternative_candidate = scored_branch_set.alternative_candidates[local_index]
        alternative_score = float(scored_branch_set.alternative_scores[local_index])
        signed_margin = float(margin)

        if signed_margin > margin_pos:
            winner_candidate = alternative_candidate
            loser_candidate = nominal_candidate
        elif signed_margin < -margin_neg:
            winner_candidate = nominal_candidate
            loser_candidate = alternative_candidate
        else:
            continue

        delta_ref = float(reference_gap_fn(context, winner_candidate, loser_candidate))

        pair = DCCPPreferencePair(
            state_index=int(scored_branch_set.state_index),
            candidate_index=int(alternative_candidate.candidate_index),
            winner_candidate=winner_candidate,
            loser_candidate=loser_candidate,
            winner_responses=winner_candidate.response_tokens,
            loser_responses=loser_candidate.response_tokens,
            weight=float(abs(signed_margin)),
            delta_ref=delta_ref,
            margin=signed_margin,
            nominal_score=nominal_score,
            alternative_score=alternative_score,
            entropy_score=float(entropy_score),
            curvature_score=float(curvature_score),
            context=dict(context),
            metadata=dict(metadata or {}),
        )

        pairs.append(pair)

    pairs = sort_and_truncate_preferences(
        pairs=pairs,
        max_pairs=config.max_pairs_per_state,
    )

    return DCCPPreferenceBuildResult(
        state_index=int(scored_branch_set.state_index),
        pairs=pairs,
    )


def sort_and_truncate_preferences(
    pairs: list[DCCPPreferencePair],
    max_pairs: Optional[int],
) -> list[DCCPPreferencePair]:
    """按 |margin| 从大到小排序，并限制每个 state 保留的 pair 数量"""
    sorted_pairs = sorted(
        pairs,
        key=lambda pair: (-float(pair.weight), int(pair.candidate_index)),
    )

    if max_pairs is None:
        return sorted_pairs

    max_pairs = int(max_pairs)
    if max_pairs <= 0:
        return []

    return sorted_pairs[:max_pairs]


def pack_preference_batch(
    preference_pairs: list[DCCPPreferencePair],
    max_pairs: int,
    padding_context: dict[str, Any],
    padding_response_tokens: Any,
    padding_response_mask: Any,
) -> dict[str, Any]:
    """将 DCCPPreferencePair 列表打包成固定大小的 pref_* batch"""
    max_pairs = int(max_pairs)

    if max_pairs <= 0:
        raise ValueError(f"max_pairs must be positive, got {max_pairs}")

    selected_pairs = list(preference_pairs[:max_pairs])
    valid_count = len(selected_pairs)
    pad_count = max_pairs - valid_count

    batch: dict[str, Any] = {}

    input_ids_values = []
    attention_mask_values = []
    pixel_values_values = []
    winner_values = []
    loser_values = []
    response_mask_values = []

    weight_values = []
    delta_ref_values = []
    margin_values = []
    valid_values = []

    nominal_score_values = []
    alternative_score_values = []
    entropy_values = []
    curvature_values = []
    state_index_values = []
    candidate_index_values = []

    for pair in selected_pairs:
        input_ids_values.append(_get_context_value(pair.context, PREF_KEYS.input_ids))
        attention_mask_values.append(_get_context_value(pair.context, PREF_KEYS.attention_mask))
        pixel_values_values.append(_get_context_value(pair.context, PREF_KEYS.pixel_values))

        winner_values.append(pair.winner_responses)
        loser_values.append(pair.loser_responses)
        response_mask_values.append(_get_response_mask(pair.context, padding_response_mask))

        weight_values.append(float(pair.weight))
        delta_ref_values.append(float(pair.delta_ref))
        margin_values.append(float(pair.margin))
        valid_values.append(True)

        nominal_score_values.append(float(pair.nominal_score))
        alternative_score_values.append(float(pair.alternative_score))
        entropy_values.append(float(pair.entropy_score))
        curvature_values.append(float(pair.curvature_score))
        state_index_values.append(int(pair.state_index))
        candidate_index_values.append(int(pair.candidate_index))

    for _ in range(pad_count):
        input_ids_values.append(_get_context_value(padding_context, PREF_KEYS.input_ids))
        attention_mask_values.append(_get_context_value(padding_context, PREF_KEYS.attention_mask))
        pixel_values_values.append(_get_context_value(padding_context, PREF_KEYS.pixel_values))

        winner_values.append(_zeros_like(padding_response_tokens))
        loser_values.append(_zeros_like(padding_response_tokens))
        response_mask_values.append(_zeros_like(padding_response_mask))

        weight_values.append(0.0)
        delta_ref_values.append(0.0)
        margin_values.append(0.0)
        valid_values.append(False)

        nominal_score_values.append(0.0)
        alternative_score_values.append(0.0)
        entropy_values.append(0.0)
        curvature_values.append(0.0)
        state_index_values.append(-1)
        candidate_index_values.append(-1)

    batch[PREF_KEYS.input_ids] = _stack_values(input_ids_values)
    batch[PREF_KEYS.attention_mask] = _stack_values(attention_mask_values)
    batch[PREF_KEYS.pixel_values] = _stack_values(pixel_values_values)

    batch[PREF_KEYS.winner_responses] = _stack_values(winner_values)
    batch[PREF_KEYS.loser_responses] = _stack_values(loser_values)
    batch[PREF_KEYS.response_mask] = _stack_values(response_mask_values)

    use_torch, device = _infer_torch_context(
        input_ids_values
        + attention_mask_values
        + pixel_values_values
        + winner_values
        + loser_values
        + response_mask_values
    )

    batch[PREF_KEYS.weight] = _make_float_vector(weight_values, use_torch=use_torch, device=device)
    batch[PREF_KEYS.delta_ref] = _make_float_vector(delta_ref_values, use_torch=use_torch, device=device)
    batch[PREF_KEYS.margin] = _make_float_vector(margin_values, use_torch=use_torch, device=device)
    batch[PREF_KEYS.valid] = _make_bool_vector(valid_values, use_torch=use_torch, device=device)

    batch[PREF_KEYS.nominal_score] = _make_float_vector(nominal_score_values, use_torch=use_torch, device=device)
    batch[PREF_KEYS.alternative_score] = _make_float_vector(alternative_score_values, use_torch=use_torch, device=device)
    batch[PREF_KEYS.entropy] = _make_float_vector(entropy_values, use_torch=use_torch, device=device)
    batch[PREF_KEYS.curvature] = _make_float_vector(curvature_values, use_torch=use_torch, device=device)
    batch[PREF_KEYS.state_index] = _make_int_vector(state_index_values, use_torch=use_torch, device=device)
    batch[PREF_KEYS.candidate_index] = _make_int_vector(candidate_index_values, use_torch=use_torch, device=device)

    return batch

def pack_preference_batch_by_rollout(
    preference_pairs_by_rollout: list[list[DCCPPreferencePair]],
    max_pairs_per_rollout: int,
    padding_context: dict[str, Any],
    padding_response_tokens: Any,
    padding_response_mask: Any,
) -> dict[str, Any]:
    """按 rollout 分组打包 DCCP preference batch

    输出形状：
        pref_valid: [B, P]
        pref_input_ids: [B, P, L]
        pref_winner_responses: [B, P, A]
    """
    max_pairs_per_rollout = int(max_pairs_per_rollout)

    if max_pairs_per_rollout <= 0:
        raise ValueError(f"max_pairs_per_rollout must be positive, got {max_pairs_per_rollout}")

    per_rollout_batches = []

    for rollout_pairs in preference_pairs_by_rollout:
        sorted_pairs = sorted(
            list(rollout_pairs),
            key=lambda pair: (-float(pair.weight), int(pair.state_index), int(pair.candidate_index)),
        )

        per_rollout_batch = pack_preference_batch(
            preference_pairs=sorted_pairs,
            max_pairs=max_pairs_per_rollout,
            padding_context=padding_context,
            padding_response_tokens=padding_response_tokens,
            padding_response_mask=padding_response_mask,
        )

        per_rollout_batches.append(per_rollout_batch)

    if len(per_rollout_batches) == 0:
        raise ValueError("preference_pairs_by_rollout must contain at least one rollout")

    output: dict[str, Any] = {}

    for key in per_rollout_batches[0].keys():
        output[key] = _stack_values([batch[key] for batch in per_rollout_batches])

    return output

def summarize_preference_pairs(preference_pairs: list[DCCPPreferencePair]) -> dict[str, float]:
    """汇总 preference pairs，用于 rollout 侧日志"""
    if len(preference_pairs) == 0:
        return {
            "dccp/valid_pairs_rollout": 0.0,
            "dccp/pref_density": 0.0,
            "dccp/margin_mean": 0.0,
            "dccp/margin_abs_mean": 0.0,
            "dccp/positive_margin_ratio": 0.0,
            "dccp/negative_margin_ratio": 0.0,
            "dccp/delta_ref_mean": 0.0,
        }

    margins = np.asarray([pair.margin for pair in preference_pairs], dtype=np.float32)
    delta_refs = np.asarray([pair.delta_ref for pair in preference_pairs], dtype=np.float32)

    return {
        "dccp/valid_pairs_rollout": float(len(preference_pairs)),
        "dccp/pref_density": 1.0,
        "dccp/margin_mean": float(np.mean(margins)),
        "dccp/margin_abs_mean": float(np.mean(np.abs(margins))),
        "dccp/positive_margin_ratio": float(np.mean(margins > 0.0)),
        "dccp/negative_margin_ratio": float(np.mean(margins < 0.0)),
        "dccp/delta_ref_mean": float(np.mean(delta_refs)),
    }


def flatten_preference_results(
    results: list[DCCPPreferenceBuildResult],
) -> list[DCCPPreferencePair]:
    """将多个 selected states 的 preference results 展平成一个列表"""
    pairs: list[DCCPPreferencePair] = []

    for result in results:
        pairs.extend(result.pairs)

    return pairs


def _get_context_value(context: dict[str, Any], key: str) -> Any:
    if key not in context:
        raise KeyError(f"context is missing required key: {key}")
    return context[key]


def _get_response_mask(context: dict[str, Any], padding_response_mask: Any) -> Any:
    if PREF_KEYS.response_mask in context:
        return context[PREF_KEYS.response_mask]
    return padding_response_mask


def _zeros_like(value: Any) -> Any:
    if torch is not None and isinstance(value, torch.Tensor):
        return torch.zeros_like(value)

    array = np.asarray(value)
    return np.zeros_like(array)


def _stack_values(values: list[Any]) -> Any:
    if len(values) == 0:
        raise ValueError("values must not be empty")

    first = values[0]

    if torch is not None and isinstance(first, torch.Tensor):
        return torch.stack(values, dim=0)

    return np.stack([np.asarray(value) for value in values], axis=0)


def _infer_torch_context(values: list[Any]) -> tuple[bool, Optional[Any]]:
    if torch is None:
        return False, None

    for value in values:
        if isinstance(value, torch.Tensor):
            return True, value.device

    return False, None


def _make_float_vector(values: list[float], use_torch: bool, device: Optional[Any]) -> Any:
    if use_torch:
        return torch.tensor(values, dtype=torch.float32, device=device)
    return np.asarray(values, dtype=np.float32)


def _make_int_vector(values: list[int], use_torch: bool, device: Optional[Any]) -> Any:
    if use_torch:
        return torch.tensor(values, dtype=torch.long, device=device)
    return np.asarray(values, dtype=np.int64)


def _make_bool_vector(values: list[bool], use_torch: bool, device: Optional[Any]) -> Any:
    if use_torch:
        return torch.tensor(values, dtype=torch.bool, device=device)
    return np.asarray(values, dtype=bool)