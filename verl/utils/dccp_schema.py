"""
DCCP batch schema.

本文件定义偏好样本与日志指标的标准字段名。
每个有效偏好样本表示同一视觉-语言状态下 winner action 与 loser action 的动作 token 偏好。

DCCP 偏好样本对应论文中的：
    b = (x, a_w, a_l, w, Δ_ref)

其中：
    a_w 表示 winner action；
    a_l 表示 loser action；
    w 表示由 progress margin 得到的偏好权重；
    Δ_ref 表示 reference policy 下 winner 与 loser 的 log-probability gap。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DCCPPrefKeys:
    """DCCP preference batch keys."""

    input_ids: str = "pref_input_ids"
    attention_mask: str = "pref_attention_mask"
    pixel_values: str = "pref_pixel_values"

    winner_responses: str = "pref_winner_responses"
    loser_responses: str = "pref_loser_responses"
    response_mask: str = "pref_response_mask"

    weight: str = "pref_weight"
    delta_ref: str = "pref_delta_ref"
    margin: str = "pref_margin"
    valid: str = "pref_valid"

    nominal_score: str = "pref_nominal_score"
    alternative_score: str = "pref_alternative_score"
    entropy: str = "pref_entropy"
    curvature: str = "pref_curvature"
    state_index: str = "pref_state_index"
    candidate_index: str = "pref_candidate_index"


@dataclass(frozen=True)
class DCCPMetricKeys:
    """DCCP metric keys."""

    valid_pairs_rollout: str = "dccp/valid_pairs_rollout"
    pref_density: str = "dccp/pref_density"
    selected_states_mean: str = "dccp/selected_states_mean"

    progress_mean: str = "dccp/progress_mean"
    curvature_mean: str = "dccp/curvature_mean"
    entropy_mean: str = "dccp/entropy_mean"

    margin_mean: str = "dccp/margin_mean"
    margin_abs_mean: str = "dccp/margin_abs_mean"
    positive_margin_ratio: str = "dccp/positive_margin_ratio"
    negative_margin_ratio: str = "dccp/negative_margin_ratio"

    delta_ref_mean: str = "dccp/delta_ref_mean"

    lrm_completion_latency: str = "dccp/lrm_completion_latency"
    lrm_progress_latency: str = "dccp/lrm_progress_latency"
    lrm_cache_hit_rate: str = "dccp/lrm_cache_hit_rate"

    loss_pref: str = "loss/dccp_pref"
    loss_total: str = "loss/total"
    valid_pairs_actor: str = "dccp/valid_pairs_actor"


PREF_KEYS = DCCPPrefKeys()
METRIC_KEYS = DCCPMetricKeys()