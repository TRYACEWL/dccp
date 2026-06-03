"""构造动作级 recovery 监督目标。

输入是局部 recovery search 的候选动作和分数，输出是训练策略需要的
``a_star`` 与置信度 ``c_t``。这里保持 action 表示无关：连续动作可加权平均，
离散/token 动作则直接选择得分最高的候选。
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def build_recovery_target(
    candidate_actions,
    candidate_scores,
    baseline_score: float,
    eps_gain: float,
    tau_gain: float,
    top_b: int = 1,
    allow_weighted_average: bool = False,
    candidate_responses: Optional[list] = None,
):
    """根据候选动作分数构造 recovery target。

    Args:
        candidate_actions: 候选动作，可为连续动作数组。
        candidate_scores: 每个候选动作的 recoverability 分数。
        baseline_score: 原 rollout 在 near-failure 状态处的局部成功分数。
        eps_gain: 候选动作相对 baseline 至少要提升多少才算有效。
        tau_gain: 对正增益做 softmax 的温度。
        top_b: 只在正增益候选里保留前 top_b 个用于置信度/加权。
        allow_weighted_average: 连续动作可设 True；token 动作必须保持 False。
        candidate_responses: 离散/token action 的 response token，用于直接监督 log_prob。

    Returns:
        无正增益候选时返回 None；否则返回 a_star、confidence、gain 等字段。
    """
    scores = np.asarray(candidate_scores, dtype=np.float32)
    if len(scores) == 0:
        return None

    # 只把“比原局部动作更有恢复希望”的候选写进 recovery buffer。
    gains = scores - float(baseline_score)
    positive = np.flatnonzero(gains > eps_gain)
    if len(positive) == 0:
        return None

    # top_b 只控制聚合候选数量；最终 best_index 仍记录得分最高的候选。
    top_b = max(int(top_b), 1)
    positive = positive[np.argsort(gains[positive])[::-1]][:top_b]
    best_index = int(positive[np.argmax(gains[positive])])
    selected_gains = gains[positive]
    tau_gain = max(float(tau_gain), 1e-6)
    weights = np.exp(selected_gains / tau_gain)
    weights = weights / weights.sum()

    if allow_weighted_average:
        # 连续动作场景：可用正增益候选的 softmax 权重做加权平均。
        a_star = np.zeros_like(np.asarray(candidate_actions[positive[0]], dtype=np.float32))
        for weight, idx in zip(weights, positive):
            a_star = a_star + float(weight) * np.asarray(candidate_actions[idx], dtype=np.float32)
    elif candidate_responses is not None:
        # 当前 OpenVLA/OFT 路径使用 token response 做监督，因此选择 best token action。
        a_star = candidate_responses[best_index]
    else:
        # 兜底路径：没有 token response 时返回原动作表示里的 best candidate。
        a_star = candidate_actions[best_index]

    # 置信度使用 top_b 内 softmax 权重最大值，越接近 1 表示最优候选更明确。
    confidence = float(np.clip(weights.max(), 0.0, 1.0))
    return {
        "a_star": a_star,
        "confidence": confidence,
        "gain": float(gains[best_index]),
        "best_index": best_index,
        "weights": weights,
    }
