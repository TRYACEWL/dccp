"""局部 recovery search 工具。

这里刻意使用 callable 接口，而不是新增一套复杂基类：
rollout worker 可以把当前已有的 policy/world model 调用方式作为函数传进来。
这样第一版改动更小，也更容易兼容后续不同 action 表示或 world model 实现。
"""

from __future__ import annotations

from typing import Callable, Iterable, Optional

import numpy as np


def sample_candidate_actions(policy, xt, lang=None, num_candidates: int = 4):
    """从当前策略采样候选第一步动作。

    xt/lang 是局部边界状态和语言条件。函数既支持直接传 callable，
    也支持传带有 ``sample_candidate_actions`` 方法的对象。
    """
    if callable(policy):
        return policy(xt, lang, num_candidates)
    if hasattr(policy, "sample_candidate_actions"):
        return policy.sample_candidate_actions(xt, lang, num_candidates)
    raise TypeError("policy must be callable or expose sample_candidate_actions")


def evaluate_recoverability(
    world_model,
    policy,
    reward_scorer,
    xt,
    lang,
    candidate_actions: Iterable,
    horizon_hr: int,
    num_mc_samples: int,
    rollout_fn: Optional[Callable] = None,
    progress_callback: Optional[Callable] = None,
    artifact_callback: Optional[Callable] = None,
) -> np.ndarray:
    """评估每个候选第一步动作的 recoverability。

    逻辑是：
    1. 对每个候选动作，固定它作为第一步；
    2. 后续 ``horizon_hr - 1`` 或短视界内的动作仍由当前 policy 采样；
    3. 用 reward_scorer 对短 rollout 打分；
    4. 多次 Monte Carlo 评估后取平均，减少单次 world model 随机性的影响。
    5. progress_callback 是可选的细粒度日志回调，主要用于 rollout worker
       在耗时 search 期间持续打印进度，避免 Ray worker 长时间无输出。
    6. artifact_callback 是可选的视频导出回调。它不会改变搜索逻辑，只在调用方
       明确打开 debug/export 时，把某次候选 rollout 的视频片段交给调用方保存或缓存。
    """
    if rollout_fn is None and hasattr(world_model, "evaluate_recoverability"):
        return np.asarray(
            world_model.evaluate_recoverability(
                policy, reward_scorer, xt, lang, candidate_actions, horizon_hr, num_mc_samples
            ),
            dtype=np.float32,
        )
    if rollout_fn is None:
        raise TypeError("rollout_fn is required when world_model has no evaluate_recoverability method")

    scores = []
    for candidate_idx, candidate in enumerate(candidate_actions):
        mc_scores = []
        for mc_idx in range(max(int(num_mc_samples), 1)):
            if progress_callback is not None:
                progress_callback(candidate_idx, mc_idx, "start", None)
            # rollout_fn 由具体 worker 提供，这里只约定返回可被 reward_scorer 打分的视频片段。
            video = rollout_fn(xt, lang, candidate, horizon_hr)
            if hasattr(reward_scorer, "score_loc_full_trajectory"):
                score = reward_scorer.score_loc_full_trajectory(video)
            else:
                score = reward_scorer.score_full_trajectory(video)
            if artifact_callback is not None:
                artifact_callback(candidate_idx, mc_idx, video, score)
            mc_scores.append(score)
            if progress_callback is not None:
                progress_callback(candidate_idx, mc_idx, "done", score)
        scores.append(float(np.mean(mc_scores)) if mc_scores else 0.0)
    return np.asarray(scores, dtype=np.float32)
