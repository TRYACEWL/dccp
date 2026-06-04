"""
DCCP action-token log-probability utilities

本文件 action-token 级别的 logprob、entropy 和 reference gap 计算

核心公式：
    log π(a | x) = Σ_u log π(z_u | x, z_<u)

用于 decision-sensitive state mining 的 entropy：
    h_t = 1/U Σ_u H_Vact(π(. | x_t, z_t,<u))

用于 preference bank 的 reference gap：
    Δ_ref = log π_ref(a_w | x) - log π_ref(a_l | x)

本文件只处理 action response positions 上的 logits 和 tokens
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn.functional as F


@dataclass
class DCCPActionLogprobResult:
    """Action-token logprob result"""

    token_logprobs: torch.Tensor
    sequence_logprobs: torch.Tensor
    response_mask: torch.Tensor


@dataclass
class DCCPActionEntropyResult:
    """Action-token entropy result"""

    token_entropies: torch.Tensor
    sequence_entropies: torch.Tensor
    response_mask: torch.Tensor


@dataclass
class DCCPReferenceGapResult:
    """Winner-loser reference gap result"""

    winner_logprobs: torch.Tensor
    loser_logprobs: torch.Tensor
    delta_ref: torch.Tensor


def build_action_vocab_mask(
    vocab_size: int,
    device: torch.device,
    action_token_ids: Optional[Any] = None,
    action_token_begin: Optional[int] = None,
    action_token_end: Optional[int] = None,
) -> torch.Tensor:
    """构造 action-token vocabulary mask"""
    vocab_size = int(vocab_size)

    if vocab_size <= 0:
        raise ValueError(f"vocab_size must be positive, got {vocab_size}")

    mask = torch.zeros(vocab_size, dtype=torch.bool, device=device)

    if action_token_ids is not None:
        ids = torch.as_tensor(action_token_ids, dtype=torch.long, device=device)
        if ids.numel() == 0:
            raise ValueError("action_token_ids must not be empty")
        if torch.any(ids < 0) or torch.any(ids >= vocab_size):
            raise ValueError("action_token_ids contains indices outside vocabulary range")
        mask[ids] = True

    if action_token_begin is not None or action_token_end is not None:
        if action_token_begin is None or action_token_end is None:
            raise ValueError("action_token_begin and action_token_end must be provided together")

        begin = int(action_token_begin)
        end = int(action_token_end)

        if begin < 0 or end > vocab_size or begin >= end:
            raise ValueError(
                f"invalid action token interval [{begin}, {end}) for vocab_size={vocab_size}"
            )

        mask[begin:end] = True

    if not torch.any(mask):
        raise ValueError("action vocabulary mask is empty")

    return mask


def compute_action_logprobs_from_logits(
    logits: torch.Tensor,
    target_tokens: torch.Tensor,
    response_mask: torch.Tensor,
    action_vocab_mask: Optional[torch.Tensor] = None,
) -> DCCPActionLogprobResult:
    """计算 action response tokens 的 logprob"""
    logits, target_tokens, response_mask = _prepare_logprob_inputs(
        logits=logits,
        target_tokens=target_tokens,
        response_mask=response_mask,
    )

    if action_vocab_mask is not None:
        logits = mask_logits_to_action_vocab(logits, action_vocab_mask)

    log_probs = F.log_softmax(logits.float(), dim=-1)
    token_logprobs = torch.gather(
        log_probs,
        dim=-1,
        index=target_tokens.unsqueeze(-1),
    ).squeeze(-1)

    masked_token_logprobs = token_logprobs * response_mask
    sequence_logprobs = masked_token_logprobs.sum(dim=-1)

    return DCCPActionLogprobResult(
        token_logprobs=masked_token_logprobs,
        sequence_logprobs=sequence_logprobs,
        response_mask=response_mask,
    )


def compute_action_entropy_from_logits(
    logits: torch.Tensor,
    response_mask: torch.Tensor,
    action_vocab_mask: Optional[torch.Tensor] = None,
) -> DCCPActionEntropyResult:
    """计算 action-token entropy"""
    logits, response_mask = _prepare_entropy_inputs(
        logits=logits,
        response_mask=response_mask,
    )

    if action_vocab_mask is not None:
        logits = mask_logits_to_action_vocab(logits, action_vocab_mask)

    log_probs = F.log_softmax(logits.float(), dim=-1)
    probs = torch.exp(log_probs)

    token_entropies = -(probs * log_probs).sum(dim=-1)
    token_entropies = torch.nan_to_num(token_entropies, nan=0.0, posinf=0.0, neginf=0.0)

    masked_token_entropies = token_entropies * response_mask
    denominator = response_mask.sum(dim=-1).clamp_min(1.0)
    sequence_entropies = masked_token_entropies.sum(dim=-1) / denominator

    return DCCPActionEntropyResult(
        token_entropies=masked_token_entropies,
        sequence_entropies=sequence_entropies,
        response_mask=response_mask,
    )


def compute_reference_gap_from_logits(
    winner_logits: torch.Tensor,
    loser_logits: torch.Tensor,
    winner_tokens: torch.Tensor,
    loser_tokens: torch.Tensor,
    winner_response_mask: torch.Tensor,
    loser_response_mask: Optional[torch.Tensor] = None,
    action_vocab_mask: Optional[torch.Tensor] = None,
) -> DCCPReferenceGapResult:
    """根据 reference policy logits 计算 Δ_ref"""
    if loser_response_mask is None:
        loser_response_mask = winner_response_mask

    winner_result = compute_action_logprobs_from_logits(
        logits=winner_logits,
        target_tokens=winner_tokens,
        response_mask=winner_response_mask,
        action_vocab_mask=action_vocab_mask,
    )

    loser_result = compute_action_logprobs_from_logits(
        logits=loser_logits,
        target_tokens=loser_tokens,
        response_mask=loser_response_mask,
        action_vocab_mask=action_vocab_mask,
    )

    delta_ref = winner_result.sequence_logprobs - loser_result.sequence_logprobs

    return DCCPReferenceGapResult(
        winner_logprobs=winner_result.sequence_logprobs,
        loser_logprobs=loser_result.sequence_logprobs,
        delta_ref=delta_ref,
    )


def compute_reference_gap_from_logprobs(
    winner_logprobs: torch.Tensor,
    loser_logprobs: torch.Tensor,
) -> torch.Tensor:
    """根据已经计算好的 sequence logprobs 计算 Δ_ref"""
    winner_logprobs = _as_float_tensor(winner_logprobs)
    loser_logprobs = _as_float_tensor(loser_logprobs)

    if winner_logprobs.shape != loser_logprobs.shape:
        raise ValueError(
            f"winner_logprobs and loser_logprobs must have the same shape, "
            f"got {tuple(winner_logprobs.shape)} and {tuple(loser_logprobs.shape)}"
        )

    return winner_logprobs - loser_logprobs


def compute_winner_loser_gap_from_logits(
    winner_logits: torch.Tensor,
    loser_logits: torch.Tensor,
    winner_tokens: torch.Tensor,
    loser_tokens: torch.Tensor,
    winner_response_mask: torch.Tensor,
    loser_response_mask: Optional[torch.Tensor] = None,
    action_vocab_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """计算当前策略下 winner-loser logprob gap"""
    if loser_response_mask is None:
        loser_response_mask = winner_response_mask

    winner_result = compute_action_logprobs_from_logits(
        logits=winner_logits,
        target_tokens=winner_tokens,
        response_mask=winner_response_mask,
        action_vocab_mask=action_vocab_mask,
    )

    loser_result = compute_action_logprobs_from_logits(
        logits=loser_logits,
        target_tokens=loser_tokens,
        response_mask=loser_response_mask,
        action_vocab_mask=action_vocab_mask,
    )

    return winner_result.sequence_logprobs - loser_result.sequence_logprobs


def mask_logits_to_action_vocab(
    logits: torch.Tensor,
    action_vocab_mask: torch.Tensor,
) -> torch.Tensor:
    """将非 action-token vocabulary 的 logits 屏蔽掉"""
    if action_vocab_mask.dtype != torch.bool:
        action_vocab_mask = action_vocab_mask.to(dtype=torch.bool)

    if action_vocab_mask.ndim != 1:
        raise ValueError(f"action_vocab_mask must be 1D, got shape {tuple(action_vocab_mask.shape)}")

    if logits.shape[-1] != action_vocab_mask.numel():
        raise ValueError(
            f"logits vocabulary size and action_vocab_mask length mismatch, "
            f"got {logits.shape[-1]} and {action_vocab_mask.numel()}"
        )

    mask = action_vocab_mask.to(device=logits.device)

    while mask.ndim < logits.ndim:
        mask = mask.unsqueeze(0)

    return logits.masked_fill(~mask, torch.finfo(logits.dtype).min)


def validate_action_tokens(
    target_tokens: torch.Tensor,
    action_vocab_mask: torch.Tensor,
    response_mask: torch.Tensor,
) -> None:
    """检查 response positions 上的 target tokens 是否属于 action-token vocabulary"""
    target_tokens = target_tokens.long()
    response_mask = response_mask.bool()

    if action_vocab_mask.dtype != torch.bool:
        action_vocab_mask = action_vocab_mask.to(dtype=torch.bool)

    action_vocab_mask = action_vocab_mask.to(device=target_tokens.device)

    if target_tokens.ndim != response_mask.ndim:
        raise ValueError(
            f"target_tokens and response_mask must have the same ndim, "
            f"got {target_tokens.ndim} and {response_mask.ndim}"
        )

    if target_tokens.shape != response_mask.shape:
        raise ValueError(
            f"target_tokens and response_mask must have the same shape, "
            f"got {tuple(target_tokens.shape)} and {tuple(response_mask.shape)}"
        )

    valid_positions = response_mask
    if not torch.any(valid_positions):
        return

    selected_tokens = target_tokens[valid_positions]

    if torch.any(selected_tokens < 0) or torch.any(selected_tokens >= action_vocab_mask.numel()):
        raise ValueError("target_tokens contains indices outside vocabulary range")

    valid_token_mask = action_vocab_mask[selected_tokens]

    if not torch.all(valid_token_mask):
        invalid_count = int((~valid_token_mask).sum().item())
        raise ValueError(f"{invalid_count} response tokens are outside action-token vocabulary")


def _prepare_logprob_inputs(
    logits: torch.Tensor,
    target_tokens: torch.Tensor,
    response_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logits = _as_float_tensor(logits)
    target_tokens = _as_long_tensor(target_tokens, device=logits.device)
    response_mask = _as_float_tensor(response_mask, device=logits.device)

    if logits.ndim != target_tokens.ndim + 1:
        raise ValueError(
            f"logits ndim must equal target_tokens ndim + 1, "
            f"got logits ndim={logits.ndim}, target_tokens ndim={target_tokens.ndim}"
        )

    if logits.shape[:-1] != target_tokens.shape:
        raise ValueError(
            f"logits prefix shape must match target_tokens shape, "
            f"got {tuple(logits.shape[:-1])} and {tuple(target_tokens.shape)}"
        )

    if target_tokens.shape != response_mask.shape:
        raise ValueError(
            f"target_tokens and response_mask must have the same shape, "
            f"got {tuple(target_tokens.shape)} and {tuple(response_mask.shape)}"
        )

    if torch.any(target_tokens < 0) or torch.any(target_tokens >= logits.shape[-1]):
        raise ValueError("target_tokens contains indices outside logits vocabulary dimension")

    response_mask = response_mask.float()
    return logits, target_tokens, response_mask


def _prepare_entropy_inputs(
    logits: torch.Tensor,
    response_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits = _as_float_tensor(logits)
    response_mask = _as_float_tensor(response_mask, device=logits.device)

    if logits.ndim != response_mask.ndim + 1:
        raise ValueError(
            f"logits ndim must equal response_mask ndim + 1, "
            f"got logits ndim={logits.ndim}, response_mask ndim={response_mask.ndim}"
        )

    if logits.shape[:-1] != response_mask.shape:
        raise ValueError(
            f"logits prefix shape must match response_mask shape, "
            f"got {tuple(logits.shape[:-1])} and {tuple(response_mask.shape)}"
        )

    return logits, response_mask.float()


def _as_float_tensor(value: Any, device: Optional[torch.device] = None) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value
        if device is not None:
            tensor = tensor.to(device=device)
        return tensor.float()

    return torch.as_tensor(value, dtype=torch.float32, device=device)


def _as_long_tensor(value: Any, device: Optional[torch.device] = None) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value
        if device is not None:
            tensor = tensor.to(device=device)
        return tensor.long()

    return torch.as_tensor(value, dtype=torch.long, device=device)