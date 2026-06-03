#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多头 Multi-Resolution VideoMAE reward 独立训练脚本。

设计目标：
1. 保持 reward 训练主入口位于 `reward_model/`，与原版 `reward_model/videomae.py` 风格一致；
2. 训练共享 encoder 的双头 reward：
   - `R_traj`：完整 rollout 成功判断
   - `R_loc`：局部 progress / viability 打分
3. 不修改 PPO / policy 主训练循环，只提供独立的 reward 训练路径。

示例：
    torchrun --standalone --nproc_per_node=8 reward_model/train_multi_resolution_videomae.py \
      --train-pattern '/path/to/train/**/*.tar' \
      --val-pattern '/path/to/val/**/*.tar' \
      --expert-pattern '/path/to/expert/**/*.tar' \
      --init-reward-checkpoint /path/to/single_or_dual_head_reward.pth \
      --ckpt-dir ckpts_multi_resolution_reward
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
from collections import OrderedDict
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import VideoMAEConfig, VideoMAEForVideoClassification

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from videomae import SuccessWindowDataset, collate_fn, get_dist_env, set_seed  # noqa: E402
from verl.models.multi_resolution_reward import MultiResolutionVideoRewardModel  # noqa: E402
from verl.utils.multi_resolution_reward_dataset import MixedProgressWindowDataset, ProgressWindowDataset  # noqa: E402


DEFAULTS = dict(
    train_pattern="",
    val_pattern="",
    expert_pattern="",
    mix_weight_rollout=0.7,
    mix_weight_expert=0.3,
    img_size=224,
    window=8,
    stride_train=8,
    stride_val=1,
    batch_size=4,
    val_batch_size=64,
    num_workers=4,
    persistent_workers=True,
    prefetch_factor=2,
    lr=1e-4,
    weight_decay=1e-4,
    max_steps=200_000,
    eval_steps=1_000,
    ckpt_dir="ckpts_multi_resolution_reward",
    seed=42,
    model_name="MCG-NJU/videomae-base",
    local_files_only=True,
    thresh_min=0.3,
    thresh_max=1.0,
    thresh_steps=20,
    use_resample_train=True,
    drop_last=True,
    init_reward_checkpoint="",
    lambda_anchor=1.0,
    lambda_rank=0.5,
    lambda_fail=0.5,
    lambda_smooth=0.0,
    failed_tail_k=2,
    rank_num_pairs=2,
    progress_curve_bins=5,
)


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Cannot parse boolean value: {value}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train dual-head multi-resolution VideoMAE reward model.")
    parser.add_argument("--train-pattern", type=str, default=DEFAULTS["train_pattern"])
    parser.add_argument("--val-pattern", type=str, default=DEFAULTS["val_pattern"])
    parser.add_argument("--expert-pattern", type=str, default=DEFAULTS["expert_pattern"])
    parser.add_argument("--mix-weight-rollout", type=float, default=DEFAULTS["mix_weight_rollout"])
    parser.add_argument("--mix-weight-expert", type=float, default=DEFAULTS["mix_weight_expert"])
    parser.add_argument("--img-size", type=int, default=DEFAULTS["img_size"])
    parser.add_argument("--window", type=int, default=DEFAULTS["window"])
    parser.add_argument("--stride-train", type=int, default=DEFAULTS["stride_train"])
    parser.add_argument("--stride-val", type=int, default=DEFAULTS["stride_val"])
    parser.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"])
    parser.add_argument("--val-batch-size", type=int, default=DEFAULTS["val_batch_size"])
    parser.add_argument("--num-workers", type=int, default=DEFAULTS["num_workers"])
    parser.add_argument("--persistent-workers", type=str2bool, default=DEFAULTS["persistent_workers"])
    parser.add_argument("--prefetch-factor", type=int, default=DEFAULTS["prefetch_factor"])
    parser.add_argument("--lr", type=float, default=DEFAULTS["lr"])
    parser.add_argument("--weight-decay", type=float, default=DEFAULTS["weight_decay"])
    parser.add_argument("--max-steps", type=int, default=DEFAULTS["max_steps"])
    parser.add_argument("--eval-steps", type=int, default=DEFAULTS["eval_steps"])
    parser.add_argument("--ckpt-dir", type=str, default=DEFAULTS["ckpt_dir"])
    parser.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    parser.add_argument("--model-name", type=str, default=DEFAULTS["model_name"])
    parser.add_argument("--local-files-only", type=str2bool, default=DEFAULTS["local_files_only"])
    parser.add_argument("--thresh-min", type=float, default=DEFAULTS["thresh_min"])
    parser.add_argument("--thresh-max", type=float, default=DEFAULTS["thresh_max"])
    parser.add_argument("--thresh-steps", type=int, default=DEFAULTS["thresh_steps"])
    parser.add_argument("--use-resample-train", type=str2bool, default=DEFAULTS["use_resample_train"])
    parser.add_argument("--drop-last", type=str2bool, default=DEFAULTS["drop_last"])
    parser.add_argument("--init-reward-checkpoint", type=str, default=DEFAULTS["init_reward_checkpoint"])
    parser.add_argument("--lambda-anchor", type=float, default=DEFAULTS["lambda_anchor"])
    parser.add_argument("--lambda-rank", type=float, default=DEFAULTS["lambda_rank"])
    parser.add_argument("--lambda-fail", type=float, default=DEFAULTS["lambda_fail"])
    parser.add_argument("--lambda-smooth", type=float, default=DEFAULTS["lambda_smooth"])
    parser.add_argument("--failed-tail-k", type=int, default=DEFAULTS["failed_tail_k"])
    parser.add_argument("--rank-num-pairs", type=int, default=DEFAULTS["rank_num_pairs"])
    parser.add_argument("--progress-curve-bins", type=int, default=DEFAULTS["progress_curve_bins"])
    return parser.parse_args()


def maybe_unwrap_state_dict(payload):
    if isinstance(payload, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in payload and isinstance(payload[key], dict):
                return payload[key]
    return payload


def binary_classification_metrics(trues, preds):
    if len(trues) != len(preds):
        raise ValueError("trues and preds must have the same length")
    total = max(len(trues), 1)
    tp = sum(int(p == 1 and t == 1) for p, t in zip(preds, trues))
    tn = sum(int(p == 0 and t == 0) for p, t in zip(preds, trues))
    fp = sum(int(p == 1 and t == 0) for p, t in zip(preds, trues))
    fn = sum(int(p == 0 and t == 1) for p, t in zip(preds, trues))
    acc = float((tp + tn) / total)
    precision = float(tp / max(tp + fp, 1))
    recall = float(tp / max(tp + fn, 1))
    if precision + recall == 0.0:
        f1 = 0.0
    else:
        f1 = float(2.0 * precision * recall / (precision + recall))
    return {
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
    }


def build_model(args, device):
    cfg = VideoMAEConfig.from_pretrained(
        args.model_name,
        num_frames=args.window,
        num_labels=2,
        local_files_only=args.local_files_only,
    )
    base_model = VideoMAEForVideoClassification.from_pretrained(
        args.model_name,
        config=cfg,
        local_files_only=args.local_files_only,
    ).to(device)
    model = MultiResolutionVideoRewardModel.from_videomae_classifier(
        base_model,
        use_multi_resolution_reward=True,
    ).to(device)

    if args.init_reward_checkpoint:
        state_dict = maybe_unwrap_state_dict(torch.load(args.init_reward_checkpoint, map_location="cpu"))
        is_dual = MultiResolutionVideoRewardModel.is_dual_head_checkpoint(state_dict)
        model.load_reward_state_dict(state_dict, strict=False)
        return model, is_dual
    return model, False


def build_train_dataset(args, train_shards, expert_shards):
    if expert_shards:
        return MixedProgressWindowDataset(
            train_shards,
            expert_shards,
            weights=(args.mix_weight_rollout, args.mix_weight_expert),
            clip_len=args.window,
            stride=args.stride_train,
            img_size=args.img_size,
            rank_num_pairs=args.rank_num_pairs,
            failed_tail_k=args.failed_tail_k,
            progress_curve_bins=args.progress_curve_bins,
        )
    return ProgressWindowDataset(
        train_shards,
        clip_len=args.window,
        stride=args.stride_train,
        img_size=args.img_size,
        rank_num_pairs=args.rank_num_pairs,
        failed_tail_k=args.failed_tail_k,
        progress_curve_bins=args.progress_curve_bins,
        use_resample=args.use_resample_train,
    )


def build_dataloaders(args):
    train_shards = sorted(glob.glob(args.train_pattern, recursive=True))
    val_shards = sorted(glob.glob(args.val_pattern, recursive=True))
    expert_shards = sorted(glob.glob(args.expert_pattern, recursive=True)) if args.expert_pattern else []

    if len(train_shards) == 0:
        raise RuntimeError(f"--train-pattern yielded no shards: {args.train_pattern}")
    if len(val_shards) == 0:
        raise RuntimeError(f"--val-pattern yielded no shards: {args.val_pattern}")
    if args.expert_pattern and len(expert_shards) == 0:
        raise RuntimeError(f"--expert-pattern yielded no shards: {args.expert_pattern}")

    train_dataset = build_train_dataset(args, train_shards, expert_shards)
    traj_val_dataset = SuccessWindowDataset(
        shard_globs=val_shards,
        window=args.window,
        stride=args.stride_val,
        img_size=args.img_size,
        mode="val",
        use_resample=False,
    )
    loc_val_dataset = ProgressWindowDataset(
        val_shards,
        clip_len=args.window,
        stride=args.stride_val,
        img_size=args.img_size,
        rank_num_pairs=args.rank_num_pairs,
        failed_tail_k=args.failed_tail_k,
        progress_curve_bins=args.progress_curve_bins,
        use_resample=False,
    )

    persistent = args.persistent_workers and args.num_workers > 0
    prefetch = args.prefetch_factor if args.num_workers > 0 else None
    common_kwargs = dict(
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=persistent,
    )
    if prefetch is not None:
        common_kwargs["prefetch_factor"] = prefetch

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        drop_last=args.drop_last,
        **common_kwargs,
    )
    traj_val_loader = DataLoader(
        traj_val_dataset,
        batch_size=args.val_batch_size,
        collate_fn=collate_fn,
        drop_last=False,
        **common_kwargs,
    )
    loc_val_loader = DataLoader(
        loc_val_dataset,
        batch_size=args.val_batch_size,
        drop_last=False,
        **common_kwargs,
    )
    return train_loader, traj_val_loader, loc_val_loader, train_shards, val_shards, expert_shards


def compute_multi_resolution_reward_losses(model, batch, device, args):
    traj_clip = batch["traj_clip"].to(device, non_blocking=True)
    traj_label = batch["traj_label"].to(device, non_blocking=True)
    traj_logits = model(pixel_values=traj_clip, head="traj").logits
    loss_done = nn.CrossEntropyLoss()(traj_logits, traj_label)

    bce = nn.BCEWithLogitsLoss()

    anchor_clip = batch["anchor_clip"].to(device, non_blocking=True)
    anchor_label = batch["anchor_label"].to(device, non_blocking=True)
    anchor_logits = model(pixel_values=anchor_clip, head="loc").logits
    loss_anchor = bce(anchor_logits.float(), anchor_label.float())

    rank_valid = batch["rank_valid"].to(device, non_blocking=True).float()
    rank_early = batch["rank_early_clips"].to(device, non_blocking=True)
    rank_late = batch["rank_late_clips"].to(device, non_blocking=True)
    batch_size, pair_count = rank_early.shape[:2]
    rank_early_logits = model(
        pixel_values=rank_early.reshape(-1, *rank_early.shape[2:]),
        head="loc",
    ).logits.reshape(batch_size, pair_count)
    rank_late_logits = model(
        pixel_values=rank_late.reshape(-1, *rank_late.shape[2:]),
        head="loc",
    ).logits.reshape(batch_size, pair_count)
    rank_margin = rank_late_logits - rank_early_logits
    rank_loss_matrix = torch.nn.functional.softplus(-rank_margin)
    rank_mask = rank_valid[:, None]
    rank_denom = rank_mask.sum().clamp_min(1.0) * pair_count
    loss_rank = (rank_loss_matrix * rank_mask).sum() / rank_denom

    fail_valid = batch["fail_valid"].to(device, non_blocking=True).float()
    fail_tail = batch["fail_tail_clips"].to(device, non_blocking=True)
    fail_k = fail_tail.shape[1]
    fail_logits = model(
        pixel_values=fail_tail.reshape(-1, *fail_tail.shape[2:]),
        head="loc",
    ).logits.reshape(batch_size, fail_k)
    fail_targets = torch.zeros_like(fail_logits)
    fail_loss_matrix = torch.nn.functional.binary_cross_entropy_with_logits(
        fail_logits.float(),
        fail_targets,
        reduction="none",
    )
    fail_mask = fail_valid[:, None]
    fail_denom = fail_mask.sum().clamp_min(1.0) * fail_k
    loss_fail = (fail_loss_matrix * fail_mask).sum() / fail_denom

    loss_smooth = loss_done.new_zeros(())
    if float(args.lambda_smooth) > 0.0:
        smooth_valid = batch["smooth_valid"].to(device, non_blocking=True).float()
        smooth_prev = batch["smooth_prev_clip"].to(device, non_blocking=True)
        smooth_next = batch["smooth_next_clip"].to(device, non_blocking=True)
        smooth_prev_logits = model(pixel_values=smooth_prev, head="loc").logits
        smooth_next_logits = model(pixel_values=smooth_next, head="loc").logits
        smooth_diff = (torch.sigmoid(smooth_next_logits) - torch.sigmoid(smooth_prev_logits)).abs()
        loss_smooth = (smooth_diff * smooth_valid).sum() / smooth_valid.sum().clamp_min(1.0)

    loss_total = (
        loss_done
        + float(args.lambda_anchor) * loss_anchor
        + float(args.lambda_rank) * loss_rank
        + float(args.lambda_fail) * loss_fail
        + float(args.lambda_smooth) * loss_smooth
    )
    return {
        "loss_total": loss_total,
        "loss_done": loss_done.detach(),
        "loss_anchor": loss_anchor.detach(),
        "loss_rank": loss_rank.detach(),
        "loss_fail": loss_fail.detach(),
        "loss_smooth": loss_smooth.detach(),
    }


@torch.no_grad()
def evaluate_traj_ddp(model, loader, device, rank, world_size, args):
    model.eval()
    logits_local, trues_local = [], []

    for vids, ys, _ in tqdm(loader, desc="ValTraj", disable=(rank != 0)):
        vids = vids.to(device, non_blocking=True)
        ys = ys.to(device, non_blocking=True)
        logits = model(pixel_values=vids, head="traj").logits
        logits_local.extend(logits.cpu().tolist())
        trues_local.extend(ys.cpu().tolist())

    logits_gather, trues_gather = [None] * world_size, [None] * world_size
    dist.all_gather_object(logits_gather, logits_local)
    dist.all_gather_object(trues_gather, trues_local)

    if rank != 0:
        return None

    logits = [x for part in logits_gather for x in part]
    trues = [x for part in trues_gather for x in part]
    probs = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()

    thresholds = np.linspace(args.thresh_min, args.thresh_max, args.thresh_steps)
    all_metrics = {}
    best = {"f1": -1.0, "thresh": float(thresholds[0])}

    for th in thresholds:
        preds = (probs >= th).astype(np.int32).tolist()
        metrics = binary_classification_metrics(trues, preds)
        all_metrics[f"thresh_{th:.2f}"] = OrderedDict(
            acc=metrics["acc"],
            precision=metrics["precision"],
            recall=metrics["recall"],
            f1=metrics["f1"],
        )
        if metrics["f1"] > best["f1"]:
            best["f1"] = float(metrics["f1"])
            best["thresh"] = float(th)
    return all_metrics, best


@torch.no_grad()
def evaluate_loc_ddp(model, loader, device, rank, world_size, curve_bins):
    model.eval()
    rank_correct_local = 0.0
    rank_total_local = 0.0
    tail_score_sum_local = 0.0
    tail_count_local = 0.0
    curve_sum_local = None
    curve_count_local = 0.0

    for batch in tqdm(loader, desc="ValLoc", disable=(rank != 0)):
        rank_valid = batch["rank_valid"].to(device, non_blocking=True).float()
        rank_early = batch["rank_early_clips"].to(device, non_blocking=True)
        rank_late = batch["rank_late_clips"].to(device, non_blocking=True)
        batch_size, pair_count = rank_early.shape[:2]
        early_logits = model(
            pixel_values=rank_early.reshape(-1, *rank_early.shape[2:]),
            head="loc",
        ).logits.reshape(batch_size, pair_count)
        late_logits = model(
            pixel_values=rank_late.reshape(-1, *rank_late.shape[2:]),
            head="loc",
        ).logits.reshape(batch_size, pair_count)
        rank_correct_local += (((late_logits > early_logits).float() * rank_valid[:, None]).sum().item())
        rank_total_local += float(rank_valid.sum().item() * pair_count)

        fail_valid = batch["fail_valid"].to(device, non_blocking=True).float()
        fail_tail = batch["fail_tail_clips"].to(device, non_blocking=True)
        fail_k = fail_tail.shape[1]
        fail_logits = model(
            pixel_values=fail_tail.reshape(-1, *fail_tail.shape[2:]),
            head="loc",
        ).logits.reshape(batch_size, fail_k)
        fail_scores = torch.sigmoid(fail_logits)
        tail_score_sum_local += float((fail_scores * fail_valid[:, None]).sum().item())
        tail_count_local += float(fail_valid.sum().item() * fail_k)

        curve_valid = batch["progress_curve_valid"].to(device, non_blocking=True).float()
        curve_clips = batch["progress_curve_clips"].to(device, non_blocking=True)
        bins = curve_clips.shape[1]
        curve_logits = model(
            pixel_values=curve_clips.reshape(-1, *curve_clips.shape[2:]),
            head="loc",
        ).logits.reshape(batch_size, bins)
        curve_scores = torch.sigmoid(curve_logits)
        if curve_sum_local is None:
            curve_sum_local = torch.zeros(bins, device=device, dtype=torch.float32)
        curve_sum_local += (curve_scores * curve_valid[:, None]).sum(dim=0)
        curve_count_local += float(curve_valid.sum().item())

    rank_tensor = torch.tensor([rank_correct_local, rank_total_local], device=device, dtype=torch.float32)
    tail_tensor = torch.tensor([tail_score_sum_local, tail_count_local], device=device, dtype=torch.float32)
    if curve_sum_local is None:
        curve_sum_local = torch.zeros(curve_bins, device=device, dtype=torch.float32)
    curve_count_tensor = torch.tensor([curve_count_local], device=device, dtype=torch.float32)

    dist.all_reduce(rank_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(tail_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(curve_sum_local, op=dist.ReduceOp.SUM)
    dist.all_reduce(curve_count_tensor, op=dist.ReduceOp.SUM)

    if rank != 0:
        return None

    curve_count = max(curve_count_tensor[0].item(), 1.0)
    return {
        "ranking_accuracy": float(rank_tensor[0].item() / max(rank_tensor[1].item(), 1.0)),
        "tail_negative_score": float(tail_tensor[0].item() / max(tail_tensor[1].item(), 1.0)),
        "progress_curve": [float(x) for x in (curve_sum_local / curve_count).detach().cpu().tolist()],
    }


def print_loc_metrics(prefix: str, metrics: Optional[Dict[str, Any]]):
    if metrics is None:
        return
    curve = ", ".join(f"{x:.4f}" for x in metrics["progress_curve"])
    print(
        f"{prefix} ranking_accuracy={metrics['ranking_accuracy']:.4f} "
        f"tail_negative_score={metrics['tail_negative_score']:.4f} "
        f"progress_curve=[{curve}]"
    )


def save_checkpoint(model, optimizer, args, global_step, traj_best, loc_metrics, rank, is_best=False):
    if rank != 0:
        return None
    os.makedirs(args.ckpt_dir, exist_ok=True)
    suffix = "best" if is_best else f"step{global_step}"
    ckpt_path = os.path.join(args.ckpt_dir, f"multi_resolution_reward_{suffix}.pth")
    payload = {
        "model_state_dict": model.module.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "global_step": global_step,
        "traj_best": traj_best,
        "loc_metrics": loc_metrics,
        "train_args": vars(args),
    }
    torch.save(payload, ckpt_path)
    print(f"[Checkpoint] saved -> {ckpt_path}")
    return ckpt_path


def main():
    args = parse_args()
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    world_size, rank, local_rank, device = get_dist_env()
    if rank == 0:
        print(f"[DDP] world_size={world_size}")
        print(
            "[multi-reward] "
            f"window={args.window} img_size={args.img_size} "
            f"lambda_anchor={args.lambda_anchor} lambda_rank={args.lambda_rank} "
            f"lambda_fail={args.lambda_fail} lambda_smooth={args.lambda_smooth}"
        )

    train_loader, traj_val_loader, loc_val_loader, train_shards, val_shards, expert_shards = build_dataloaders(args)
    if rank == 0:
        print(f"Train shards: {len(train_shards)}")
        print(f"Val   shards: {len(val_shards)}")
        print(f"Expert shards: {len(expert_shards)}")

    model, init_was_dual = build_model(args, device)
    if rank == 0 and args.init_reward_checkpoint:
        init_mode = "dual-head checkpoint" if init_was_dual else "single-head checkpoint (bootstrap loc_head)"
        print(f"[Init] loaded {init_mode}: {args.init_reward_checkpoint}")

    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    os.makedirs(args.ckpt_dir, exist_ok=True) if rank == 0 else None
    global_step = 0
    best_traj_f1 = -1.0

    train_iter = iter(train_loader)
    while global_step < args.max_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            continue

        model.train()
        losses = compute_multi_resolution_reward_losses(model.module, batch, device, args)
        optimizer.zero_grad(set_to_none=True)
        losses["loss_total"].backward()
        optimizer.step()

        global_step += 1
        if rank == 0 and global_step % 10 == 0:
            print(
                f"[step {global_step}] "
                f"loss_total={losses['loss_total'].item():.4f} "
                f"loss_done={losses['loss_done'].item():.4f} "
                f"loss_anchor={losses['loss_anchor'].item():.4f} "
                f"loss_rank={losses['loss_rank'].item():.4f} "
                f"loss_fail={losses['loss_fail'].item():.4f} "
                f"loss_smooth={losses['loss_smooth'].item():.4f}"
            )

        if global_step % args.eval_steps != 0:
            continue

        traj_out = evaluate_traj_ddp(model.module, traj_val_loader, device, rank, world_size, args)
        loc_out = evaluate_loc_ddp(model.module, loc_val_loader, device, rank, world_size, args.progress_curve_bins)
        if rank == 0 and traj_out is not None:
            all_metrics, best = traj_out
            print(f"\n[Val @ step {global_step}]")
            for key, value in all_metrics.items():
                print(
                    f"{key}: acc={value['acc']:.4f} "
                    f"prec={value['precision']:.4f} rec={value['recall']:.4f} f1={value['f1']:.4f}"
                )
            print(f"[reward_traj] best_f1={best['f1']:.4f} @ thresh={best['thresh']:.2f}")
            print_loc_metrics("[reward_loc]", loc_out)
            save_checkpoint(model, optimizer, args, global_step, best, loc_out, rank, is_best=False)
            if best["f1"] > best_traj_f1:
                best_traj_f1 = best["f1"]
                save_checkpoint(model, optimizer, args, global_step, best, loc_out, rank, is_best=True)
        dist.barrier()

    if rank == 0:
        print(f"[Done] total steps={global_step}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
