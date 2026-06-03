import argparse
import glob
import io
import json
import os
import tarfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import VideoMAEConfig, VideoMAEForVideoClassification

try:
    from transformers import VideoMAEImageProcessor as VideoMAEFeatureExtractor
except Exception:
    from transformers import VideoMAEFeatureExtractor


def iter_tar_samples(patterns):
    paths = []
    for pattern in patterns:
        paths.extend(glob.glob(pattern, recursive=True))
    for path in sorted(set(paths)):
        with tarfile.open(path) as tar:
            members = {m.name: m for m in tar.getmembers() if m.isfile()}
            keys = sorted(name[: -len(".video.npy")] for name in members if name.endswith(".video.npy"))
            for key in keys:
                video = np.load(io.BytesIO(tar.extractfile(members[f"{key}.video.npy"]).read()), allow_pickle=False)
                meta = json.loads(tar.extractfile(members[f"{key}.meta.json"]).read().decode("utf-8"))
                yield path, key, video, meta


def make_clips(video, window, stride, min_steps):
    total_frames = len(video)
    clips = []
    for end in range(total_frames, min_steps + window - 1, -stride):
        clips.append((video[end - window : end], end - window, end))
    clips.reverse()
    return clips


@torch.no_grad()
def predict_video(video, model, fe, device, args):
    clips = make_clips(video, args.window, args.stride, args.min_steps)
    best_prob = 0.0
    first_finish = len(video) - 1
    pred_complete = False
    for i in range(0, len(clips), args.batch_size):
        batch = clips[i : i + args.batch_size]
        frames = [[Image.fromarray(f.astype(np.uint8)).convert("RGB") for f in clip] for clip, _, _ in batch]
        inputs = fe(frames, return_tensors="pt")["pixel_values"].to(device)
        logits = model(pixel_values=inputs).logits
        if args.prob_mode == "softmax":
            probs = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()
        else:
            probs = torch.sigmoid(logits)[:, 1].detach().cpu().numpy()
        for (_, _start, end), prob in zip(batch, probs):
            best_prob = max(best_prob, float(prob))
            if prob >= args.threshold:
                pred_complete = True
                first_finish = end - 1
                return pred_complete, first_finish, best_prob
    return pred_complete, first_finish, best_prob


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patterns", nargs="+", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--prob-mode", choices=["softmax", "sigmoid"], default="softmax")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--min-steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-videos", type=int, default=0)
    parser.add_argument("--out-json", default="")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fe = VideoMAEFeatureExtractor.from_pretrained("MCG-NJU/videomae-base", size=args.img_size)
    cfg = VideoMAEConfig.from_pretrained("MCG-NJU/videomae-base", num_frames=args.window, num_labels=2)
    model = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-base", config=cfg).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu"), strict=True)
    model.eval()

    rows = []
    for idx, (path, key, video, meta) in enumerate(iter_tar_samples(args.patterns), 1):
        true_complete = bool(meta.get("complete", False))
        pred_complete, pred_finish, best_prob = predict_video(video, model, fe, device, args)
        rows.append(
            {
                "path": path,
                "key": key,
                "true_complete": true_complete,
                "pred_complete": pred_complete,
                "true_finish_step": int(meta.get("finish_step", -1)),
                "pred_finish_step": int(pred_finish),
                "best_prob": best_prob,
            }
        )
        if args.max_videos and idx >= args.max_videos:
            break

    y_true = [int(r["true_complete"]) for r in rows]
    y_pred = [int(r["pred_complete"]) for r in rows]
    pred_success = sum(y_pred)
    true_success = sum(y_true)
    metrics = {
        "ckpt": args.ckpt,
        "threshold": args.threshold,
        "prob_mode": args.prob_mode,
        "total": len(rows),
        "true_success": true_success,
        "true_success_rate": true_success / len(rows) if rows else 0.0,
        "pred_success": pred_success,
        "pred_success_rate": pred_success / len(rows) if rows else 0.0,
        "accuracy": accuracy_score(y_true, y_pred) if rows else 0.0,
        "precision": precision_score(y_true, y_pred, zero_division=0) if rows else 0.0,
        "recall": recall_score(y_true, y_pred, zero_division=0) if rows else 0.0,
        "f1": f1_score(y_true, y_pred, zero_division=0) if rows else 0.0,
    }
    print(json.dumps(metrics, indent=2))
    if args.out_json:
        Path(os.path.dirname(args.out_json) or ".").mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump({"metrics": metrics, "rows": rows}, f, indent=2)


if __name__ == "__main__":
    main()
