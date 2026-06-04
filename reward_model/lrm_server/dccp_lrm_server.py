"""
DCCP LRM server adapter

本文件为提供 completion 和 progress HTTP 接口
动态加载 Large-Reward-Models 官方 vlm_reward_server.py 中的模型加载与推理函数

主要接口：
    GET  /health
    GET  /completion
    POST /completion
    POST /completion_batch
    GET  /progress
    POST /progress
    POST /progress_batch

completion 对应 DCCP 中的 S_comp
progress 对应 DCCP 中的 S_prog
"""

from __future__ import annotations

import argparse
import base64
import importlib
import importlib.util
import io
import json
import os
import threading
from typing import Any

import numpy as np
from flask import Flask, jsonify, request
from PIL import Image


app = Flask(__name__)

backend = None
backend_lock = threading.Lock()
server_mode = "progress"
progress_backend = "robometer"


def import_official_lrm_backend(official_server_py: str | None):
    """导入官方 Large-Reward-Models 推理后端"""
    if official_server_py:
        official_server_py = os.path.abspath(official_server_py)
        if not os.path.exists(official_server_py):
            raise FileNotFoundError(f"official_server_py does not exist: {official_server_py}")

        spec = importlib.util.spec_from_file_location("dccp_official_lrm_backend", official_server_py)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"failed to create import spec for {official_server_py}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    return importlib.import_module("vlm_reward_server")


def load_backend(
    official_server_py: str | None,
    model_path: str,
    base_model_path: str,
    gpu_id: int,
):
    """加载官方 LRM 模型"""
    global backend

    backend = import_official_lrm_backend(official_server_py)

    if not hasattr(backend, "load_vlm_model"):
        raise AttributeError("official LRM backend must provide load_vlm_model")

    backend.load_vlm_model(
        model_path=model_path,
        base_model_path=base_model_path,
        gpu_id=gpu_id,
    )

    return backend


def extract_instruction(payload: dict[str, Any]) -> str:
    """从请求中提取语言指令"""
    for key in ["instruction", "task_description", "task", "prompt"]:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    metadata = payload.get("metadata")
    if isinstance(metadata, dict):
        for key in ["instruction", "task_description", "task"]:
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    raise KeyError("request payload is missing instruction or task_description")


def decode_image_item(item: Any) -> np.ndarray:
    """将单张图像解码成 uint8 HWC numpy array"""
    if isinstance(item, np.ndarray):
        return normalize_image_array(item)

    if isinstance(item, Image.Image):
        return np.asarray(item.convert("RGB"), dtype=np.uint8)

    if isinstance(item, dict):
        if "array" in item:
            return normalize_image_array(np.asarray(item["array"]))

        if "data" in item and "shape" in item:
            raw = base64.b64decode(item["data"])
            dtype = np.dtype(item.get("dtype", "uint8"))
            shape = tuple(item["shape"])
            array = np.frombuffer(raw, dtype=dtype).reshape(shape)
            return normalize_image_array(array)

        if "image" in item:
            return decode_image_item(item["image"])

    if isinstance(item, str):
        text = item
        if "," in text and text.strip().startswith("data:"):
            text = text.split(",", 1)[1]

        raw = base64.b64decode(text)

        try:
            image = Image.open(io.BytesIO(raw)).convert("RGB")
            return np.asarray(image, dtype=np.uint8)
        except Exception:
            raise ValueError("base64 string image must be encoded image bytes or dict with shape")

    raise TypeError(f"unsupported image item type: {type(item)}")


def normalize_image_array(array: np.ndarray) -> np.ndarray:
    """规范图像数组为 uint8 HWC"""
    array = np.asarray(array)

    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]

    if array.ndim != 3:
        raise ValueError(f"image array must be HWC, got shape {array.shape}")

    if array.shape[0] in [1, 3] and array.shape[-1] not in [1, 3]:
        array = np.transpose(array, (1, 2, 0))

    if array.dtype != np.uint8:
        max_value = float(array.max()) if array.size > 0 else 0.0
        if max_value <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)

    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)

    if array.shape[-1] != 3:
        raise ValueError(f"image array must have 3 channels, got shape {array.shape}")

    return array


def extract_frames(payload: dict[str, Any]) -> list[np.ndarray]:
    """从请求中提取视频帧"""
    for key in [
        "frames",
        "keyframes",
        "images",
        "video",
        "rollout_video",
        "branch_or_suffix_video",
        "branch_video",
        "suffix_video",
    ]:
        value = payload.get(key)
        if value is None:
            continue

        if isinstance(value, dict) and "frames" in value:
            value = value["frames"]

        if isinstance(value, list):
            frames = [decode_image_item(item) for item in value]
            if len(frames) > 0:
                return frames

        if isinstance(value, dict) and "data" in value and "shape" in value:
            array = decode_video_array(value)
            return [normalize_image_array(frame) for frame in array]

    if "image" in payload:
        return [decode_image_item(payload["image"])]

    raise KeyError("request payload is missing frames, video, keyframes, images or image")


def decode_video_array(item: dict[str, Any]) -> np.ndarray:
    """解码带 shape 的视频数组"""
    raw = base64.b64decode(item["data"])
    dtype = np.dtype(item.get("dtype", "uint8"))
    shape = tuple(item["shape"])
    array = np.frombuffer(raw, dtype=dtype).reshape(shape)

    if array.ndim != 4:
        raise ValueError(f"video array must be THWC, got shape {array.shape}")

    return array


def frame_list_to_pil(frames: list[np.ndarray]) -> list[Image.Image]:
    """将 numpy 帧列表转换为 PIL 图像列表"""
    return [
        Image.fromarray(normalize_image_array(frame)).convert("RGB")
        for frame in frames
    ]


def extract_batch_items(payload: Any) -> list[dict[str, Any]]:
    """从 batch 请求中提取 item 列表"""
    if isinstance(payload, list):
        return [item if isinstance(item, dict) else {"video": item} for item in payload]

    if not isinstance(payload, dict):
        raise TypeError("batch payload must be a dict or list")

    for key in ["items", "batch", "requests"]:
        value = payload.get(key)
        if isinstance(value, list):
            return [item if isinstance(item, dict) else {"video": item} for item in value]

    videos = (
        payload.get("videos")
        or payload.get("rollout_videos")
        or payload.get("branch_or_suffix_videos")
        or payload.get("branch_videos")
        or payload.get("suffix_videos")
    )
    instructions = payload.get("instructions") or payload.get("task_descriptions")
    metadata_list = payload.get("batch_metadata") or payload.get("metadata")

    if isinstance(videos, list):
        items = []
        for index, video in enumerate(videos):
            item = {"video": video}

            if isinstance(instructions, list) and index < len(instructions):
                item["instruction"] = instructions[index]
            elif isinstance(instructions, str):
                item["instruction"] = instructions
            elif "instruction" in payload:
                item["instruction"] = payload["instruction"]
            elif "task_description" in payload:
                item["task_description"] = payload["task_description"]

            if isinstance(metadata_list, list) and index < len(metadata_list):
                item["metadata"] = metadata_list[index]

            items.append(item)

        return items

    return [payload]


def score_completion_item(payload: dict[str, Any]) -> dict[str, Any]:
    """计算单条完整轨迹的 completion score"""
    if backend is None:
        raise RuntimeError("LRM backend is not loaded")

    instruction = extract_instruction(payload)
    frames = extract_frames(payload)
    final_frame = frames[-1]

    if not hasattr(backend, "compute_vlm_completion"):
        raise AttributeError("official LRM backend must provide compute_vlm_completion for completion mode")

    with backend_lock:
        result = backend.compute_vlm_completion(final_frame, instruction)

    score = float(result.get("score", 0.0))
    completed = bool(result.get("completed", score >= 0.5))

    return {
        "success": True,
        "score": int(score >= 0.5),
        "completion": int(score >= 0.5),
        "completion_score": int(score >= 0.5),
        "complete": completed,
        "response": result.get("response", ""),
        "reasoning": result.get("reasoning", ""),
    }


def score_progress_item(payload: dict[str, Any]) -> dict[str, Any]:
    """计算单条局部分支或 suffix 的 progress score"""
    if backend is None:
        raise RuntimeError("LRM backend is not loaded")

    instruction = extract_instruction(payload)
    frames = extract_frames(payload)

    with backend_lock:
        if progress_backend == "robometer" and hasattr(backend, "compute_vlm_robometer"):
            result = backend.compute_vlm_robometer(frame_list_to_pil(frames), instruction)
        elif progress_backend == "roboreward" and hasattr(backend, "compute_vlm_roboreward"):
            result = backend.compute_vlm_roboreward(frame_list_to_pil(frames), instruction)
        elif hasattr(backend, "compute_vlm_reward"):
            result = backend.compute_vlm_reward(
                image=frames[-1],
                task_description=instruction,
                reward_type="progress",
                initial_image=frames[0],
            )
        else:
            raise AttributeError(
                "official LRM backend must provide compute_vlm_robometer, "
                "compute_vlm_roboreward or compute_vlm_reward"
            )

    score = float(result.get("score", 0.0))
    score = max(0.0, min(1.0, score))

    return {
        "success": True,
        "score": score,
        "progress": score,
        "progress_score": score,
        "raw_score": result.get("raw_score", None),
        "response": result.get("response", ""),
        "progress_per_frame": result.get("progress_per_frame", []),
    }


@app.get("/health")
def health():
    """健康检查"""
    return jsonify(
        {
            "status": "healthy",
            "mode": server_mode,
            "progress_backend": progress_backend,
            "model_loaded": backend is not None,
        }
    )


@app.get("/completion")
def completion_get():
    """completion endpoint 检查"""
    return jsonify(
        {
            "status": "ok",
            "endpoint": "completion",
            "note": "Use POST for scoring",
        }
    )


@app.get("/progress")
def progress_get():
    """progress endpoint 检查"""
    return jsonify(
        {
            "status": "ok",
            "endpoint": "progress",
            "note": "Use POST for scoring",
        }
    )


@app.post("/completion")
def completion_post():
    """DCCP completion scoring endpoint"""
    try:
        payload = request.get_json(force=True)
        return jsonify(score_completion_item(payload))
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500


@app.post("/progress")
def progress_post():
    """DCCP progress scoring endpoint"""
    try:
        payload = request.get_json(force=True)
        return jsonify(score_progress_item(payload))
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500


@app.post("/completion_batch")
def completion_batch_post():
    """DCCP completion batch endpoint"""
    try:
        payload = request.get_json(force=True)
        items = extract_batch_items(payload)
        results = [score_completion_item(item) for item in items]
        scores = [int(result["score"]) for result in results]

        return jsonify(
            {
                "success": True,
                "scores": scores,
                "completion_scores": scores,
                "completions": scores,
                "results": results,
            }
        )
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500


@app.post("/progress_batch")
def progress_batch_post():
    """DCCP progress batch endpoint"""
    try:
        payload = request.get_json(force=True)
        items = extract_batch_items(payload)
        results = [score_progress_item(item) for item in items]
        scores = [float(result["score"]) for result in results]

        return jsonify(
            {
                "success": True,
                "scores": scores,
                "progress_scores": scores,
                "progress": scores,
                "results": results,
            }
        )
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500


def main() -> None:
    parser = argparse.ArgumentParser(description="DCCP LRM server adapter")
    parser.add_argument("--mode", type=str, choices=["completion", "progress"], required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--base_model_path", type=str, default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--official_server_py", type=str, default="")
    parser.add_argument(
        "--progress_backend",
        type=str,
        choices=["reward", "roboreward", "robometer"],
        default="reward",
    )

    args = parser.parse_args()

    global server_mode
    global progress_backend

    server_mode = args.mode
    progress_backend = args.progress_backend

    official_server_py = args.official_server_py.strip() or None

    load_backend(
        official_server_py=official_server_py,
        model_path=args.model_path,
        base_model_path=args.base_model_path,
        gpu_id=int(args.gpu_id),
    )

    print(
        json.dumps(
            {
                "event": "dccp_lrm_server_start",
                "mode": server_mode,
                "host": args.host,
                "port": args.port,
                "model_path": args.model_path,
                "base_model_path": args.base_model_path,
                "gpu_id": args.gpu_id,
                "progress_backend": progress_backend,
                "official_server_py": official_server_py,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )

    app.run(
        host=args.host,
        port=int(args.port),
        threaded=False,
    )


if __name__ == "__main__":
    main()