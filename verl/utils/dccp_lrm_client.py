"""
DCCP LRM client.

本文件实现 LRM scoring interface。
训练进程通过 HTTP endpoint 请求 completion 和 progress 分数，不在本进程加载 LRM 权重。

接口含义：
    score_completion: S_comp(Γ_traj(τ), instruction) -> {0, 1}
    score_progress:   S_prog(Γ_loc(ρ), instruction) -> [0, 1]

completion 用于完整 imagined trajectory 的任务完成判断。
progress 用于 short suffix 或 counterfactual branch 的局部进度判断。
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
from PIL import Image


@dataclass
class DCCPLRMClientConfig:
    completion_endpoint: str
    progress_endpoint: str

    completion_batch_endpoint: Optional[str] = None
    progress_batch_endpoint: Optional[str] = None

    timeout_sec: float = 60.0
    max_retries: int = 2
    retry_sleep_sec: float = 1.0

    use_cache: bool = True
    cache_dir: str = "./cache/lrm_scores"
    jpeg_quality: int = 95


class DCCPLRMClient:
    """HTTP client for DCCP completion/progress scoring."""

    def __init__(self, config: DCCPLRMClientConfig):
        self.config = config
        self.cache_dir = Path(config.cache_dir)

        if self.config.use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.cache_hits = 0
        self.cache_misses = 0

    @classmethod
    def from_config_dict(cls, cfg: dict[str, Any]) -> "DCCPLRMClient":
        return cls(
            DCCPLRMClientConfig(
                completion_endpoint=str(cfg.get("completion_endpoint", "")),
                progress_endpoint=str(cfg.get("progress_endpoint", "")),
                completion_batch_endpoint=cfg.get("completion_batch_endpoint"),
                progress_batch_endpoint=cfg.get("progress_batch_endpoint"),
                timeout_sec=float(cfg.get("timeout_sec", 60.0)),
                max_retries=int(cfg.get("max_retries", 2)),
                retry_sleep_sec=float(cfg.get("retry_sleep_sec", 1.0)),
                use_cache=bool(cfg.get("use_cache", True)),
                cache_dir=str(cfg.get("cache_dir", "./cache/lrm_scores")),
                jpeg_quality=int(cfg.get("jpeg_quality", 95)),
            )
        )

    def score_completion(
        self,
        frames: Iterable[Any],
        instruction: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> int:
        """Return binary completion label for a full trajectory."""
        payload = self._build_payload(frames, instruction, metadata)
        cached = self._read_cache("completion", payload)
        if cached is not None:
            return int(cached["value"])

        response = self._post_json(self.config.completion_endpoint, payload)
        value = self._parse_completion_response(response)
        self._write_cache("completion", payload, {"value": int(value), "raw": response})
        return int(value)

    def score_progress(
        self,
        frames: Iterable[Any],
        instruction: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> float:
        """Return continuous progress score for a local suffix or branch."""
        payload = self._build_payload(frames, instruction, metadata)
        cached = self._read_cache("progress", payload)
        if cached is not None:
            return float(cached["value"])

        response = self._post_json(self.config.progress_endpoint, payload)
        value = self._parse_progress_response(response)
        value = float(np.clip(value, 0.0, 1.0))
        self._write_cache("progress", payload, {"value": value, "raw": response})
        return value

    def score_completion_batch(
        self,
        batch_frames: list[Iterable[Any]],
        batch_instructions: list[str],
        batch_metadata: Optional[list[dict[str, Any]]] = None,
    ) -> list[int]:
        """Return binary completion labels for a batch of trajectories."""
        self._check_batch_lengths(batch_frames, batch_instructions, batch_metadata)

        if batch_metadata is None:
            batch_metadata = [{} for _ in batch_frames]

        if self.config.completion_batch_endpoint:
            payloads = [
                self._build_payload(frames, instruction, metadata)
                for frames, instruction, metadata in zip(batch_frames, batch_instructions, batch_metadata)
            ]
            response = self._post_json(self.config.completion_batch_endpoint, {"items": payloads})
            return [int(self._parse_completion_response(item)) for item in self._as_response_list(response)]

        return [
            self.score_completion(frames, instruction, metadata)
            for frames, instruction, metadata in zip(batch_frames, batch_instructions, batch_metadata)
        ]

    def score_progress_batch(
        self,
        batch_frames: list[Iterable[Any]],
        batch_instructions: list[str],
        batch_metadata: Optional[list[dict[str, Any]]] = None,
    ) -> list[float]:
        """Return continuous progress scores for a batch of local suffixes or branches."""
        self._check_batch_lengths(batch_frames, batch_instructions, batch_metadata)

        if batch_metadata is None:
            batch_metadata = [{} for _ in batch_frames]

        if self.config.progress_batch_endpoint:
            payloads = [
                self._build_payload(frames, instruction, metadata)
                for frames, instruction, metadata in zip(batch_frames, batch_instructions, batch_metadata)
            ]
            response = self._post_json(self.config.progress_batch_endpoint, {"items": payloads})
            return [
                float(np.clip(self._parse_progress_response(item), 0.0, 1.0))
                for item in self._as_response_list(response)
            ]

        return [
            self.score_progress(frames, instruction, metadata)
            for frames, instruction, metadata in zip(batch_frames, batch_instructions, batch_metadata)
        ]

    @staticmethod
    def _check_batch_lengths(
        batch_frames: list[Iterable[Any]],
        batch_instructions: list[str],
        batch_metadata: Optional[list[dict[str, Any]]],
    ) -> None:
        if len(batch_frames) != len(batch_instructions):
            raise ValueError(
                f"batch_frames and batch_instructions must have the same length, "
                f"got {len(batch_frames)} and {len(batch_instructions)}."
            )
        if batch_metadata is not None and len(batch_metadata) != len(batch_frames):
            raise ValueError(
                f"batch_metadata and batch_frames must have the same length, "
                f"got {len(batch_metadata)} and {len(batch_frames)}."
            )

    def _build_payload(
        self,
        frames: Iterable[Any],
        instruction: str,
        metadata: Optional[dict[str, Any]],
    ) -> dict[str, Any]:
        encoded_frames = [self._encode_frame_to_base64_jpeg(frame) for frame in frames]

        if len(encoded_frames) == 0:
            raise ValueError("LRM scoring requires at least one frame.")

        return {
            "instruction": str(instruction),
            "frames": encoded_frames,
            "frame_format": "base64_jpeg",
            "metadata": metadata or {},
        }

    def _encode_frame_to_base64_jpeg(self, frame: Any) -> str:
        image = self._to_pil_rgb(frame)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=int(self.config.jpeg_quality))
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    @staticmethod
    def _to_pil_rgb(frame: Any) -> Image.Image:
        if isinstance(frame, Image.Image):
            return frame.convert("RGB")

        array = np.asarray(frame)

        if array.ndim != 3:
            raise ValueError(f"Expected frame shape [H, W, C] or [C, H, W], got {array.shape}.")

        if array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
            array = np.transpose(array, (1, 2, 0))

        if array.shape[-1] == 1:
            array = np.repeat(array, 3, axis=-1)
        elif array.shape[-1] == 4:
            array = array[..., :3]
        elif array.shape[-1] != 3:
            raise ValueError(f"Expected 1, 3, or 4 channels, got frame shape {array.shape}.")

        if array.dtype != np.uint8:
            array = array.astype(np.float32)
            max_value = float(array.max()) if array.size > 0 else 0.0
            if max_value <= 1.0:
                array = array * 255.0
            array = np.clip(array, 0, 255).astype(np.uint8)

        return Image.fromarray(array).convert("RGB")

    def _post_json(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        if not endpoint:
            raise RuntimeError("LRM endpoint is empty.")

        body = json.dumps(payload).encode("utf-8")
        last_error: Optional[Exception] = None

        for attempt in range(int(self.config.max_retries) + 1):
            request = urllib.request.Request(
                endpoint,
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )

            try:
                with urllib.request.urlopen(request, timeout=float(self.config.timeout_sec)) as response:
                    response_body = response.read().decode("utf-8")
                    parsed = json.loads(response_body)
                    if not isinstance(parsed, dict):
                        raise TypeError(f"Expected JSON object from LRM server, got {type(parsed)}.")
                    return parsed
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError, TypeError) as exc:
                last_error = exc
                if attempt < int(self.config.max_retries):
                    time.sleep(float(self.config.retry_sleep_sec))

        raise RuntimeError(f"LRM request failed after retries. endpoint={endpoint}, error={last_error}")

    @staticmethod
    def _parse_completion_response(response: dict[str, Any]) -> int:
        for key in ("completion", "complete", "success", "label", "value", "score"):
            if key not in response:
                continue

            value = response[key]

            if isinstance(value, bool):
                return int(value)
            if isinstance(value, (int, np.integer)):
                return int(value > 0)
            if isinstance(value, (float, np.floating)):
                return int(float(value) >= 0.5)
            if isinstance(value, str):
                lowered = value.strip().lower()
                if lowered in ("1", "true", "yes", "success", "complete", "completed"):
                    return 1
                if lowered in ("0", "false", "no", "failure", "incomplete", "failed"):
                    return 0

        raise KeyError(f"Cannot parse completion response: {response}")

    @staticmethod
    def _parse_progress_response(response: dict[str, Any]) -> float:
        for key in ("progress", "score", "value"):
            if key in response:
                return float(response[key])

        raise KeyError(f"Cannot parse progress response: {response}")

    @staticmethod
    def _as_response_list(response: dict[str, Any]) -> list[dict[str, Any]]:
        for key in ("items", "results", "scores", "outputs"):
            if key in response:
                items = response[key]
                if not isinstance(items, list):
                    raise TypeError(f"Expected response['{key}'] to be a list, got {type(items)}.")
                return items

        raise KeyError(f"Cannot parse batch response: {response}")

    def _cache_key(self, mode: str, payload: dict[str, Any]) -> str:
        digest_payload = {
            "mode": mode,
            "instruction": payload.get("instruction", ""),
            "frames_sha256": [
                hashlib.sha256(frame.encode("utf-8")).hexdigest()
                for frame in payload.get("frames", [])
            ],
            "metadata": payload.get("metadata", {}),
        }
        raw = json.dumps(digest_payload, sort_keys=True).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def _cache_path(self, mode: str, payload: dict[str, Any]) -> Path:
        return self.cache_dir / f"{mode}_{self._cache_key(mode, payload)}.json"

    def _read_cache(self, mode: str, payload: dict[str, Any]) -> Optional[dict[str, Any]]:
        if not self.config.use_cache:
            return None

        path = self._cache_path(mode, payload)

        if not path.exists():
            self.cache_misses += 1
            return None

        try:
            with path.open("r", encoding="utf-8") as f:
                record = json.load(f)
            self.cache_hits += 1
            return record
        except json.JSONDecodeError:
            self.cache_misses += 1
            return None

    def _write_cache(self, mode: str, payload: dict[str, Any], record: dict[str, Any]) -> None:
        if not self.config.use_cache:
            return

        os.makedirs(self.cache_dir, exist_ok=True)
        path = self._cache_path(mode, payload)
        tmp_path = path.with_suffix(".tmp")

        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)

        tmp_path.replace(path)

    def cache_hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        if total == 0:
            return 0.0
        return float(self.cache_hits / total)