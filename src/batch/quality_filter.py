"""
Clip-level quality filter — fast sampling to detect garbage footage.

Samples a few frames per clip and scores blur, exposure, and stability.
Inspired by OpenMontage source_media_review + slideshow_risk.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None  # degrade gracefully; blur/exposure checks skipped


@dataclass
class QualityResult:
    clip_id: str
    score: float = 0.0          # 0-100, higher = better
    flags: list[str] = field(default_factory=list)
    is_valid: bool = True
    details: dict = field(default_factory=dict)


class QualityFilter:
    """
    Fast clip-level quality assessment.

    Sampling strategy: extract N frames evenly distributed across the clip
    via ffmpeg, then compute per-frame metrics.
    """

    def __init__(
        self,
        clip_threshold: int = 30,
        blur_threshold: float = 50.0,
        dark_threshold: float = 30.0,
        bright_threshold: float = 225.0,
        min_duration_sec: float = 1.0,
    ):
        self.clip_threshold = clip_threshold
        self.blur_threshold = blur_threshold
        self.dark_threshold = dark_threshold
        self.bright_threshold = bright_threshold
        self.min_duration_sec = min_duration_sec

    def evaluate_clip(
        self,
        clip_path: str,
        clip_id: str = "",
        duration_sec: float = 0.0,
        sample_frames: int = 10,
    ) -> QualityResult:
        """
        Evaluate a single clip by sampling frames.

        Returns QualityResult with score (0-100) and flags.
        """
        if not clip_id:
            clip_id = os.path.splitext(os.path.basename(clip_path))[0]

        result = QualityResult(clip_id=clip_id)

        # Duration check
        if duration_sec > 0 and duration_sec < self.min_duration_sec:
            result.flags.append("too_short")
            result.score = 0
            result.is_valid = False
            return result

        # Extract sample frames
        frames = self._extract_sample_frames(clip_path, sample_frames)
        if not frames:
            result.flags.append("no_frames_extracted")
            result.score = 0
            result.is_valid = False
            return result

        # Compute metrics
        blur_scores = []
        exposure_scores = []

        for frame in frames:
            if cv2 is not None:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
                # Blur: Laplacian variance
                laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
                blur_scores.append(laplacian_var)

                # Exposure: mean brightness
                mean_brightness = float(np.mean(gray))
                exposure_scores.append(mean_brightness)
            else:
                # Fallback: numpy-based rough estimate
                gray = np.mean(frame, axis=2) if len(frame.shape) == 3 else frame
                mean_brightness = float(np.mean(gray))
                exposure_scores.append(mean_brightness)

        # Scoring
        score = 100.0

        # Blur penalty
        if blur_scores:
            avg_blur = np.mean(blur_scores)
            result.details["avg_blur_laplacian"] = round(float(avg_blur), 2)
            if avg_blur < self.blur_threshold:
                penalty = min(40, (self.blur_threshold - avg_blur) / self.blur_threshold * 40)
                score -= penalty
                if avg_blur < self.blur_threshold * 0.3:
                    result.flags.append("very_blurry")
                else:
                    result.flags.append("blurry")

        # Exposure penalty
        if exposure_scores:
            avg_exposure = np.mean(exposure_scores)
            result.details["avg_brightness"] = round(float(avg_exposure), 2)
            if avg_exposure < self.dark_threshold:
                penalty = min(40, (self.dark_threshold - avg_exposure) / self.dark_threshold * 40)
                score -= penalty
                result.flags.append("too_dark")
            elif avg_exposure > self.bright_threshold:
                penalty = min(30, (avg_exposure - self.bright_threshold) / (255 - self.bright_threshold) * 30)
                score -= penalty
                result.flags.append("overexposed")

        result.score = round(max(0, score), 1)
        result.is_valid = result.score >= self.clip_threshold

        if not result.is_valid:
            result.flags.append("below_threshold")

        return result

    def _extract_sample_frames(self, clip_path: str, n: int) -> list[np.ndarray]:
        """Extract N evenly-spaced frames via ffmpeg."""
        if not os.path.exists(clip_path):
            return []

        # Get video duration first
        probe_cmd = [
            "ffprobe", "-v", "quiet",
            "-show_entries", "format=duration",
            "-of", "json",
            clip_path,
        ]
        try:
            probe_result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=15)
            probe_data = json.loads(probe_result.stdout)
            duration = float(probe_data.get("format", {}).get("duration", 0))
        except Exception:
            duration = 10.0

        if duration <= 0:
            return []

        # Calculate interval, skip first/last 5%
        start_t = duration * 0.05
        end_t = duration * 0.95
        if n <= 1:
            timestamps = [(start_t + end_t) / 2]
        else:
            step = (end_t - start_t) / (n - 1)
            timestamps = [start_t + i * step for i in range(n)]

        frames = []
        with tempfile.TemporaryDirectory(prefix="cutclaw_qf_") as tmpdir:
            for idx, ts in enumerate(timestamps):
                out_path = os.path.join(tmpdir, f"f{idx:03d}.jpg")
                cmd = [
                    "ffmpeg", "-v", "quiet",
                    "-ss", f"{ts:.3f}",
                    "-i", clip_path,
                    "-frames:v", "1",
                    "-vf", "scale=320:180",
                    "-y", out_path,
                ]
                try:
                    subprocess.run(cmd, capture_output=True, timeout=10)
                    if os.path.exists(out_path):
                        if cv2 is not None:
                            img = cv2.imread(out_path)
                            if img is not None:
                                frames.append(img)
                        else:
                            # Fallback: use PIL for image decoding
                            try:
                                from PIL import Image
                                img = np.array(Image.open(out_path))
                                if img is not None:
                                    frames.append(img)
                            except ImportError:
                                pass
                except Exception:
                    continue

        return frames


def evaluate_clip_quick(clip_path: str, clip_id: str = "", duration_sec: float = 0.0) -> QualityResult:
    """Convenience function for one-off clip evaluation."""
    qf = QualityFilter()
    return qf.evaluate_clip(clip_path, clip_id=clip_id, duration_sec=duration_sec)
