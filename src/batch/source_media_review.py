"""
Source media audit — per-project ffprobe review of all clips.

Checks for codec/resolution/fps/colorspace consistency across the project
and flags issues that would require normalization during rendering.
Inspired by OpenMontage/lib/source_media_review.py.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, field, asdict
from typing import Optional

from src.project.project import Project, Clip


@dataclass
class SourceReviewReport:
    project_id: str
    total_clips: int = 0
    issues: list[dict] = field(default_factory=list)
    resolution_variants: list[str] = field(default_factory=list)
    codec_variants: list[str] = field(default_factory=list)
    fps_variants: list[float] = field(default_factory=list)
    color_spaces: list[str] = field(default_factory=list)
    needs_normalization: bool = False


class SourceMediaReview:
    """
    Audit all clips in a project for consistency.

    For each clip:
    1. ffprobe deep probe (codec, resolution, fps, pixel format, colorspace)
    2. Flag: low res (<720p), mono audio, ultra-short (<3s), HDR/SDR mix
    3. Aggregate variants across project to decide if normalization is needed
    """

    def review_project(self, project: Project) -> SourceReviewReport:
        report = SourceReviewReport(project_id=project.project_id, total_clips=len(project.clips))

        resolutions = set()
        codecs = set()
        fps_vals = set()
        color_spaces = set()

        for clip in project.clips:
            self._review_clip(clip, report, resolutions, codecs, fps_vals, color_spaces)

        report.resolution_variants = sorted(resolutions)
        report.codec_variants = sorted(codecs)
        report.fps_variants = sorted(fps_vals)
        report.color_spaces = sorted(color_spaces)

        # Need normalization if >1 variant in any dimension
        report.needs_normalization = (
            len(resolutions) > 1 or
            len(codecs) > 1 or
            len(fps_vals) > 1 or
            len(color_spaces) > 1
        )

        return report

    def _review_clip(
        self,
        clip: Clip,
        report: SourceReviewReport,
        resolutions: set,
        codecs: set,
        fps_vals: set,
        color_spaces: set,
    ):
        """Probe a single clip and flag issues."""
        fp = clip.file_path
        if not os.path.exists(fp):
            report.issues.append({
                "clip_id": clip.clip_id,
                "severity": "error",
                "message": f"File not found: {fp}",
            })
            return

        try:
            info = self._ffprobe(fp)
        except Exception as e:
            report.issues.append({
                "clip_id": clip.clip_id,
                "severity": "error",
                "message": f"ffprobe failed: {e}",
            })
            return

        streams = info.get("streams", [])
        fmt = info.get("format", {})
        video = next((s for s in streams if s.get("codec_type") == "video"), {})
        audio = next((s for s in streams if s.get("codec_type") == "audio"), {})

        # Resolution
        w = int(video.get("width", 0) or 0)
        h = int(video.get("height", 0) or 0)
        res_str = f"{w}x{h}"
        if w > 0 and h > 0:
            resolutions.add(res_str)
            if h < 720:
                report.issues.append({
                    "clip_id": clip.clip_id,
                    "severity": "warning",
                    "message": f"Low resolution: {res_str}",
                })

        # Codec
        codec_name = video.get("codec_name", "")
        if codec_name:
            codecs.add(codec_name)

        # FPS
        rfr = video.get("r_frame_rate", "0/1")
        fps = 0.0
        if "/" in rfr:
            parts = rfr.split("/")
            if float(parts[1]) > 0:
                fps = round(float(parts[0]) / float(parts[1]), 3)
        else:
            fps = float(rfr)
        if fps > 0:
            fps_vals.add(round(fps, 1))

        # Color space / transfer
        colorspace = video.get("color_space", "") or video.get("color_transfer", "") or ""
        pix_fmt = video.get("pix_fmt", "")
        # Detect HDR via pixel format or transfer characteristics
        is_hdr = "bt2020" in colorspace.lower() or "pq" in colorspace.lower() or "hlg" in colorspace.lower()
        if is_hdr:
            color_spaces.add("bt2020_hdr")
        elif "bt709" in colorspace.lower():
            color_spaces.add("bt709")
        elif pix_fmt:
            color_spaces.add(f"unknown({pix_fmt})")

        # Audio check
        if not audio:
            report.issues.append({
                "clip_id": clip.clip_id,
                "severity": "warning",
                "message": "No audio stream",
            })
        else:
            channels = int(audio.get("channels", 0) or 0)
            if channels == 1:
                report.issues.append({
                    "clip_id": clip.clip_id,
                    "severity": "info",
                    "message": "Mono audio",
                })

        # Duration
        duration = float(fmt.get("duration", 0) or 0)
        if duration < 3.0:
            report.issues.append({
                "clip_id": clip.clip_id,
                "severity": "info",
                "message": f"Very short clip: {duration:.1f}s",
            })

    @staticmethod
    def _ffprobe(file_path: str) -> dict:
        cmd = [
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_format", "-show_streams",
            file_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            raise RuntimeError(f"ffprobe error: {result.stderr}")
        return json.loads(result.stdout)

    @staticmethod
    def save_report(report: SourceReviewReport, output_path: str):
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(asdict(report), f, ensure_ascii=False, indent=2)

    @staticmethod
    def format_report(report: SourceReviewReport) -> str:
        lines = [
            f"Source Media Review: {report.project_id}",
            f"  Total clips: {report.total_clips}",
            f"  Resolutions: {', '.join(report.resolution_variants) or 'N/A'}",
            f"  Codecs:      {', '.join(report.codec_variants) or 'N/A'}",
            f"  FPS:         {', '.join(str(f) for f in report.fps_variants) or 'N/A'}",
            f"  Color space: {', '.join(report.color_spaces) or 'N/A'}",
            f"  Needs normalization: {'YES' if report.needs_normalization else 'No'}",
        ]
        if report.issues:
            lines.append(f"  Issues ({len(report.issues)}):")
            for issue in report.issues:
                lines.append(f"    [{issue['severity']}] {issue['clip_id']}: {issue['message']}")
        return "\n".join(lines)
