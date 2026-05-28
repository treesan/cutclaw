"""
Platform-specific media output profiles.

Inspired by OpenMontage/lib/media_profiles.py — frozen dataclass with
ffmpeg_output_args() to auto-generate encoder CLI flags.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class AspectRatio(str, Enum):
    LANDSCAPE_16_9 = "16:9"
    PORTRAIT_9_16 = "9:16"
    SQUARE_1_1 = "1:1"
    STANDARD_4_3 = "4:3"
    CINEMATIC_21_9 = "21:9"


@dataclass(frozen=True)
class MediaProfile:
    name: str
    width: int
    height: int
    aspect_ratio: AspectRatio
    fps: int
    codec: str              # ffmpeg encoder: "libx265" / "libx264"
    audio_codec: str        # "aac" / "mp3"
    crf: int
    pixel_format: str = "yuv420p"
    preset: str = "medium"
    max_file_size_mb: Optional[float] = None
    max_duration_seconds: Optional[float] = None
    notes: str = ""


# ── Built-in Profiles ──────────────────────────────────────────────

BILIBILI_4K = MediaProfile(
    name="bilibili_4k",
    width=3840, height=2160,
    aspect_ratio=AspectRatio.LANDSCAPE_16_9,
    fps=60, codec="libx265", audio_codec="aac", crf=16,
    pixel_format="yuv420p10le", preset="slow",
    max_duration_seconds=900,
    notes="B站超高清, HEVC源获超高清标签流量加持",
)

BILIBILI_1080P = MediaProfile(
    name="bilibili_1080p",
    width=1920, height=1080,
    aspect_ratio=AspectRatio.LANDSCAPE_16_9,
    fps=60, codec="libx264", audio_codec="aac", crf=20,
    max_duration_seconds=900,
    notes="B站高清, 兼容性最好",
)

DOUYIN = MediaProfile(
    name="douyin",
    width=1080, height=1920,
    aspect_ratio=AspectRatio.PORTRAIT_9_16,
    fps=30, codec="libx264", audio_codec="aac", crf=22,
    max_duration_seconds=60,
    notes="抖音短视频",
)

XIAOHONGSHU = MediaProfile(
    name="xiaohongshu",
    width=1080, height=1440,
    aspect_ratio=AspectRatio.STANDARD_4_3,
    fps=30, codec="libx264", audio_codec="aac", crf=22,
    max_duration_seconds=300,
    notes="小红书图文视频",
)

ALL_PROFILES: dict[str, MediaProfile] = {
    p.name: p for p in [BILIBILI_4K, BILIBILI_1080P, DOUYIN, XIAOHONGSHU]
}


def get_profile(name: str) -> MediaProfile:
    if name not in ALL_PROFILES:
        available = ", ".join(ALL_PROFILES.keys())
        raise ValueError(f"Unknown profile {name!r}. Available: {available}")
    return ALL_PROFILES[name]


def ffmpeg_output_args(profile: MediaProfile) -> list[str]:
    """Generate ffmpeg output encoding args from a MediaProfile."""
    return [
        "-c:v", profile.codec,
        "-c:a", profile.audio_codec,
        "-crf", str(profile.crf),
        "-preset", profile.preset,
        "-pix_fmt", profile.pixel_format,
        "-r", str(profile.fps),
    ]
