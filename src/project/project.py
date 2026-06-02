"""
Project model and ProjectManager for batch video editing.

A Project represents one trip/activity containing multiple raw video clips.
ProjectManager handles scanning directories, extracting metadata via ffprobe,
grouping clips by day, and persisting project.json.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass, field, asdict
from datetime import datetime, date
from pathlib import Path
from typing import Optional


# ── Data Classes ────────────────────────────────────────────────────

@dataclass
class BGMConfig:
    strategy: str = "multi_segment"         # "multi_segment" / "single"
    segments: list[BGMSegment] = field(default_factory=list)
    short_video_bgm: Optional[str] = None


@dataclass
class BGMSegment:
    segment_id: str
    audio_path: str
    day_idx: Optional[int] = None
    start_sec: float = 0.0
    duration_sec: float = 0.0
    fade_in_sec: float = 1.0
    fade_out_sec: float = 2.0


@dataclass
class Clip:
    clip_id: str                            # e.g. "DJI_20251002182137_0005_D"
    file_path: str                          # absolute path
    file_size_mb: float = 0.0
    duration_sec: float = 0.0
    width: int = 0
    height: int = 0
    resolution: str = ""                    # e.g. "3840x2160"
    fps: float = 0.0
    codec: str = ""
    bitrate_kbps: int = 0
    creation_time: str = ""                 # ISO 8601
    device: str = ""                        # "dji" / "nikon" / "unknown"
    has_audio: bool = False                 # 是否有音频流（DJI 航拍通常无音频）

    # preprocess result paths (relative to output_dir)
    scene_summaries_dir: Optional[str] = None
    shot_scenes_path: Optional[str] = None
    asr_path: Optional[str] = None
    captions_path: Optional[str] = None

    # quality
    quality_score: Optional[float] = None
    quality_flags: list[str] = field(default_factory=list)
    is_valid: bool = True


@dataclass
class Day:
    day_idx: int                            # 1-based
    date: str                               # "2025-10-02"
    title: str = ""                         # user-editable, e.g. "西宁 → 青海湖"
    day_label: str = ""                     # 自动生成的标签，如 "Day 1 (11月14日)"
    clip_ids: list[str] = field(default_factory=list)
    summary: Optional[str] = None


@dataclass
class Project:
    project_id: str
    name: str
    base_dir: str                           # source video directory
    output_dir: str                         # "Output/Projects/<project_id>"
    created_at: str = ""                    # ISO 8601

    clips: list[Clip] = field(default_factory=list)
    days: list[Day] = field(default_factory=list)

    output_profiles: list[str] = field(default_factory=lambda: ["bilibili_4k"])
    bgm_config: BGMConfig = field(default_factory=BGMConfig)

    preprocess_progress: dict[str, str] = field(default_factory=dict)  # clip_id -> status
    metadata_path: str = ""                 # project.json path

    def get_day_summary(self) -> str:
        """Return a human-readable summary grouped by day, for instruction generation."""
        total_dur = sum(c.duration_sec for c in self.clips)
        total_dur_min = total_dur / 60
        lines = [
            f"项目: {self.name} ({len(self.clips)} 个视频, 总时长 {total_dur_min:.1f} 分钟)",
        ]
        clip_map = {c.clip_id: c for c in self.clips}
        for day in self.days:
            day_clips = [clip_map[cid] for cid in day.clip_ids if cid in clip_map]
            day_dur = sum(c.duration_sec for c in day_clips) / 60
            title = day.title or day.date or f"Day {day.day_idx}"
            lines.append(f"Day {day.day_idx} ({title}): {len(day_clips)} 个视频, {day_dur:.1f} 分钟")
        return "\n".join(lines)


# ── FFprobe helpers ─────────────────────────────────────────────────

VIDEO_EXTENSIONS = {".mp4", ".mov"}


def _run_ffprobe(file_path: str) -> dict:
    """Run ffprobe and return parsed JSON metadata."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_format", "-show_streams",
        file_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {file_path}: {result.stderr}")
    return json.loads(result.stdout)


def _extract_clip_metadata(file_path: str, clip_id: str) -> Clip:
    """Extract metadata from a single video file via ffprobe."""
    info = _run_ffprobe(file_path)
    fmt = info.get("format", {})
    streams = info.get("streams", [])

    video_stream = next((s for s in streams if s.get("codec_type") == "video"), {})
    audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), {})

    # duration
    duration = float(fmt.get("duration", 0) or video_stream.get("duration", 0) or 0)

    # resolution
    width = int(video_stream.get("width", 0) or 0)
    height = int(video_stream.get("height", 0) or 0)

    # fps — r_frame_rate is "60000/1001" style
    fps = 0.0
    rfr = video_stream.get("r_frame_rate", "0/1")
    if "/" in rfr:
        num, den = rfr.split("/")
        if float(den) > 0:
            fps = round(float(num) / float(den), 3)
    else:
        fps = float(rfr)

    # codec
    codec = video_stream.get("codec_name", "")

    # bitrate
    bitrate = int(fmt.get("bit_rate", 0) or 0) // 1000  # kbps

    # creation_time — from stream tags or format tags
    creation_time = ""
    for tag_source in [video_stream, fmt]:
        tags = tag_source.get("tags", {})
        ct = tags.get("creation_time", "") or tags.get("date", "") or tags.get("com.apple.quicktime.creationdate", "")
        if ct:
            creation_time = ct
            break
    # Try to parse from filename if not in metadata
    if not creation_time:
        creation_time = _parse_creation_time_from_filename(file_path)

    # device detection
    device = _detect_device(file_path, fmt, video_stream)

    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)

    # resolution 字符串（如 "3840x2160"）
    resolution = f"{width}x{height}" if width and height else ""

    # has_audio 检测
    has_audio = bool(audio_stream) and audio_stream.get("codec_name") not in ("", None)

    return Clip(
        clip_id=clip_id,
        file_path=os.path.abspath(file_path),
        file_size_mb=round(file_size_mb, 2),
        duration_sec=round(duration, 3),
        width=width,
        height=height,
        resolution=resolution,
        fps=fps,
        codec=codec,
        bitrate_kbps=bitrate,
        creation_time=creation_time,
        device=device,
        has_audio=has_audio,
    )


def _parse_creation_time_from_filename(file_path: str) -> str:
    """Try to extract date from DJI-style filename: DJI_YYYYMMDDHHMMSS_XXXX_D.MP4"""
    basename = os.path.basename(file_path)
    m = re.search(r'(\d{8})(\d{6})', basename)
    if m:
        try:
            dt = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
            return dt.isoformat()
        except ValueError:
            pass
    return ""


def _detect_device(file_path: str, fmt: dict, video_stream: dict) -> str:
    """Detect camera device from filename or metadata."""
    basename = os.path.basename(file_path).upper()
    if basename.startswith("DJI"):
        return "dji"
    if basename.startswith("DSC") or basename.startswith("DSCN"):
        return "nikon"
    make = video_stream.get("tags", {}).get("make", "").lower()
    if "dji" in make:
        return "dji"
    if "nikon" in make:
        return "nikon"
    return "unknown"


def _generate_clip_id(file_path: str) -> str:
    """Generate a clip_id from filename, stripping extension."""
    stem = Path(file_path).stem
    # Remove leading dots (macOS hidden files) and replace spaces
    stem = re.sub(r'^\.+', '', stem)
    return stem.replace(" ", "_")


# ── ProjectManager ──────────────────────────────────────────────────

class ProjectManager:
    """Create, load, and save Projects."""

    @staticmethod
    def create_from_directory(
        video_dir: str,
        project_name: str = "",
        group_by: str = "day",
        output_root: str = "Output/Projects",
    ) -> Project:
        """
        Scan a directory of video files, extract metadata, group by day,
        and produce a Project.
        """
        video_dir = os.path.abspath(video_dir)
        if not os.path.isdir(video_dir):
            raise FileNotFoundError(f"Video directory not found: {video_dir}")

        # Scan for video files, skip macOS hidden files
        video_files = []
        for root, _dirs, files in os.walk(video_dir):
            for f in sorted(files):
                if f.startswith("._"):
                    continue
                ext = os.path.splitext(f)[1].lower()
                if ext in VIDEO_EXTENSIONS:
                    video_files.append(os.path.join(root, f))

        if not video_files:
            raise ValueError(f"No video files found in {video_dir}")

        # Derive project_id from directory name
        dir_name = os.path.basename(video_dir)
        # Clean: "20251001青甘小环线" → "20251001_青甘小环线"
        project_id = re.sub(r'(\d{8})', r'\1_', dir_name, count=1).rstrip("_")
        if not project_name:
            project_name = dir_name

        output_dir = os.path.join(output_root, project_id)
        os.makedirs(output_dir, exist_ok=True)

        # Extract metadata for each clip
        clips: list[Clip] = []
        for fp in video_files:
            clip_id = _generate_clip_id(fp)
            print(f"  📎 Probing: {os.path.basename(fp)} ... ", end="", flush=True)
            try:
                clip = _extract_clip_metadata(fp, clip_id)
                clips.append(clip)
                print(f"{clip.duration_sec:.1f}s, {clip.width}x{clip.height}, {clip.codec}")
            except Exception as e:
                print(f"⚠️ skipped ({e})")

        if not clips:
            raise ValueError("No valid video clips could be read")

        # Group by day
        days = _group_clips_by_day(clips) if group_by == "day" else []

        project = Project(
            project_id=project_id,
            name=project_name,
            base_dir=video_dir,
            output_dir=output_dir,
            created_at=datetime.now().isoformat(),
            clips=clips,
            days=days,
            metadata_path=os.path.join(output_dir, "project.json"),
        )

        # Initialize preprocess progress
        for clip in clips:
            project.preprocess_progress[clip.clip_id] = "pending"

        ProjectManager.save(project)
        return project

    @staticmethod
    def load(project_path: str) -> Project:
        """Load a Project from project.json (支持目录路径自动拼接)。"""
        # 如果传入的是目录，自动拼接 project.json
        if os.path.isdir(project_path):
            project_path = os.path.join(project_path, "project.json")

        if not os.path.exists(project_path):
            raise FileNotFoundError(f"project.json 不存在: {project_path}")

        with open(project_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 修正：None 字段 fallback 到空 list，避免反序列化时 TypeError
        clips = [Clip(**c) for c in (data.get("clips") or [])]
        days = [Day(**d) for d in (data.get("days") or [])]
        bgm_data = data.get("bgm_config") or {}
        segments = [BGMSegment(**s) for s in (bgm_data.get("segments") or [])]
        bgm_config = BGMConfig(
            strategy=bgm_data.get("strategy", "multi_segment"),
            segments=segments,
            short_video_bgm=bgm_data.get("short_video_bgm"),
        )

        project = Project(
            project_id=data["project_id"],
            name=data["name"],
            base_dir=data["base_dir"],
            output_dir=data["output_dir"],
            created_at=data.get("created_at", ""),
            clips=clips,
            days=days,
            output_profiles=data.get("output_profiles", ["bilibili_4k"]),
            bgm_config=bgm_config,
            preprocess_progress=data.get("preprocess_progress", {}),
            metadata_path=project_path,
        )
        return project

    @staticmethod
    def save(project: Project):
        """Persist project to project.json."""
        os.makedirs(project.output_dir, exist_ok=True)
        path = project.metadata_path or os.path.join(project.output_dir, "project.json")
        data = asdict(project)
        data["metadata_path"] = path
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        project.metadata_path = path

    @staticmethod
    def status_summary(project: Project) -> str:
        """Return a human-readable status summary."""
        total = len(project.clips)
        done = sum(1 for s in project.preprocess_progress.values() if s == "done")
        failed = sum(1 for s in project.preprocess_progress.values() if s == "failed")
        pending = total - done - failed

        total_dur = sum(c.duration_sec for c in project.clips)
        total_size = sum(c.file_size_mb for c in project.clips)

        lines = [
            f"Project: {project.name} ({project.project_id})",
            f"Source:  {project.base_dir}",
            f"Output:  {project.output_dir}",
            f"Clips:   {total} files, {total_dur / 60:.1f} min, {total_size / 1024:.1f} GB",
            f"Days:    {len(project.days)}",
            f"Progress: {done} done, {failed} failed, {pending} pending",
            f"Profiles: {', '.join(project.output_profiles)}",
        ]

        # Day breakdown
        for day in project.days:
            day_clips = [c for c in project.clips if c.clip_id in day.clip_ids]
            day_dur = sum(c.duration_sec for c in day_clips)
            title = day.title or day.date or f"Day {day.day_idx}"
            lines.append(f"  Day {day.day_idx} ({title}): {len(day_clips)} clips, {day_dur / 60:.1f} min")

        return "\n".join(lines)


# ── Grouping ────────────────────────────────────────────────────────

def _group_clips_by_day(clips: list[Clip]) -> list[Day]:
    """按创建日期分组。无 creation_time 时 fallback 到文件 mtime。"""
    date_map: dict[str, list[str]] = {}

    for clip in clips:
        dt_str = clip.creation_time
        d = ""
        if dt_str:
            try:
                # Handle ISO format with or without timezone
                dt_str_clean = dt_str.split("+")[0].split("Z")[0]
                dt = datetime.fromisoformat(dt_str_clean)
                d = dt.date().isoformat()
            except (ValueError, AttributeError):
                d = ""

        # Fallback: 使用文件 mtime
        if not d and clip.file_path and os.path.exists(clip.file_path):
            try:
                mtime = os.path.getmtime(clip.file_path)
                d = datetime.fromtimestamp(mtime).date().isoformat()
            except OSError:
                d = "unsorted"

        # 最终 fallback: unsorted 组
        if not d:
            d = "unsorted"

        date_map.setdefault(d, []).append(clip.clip_id)

    # Sort by date, unsorted goes last
    sorted_dates = sorted(date_map.keys(), key=lambda x: (x in ("unknown", "unsorted"), x))

    days = []
    for idx, d in enumerate(sorted_dates, start=1):
        # 自动生成 day_label
        if d in ("unknown", "unsorted"):
            day_label = f"Day {idx} (未知日期)"
        else:
            day_label = f"Day {idx} ({d})"

        days.append(Day(
            day_idx=idx,
            date=d,
            day_label=day_label,
            clip_ids=date_map[d],
        ))

    return days
