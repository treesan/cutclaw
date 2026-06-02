"""
MaterialIndex — global flat scene index for the planner agent.

Aggregates scene summaries from all valid clips into a single searchable
index, breaking clip boundaries so the planner can select shots across
the entire project.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional

from src.project.project import Project


@dataclass
class SceneEntry:
    """One scene in the global material index."""
    scene_id: str                       # e.g. "S001_003" (globally unique)
    clip_id: str                        # source clip
    clip_file_path: str                 # absolute path for rendering
    scene_idx_in_clip: int

    start_sec: float
    end_sec: float
    duration_sec: float

    caption: str = ""
    location: str = ""
    emotion: str = ""
    shot_type: str = ""                 # "drone_aerial" / "WS" / "MS" / "CU"
    has_protagonist: bool = False
    has_dialogue: bool = False
    quality_score: float = 0.0

    day_idx: int = 0
    tags: list[str] = field(default_factory=list)


@dataclass
class MaterialIndex:
    """Global material index — the planner's data source."""
    project_id: str
    build_time: str = ""
    total_clips: int = 0
    valid_clips: int = 0
    total_scenes: int = 0
    total_duration_sec: float = 0.0
    scenes: list[SceneEntry] = field(default_factory=list)

    def __post_init__(self):
        if self.scenes and self.total_scenes == 0:
            self.total_scenes = len(self.scenes)
        if self.scenes and self.total_duration_sec == 0:
            self.total_duration_sec = sum(s.duration_sec for s in self.scenes)


def build_material_index(project: Project) -> MaterialIndex:
    """
    Scan all clips' scene_summaries_video directories, parse scene JSONs,
    and build a flat MaterialIndex.
    """
    # 修正：过滤掉极短视频（< 2s），避免进入锦书策划浪费 LLM 调用
    MIN_CLIP_DURATION_SEC = 2.0
    skipped_clips: list[str] = []
    valid_clips = []
    for c in project.clips:
        if c.duration_sec < MIN_CLIP_DURATION_SEC:
            skipped_clips.append(f"{c.clip_id} ({c.duration_sec:.1f}s)")
            continue
        valid_clips.append(c)

    if skipped_clips:
        print(f"⚠️  [material_index] 跳过 {len(skipped_clips)} 个极短视频: {skipped_clips}")

    index = MaterialIndex(
        project_id=project.project_id,
        build_time=datetime.now().isoformat(),
    )

    # Build day lookup: clip_id -> day_idx
    clip_to_day: dict[str, int] = {}
    for day in project.days:
        for cid in day.clip_ids:
            clip_to_day[cid] = day.day_idx

    global_scene_counter = 0

    for clip in valid_clips:
        if not clip.is_valid:
            continue

        index.total_clips += 1
        clip_scene_counter = 0  # 修正：每个 clip 内的 scene 独立计数

        scene_dir = clip.scene_summaries_dir
        if not scene_dir or not os.path.isdir(scene_dir):
            continue

        index.valid_clips += 1

        # Read scene summary JSONs from this clip
        scene_files = sorted(
            [f for f in os.listdir(scene_dir) if f.endswith(".json")],
            key=_natural_sort_key,
        )

        for sf in scene_files:
            scene_data = _load_scene_json(os.path.join(scene_dir, sf))
            if scene_data is None:
                continue

            global_scene_counter += 1
            clip_scene_counter += 1  # 每个 clip 内的独立计数器
            scene_id = f"S{global_scene_counter:03d}"

            entry = _scene_data_to_entry(
                scene_id=scene_id,
                clip=clip,
                scene_data=scene_data,
                scene_idx=clip_scene_counter,  # 修正：使用 clip 内独立索引
                day_idx=clip_to_day.get(clip.clip_id, 0),
            )
            index.scenes.append(entry)
            index.total_duration_sec += entry.duration_sec

    index.total_scenes = len(index.scenes)
    return index


def save_material_index(index: MaterialIndex, output_path: str):
    """Persist material_index.json."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(asdict(index), f, ensure_ascii=False, indent=2)


def load_material_index(path: str) -> MaterialIndex:
    """Load material_index.json."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    scenes = [SceneEntry(**s) for s in data.pop("scenes", [])]
    return MaterialIndex(**data, scenes=scenes)


def _scene_data_to_entry(
    scene_id: str,
    clip,
    scene_data: dict,
    scene_idx: int,
    day_idx: int,
) -> SceneEntry:
    """Convert a scene summary JSON dict into a SceneEntry."""
    # 场景 JSON 格式（scene_analysis_video 产出）：
    # {
    #   "scene_id": "scene_0",
    #   "time_range": {"start_seconds": 0.0, "end_seconds": 3.5},
    #   "video_analysis": {
    #     "scene_caption": {
    #       "scene_summary": {"narrative": "...", "key_event": "...", "location": "...", "emotion": "..."},
    #       "scene_classification": {"is_usable": true, "importance_score": 5, "scene_type": "..."}
    #     }
    #   }
    # }

    # 时间范围（支持浮秒或 "HH:MM:SS.SSS" 字符串格式）
    time_range = scene_data.get("time_range", {})
    start_sec = _parse_time_value(time_range.get("start_seconds", 0))
    end_sec = _parse_time_value(time_range.get("end_seconds", 0))
    duration = max(0, end_sec - start_sec)

    # 场景描述（嵌套在 video_analysis.scene_caption.scene_summary 中）
    caption = ""
    location = ""
    emotion = ""
    shot_type = ""
    has_protagonist = False

    video_analysis = scene_data.get("video_analysis", {})
    scene_caption = video_analysis.get("scene_caption", {})
    scene_summary = scene_caption.get("scene_summary", {})
    scene_classification = scene_caption.get("scene_classification", {})

    if isinstance(scene_summary, dict):
        caption = scene_summary.get("narrative", "") or scene_summary.get("key_event", "")
        location = scene_summary.get("location", "")
        emotion = scene_summary.get("emotion", "")

    if isinstance(scene_classification, dict):
        shot_type = scene_classification.get("scene_type", "")

    # 标签
    tags = []
    env = scene_summary.get("environment", "") if isinstance(scene_summary, dict) else ""
    if env:
        tags.append(env)

    dialogue = scene_data.get("dialogue", "")
    has_dialogue = bool(dialogue and dialogue.strip())

    return SceneEntry(
        scene_id=scene_id,  # 修正：使用调用方传入的全局 scene_id
        clip_id=clip.clip_id,
        clip_file_path=clip.file_path,
        scene_idx_in_clip=scene_idx,  # clip 内独立计数（per-clip）
        start_sec=start_sec,
        end_sec=end_sec,
        duration_sec=round(duration, 3),
        caption=caption,
        location=location,
        emotion=emotion,
        shot_type=shot_type,
        has_dialogue=has_dialogue,
        day_idx=day_idx,
        tags=tags,
    )


def _load_scene_json(path: str) -> Optional[dict]:
    """Load a scene summary JSON, return None on failure."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _natural_sort_key(s: str):
    """Sort key for natural ordering of filenames like scene_1.json, scene_10.json."""
    import re
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]


def _parse_time_value(value) -> float:
    """
    解析时间字段为浮点秒数。
    支持: 数字（int/float）或 "HH:MM:SS.SSS" 字符串
    """
    if value is None or value == "":
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        # 尝试直接转 float
        try:
            return float(value)
        except ValueError:
            pass
        # 解析 "HH:MM:SS.SSS" 格式
        try:
            parts = value.split(":")
            if len(parts) == 3:
                h, m, s = parts
                return int(h) * 3600 + int(m) * 60 + float(s)
            elif len(parts) == 2:
                m, s = parts
                return int(m) * 60 + float(s)
        except (ValueError, IndexError):
            pass
    return 0.0
