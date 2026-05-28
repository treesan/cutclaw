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

    for clip in project.clips:
        if not clip.is_valid:
            continue

        index.total_clips += 1

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
            scene_id = f"S{global_scene_counter:03d}"

            entry = _scene_data_to_entry(
                scene_data=scene_id,
                clip=clip,
                scene_data_full=scene_data,
                scene_idx=global_scene_counter,
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
    scene_data: str,
    clip,
    scene_data_full: dict,
    scene_idx: int,
    day_idx: int,
) -> SceneEntry:
    """Convert a scene summary JSON dict into a SceneEntry."""
    # Scene summary JSONs from scene_analysis_video typically have:
    # time_start, time_end, scene_caption, environment, dialogue, etc.
    start_sec = float(scene_data_full.get("time_start", 0) or scene_data_full.get("start_sec", 0) or 0)
    end_sec = float(scene_data_full.get("time_end", 0) or scene_data_full.get("end_sec", 0) or 0)
    duration = max(0, end_sec - start_sec)

    caption = ""
    scene_caption = scene_data_full.get("scene_caption", {})
    if isinstance(scene_caption, dict):
        caption = scene_caption.get("caption", "") or scene_caption.get("description", "")
    elif isinstance(scene_caption, str):
        caption = scene_caption

    # Extract tags from environment / shot_type fields
    tags = []
    env = scene_data_full.get("environment", "")
    if env:
        tags.append(env)
    shot_type = scene_data_full.get("shot_type", "") or ""
    location = scene_data_full.get("location", "") or ""
    emotion = scene_data_full.get("emotion", "") or ""

    dialogue = scene_data_full.get("dialogue", "")
    has_dialogue = bool(dialogue and dialogue.strip())

    return SceneEntry(
        scene_id=f"S{scene_idx:03d}",
        clip_id=clip.clip_id,
        clip_file_path=clip.file_path,
        scene_idx_in_clip=scene_idx,
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
