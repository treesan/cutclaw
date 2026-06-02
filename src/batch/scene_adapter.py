"""
Scene adapter — converts MaterialIndex SceneEntry objects into the nested
scene summary JSON format expected by Screenwriter and DirectShotSelector.

The on-disk scene summary format (consumed by Screenwriter/DirectShotSelector):
{
    "scene_id": "scene_0",
    "time_range": {"start_seconds": 0.0, "end_seconds": 3.5},
    "video_analysis": {
        "scene_caption": {
            "scene_summary": {"narrative": "...", "key_event": "...", "location": "..."},
            "scene_classification": {"is_usable": true, "importance_score": 5, "scene_type": "..."}
        }
    }
}
"""

from __future__ import annotations

import json
import os
import shutil
from typing import Optional

from src.batch.material_index import MaterialIndex, SceneEntry


def _quality_to_importance(quality_score: float) -> int:
    """Map quality_score (0-100) to importance_score (1-5)."""
    if quality_score >= 70:
        return 5
    if quality_score >= 50:
        return 4
    if quality_score >= 30:
        return 3
    if quality_score >= 15:
        return 2
    return 1


def scene_entry_to_json(entry: SceneEntry, global_idx: int) -> dict:
    """将 SceneEntry 转换为 Screenwriter/DirectShotSelector 期望的嵌套 JSON 格式。"""
    # SceneEntry 没有独立的 key_event 字段，用 caption 的前 50 字符作为摘要
    key_event = entry.caption[:50] if entry.caption else ""
    return {
        "scene_id": f"scene_{global_idx}",
        "time_range": {
            "start_seconds": entry.start_sec,
            "end_seconds": entry.end_sec,
        },
        "video_analysis": {
            "scene_caption": {
                "scene_summary": {
                    "narrative": entry.caption,
                    "key_event": key_event,
                    "location": entry.location,
                    "emotion": entry.emotion,
                },
                "scene_classification": {
                    "is_usable": True,  # MaterialIndex 只包含 valid clips 的 scenes
                    "importance_score": _quality_to_importance(entry.quality_score),
                    "scene_type": entry.shot_type,
                },
            }
        },
        # 跨 clip 追踪元数据（Screenwriter 会忽略这些字段）
        "_clip_id": entry.clip_id,
        "_scene_id": entry.scene_id,
        "_day_idx": entry.day_idx,
    }


def write_virtual_scene_dir(
    index: MaterialIndex,
    output_dir: str,
) -> dict[int, SceneEntry]:
    """Write MaterialIndex scenes to a virtual scene_summaries directory.

    Creates ``output_dir/scene_0.json``, ``scene_1.json``, … in the nested
    format expected by Screenwriter and DirectShotSelector.

    Args:
        index: The material index containing all scenes.
        output_dir: Target directory for the virtual scene files.

    Returns:
        Mapping from global sequential index to the original SceneEntry.
        Used by the caller to inject ``source_clip``/``scene_id`` into the
        generated shot_plan.
    """
    os.makedirs(output_dir, exist_ok=True)

    scene_mapping: dict[int, SceneEntry] = {}

    for idx, entry in enumerate(index.scenes):
        scene_data = scene_entry_to_json(entry, idx)
        scene_path = os.path.join(output_dir, f"scene_{idx}.json")
        with open(scene_path, "w", encoding="utf-8") as f:
            json.dump(scene_data, f, ensure_ascii=False, indent=2)
        scene_mapping[idx] = entry

    return scene_mapping


def cleanup_virtual_scene_dir(output_dir: str):
    """Remove a virtual scene directory created by write_virtual_scene_dir."""
    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
