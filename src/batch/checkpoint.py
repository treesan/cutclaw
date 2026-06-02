"""
Per-stage checkpoint manager for batch processing.

Inspired by OpenMontage/lib/checkpoint.py — each stage writes a JSON
checkpoint so interrupted runs can resume from where they left off.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import Optional


@dataclass
class StageCheckpoint:
    """Snapshot of one stage's state for one clip or the whole project."""
    project_id: str
    stage: str              # "preprocess" / "build_index" / "shot_plan" / "shot_point" / "render"
    status: str             # "completed" / "failed" / "in_progress"
    clip_id: Optional[str] = None       # set for per-clip checkpoints
    artifacts: list[str] = field(default_factory=list)   # output file paths
    error: Optional[str] = None
    timestamp: str = ""


class CheckpointManager:
    """
    Manages project-level checkpoints for resumable batch processing.

    Checkpoint files live in <project_dir>/checkpoints/.
    Each clip gets its own file during preprocess; project-wide stages
    get a single file per stage.
    """

    def __init__(self, project_dir: str, project_id: str = ""):
        self.project_id = project_id
        self.checkpoint_dir = os.path.join(project_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def _path_for(self, stage: str, clip_id: Optional[str] = None) -> str:
        if clip_id:
            return os.path.join(self.checkpoint_dir, f"{stage}_{clip_id}.json")
        return os.path.join(self.checkpoint_dir, f"{stage}.json")

    def write(self, checkpoint: StageCheckpoint):
        """Write or overwrite a checkpoint JSON."""
        checkpoint.timestamp = datetime.now().isoformat()
        path = self._path_for(checkpoint.stage, checkpoint.clip_id)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(checkpoint), f, ensure_ascii=False, indent=2)

    def read(self, stage: str, clip_id: Optional[str] = None) -> Optional[StageCheckpoint]:
        """Read a checkpoint if it exists, else None."""
        path = self._path_for(stage, clip_id)
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return StageCheckpoint(**data)

    def mark_completed(self, stage: str, clip_id: Optional[str] = None, artifacts: list[str] | None = None):
        """Convenience: mark a stage as completed."""
        self.write(StageCheckpoint(
            project_id=self.project_id,
            stage=stage,
            status="completed",
            clip_id=clip_id,
            artifacts=artifacts or [],
        ))

    def mark_failed(self, stage: str, clip_id: Optional[str] = None, error: str = ""):
        """Convenience: mark a stage as failed."""
        self.write(StageCheckpoint(
            project_id=self.project_id,
            stage=stage,
            status="failed",
            clip_id=clip_id,
            error=error,
        ))

    def get_completed_clips(self, stage: str = "preprocess") -> set[str]:
        """Return set of clip_ids that completed the given stage."""
        completed = set()
        prefix = f"{stage}_"
        for fname in os.listdir(self.checkpoint_dir):
            if fname.startswith(prefix) and fname.endswith(".json"):
                clip_id = fname[len(prefix):-5]
                cp = self.read(stage, clip_id)
                if cp and cp.status == "completed":
                    completed.add(clip_id)
        return completed

    def get_next_stage(self, stages: list[str]) -> Optional[str]:
        """Return the first stage that has no completed checkpoint, or None if all done."""
        for stage in stages:
            cp = self.read(stage)
            if cp is None or cp.status != "completed":
                return stage
        return None

    def is_clip_done(self, stage: str, clip_id: str) -> bool:
        """Check if a specific clip has completed a stage."""
        cp = self.read(stage, clip_id)
        return cp is not None and cp.status == "completed"

    def cleanup(self, valid_clip_ids: set[str] | None = None, stage: str | None = None) -> int:
        """
        清理过期 checkpoint 文件。

        Args:
            valid_clip_ids: 仍然有效的 clip_id 集合。如果为 None，则清理指定 stage 的所有 checkpoint。
            stage: 只清理指定 stage 的 checkpoint。如果为 None，则清理所有 stage。

        Returns:
            删除的文件数
        """
        deleted = 0
        if not os.path.isdir(self.checkpoint_dir):
            return 0

        for fname in os.listdir(self.checkpoint_dir):
            if not fname.endswith(".json"):
                continue

            # 解析文件名: {stage}_{clip_id}.json 或 {stage}.json
            stem = fname[:-5]  # 去掉 .json
            parts = stem.split("_", 1)
            fname_stage = parts[0]
            fname_clip_id = parts[1] if len(parts) > 1 else None

            # 过滤 stage
            if stage and fname_stage != stage:
                continue

            # 过滤 clip_id
            if valid_clip_ids is not None:
                if fname_clip_id is None:
                    # 项目级 checkpoint 保留
                    continue
                if fname_clip_id not in valid_clip_ids:
                    fp = os.path.join(self.checkpoint_dir, fname)
                    os.remove(fp)
                    deleted += 1

        return deleted

    def clear_stage(self, stage: str) -> int:
        """
        清除指定 stage 的所有 checkpoint（包括项目级和 clip 级）。
        """
        return self.cleanup(stage=stage)
