"""
Batch preprocessing orchestrator.

Runs the existing per-clip preprocessing pipeline (decode → caption →
scene_merge → scene_analysis) across all clips in a project, with
ThreadPoolExecutor parallelism and checkpoint-based resume.
"""

from __future__ import annotations

import os
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import src.config as config
from src.project.project import Project, Clip, ProjectManager
from src.batch.checkpoint import CheckpointManager
from src.batch.quality_filter import QualityFilter


class BatchPreprocessor:
    """
    Parallel preprocessor for all clips in a Project.

    For each clip, runs the same pipeline as local_run.py's preprocessing:
    1. decode_video_to_frames (shot detection + frame sampling)
    2. ASR (film mode only)
    3. video_caption + scene_merge + scene_analysis

    Uses checkpoint per clip so interrupted runs resume cleanly.
    """

    def __init__(
        self,
        project: Project,
        max_workers: int = 2,
        video_type: str = "vlog",
        skip_existing: bool = True,
    ):
        self.project = project
        self.max_workers = max_workers
        self.video_type = video_type
        self.skip_existing = skip_existing
        self.checkpoint = CheckpointManager(project.output_dir)
        self.quality_filter = QualityFilter()

    def run(self, clip_ids: Optional[list[str]] = None) -> dict:
        """
        Preprocess all (or specified) clips in parallel.

        Returns summary dict with counts.
        """
        clips = self.project.clips
        if clip_ids:
            clips = [c for c in clips if c.clip_id in clip_ids]

        # Filter out already-done clips
        pending = []
        for clip in clips:
            if self.skip_existing and self.checkpoint.is_clip_done("preprocess", clip.clip_id):
                print(f"  ⏭️  {clip.clip_id}: already done, skipping")
                continue
            pending.append(clip)

        if not pending:
            print("✅ All clips already preprocessed.")
            return {"total": len(clips), "skipped": len(clips), "success": 0, "failed": 0}

        print(f"\n🚀 Preprocessing {len(pending)} clips with {self.max_workers} workers...")
        results = {"total": len(clips), "skipped": len(clips) - len(pending), "success": 0, "failed": 0, "errors": []}

        if self.max_workers <= 1:
            for clip in pending:
                self._process_one_clip(clip, results)
        else:
            with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                futures = {pool.submit(self._process_one_clip, clip, results): clip for clip in pending}
                for future in as_completed(futures):
                    clip = futures[future]
                    try:
                        future.result()
                    except Exception as e:
                        print(f"  ❌ {clip.clip_id}: unexpected error: {e}")

        # Update project.json with progress
        ProjectManager.save(self.project)

        print(f"\n{'='*60}")
        print(f"📊 Preprocessing complete:")
        print(f"   Total: {results['total']}, Success: {results['success']}, "
              f"Skipped: {results['skipped']}, Failed: {results['failed']}")
        if results["errors"]:
            print(f"   Errors:")
            for err in results["errors"][:5]:
                print(f"     - {err}")
        print(f"{'='*60}\n")

        return results

    def _process_one_clip(self, clip: Clip, results: dict):
        """Run full preprocessing pipeline for one clip."""
        clip_id = clip.clip_id
        fp = clip.file_path

        if not os.path.exists(fp):
            self._mark_failed(clip_id, f"File not found: {fp}")
            results["failed"] += 1
            results["errors"].append(f"{clip_id}: file not found")
            return

        # Set up output directories for this clip
        clip_dir = os.path.join(self.project.output_dir, "Clips", clip_id)
        frames_dir = os.path.join(clip_dir, "frames")
        captions_dir = os.path.join(clip_dir, "captions")
        scenes_dir = os.path.join(captions_dir, "scenes")
        scene_summaries_dir = os.path.join(captions_dir, "scene_summaries_video")
        shot_scenes_file = os.path.join(frames_dir, "shot_scenes.txt")

        os.makedirs(frames_dir, exist_ok=True)
        os.makedirs(captions_dir, exist_ok=True)

        t0 = time.time()

        try:
            # Step 1: Decode video to frames + shot detection
            print(f"  🎞️  [{clip_id}] Decoding & shot detection...")
            from src.video.preprocess import decode_video_to_frames
            vr = decode_video_to_frames(
                fp, frames_dir,
                config.VIDEO_FPS,
                config.VIDEO_RESOLUTION,
                max_minutes=getattr(config, 'VIDEO_MAX_MINUTES', None),
                shot_detection_threshold=config.SHOT_DETECTION_THRESHOLD,
                shot_detection_min_scene_len=config.SHOT_DETECTION_MIN_SCENE_LEN,
                save_frames_to_disk=getattr(config, 'VIDEO_SAVE_DEBUG_FRAMES', False),
                image_format='jpg', jpeg_quality=80,
            )

            # Step 2: ASR (film mode only)
            srt_path = os.path.join(frames_dir, "subtitles.srt")
            srt_with_characters = os.path.join(frames_dir, "subtitles_with_characters.srt")

            if self.video_type == "film":
                print(f"  🎙️  [{clip_id}] Running ASR...")
                from src.video.preprocess.asr import run_asr, assign_speakers_to_srt
                from src.video.deconstruction.get_character import analyze_subtitles

                run_asr(
                    video_path=fp,
                    output_dir=frames_dir,
                    srt_path=srt_path,
                    backend=config.ASR_BACKEND,
                    asr_device=config.ASR_DEVICE,
                    asr_language=config.ASR_LANGUAGE,
                    whisper_cpp_model_name=getattr(config, 'ASR_WHISPER_CPP_MODEL', 'base'),
                    whisper_cpp_n_threads=getattr(config, 'ASR_WHISPER_CPP_N_THREADS', 4),
                    litellm_model=getattr(config, 'ASR_LITELLM_MODEL', None),
                    litellm_api_key=getattr(config, 'ASR_LITELLM_API_KEY', None),
                    litellm_api_base=getattr(config, 'ASR_LITELLM_API_BASE', None),
                    litellm_max_segment_mb=getattr(config, 'ASR_LITELLM_MAX_SEGMENT_MB', 25.0),
                    litellm_batch_size=getattr(config, 'ASR_LITELLM_BATCH_SIZE', 8),
                )

                # Character identification
                if os.path.exists(srt_path):
                    character_info_path = os.path.join(frames_dir, "character_info.json")
                    if not os.path.exists(character_info_path):
                        print(f"  👥 [{clip_id}] Identifying characters...")
                        analyze_subtitles(
                            srt_path=srt_path,
                            movie_name=clip_id.replace('_', ' '),
                            output_dir=frames_dir,
                            use_full_subtitles=True,
                            model=config.VIDEO_ANALYSIS_MODEL,
                            api_base=config.VIDEO_ANALYSIS_ENDPOINT,
                            api_key=config.VIDEO_ANALYSIS_API_KEY,
                            max_tokens=config.VIDEO_ANALYSIS_MODEL_MAX_TOKEN,
                        )

            # Step 3: Video captioning
            caption_file = os.path.join(captions_dir, "captions.json")
            if not os.path.exists(caption_file):
                print(f"  🎬 [{clip_id}] Video captioning...")
                from src.video.deconstruction.video_caption import process_video
                subtitle_to_use = None
                if self.video_type == "film":
                    subtitle_to_use = srt_with_characters if os.path.exists(srt_with_characters) else srt_path
                process_video(
                    video=vr,
                    output_caption_folder=captions_dir,
                    subtitle_file_path=subtitle_to_use,
                    long_shots_path=shot_scenes_file if os.path.exists(shot_scenes_file) else None,
                    video_type=self.video_type,
                    frames_dir=frames_dir,
                )

            # Step 4: Scene merge
            scenes_output = os.path.join(scenes_dir, "scene_0.json")
            shots_dir = os.path.join(captions_dir, "ckpt")
            if os.path.exists(shots_dir) and not os.path.exists(scenes_output):
                print(f"  🧩 [{clip_id}] Scene merge...")
                from src.video.deconstruction.scene_merge import OptimizedSceneSegmenter, load_shots, save_scenes
                shots = load_shots(shots_dir)
                if shots:
                    segmenter = OptimizedSceneSegmenter()
                    merged = segmenter.segment(
                        shots,
                        threshold=getattr(config, 'SCENE_SIMILARITY_THRESHOLD', 0.5),
                        max_scene_duration_secs=getattr(config, 'MAX_SCENE_DURATION_SECS', 300),
                    )
                    save_scenes(merged, scenes_dir)

            # Step 5: Scene analysis
            if os.path.exists(scenes_dir) and os.path.exists(scenes_output):
                if not os.path.isdir(scene_summaries_dir) or not os.listdir(scene_summaries_dir):
                    print(f"  🔍 [{clip_id}] Scene analysis...")
                    from src.video.deconstruction.scene_analysis_video import SceneVideoAnalyzer
                    subtitle_to_use = None
                    if self.video_type == "film":
                        subtitle_to_use = srt_with_characters if os.path.exists(srt_with_characters) else srt_path
                    analyzer = SceneVideoAnalyzer(vr=vr, subtitle_file=subtitle_to_use)
                    result = analyzer.analyze_scenes_dir(
                        scenes_dir=scenes_dir,
                        output_dir=scene_summaries_dir,
                        max_workers=getattr(config, 'CAPTION_BATCH_SIZE', 8),
                        overwrite=False,
                    )

            # Step 6: Quick quality check
            quality_result = self.quality_filter.evaluate_clip(
                fp, clip_id=clip_id, duration_sec=clip.duration_sec,
            )

            # Update clip with preprocess results
            clip.scene_summaries_dir = scene_summaries_dir
            clip.shot_scenes_path = shot_scenes_file
            clip.asr_path = srt_path if os.path.exists(srt_path) else None
            clip.captions_path = caption_file if os.path.exists(caption_file) else None
            clip.quality_score = quality_result.score
            clip.quality_flags = quality_result.flags
            clip.is_valid = quality_result.is_valid

            elapsed = time.time() - t0
            self.checkpoint.mark_completed("preprocess", clip_id, artifacts=[scene_summaries_dir])
            self.project.preprocess_progress[clip_id] = "done"
            results["success"] += 1
            status = "✅" if quality_result.is_valid else "⚠️"
            print(f"  {status} [{clip_id}] Done in {elapsed:.1f}s (quality={quality_result.score:.0f})")

        except Exception as e:
            elapsed = time.time() - t0
            error_msg = f"{clip_id}: {e}"
            self._mark_failed(clip_id, str(e))
            self.project.preprocess_progress[clip_id] = "failed"
            results["failed"] += 1
            results["errors"].append(error_msg)
            print(f"  ❌ [{clip_id}] Failed after {elapsed:.1f}s: {e}")

    def _mark_failed(self, clip_id: str, error: str):
        self.checkpoint.mark_failed("preprocess", clip_id, error=error)
