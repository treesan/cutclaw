import src.config as config
import os
import argparse
import time
import threading

from src.video.preprocess import decode_video_to_frames
from src.video.preprocess.asr import run_asr, assign_speakers_to_srt
from src.video.deconstruction.get_character import analyze_subtitles

from src.video.deconstruction.video_caption import process_video
from src.video.deconstruction.scene_merge import OptimizedSceneSegmenter, load_shots, save_scenes
from src.video.deconstruction.scene_analysis_video import SceneVideoAnalyzer

from src.audio.audio_caption_madmom import caption_audio_with_madmom_segments

# import src.config as config

def parse_config_overrides(unknown_args):
    """
    Parse config override arguments in the format --config.PARAM_NAME value

    Args:
        unknown_args: List of unknown arguments from argparse

    Returns:
        None (modifies config module in place)
    """
    i = 0
    while i < len(unknown_args):
        arg = unknown_args[i]
        if arg.startswith('--config.'):
            param_name = arg[9:]  # Remove '--config.' prefix
            if i + 1 < len(unknown_args) and not unknown_args[i + 1].startswith('--'):
                value_str = unknown_args[i + 1]

                # Auto-detect type based on existing config value or infer from string
                if hasattr(config, param_name):
                    original_value = getattr(config, param_name)
                    # Preserve original type (handle None by inferring from string)
                    if original_value is None:
                        # None default → infer type from value string
                        try:
                            if '.' in value_str:
                                value = float(value_str)
                            else:
                                value = int(value_str)
                        except ValueError:
                            if value_str.lower() in ('true', 'false', 'yes', 'no'):
                                value = value_str.lower() in ('true', 'yes')
                            else:
                                value = value_str
                    elif isinstance(original_value, bool):
                        value = value_str.lower() in ('true', '1', 'yes')
                    elif isinstance(original_value, int):
                        value = int(value_str)
                    elif isinstance(original_value, float):
                        value = float(value_str)
                    else:
                        value = value_str
                else:
                    # Infer type from string (new param)
                    try:
                        if '.' in value_str:
                            value = float(value_str)
                        else:
                            value = int(value_str)
                    except ValueError:
                        if value_str.lower() in ('true', 'false'):
                            value = value_str.lower() == 'true'
                        else:
                            value = value_str

                setattr(config, param_name, value)
                print(f"✅ Config override: {param_name} = {value} (type: {type(value).__name__})")
                i += 2
            else:
                print(f"⚠️ Warning: --config.{param_name} specified but no value provided")
                i += 1
        else:
            print(f"⚠️ Warning: Unknown argument '{arg}' ignored")
            i += 1

def _run_project_command(args):
    """Handle --project subcommands for batch editing."""
    from src.project.project import ProjectManager

    if args.project == "create":
        if not args.video_dir:
            print("❌ --video-dir is required for 'create'")
            return
        print(f"📂 Creating project from: {args.video_dir}")
        project = ProjectManager.create_from_directory(
            video_dir=args.video_dir,
            project_name=args.project_name or "",
        )
        print(f"\n✅ Project created: {project.metadata_path}")
        print(ProjectManager.status_summary(project))

    elif args.project == "status":
        if not args.project_path:
            print("❌ --project-path is required for 'status'")
            return
        project = ProjectManager.load(args.project_path)
        print(ProjectManager.status_summary(project))

    elif args.project == "review-sources":
        if not args.project_path:
            print("❌ --project-path is required for 'review-sources'")
            return
        project = ProjectManager.load(args.project_path)
        from src.batch.source_media_review import SourceMediaReview
        reviewer = SourceMediaReview()
        report = reviewer.review_project(project)
        print(reviewer.format_report(report))
        report_path = os.path.join(project.output_dir, "source_review.json")
        reviewer.save_report(report, report_path)
        print(f"\n💾 Report saved to: {report_path}")

    elif args.project == "preprocess":
        if not args.project_path:
            print("❌ --project-path is required for 'preprocess'")
            return
        project = ProjectManager.load(args.project_path)
        from src.batch.batch_preprocess import BatchPreprocessor
        video_type = getattr(args, "type", "vlog")
        max_workers = getattr(args, "max_workers", 2) or 2
        preprocessor = BatchPreprocessor(
            project=project,
            max_workers=max_workers,
            video_type=video_type,
            skip_existing=True,
        )
        preprocessor.run()

    elif args.project == "build-index":
        if not args.project_path:
            print("❌ --project-path is required for 'build-index'")
            return
        project = ProjectManager.load(args.project_path)
        from src.batch.material_index import build_material_index, save_material_index
        print(f"📊 Building material index for {project.project_id}...")
        index = build_material_index(project)
        index_path = os.path.join(project.output_dir, "material_index.json")
        save_material_index(index, index_path)
        print(f"✅ Material index built: {index.total_scenes} scenes from {index.valid_clips} clips")
        print(f"💾 Saved to: {index_path}")


def main():
    parser = argparse.ArgumentParser(description="Run VideoCaptioningAgent on a video.")
    parser.add_argument("--Video_Path", help="The URL of the video to process.", default="Dataset/Video/Movie/La_La_Land.mkv")
    parser.add_argument("--Audio_Path", help="The URL of the video to process (optional: skip for analysis-only mode).", default="")
    parser.add_argument("--Instruction", help="The Instruction to cutting the video.", default="Mia and Sebastian's relationship evolves through sweet to break moments.")
    parser.add_argument("--instruction_type", help="Type of instruction: 'object' for Object-centric or 'narrative' for Narrative-driven", default="object", choices=["object", "narrative"])
    parser.add_argument("--type", help="film or vlog", default="film")
    parser.add_argument("--preprocess-only", action="store_true", help="只做视频+BGM解析，跳过 Screenwriter 和 EditorCoreAgent")
    parser.add_argument("--SRT_Path", type=str,
                        help="Path to existing SRT file. Skips ASR transcription; diarization still runs to assign speakers.")

    # Batch project mode (new, incremental — does not affect existing flow)
    parser.add_argument("--project", choices=["create", "status", "preprocess", "build-index", "review-sources"],
                        help="Batch project subcommand. When set, runs project management instead of single-video pipeline.")
    parser.add_argument("--project-path", type=str, help="Path to project.json for batch operations.")
    parser.add_argument("--video-dir", type=str, help="Source video directory for 'create' command.")
    parser.add_argument("--project-name", type=str, default="", help="Friendly project name (auto-derived from directory if omitted).")
    parser.add_argument("--max-workers", type=int, default=2, help="Parallel workers for batch preprocess (default: 2).")

    # Parse known args and capture unknown args for config overrides
    args, unknown = parser.parse_known_args()

    # Apply config overrides
    parse_config_overrides(unknown)

    # ── Route to batch project commands if --project is set ──
    if args.project:
        _run_project_command(args)
        return

    config.VIDEO_TYPE = args.type

    Video_Path = args.Video_Path
    Audio_Path = args.Audio_Path
    Instruction = args.Instruction
    instruction_type = args.instruction_type

    video_id = os.path.splitext(os.path.basename(Video_Path))[0].replace('.', '_').replace(' ', '_')
    audio_id = os.path.splitext(os.path.basename(Audio_Path))[0].replace('.', '_').replace(' ', '_') if Audio_Path else "no_audio"

    # Generate a safe filename from instruction
    import re
    import hashlib
    # Create a short hash of the instruction for uniqueness
    instruction_hash = hashlib.md5(Instruction.encode('utf-8')).hexdigest()[:8]
    # Create a more readable version (up to 50 characters, sanitized)
    instruction_safe = re.sub(r'[^\w\s-]', '', Instruction)[:50].strip().replace(' ', '_')
    # If instruction is too long or empty, use a more informative format
    if len(instruction_safe) > 0:
        instruction_id = f"{instruction_safe}_{instruction_hash}"
    else:
        instruction_id = f"instruction_{instruction_hash}"

    # ===== All Path Definitions =====
    # Raw video output
    output_path = os.path.join(config.VIDEO_DATABASE_FOLDER, "raw", f"{video_id}.mp4")

    # Video-related paths
    frames_dir = os.path.join(config.VIDEO_DATABASE_FOLDER, 'Video', video_id, "frames")
    video_captions_dir = os.path.join(config.VIDEO_DATABASE_FOLDER, 'Video', video_id, "captions")
    video_db_path = os.path.join(config.VIDEO_DATABASE_FOLDER, 'Video', video_id, "database.json")
    srt_path = os.path.join(config.VIDEO_DATABASE_FOLDER, 'Video', video_id, "subtitles.srt")
    srt_with_characters = os.path.join(config.VIDEO_DATABASE_FOLDER, 'Video', video_id, "subtitles_with_characters.srt")
    character_info_path = os.path.join(config.VIDEO_DATABASE_FOLDER, 'Video', video_id, "character_info.json")

    # Shot and scene paths
    shot_scenes_file = os.path.join(frames_dir, "shot_scenes.txt")
    caption_file = os.path.join(video_captions_dir, "captions.json")
    shots_dir = os.path.join(config.VIDEO_DATABASE_FOLDER, 'Video', video_id, "captions", "ckpt")
    scenes_dir = os.path.join(config.VIDEO_DATABASE_FOLDER, 'Video', video_id, "captions", "scenes")
    scenes_output = os.path.join(scenes_dir, "scene_0.json")
    scene_summaries_dir = os.path.join(config.VIDEO_DATABASE_FOLDER, 'Video', video_id, "captions", "scene_summaries_video")

    # Audio-related paths (only if Audio_Path provided; used as flag for analysis-only mode)
    audio_captions_dir = os.path.join(config.VIDEO_DATABASE_FOLDER, 'Audio', audio_id, "captions")
    audio_caption_file = os.path.join(audio_captions_dir, "captions.json") if Audio_Path else None

    # Output paths (include instruction type and instruction ID for different editing tasks)
    shot_plan_output_path = os.path.join(
        config.VIDEO_DATABASE_FOLDER,
        'Output',
        f"{video_id}_{audio_id}",
        f"shot_plan_{instruction_id}.json"
    )
    shot_point_output_path = os.path.join(
        config.VIDEO_DATABASE_FOLDER,
        'Output',
        f"{video_id}_{audio_id}",
        f"shot_point_{instruction_id}.json"
    )
    start_time = time.time()
    stage_times = {}

    print(f"\n{'='*80}")
    print(f"🎬 Starting VideoCuttingAgent Pipeline")
    print(f"📽️  Video: {Video_Path}")
    print(f"🎵 Audio: {Audio_Path if Audio_Path else '(none - analysis-only mode)'}")
    print(f"📝 Instruction: {Instruction}")
    print(f"{'='*80}\n")

    # Step 1: Decode video to frames and perform shot detection
    print(f"🎞️ [Step 1] Extracting video frames format in {frames_dir}...")
    t0 = time.time()
    vr = decode_video_to_frames(
        Video_Path,
        frames_dir,
        config.VIDEO_FPS,
        config.VIDEO_RESOLUTION,
        max_minutes=getattr(config, 'VIDEO_MAX_MINUTES', None),
        shot_detection_threshold=config.SHOT_DETECTION_THRESHOLD,
        shot_detection_min_scene_len=config.SHOT_DETECTION_MIN_SCENE_LEN,
        save_frames_to_disk=getattr(config, 'VIDEO_SAVE_DEBUG_FRAMES', False),
        image_format='jpg',
        jpeg_quality=80,
    )
    stage_times['shot_detection'] = time.time() - t0
    print(f"✅ [Step 1] Shot detection completed in {stage_times['shot_detection']:.1f}s")


    thread_errors = {}
    asr_done_event = threading.Event()  # used only for film type

    def run_asr_and_character_id():
        """Thread A: ASR + Character ID (film only). Sets asr_done_event when complete."""
        try:
            t0 = time.time()
            if args.type != "vlog":
                if args.SRT_Path is not None:
                    print(f"🔤 [Thread A: ASR] External SRT provided, skipping ASR transcription: {args.SRT_Path}")
                    if not os.path.exists(srt_path):
                        enable_diarization = getattr(config, 'ASR_ENABLE_DIARIZATION', False)
                        if enable_diarization:
                            from src.video.preprocess.asr import extract_audio_mp3_16k
                            extracted_audio_path = os.path.join(frames_dir, "audio_16k_mono.mp3")
                            if not os.path.exists(extracted_audio_path):
                                print("[Thread A: ASR] 🔊 Extracting audio for diarization...")
                                extract_audio_mp3_16k(Video_Path, extracted_audio_path)
                            assign_speakers_to_srt(
                                srt_path=args.SRT_Path,
                                audio_path=extracted_audio_path,
                                output_srt_path=srt_path,
                                device=config.ASR_DEVICE,
                            )
                        else:
                            import shutil
                            shutil.copy(args.SRT_Path, srt_path)
                            print(f"[Thread A: ASR] 📋 Diarization disabled, copied SRT to {srt_path}")
                    else:
                        print(f"[Thread A: ASR] ⏭️ SRT already exists at {srt_path}, skipping.")
                else:
                    print("[Thread A: ASR] 🎙️ Running ASR to generate subtitles...")
                    run_asr(
                        video_path=Video_Path,
                        output_dir=frames_dir,
                        srt_path=srt_path,
                        backend=config.ASR_BACKEND,
                        asr_device=config.ASR_DEVICE,
                        asr_language=config.ASR_LANGUAGE,
                        whisper_cpp_model_name=getattr(config, 'ASR_WHISPER_CPP_MODEL', 'base.en'),
                        whisper_cpp_n_threads=getattr(config, 'ASR_WHISPER_CPP_N_THREADS', 4),
                        litellm_model=getattr(config, 'ASR_LITELLM_MODEL', None),
                        litellm_api_key=getattr(config, 'ASR_LITELLM_API_KEY', None),
                        litellm_api_base=getattr(config, 'ASR_LITELLM_API_BASE', None),
                        litellm_max_segment_mb=getattr(config, 'ASR_LITELLM_MAX_SEGMENT_MB', 25.0),
                        litellm_batch_size=getattr(config, 'ASR_LITELLM_BATCH_SIZE', 8),
                        litellm_debug_dir=os.path.join(config.VIDEO_DATABASE_FOLDER, 'Video', video_id, "subtitles_segments"),
                    )
                print("[Thread A: ASR] ✅ ASR/SRT step completed.")
                if os.path.exists(srt_path) and not os.path.exists(character_info_path):
                    print("[Thread A: CharID] 👥 Analyzing subtitles to identify characters...")
                    video_name = video_id.replace('_', ' ')
                    speaker_mapping, _character_info = analyze_subtitles(
                        srt_path=srt_path,
                        movie_name=video_name,
                        output_dir=os.path.join(config.VIDEO_DATABASE_FOLDER, 'Video', video_id),
                        use_full_subtitles=True,
                        model=config.VIDEO_ANALYSIS_MODEL,
                        api_base=config.VIDEO_ANALYSIS_ENDPOINT,
                        api_key=config.VIDEO_ANALYSIS_API_KEY,
                        max_tokens=config.VIDEO_ANALYSIS_MODEL_MAX_TOKEN,
                    )
                    print(f"[Thread A: CharID] ✅ Character identification completed. Found {len(speaker_mapping)} characters.")
                elif os.path.exists(character_info_path):
                    print(f"[Thread A: CharID] ⏭️ Character info already exists at {character_info_path}.")
                else:
                    print(f"[Thread A: CharID] ⚠️ Subtitle file not found at {srt_path}, skipping character identification.")
            else:
                print("[Thread A] ⏭️ Skipping ASR/character ID for vlog type.")
            stage_times['asr_character_id'] = time.time() - t0
            print(f"[Thread A] ✨ Completed in {stage_times['asr_character_id']:.1f}s")
        except Exception as e:
            thread_errors['asr'] = e
            print(f"[Thread A] ❌ ERROR: {e}")
        finally:
            asr_done_event.set()  # always signal, even on error, so Thread B doesn't hang

    def run_video_captioning():
        """Thread B: Video Captioning → Scene Merge → Scene Analysis.
        For film: waits for ASR to complete first. For vlog: starts immediately."""
        try:
            t0 = time.time()
            if args.type != "vlog":
                print("[Thread B: Video] ⏳ Waiting for ASR/Character ID to complete...")
                asr_done_event.wait()
                if 'asr' in thread_errors:
                    raise RuntimeError("ASR failed, cannot proceed with video captioning")

            if not os.path.exists(caption_file):
                print("[Thread B: Video] 🎬 Processing video to get captions...")
                if args.type == "vlog":
                    subtitle_to_use = None
                    print("[Thread B: Video] 🎬 Processing vlog without subtitles.")
                else:
                    subtitle_to_use = srt_with_characters if os.path.exists(srt_with_characters) else srt_path
                    print(f"[Thread B: Video] 🎬 Processing video with subtitle file: {subtitle_to_use}")
                process_video(
                    video=vr,
                    output_caption_folder=video_captions_dir,
                    subtitle_file_path=subtitle_to_use,
                    long_shots_path=shot_scenes_file if os.path.exists(shot_scenes_file) else None,
                    video_type=args.type,
                    frames_dir=frames_dir,
                )
            else:
                print(f"[Thread B: Video] ⏭️ Captions already exist at {caption_file}.")

            # Scene Merge
            if os.path.exists(shots_dir) and not os.path.exists(scenes_output):
                print("[Thread B: Scene] 🧩 Merging shots into scenes...")
                shots = load_shots(shots_dir)
                print(f"[Thread B: Scene] 📄 Loaded {len(shots)} shots")
                if shots:
                    segmenter = OptimizedSceneSegmenter()
                    merged_scenes = segmenter.segment(
                        shots,
                        threshold=config.SCENE_SIMILARITY_THRESHOLD if hasattr(config, 'SCENE_SIMILARITY_THRESHOLD') else 0.5,
                        max_scene_duration_secs=config.MAX_SCENE_DURATION_SECS if hasattr(config, 'MAX_SCENE_DURATION_SECS') else 300
                    )
                    print(f"[Thread B: Scene] ✅ Merged {len(shots)} shots into {len(merged_scenes)} scenes")
                    save_scenes(merged_scenes, scenes_dir)
                    print(f"[Thread B: Scene] 💾 Scenes saved to {scenes_dir}")
                else:
                    print("[Thread B: Scene] ⚠️ No shots found to merge")
            elif os.path.exists(scenes_output):
                print(f"[Thread B: Scene] ⏭️ Scenes already exist at {scenes_dir}")
            else:
                print(f"[Thread B: Scene] ⚠️ Shots directory not found at {shots_dir}, skipping scene merge")

            # Scene Analysis
            if os.path.exists(scenes_dir) and os.path.exists(scenes_output):
                if args.type == "vlog":
                    subtitle_to_use = None
                    print("[Thread B: Analysis] 🔍 Analyzing scenes without subtitles for vlog.")
                else:
                    subtitle_to_use = srt_with_characters if os.path.exists(srt_with_characters) else srt_path
                analyzer = SceneVideoAnalyzer(vr=vr, subtitle_file=subtitle_to_use)
                result = analyzer.analyze_scenes_dir(
                    scenes_dir=scenes_dir,
                    output_dir=scene_summaries_dir,
                    max_workers=config.CAPTION_BATCH_SIZE,
                    overwrite=False,
                )
                if result["status"] == "invalid":
                    print(f"[Thread B: Analysis] ❌ {result['errors'][0]}")
                elif result["status"] == "skipped":
                    print(f"[Thread B: Analysis] ⏭️ Scene summaries already exist ({result['already_analyzed']} files)")
                else:
                    print(f"[Thread B: Analysis] ✅ Scene analysis completed: {result['success']} success, {result['skipped']} skipped")
                    if result["errors"]:
                        print(f"[Thread B: Analysis] ⚠️ Errors: {len(result['errors'])}")
                        for e in result["errors"][:3]:
                            print(f"   > {e}")
            else:
                print(f"[Thread B: Analysis] ⚠️ Scenes directory not found or empty at {scenes_dir}, skipping scene analysis")
            stage_times['video_captioning'] = time.time() - t0
            print(f"[Thread B] ✨ Completed in {stage_times['video_captioning']:.1f}s")
        except Exception as e:
            thread_errors['video'] = e
            print(f"[Thread B] ❌ ERROR: {e}")

    def run_audio_analysis():
        """Thread C: Audio Analysis — fully independent of video pipeline."""
        try:
            t0 = time.time()
            if not os.path.exists(audio_caption_file):
                print("[Thread C: Audio] 🎵 Processing audio to get captions...")
                caption_audio_with_madmom_segments(
                    audio_path=Audio_Path,
                    output_path=audio_caption_file,
                    max_tokens=config.AUDIO_KEYPOINT_MAX_TOKENS,
                    temperature=config.AUDIO_KEYPOINT_TEMPERATURE,
                    top_p=config.AUDIO_KEYPOINT_TOP_P,
                    max_workers=config.AUDIO_BATCH_SIZE,
                    detection_methods=config.AUDIO_DETECTION_METHODS,
                    beats_per_bar=[config.AUDIO_BEATS_PER_BAR],
                    min_bpm=config.AUDIO_MIN_BPM,
                    max_bpm=config.AUDIO_MAX_BPM,
                    pitch_tolerance=config.AUDIO_PITCH_TOLERANCE,
                    pitch_threshold=config.AUDIO_PITCH_THRESHOLD,
                    pitch_min_distance=config.AUDIO_PITCH_MIN_DISTANCE,
                    pitch_nms_method=config.AUDIO_PITCH_NMS_METHOD,
                    pitch_max_points=config.AUDIO_PITCH_MAX_POINTS,
                    mel_win_s=config.AUDIO_MEL_WIN_S,
                    mel_n_filters=config.AUDIO_MEL_N_FILTERS,
                    mel_threshold_ratio=config.AUDIO_MEL_THRESHOLD_RATIO,
                    mel_min_distance=config.AUDIO_MEL_MIN_DISTANCE,
                    mel_nms_method=config.AUDIO_MEL_NMS_METHOD,
                    mel_max_points=config.AUDIO_MEL_MAX_POINTS,
                    merge_close=config.AUDIO_MERGE_CLOSE,
                    min_interval=config.AUDIO_MIN_INTERVAL,
                    top_k_keypoints=config.AUDIO_TOP_K,
                    energy_percentile=config.AUDIO_ENERGY_PERCENTILE,
                    min_segment_duration=config.AUDIO_MIN_SEGMENT_DURATION,
                    max_segment_duration=config.AUDIO_MAX_SEGMENT_DURATION,
                    use_stage1_sections=config.AUDIO_USE_STAGE1_SECTIONS,
                    section_min_interval=config.AUDIO_SECTION_MIN_INTERVAL,
                )
            else:
                print(f"[Thread C: Audio] ⏭️ Audio captions already exist at {audio_caption_file}.")
            stage_times['audio_analysis'] = time.time() - t0
            print(f"[Thread C] ✨ Completed in {stage_times['audio_analysis']:.1f}s")
        except Exception as e:
            thread_errors['audio'] = e
            print(f"[Thread C] ❌ ERROR: {e}")

    # Launch threads
    thread_a = threading.Thread(target=run_asr_and_character_id, name="ASR-CharID",    daemon=False)
    thread_b = threading.Thread(target=run_video_captioning,     name="VideoCaptions", daemon=False)

    thread_a.start()
    thread_b.start()

    if Audio_Path:
        print("[main] 🎵 Audio file provided, starting audio analysis...")
        thread_c = threading.Thread(target=run_audio_analysis, name="AudioAnalysis", daemon=False)
        thread_c.start()
    else:
        print("[main] 🎵 No Audio_Path provided → skipping audio analysis (analysis-only mode)")
        # Set audio_done flag so downstream checks work correctly
        thread_c = None

    thread_a.join()
    thread_b.join()
    if thread_c:
        thread_c.join()

    if thread_errors:
        for name, err in thread_errors.items():
            print(f"❌ Pipeline stage '{name}' failed: {err}")
        raise RuntimeError(f"Pipeline failed in stages: {list(thread_errors.keys())}")

    print("\n🚀 All parallel stages completed.")

    end_time = time.time()
    print(f"\n{'='*60}")
    print(f"⏱️  Stage Timing Summary:")
    for stage, elapsed in stage_times.items():
        print(f"  {stage:<30} {elapsed:>8.1f}s")
    print(f"  {'total (wall clock)':<30} {end_time - start_time:>8.1f}s")
    print(f"{'='*60}\n")

    



    # ── --preprocess-only 模式：只做解析，到此结束 ──
    if getattr(args, "preprocess_only", False):
        print("\n" + "="*60)
        print("⏭️  --preprocess-only: 预处理完成，跳过 Screenwriter + EditorCoreAgent")
        print("    视频解析 → Output/Video/ 目录")
        print("    BGM 分析 → Output/Audio/ 目录")
        print("="*60 + "\n")
        return

    # Step 5: Run Screenwriter to generate shot plan
    if os.path.exists(scene_summaries_dir) and os.path.exists(audio_caption_file):
        print("\n" + "="*80)
        if os.path.exists(shot_plan_output_path):
            print("✍️  Running Screenwriter to validate/complete existing shot plan...")
            print(f"📄 Existing shot plan detected: {shot_plan_output_path}")
        else:
            print("✍️  Running Screenwriter to generate shot plan...")
        print("="*80)

        from src.Screenwriter_scene_short import Screenwriter

        # Create output directory
        os.makedirs(os.path.dirname(shot_plan_output_path), exist_ok=True)

        # Initialize Screenwriter agent
        screenwriter = Screenwriter(
            video_scene_path=scene_summaries_dir,
            audio_caption_path=audio_caption_file,
            output_path=shot_plan_output_path,
            video_path=Video_Path,
            subtitle_path=srt_with_characters if config.VIDEO_TYPE == "film" and os.path.exists(srt_with_characters) else None,
            main_character=config.MAIN_CHARACTER_NAME if config.MAIN_CHARACTER_NAME else None,
            max_iterations=20,
        )

        # Run the screenwriter with the Instruction
        print(f"📝 Instruction: '{Instruction}'")
        t0 = time.time()
        _shot_plan = screenwriter.run(Instruction)
        stage_times['screenwriter'] = time.time() - t0

        print(f"\n{'='*80}")
        print(f"✅ Shot plan generated successfully in {stage_times['screenwriter']:.1f}s!")
        print(f"💾 Output saved to: {shot_plan_output_path}")
        print(f"{'='*80}\n")
    # Step 6: Run EditorCoreAgent to select video clips based on shot plan
    # Check if we have all required files for core agent
    if os.path.exists(scene_summaries_dir) and os.path.exists(audio_caption_file) and os.path.exists(shot_plan_output_path):
        print("\n" + "="*80)
        print("✂️  Running EditorCoreAgent to select video clips...")
        print("="*80)

        if config.VIDEO_TYPE == "film":
            from src.core import EditorCoreAgent, ParallelShotOrchestrator
        elif config.VIDEO_TYPE == "vlog":
            from src.core_vlog import EditorCoreAgent

        # Create output directory
        os.makedirs(os.path.dirname(shot_point_output_path), exist_ok=True)

        max_iterations = config.AGENT_MAX_ITERATIONS if hasattr(config, 'AGENT_MAX_ITERATIONS') else 20
        use_parallel_shot = (
            config.VIDEO_TYPE == "film" and
            getattr(config, "PARALLEL_SHOT_ENABLED", True)
        )

        print(f"🚀 Running editor agent with instruction: '{Instruction}'")
        print(f"📂 Using shot plan from: {shot_plan_output_path}")

        if use_parallel_shot:
            max_workers = getattr(config, "PARALLEL_SHOT_MAX_WORKERS", 4)
            max_reruns = getattr(config, "PARALLEL_SHOT_MAX_RERUNS", 2)
            print(f"⚡ Parallel mode enabled (workers: {max_workers}, max_reruns: {max_reruns})")
            orchestrator = ParallelShotOrchestrator(
                video_caption_path=caption_file,
                video_scene_path=scene_summaries_dir,
                audio_caption_path=audio_caption_file,
                output_path=shot_point_output_path,
                max_iterations=max_iterations,
                video_path=Video_Path,
                frame_folder_path=frames_dir,
                transcript_path=srt_with_characters if os.path.exists(srt_with_characters) else srt_path,
                max_workers=max_workers,
                max_reruns=max_reruns,
            )
            _results = orchestrator.run_parallel(shot_plan_path=shot_plan_output_path)
            print(f"✅ Parallel mode completed, selected {len(_results)} shots.")
        else:
            print("🚶 Sequential mode enabled (EditorCoreAgent.run).")
            editor_agent = EditorCoreAgent(
                video_caption_path=caption_file,
                video_scene_path=scene_summaries_dir,
                audio_caption_path=audio_caption_file,
                output_path=shot_point_output_path,
                max_iterations=max_iterations,
                video_path=Video_Path,
                video_reader=vr.get("video_reader") if isinstance(vr, dict) else vr,
                frame_folder_path=frames_dir,
                transcript_path=srt_with_characters if os.path.exists(srt_with_characters) else srt_path
            )
            _messages = editor_agent.run(shot_plan_path=shot_plan_output_path)

        print(f"\n{'='*80}")
        print(f"🎉 Video clip selection completed!")
        print(f"💾 Output saved to: {shot_point_output_path}")
        print(f"{'='*80}\n")
    else:
        print("\n" + "="*80)
        print("❌ Cannot run EditorCoreAgent - missing required files:")
        if not os.path.exists(scene_summaries_dir):
            print(f"  ❌ Scene summaries directory not found at {scene_summaries_dir}")
        if audio_caption_file and not os.path.exists(audio_caption_file):
            print(f"  ❌ Audio caption file not found at {audio_caption_file}")
        if not audio_caption_file:
            print(f"  ℹ️  Audio not provided (analysis-only mode) - skipped")
        if not os.path.exists(shot_plan_output_path):
            print(f"  ❌ Shot plan file not found at {shot_plan_output_path}")
        print("="*80 + "\n")


if __name__ == "__main__":
    main()
