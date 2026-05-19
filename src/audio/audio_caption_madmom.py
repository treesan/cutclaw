# Original Author: Shifang Zhao
# Optimized by: Pengo0o (Focus: Acceleration)
"""
Audio Caption Module using Madmom-based Segmentation

This module provides audio captioning functionality using cloud API (litellm)
with Madmom keypoint detection for intelligent audio segmentation.

Features:
    - Madmom integration: use music keypoints (beats, onsets) for segmentation
    - Concurrent cloud API calls for efficient parallel analysis
    - Configurable parameters with config file defaults

Requirements:
    - litellm
    - python-dotenv
    - soundfile
    - madmom
    - numpy

Usage:
    from vca.build_database.audio_caption_madmom import caption_audio_with_madmom_segments

    result = caption_audio_with_madmom_segments(
        audio_path="/path/to/audio.mp3",
        output_path="output.json",
    )
"""

import json
import os
import re
import sys
import tempfile
from typing import Dict, List, Optional

import soundfile as sf
import numpy as np

from src.audio.litellm_client import call_audio_api, call_audio_api_batch
from src.audio.litellm_client import AUDIO_MODEL as _LITELLM_AUDIO_MODEL
from .. import config


# --------------------------------------------------------------------------- #
def _get_model_name() -> str:
    """Return the model name configured for audio calls, for error messages."""
    return _LITELLM_AUDIO_MODEL


# --------------------------------------------------------------------------- #
#                              Prompt templates                               #
# --------------------------------------------------------------------------- #

AUDIO_OVERALL_PROMPT = """
You are a professional music analyst. Analyze this audio and identify its structural sections with PRECISE timestamps.

## Task
1. Identify the song structure by detecting changes in: melody, instrumentation, lyrics, energy level, rhythm patterns
2. Label each section with standard music terminology

## Standard Song Structure Types (use these names):
- Intro: Opening instrumental/ambient section
- Verse (Verse 1, Verse 2, etc.): Main storytelling sections with lyrics
- Chorus (Chorus 1, Chorus 2, etc.): Main hook/repeated section
- Bridge: Contrasting section, usually appears once
- Build-up: Rising tension section
- Drop: High-energy climax section (EDM/electronic)
- Outro: Closing section

## CRITICAL TIME CONSTRAINTS (MUST FOLLOW):
- MAXIMUM section length: 45 seconds
- MINIMUM section length: 15 seconds
- If you detect a section longer than 45 seconds, you MUST split it (e.g., "Verse 1 Part A" 00:30-01:05, "Verse 1 Part B" 01:05-01:40)

## Detection Tips:
- Listen for lyrical changes (new verse = new lyrics)
- Listen for melodic changes (chorus usually has different melody than verse)
- Listen for instrumental changes (intro/outro often instrumental)
- Listen for energy/dynamics changes (build-ups, drops, breakdowns)
- Listen for repeated sections (chorus repeats multiple times)

Output ONLY valid JSON:
{
  "summary": "Genre, overall mood, and key musical characteristics",
  "sections": [
    {
      "name": "Intro",
      "description": "Brief description of what happens in this section",
      "Start_Time": "00:00",
      "End_Time": "00:25"
    },
    {
      "name": "Verse 1",
      "description": "Description of lyrics and instrumentation",
      "Start_Time": "00:25",
      "End_Time": "01:05"
    }
  ]
}
"""

AUDIO_SEG_KEYPOINT_PROMPT = """You are a professional music analyst for video editing. Analyze this audio segment and describe its characteristics for matching with video footage.

Focus on:
- Musical style and instrumentation
- Emotional atmosphere and mood
- Energy dynamics and intensity changes
- Rhythmic patterns and tempo feel

Output ONLY valid JSON (no markdown, no explanation):
{
  "summary": "Description of genre, instrumentation, and overall mood",
  "emotion": "Primary emotional tone (e.g., energetic, melancholic, uplifting, tense, romantic, triumphant, mysterious, nostalgic)",
  "energy": "Energy level 1-10 with trend (e.g., '7, building intensity', '3, calm and steady', '9, explosive climax', '5, gradually fading')",
  "rhythm": "Tempo and rhythmic feel (e.g., '128 BPM, driving electronic beat', '85 BPM, relaxed groove', '60 BPM, slow ambient pulse', 'free tempo, atmospheric')"
}"""



# --------------------------------------------------------------------------- #
#                           Time Format Helper                                #
# --------------------------------------------------------------------------- #
def seconds_to_mmss(seconds: float) -> str:
    """
    Convert seconds to MM:SS.f format (with one decimal place).

    Args:
        seconds: Time in seconds

    Returns:
        Time string in MM:SS.f format
    """
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes:02d}:{secs:04.1f}"


# --------------------------------------------------------------------------- #
#                           JSON Parsing Helper                               #
# --------------------------------------------------------------------------- #
def extract_json_from_text(text: str) -> Optional[Dict]:
    """
    Extract and parse JSON from text that may contain additional content.

    Args:
        text: Text that may contain JSON

    Returns:
        Parsed JSON dict or None if no valid JSON found
    """
    # Try to find JSON block in the text
    # Look for content between { and }
    json_match = re.search(r'\{[\s\S]*\}', text)

    if json_match:
        json_str = json_match.group(0)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"Warning: Found JSON-like structure but failed to parse: {e}")
            return None

    # If no JSON found, return None
    return None


# --------------------------------------------------------------------------- #
#                           Audio Segmentation Helper                         #
# --------------------------------------------------------------------------- #
def segment_audio_file(
    audio_path: str,
    start_time: float,
    end_time: float,
    output_path: str = None
) -> str:
    """
    Extract a segment from an audio file.

    Args:
        audio_path: Path to the source audio file
        start_time: Start time in seconds
        end_time: End time in seconds
        output_path: Optional output path (if None, creates a temp file)

    Returns:
        Path to the segmented audio file
    """
    # Read the audio file
    audio_data, sample_rate = sf.read(audio_path)

    # Calculate start and end samples
    start_sample = int(start_time * sample_rate)
    end_sample = int(end_time * sample_rate)

    # Extract the segment
    segment = audio_data[start_sample:end_sample]

    # Create output path if not specified
    if output_path is None:
        # Create a temporary file
        temp_fd, output_path = tempfile.mkstemp(suffix=".wav")
        os.close(temp_fd)  # Close the file descriptor

    # Write the segment to file
    sf.write(output_path, segment, sample_rate)

    return output_path


# --------------------------------------------------------------------------- #
#                    Batch Caption Function for Multiple Segments             #
# --------------------------------------------------------------------------- #
def generate_audio_captions_batch(
    audio_paths: List[str],
    prompt: str,
    temperature: float = 0.7,
    top_p: float = 0.95,
    max_tokens: int = 4096,
    max_workers: int = 5,
) -> List[str]:
    """
    Generate audio captions for multiple audio files via concurrent cloud API calls.

    Args:
        audio_paths: List of paths to audio files
        prompt: User prompt for caption generation
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        max_tokens: Maximum tokens to generate
        max_workers: Max concurrent API requests

    Returns:
        List of generated caption strings
    """
    return call_audio_api_batch(
        audio_paths=audio_paths,
        prompt=prompt,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        max_workers=max_workers,
    )


# --------------------------------------------------------------------------- #
#                    Generate Overall Audio Analysis                          #
# --------------------------------------------------------------------------- #
def generate_overall_analysis(
    audio_path: str,
    temperature: float = 0.7,
    top_p: float = 0.95,
    max_tokens: int = 4096,
    audio_duration: float = None,
) -> str:
    """
    Generate overall analysis for the entire audio file via cloud API.

    Args:
        audio_path: Path to the audio file
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        max_tokens: Maximum tokens to generate
        audio_duration: Audio duration in seconds (optional, for constraining timestamps)

    Returns:
        Generated overall analysis text
    """
    prompt = AUDIO_OVERALL_PROMPT
    if audio_duration:
        duration_str = f"{int(audio_duration // 60):02d}:{int(audio_duration % 60):02d}"
        duration_constraint = (
            f"\n\n## IMPORTANT: Audio Duration Constraint\n"
            f"This audio is exactly {duration_str} ({audio_duration:.1f} seconds) long.\n"
            f"ALL timestamps MUST be within 00:00 to {duration_str}. "
            f"Do NOT generate any timestamp beyond {duration_str}!\n"
            f"The last section MUST end at {duration_str}.\n"
        )
        prompt = prompt + duration_constraint

    return call_audio_api(audio_path, prompt, temperature, top_p, max_tokens)


# --------------------------------------------------------------------------- #
#                    Time Parsing Helper                                      #
# --------------------------------------------------------------------------- #
def mmss_to_seconds(mmss: str) -> float:
    """
    Convert MM:SS or MM:SS.f format to seconds.

    Args:
        mmss: Time string in MM:SS or MM:SS.f format

    Returns:
        Time in seconds
    """
    try:
        parts = mmss.split(':')
        if len(parts) == 2:
            minutes = int(parts[0])
            secs = float(parts[1])
            return minutes * 60 + secs
        else:
            return float(mmss)
    except (ValueError, AttributeError):
        return 0.0


def validate_sections_within_duration(sections: List[Dict], audio_duration: float, tolerance: float = 1.0) -> tuple:
    """
    Validate that all sections are within the audio duration.

    Args:
        sections: List of section dicts with Start_Time and End_Time
        audio_duration: Audio duration in seconds
        tolerance: Allowed tolerance for boundary checking (seconds)

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not sections or not audio_duration:
        return True, ""

    max_allowed = audio_duration + tolerance

    for idx, sec in enumerate(sections):
        try:
            start_time = mmss_to_seconds(sec.get("Start_Time", "00:00"))
            end_time = mmss_to_seconds(sec.get("End_Time", "00:00"))

            if start_time > max_allowed:
                return False, f"Section {idx+1} '{sec.get('name', 'Unknown')}' start_time ({start_time:.1f}s) exceeds audio duration ({audio_duration:.1f}s)"

            if end_time > max_allowed:
                return False, f"Section {idx+1} '{sec.get('name', 'Unknown')}' end_time ({end_time:.1f}s) exceeds audio duration ({audio_duration:.1f}s)"

        except Exception as e:
            return False, f"Section {idx+1} time parsing error: {e}"

    return True, ""


def validate_section_durations(sections: List[Dict], min_duration: float = 5.0, max_duration: float = 90.0) -> tuple:
    """
    Validate that all sections have duration within the specified range.

    Args:
        sections: List of section dicts with Start_Time and End_Time
        min_duration: Minimum allowed section duration in seconds (default: 5s)
        max_duration: Maximum allowed section duration in seconds (default: 90s)

    Returns:
        Tuple of (is_valid, error_message, invalid_sections_info)
    """
    if not sections:
        return True, "", []

    invalid_sections = []

    for idx, sec in enumerate(sections):
        try:
            start_time = mmss_to_seconds(sec.get("Start_Time", "00:00"))
            end_time = mmss_to_seconds(sec.get("End_Time", "00:00"))
            duration = end_time - start_time

            if duration < min_duration:
                invalid_sections.append({
                    'index': idx + 1,
                    'name': sec.get('name', 'Unknown'),
                    'duration': duration,
                    'issue': 'too_short'
                })
            elif duration > max_duration:
                invalid_sections.append({
                    'index': idx + 1,
                    'name': sec.get('name', 'Unknown'),
                    'duration': duration,
                    'issue': 'too_long'
                })

        except Exception as e:
            invalid_sections.append({
                'index': idx + 1,
                'name': sec.get('name', 'Unknown'),
                'duration': 0,
                'issue': f'parse_error: {e}'
            })

    if invalid_sections:
        error_msgs = []
        for inv in invalid_sections:
            if inv['issue'] == 'too_short':
                error_msgs.append(f"Section {inv['index']} '{inv['name']}' is too short ({inv['duration']:.1f}s < {min_duration}s)")
            elif inv['issue'] == 'too_long':
                error_msgs.append(f"Section {inv['index']} '{inv['name']}' is too long ({inv['duration']:.1f}s > {max_duration}s)")
            else:
                error_msgs.append(f"Section {inv['index']} '{inv['name']}': {inv['issue']}")
        return False, "; ".join(error_msgs), invalid_sections

    return True, "", []


# --------------------------------------------------------------------------- #
#                    Split Point Search Helper                                #
# --------------------------------------------------------------------------- #
def _find_split_points_near_midpoints(
    start_time: float,
    end_time: float,
    num_parts: int,
    all_keypoints: List[Dict],
    search_radius: float = 3.0
) -> List[float]:
    """
    在均分点附近搜索最接近的关键点作为分割点（参考 interactive 逻辑）。

    Args:
        start_time: 片段开始时间
        end_time: 片段结束时间
        num_parts: 需要分成的部分数
        all_keypoints: 所有原始关键点列表
        search_radius: 搜索半径（秒），在均分点 ± search_radius 范围内搜索

    Returns:
        分割点时间列表（包含 start_time 和 end_time）
    """
    duration = end_time - start_time
    part_duration = duration / num_parts

    # 计算理想的均分点
    ideal_midpoints = []
    for i in range(1, num_parts):
        ideal_midpoints.append(start_time + i * part_duration)

    # 找到位于这个间隔内的所有候选点（从原始 keypoints 中）
    candidates = [
        kp for kp in all_keypoints
        if start_time < kp['time'] < end_time
    ]

    # 对每个均分点，找到最接近的候选点
    actual_split_points = []
    used_indices = set()  # 记录已使用的候选点索引，避免重复使用

    for midpoint in ideal_midpoints:
        # 首先尝试在 search_radius 范围内找最强的点
        nearby_candidates = [
            (idx, kp) for idx, kp in enumerate(candidates)
            if idx not in used_indices and midpoint - search_radius <= kp['time'] <= midpoint + search_radius
        ]

        if nearby_candidates:
            # 在附近找到了候选点，选择强度最高的
            best_idx, best = max(nearby_candidates, key=lambda x: x[1].get('normalized_intensity', x[1].get('intensity', 0)))
            actual_split_points.append(best['time'])
            used_indices.add(best_idx)
            offset = best['time'] - midpoint
            offset_str = f"+{offset:.2f}s" if offset >= 0 else f"{offset:.2f}s"
            print(f"      ✓ midpoint {midpoint:.2f}s → keypoint {best['time']:.2f}s ({offset_str}, "
                  f"type={best.get('type', 'Unknown')[:20]}, intensity={best.get('intensity', 0):.3f})")
        else:
            # 附近没找到，从所有候选点中找最接近的
            available_candidates = [
                (idx, kp) for idx, kp in enumerate(candidates)
                if idx not in used_indices
            ]

            if available_candidates:
                # 找到最接近理想时间的候选点
                best_idx, best = min(available_candidates, key=lambda x: abs(x[1]['time'] - midpoint))
                actual_split_points.append(best['time'])
                used_indices.add(best_idx)
                offset = best['time'] - midpoint
                offset_str = f"+{offset:.2f}s" if offset >= 0 else f"{offset:.2f}s"
                print(f"      ○ midpoint {midpoint:.2f}s → closest keypoint {best['time']:.2f}s ({offset_str}, "
                      f"type={best.get('type', 'Unknown')[:20]}, intensity={best.get('intensity', 0):.3f})")
            else:
                # 所有候选点都用完了，使用均分点（这种情况很少见）
                actual_split_points.append(midpoint)
                print(f"      ✗ midpoint {midpoint:.2f}s → no available keypoints, using midpoint")

    # 去重并排序
    actual_split_points = sorted(set(actual_split_points))

    # 构建完整的分割点列表
    return [start_time] + actual_split_points + [end_time]


# --------------------------------------------------------------------------- #
#                    Madmom-based Segmentation Function                       #
# --------------------------------------------------------------------------- #
def caption_audio_with_madmom_segments(
    audio_path: str,
    output_path: Optional[str] = None,
    max_tokens: int = 4096,
    temperature: float = 0.7,
    top_p: float = 0.95,
    max_workers: int = None,
    batch_size: int = None,  # deprecated alias for max_workers
    # Detection method selection (NEW: supports multiple methods like interactive)
    detection_methods: List[str] = None,  # ["downbeat", "pitch", "mel_energy"]
    # Madmom detection parameters (downbeat)
    onset_threshold: float = None,  # DEPRECATED
    onset_smooth: float = 0.5,  # DEPRECATED
    onset_pre_avg: float = 0.5,  # DEPRECATED
    onset_post_avg: float = 0.5,  # DEPRECATED
    onset_pre_max: float = 0.5,  # DEPRECATED
    onset_post_max: float = 0.5,  # DEPRECATED
    onset_combine: float = None,  # DEPRECATED
    beats_per_bar: list = None,
    min_bpm: float = None,
    max_bpm: float = None,
    # Pitch detection parameters
    pitch_tolerance: float = None,
    pitch_threshold: float = None,
    pitch_min_distance: float = None,
    pitch_nms_method: str = None,
    pitch_max_points: int = None,
    # Mel energy detection parameters
    mel_win_s: int = None,
    mel_n_filters: int = None,
    mel_threshold_ratio: float = None,
    mel_min_distance: float = None,
    mel_nms_method: str = None,
    mel_max_points: int = None,
    # Filtering parameters
    min_segment_duration: float = None,
    max_segment_duration: float = None,
    merge_close: float = None,
    min_interval: float = 0.0,
    top_k_keypoints: int = 0,
    energy_percentile: float = 0.0,
    # Section-based filtering (if using stage1 sections)
    use_stage1_sections: bool = None,
    section_min_interval: float = None,
) -> Dict:
    """
    Generate caption for an audio file using Madmom keypoints for segmentation.

    This function produces a two-level hierarchical structure:
    - Level 1: High-level sections from overall audio analysis (Intro, Verse, Chorus, etc.)
    - Level 2: Fine-grained sub-segments within each section based on Madmom keypoints

    The processing pipeline:
    1. Detect audio keypoints using Madmom (beats, onsets, spectral changes) + Rule-based filtering
       - Apply merge_close, min_interval, top_k, energy_percentile filters
    2. Use AI model to generate overall analysis and identify Level 1 sections
    3. For each Level 1 section, create Level 2 sub-segments using filtered Madmom keypoints
    4. Use AI model to analyze each sub-segment in detail (caption generation)
    5. Merge into two-level output format

    Args:
        audio_path: Path to the audio file
        output_path: Optional path to save the caption as JSON
        model_path: Model checkpoint path (default: from config)
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        top_k: Top-k sampling parameter
        use_flash_attn2: Whether to use flash attention 2
        batch_size: Number of audio segments to process in parallel

        # Madmom detection parameters
        onset_threshold: Onset detection threshold (higher = fewer onsets)
        onset_smooth: Smoothing window size for onset activation
        onset_pre_avg: Pre-averaging window for onset detection
        onset_post_avg: Post-averaging window for onset detection
        onset_pre_max: Pre-max window for onset peak picking
        onset_post_max: Post-max window for onset peak picking
        onset_combine: Time window for combining nearby onsets
        beats_per_bar: Beats per bar for rhythm detection (default: [4])
        min_bpm: Minimum BPM for beat detection
        max_bpm: Maximum BPM for beat detection

        # Filtering parameters
        min_segment_duration: Minimum segment duration in seconds
        max_segment_duration: Maximum segment duration in seconds
        merge_close: Merge keypoints closer than this threshold
        min_interval: Minimum interval between keypoints
        top_k_keypoints: Keep only top K keypoints by intensity (0 = no limit)
        energy_percentile: Keep only keypoints above this energy percentile

        # Section-based filtering (alternative to simple filtering)
        use_stage1_sections: If True, first identify sections then find keypoints per section
        section_min_interval: Minimum interval between keypoints (global, across all sections)

    Returns:
        Dictionary containing the complete caption with segment details
    """
    # Import madmom filtering functions (detection now uses madmom_api)
    madmom_module_path = os.path.join(os.path.dirname(__file__))
    if madmom_module_path not in sys.path:
        sys.path.insert(0, madmom_module_path)

    from audio_Madmom import (
        filter_significant_keypoints,
        filter_by_sections,
    )

    # Resolve max_workers (batch_size is a deprecated alias)
    if max_workers is None:
        max_workers = batch_size if batch_size is not None else getattr(config, 'AUDIO_BATCH_SIZE', 5)

    # Detection methods (NEW: support multiple methods like interactive)
    if detection_methods is None:
        detection_methods = ["downbeat"]  # Default to downbeat only
    if isinstance(detection_methods, str):
        detection_methods = [detection_methods]
    # Validate methods
    valid_methods = {"downbeat", "pitch", "mel_energy"}
    detection_methods = [m for m in detection_methods if m in valid_methods]
    if not detection_methods:
        detection_methods = ["downbeat"]

    # Downbeat parameters
    if min_bpm is None:
        min_bpm = getattr(config, 'AUDIO_MIN_BPM', 55.0)
    if max_bpm is None:
        max_bpm = getattr(config, 'AUDIO_MAX_BPM', 215.0)
    if beats_per_bar is None:
        beats_per_bar = [4]

    # Pitch parameters
    if pitch_tolerance is None:
        pitch_tolerance = getattr(config, 'AUDIO_PITCH_TOLERANCE', 0.8)
    if pitch_threshold is None:
        pitch_threshold = getattr(config, 'AUDIO_PITCH_THRESHOLD', 0.8)
    if pitch_min_distance is None:
        pitch_min_distance = getattr(config, 'AUDIO_PITCH_MIN_DISTANCE', 0.3)
    if pitch_nms_method is None:
        pitch_nms_method = getattr(config, 'AUDIO_PITCH_NMS_METHOD', "basic")
    if pitch_max_points is None:
        pitch_max_points = getattr(config, 'AUDIO_PITCH_MAX_POINTS', 50)

    # Mel energy parameters
    if mel_win_s is None:
        mel_win_s = getattr(config, 'AUDIO_MEL_WIN_S', 512)
    if mel_n_filters is None:
        mel_n_filters = getattr(config, 'AUDIO_MEL_N_FILTERS', 40)
    if mel_threshold_ratio is None:
        mel_threshold_ratio = getattr(config, 'AUDIO_MEL_THRESHOLD_RATIO', 0.3)
    if mel_min_distance is None:
        mel_min_distance = getattr(config, 'AUDIO_MEL_MIN_DISTANCE', 0.3)
    if mel_nms_method is None:
        mel_nms_method = getattr(config, 'AUDIO_MEL_NMS_METHOD', "basic")
    if mel_max_points is None:
        mel_max_points = getattr(config, 'AUDIO_MEL_MAX_POINTS', 50)

    # Other parameters
    if min_segment_duration is None:
        min_segment_duration = getattr(config, 'AUDIO_MIN_SEGMENT_DURATION', 3.0)
    if max_segment_duration is None:
        max_segment_duration = getattr(config, 'AUDIO_MAX_SEGMENT_DURATION', 30.0)
    if merge_close is None:
        merge_close = getattr(config, 'AUDIO_MERGE_CLOSE', 0.1)
    if use_stage1_sections is None:
        use_stage1_sections = getattr(config, 'AUDIO_USE_STAGE1_SECTIONS', False)
    if section_min_interval is None:
        section_min_interval = getattr(config, 'AUDIO_SECTION_MIN_INTERVAL', 3.0)

    # Check if audio file exists
    if not audio_path.startswith("http://") and not audio_path.startswith("https://"):
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

    print("\n" + "="*80)
    print("MADMOM-BASED SEGMENTATION ANALYSIS")
    print("="*80)
    print("\nProcessing Pipeline:")
    print("  1. Madmom keypoint detection + Rule-based filtering")
    print("  2. AI-based Level 1 section segmentation")
    print("  3. AI-based caption for each sub-segment")
    print("  4. Merge into two-level output format")

    # Get audio duration
    audio_duration = None
    try:
        if not audio_path.startswith("http://") and not audio_path.startswith("https://"):
            info = sf.info(audio_path)
            audio_duration = info.duration
            duration_str = f"{int(audio_duration // 60):02d}:{int(audio_duration % 60):02d}"
            print(f"\n✓ Audio duration: {duration_str} ({audio_duration:.2f} seconds)")
    except Exception as e:
        print(f"⚠ Warning: Could not determine audio duration: {e}")

    # Stage 1: Detect keypoints using Madmom (support multiple methods)
    print("\n" + "="*80)
    print("STAGE 1: Madmom keypoint detection + Rule-based filtering")
    print("="*80)
    print(f"\n[Step 1.1] Detecting audio keypoints with Madmom (methods: {', '.join(detection_methods)})...")
    print("  Using madmom_api to ensure alignment with interactive interface")

    # Import madmom_api to ensure 100% alignment with interactive version
    from src.audio.madmom_api import detect_keypoints_madmom

    # Run selected methods and merge keypoints (using madmom_api for consistency)
    merged_keypoints = []
    for method in detection_methods:
        print(f"\n  → Running {method} detection...")

        # Use madmom_api (same as interactive interface)
        result = detect_keypoints_madmom(
            audio_path=audio_path,
            detection_method=method,
            # Downbeat parameters (used only if method="downbeat")
            beats_per_bar=beats_per_bar[0] if beats_per_bar else 4,  # Convert list to int
            min_bpm=min_bpm,
            max_bpm=max_bpm,
            num_tempi=60,
            transition_lambda=100,
            observation_lambda=16,
            dbn_threshold=0.05,
            correct_beats=True,
            fps=100,
            # Pitch parameters (used only if method="pitch")
            pitch_tolerance=pitch_tolerance,
            pitch_threshold=pitch_threshold,
            pitch_min_distance=pitch_min_distance,
            pitch_nms_method=pitch_nms_method,
            pitch_max_points=pitch_max_points,
            # Mel energy parameters (used only if method="mel_energy")
            mel_win_s=mel_win_s,
            mel_n_filters=mel_n_filters,
            mel_threshold_ratio=mel_threshold_ratio,
            mel_min_distance=mel_min_distance,
            mel_nms_method=mel_nms_method,
            mel_max_points=mel_max_points,
            # Silence filtering (from config)
            silence_filter=True,
            silence_threshold_db=getattr(config, 'AUDIO_SILENCE_THRESHOLD_DB', -45.0),
            # Disable post-filtering here (will apply later in unified manner)
            min_interval=0.0,
            top_k=0,
            energy_percentile=0.0,
            return_python_types=True,
        )

        method_keypoints = result.get('keypoints', [])
        merged_keypoints.extend(method_keypoints)
        print(f"    ✓ Detected {len(method_keypoints)} {method} keypoints")

    # Use merged keypoints
    keypoints = merged_keypoints
    print(f"\n✓ Total detected keypoints: {len(keypoints)} (from {len(detection_methods)} method(s))")

    # Stage 1.5: 规则过滤 - 按照配置参数过滤分割点
    print("\n[Step 1.2] Applying rule-based filtering...")
    print(f"  Parameters: min_interval={min_interval}s, top_k={top_k_keypoints}, energy_percentile={energy_percentile}")

    # 应用所有规则过滤
    filtered_keypoints = filter_significant_keypoints(
        keypoints=keypoints,
        min_interval=min_interval,
        top_k=top_k_keypoints,
        energy_percentile=energy_percentile,
        use_normalized_intensity=True
    )

    print(f"✓ After rule-based filtering: {len(filtered_keypoints)} keypoints")

    # Stage 2: Generate overall analysis (Level 1 sections) using AI model
    print("\n" + "="*80)
    print("STAGE 2: AI-based Level 1 section segmentation")
    print("="*80)
    print("\nUsing AI model to identify high-level sections (Intro, Verse, Chorus, etc.)...")

    # Retry logic for generating valid sections
    MAX_RETRIES = 5  # Increased retries for duration validation
    SECTION_MIN_DURATION = 5.0   # Minimum section duration in seconds (relaxed from 15s in prompt)
    SECTION_MAX_DURATION = 60.0  # Maximum section duration in seconds (MUST match prompt!)
    overall_summary = ""
    stage1_sections = []

    for retry_attempt in range(MAX_RETRIES):
        if retry_attempt > 0:
            print(f"\n⚠ Retry attempt {retry_attempt + 1}/{MAX_RETRIES}...")

        print("\nGenerating overall analysis for the entire audio...")
        try:
            overall_analysis_text = generate_overall_analysis(
                audio_path=audio_path,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                audio_duration=audio_duration,
            )
        except Exception as e:
            print(f"⚠ AI overall analysis failed: {e}")
            print(f"  The current model '{_get_model_name()}' may not support audio input.")
            print(f"  Falling back to default sections based on Madmom keypoints.")
            overall_analysis_text = ""
            stage1_sections = []
            overall_summary = ""
            break  # Exit retry loop immediately — model clearly doesn't support audio

        # Try to parse JSON from overall analysis
        overall_json = extract_json_from_text(overall_analysis_text)
        if overall_json and isinstance(overall_json, dict):
            overall_summary = overall_json.get("summary", "")
            stage1_sections = overall_json.get("sections", [])
            print(f"✓ Overall analysis generated successfully")
            print(f"  Found {len(stage1_sections)} Level 1 sections from overall analysis")

            # Validation 1: Check sections are within audio duration
            if audio_duration and stage1_sections:
                is_valid, error_msg = validate_sections_within_duration(stage1_sections, audio_duration)
                if not is_valid:
                    print(f"⚠ Section boundary validation failed: {error_msg}")
                    if retry_attempt < MAX_RETRIES - 1:
                        print(f"  Will retry generation...")
                        stage1_sections = []  # Clear invalid sections
                        continue
                    else:
                        print(f"  Max retries reached, will auto-fix boundaries")
                else:
                    print(f"✓ All sections are within audio duration ({audio_duration:.1f}s)")

            # Validation 2: Check section durations are within 5-90 seconds
            if stage1_sections:
                is_duration_valid, duration_error_msg, invalid_sections = validate_section_durations(
                    stage1_sections,
                    min_duration=SECTION_MIN_DURATION,
                    max_duration=SECTION_MAX_DURATION
                )
                if not is_duration_valid:
                    print(f"⚠ Section duration validation failed:")
                    for inv in invalid_sections:
                        if inv['issue'] == 'too_short':
                            print(f"    - Section {inv['index']} '{inv['name']}': {inv['duration']:.1f}s < {SECTION_MIN_DURATION}s (too short)")
                        elif inv['issue'] == 'too_long':
                            print(f"    - Section {inv['index']} '{inv['name']}': {inv['duration']:.1f}s > {SECTION_MAX_DURATION}s (too long)")
                    if retry_attempt < MAX_RETRIES - 1:
                        print(f"  Will retry generation...")
                        stage1_sections = []  # Clear invalid sections
                        continue
                    else:
                        print(f"  Max retries reached, will use current sections")
                else:
                    print(f"✓ All sections have valid duration ({SECTION_MIN_DURATION}s - {SECTION_MAX_DURATION}s)")

            # All validations passed
            break
        else:
            overall_summary = overall_analysis_text
            stage1_sections = []
            print(f"⚠ Overall analysis generated but JSON parsing failed")
            if retry_attempt < MAX_RETRIES - 1:
                print(f"  Will retry generation...")
            else:
                print(f"  Max retries reached, will create default sections")

    # Auto-fix sections that exceed audio duration (after max retries)
    if audio_duration and stage1_sections:
        fixed_sections = []
        for sec in stage1_sections:
            start_time = mmss_to_seconds(sec.get("Start_Time", "00:00"))
            end_time = mmss_to_seconds(sec.get("End_Time", "00:00"))

            # Skip sections that start beyond audio duration
            if start_time >= audio_duration:
                print(f"  ⚠ Removing section '{sec.get('name', 'Unknown')}' (starts at {start_time:.1f}s, beyond audio duration)")
                continue

            # Clamp end time to audio duration
            if end_time > audio_duration:
                print(f"  ⚠ Clamping section '{sec.get('name', 'Unknown')}' end_time from {end_time:.1f}s to {audio_duration:.1f}s")
                sec["End_Time"] = seconds_to_mmss(audio_duration)

            fixed_sections.append(sec)

        stage1_sections = fixed_sections
        print(f"✓ After auto-fix: {len(stage1_sections)} valid sections")

    # If no sections from stage1, create default sections based on audio duration
    if not stage1_sections:
        print("\n  Creating default Level 1 sections based on audio duration...")
        # 每30秒为一个默认段落
        default_section_duration = 30.0
        section_start = 0.0
        section_idx = 1
        while section_start < audio_duration:
            section_end = min(section_start + default_section_duration, audio_duration)
            stage1_sections.append({
                "name": f"Section {section_idx}",
                "description": "",
                "Start_Time": seconds_to_mmss(section_start),
                "End_Time": seconds_to_mmss(section_end)
            })
            section_start = section_end
            section_idx += 1
        print(f"  Created {len(stage1_sections)} default sections (每{default_section_duration}s一段)")

    # Stage 2.5: 基于 sections 的分割点过滤
    # 如果启用 use_stage1_sections，使用基于段落的过滤方法
    if use_stage1_sections and stage1_sections:
        print("\n" + "-"*80)
        print("[Step 2.5] Applying section-based keypoint filtering...")
        print("-"*80)

        # 将 stage1_sections 转换为 filter_by_sections 需要的格式
        sections_for_filter = []
        for sec in stage1_sections:
            try:
                start_time = mmss_to_seconds(sec.get("Start_Time", "00:00"))
                end_time = mmss_to_seconds(sec.get("End_Time", "00:00"))
                if end_time > start_time:
                    sections_for_filter.append({
                        'name': sec.get('name', 'Unknown'),
                        'start_time': start_time,
                        'end_time': end_time,
                        'duration': end_time - start_time
                    })
            except Exception as e:
                print(f"⚠ Warning: Failed to parse section: {sec.get('name', 'Unknown')} - {e}")

        if sections_for_filter:
            # 如果 section_min_interval 未设置（0），使用 min_segment_duration
            # 这样确保选择的关键点之间间隔足够，不会产生太短的片段
            effective_min_interval = section_min_interval if section_min_interval > 0 else min_segment_duration

            print(f"  Filtering based on {len(sections_for_filter)} sections")
            print(f"  (section_min_interval={effective_min_interval}s to avoid short segments)")
            if config.AUDIO_TOTAL_SHOTS is not None:
                print(f"  (total_shots={config.AUDIO_TOTAL_SHOTS} for proportional allocation)")
            filtered_keypoints = filter_by_sections(
                keypoints=filtered_keypoints,
                sections=sections_for_filter,
                section_min_interval=effective_min_interval,
                use_normalized_intensity=True,
                min_segment_duration=min_segment_duration,
                max_segment_duration=max_segment_duration,
                total_shots=config.AUDIO_TOTAL_SHOTS,
                audio_duration=audio_duration,  # CRITICAL: needed for boundary checks
                weight_downbeat=config.AUDIO_WEIGHT_DOWNBEAT,
                weight_pitch=config.AUDIO_WEIGHT_PITCH,
                weight_mel_energy=config.AUDIO_WEIGHT_MEL_ENERGY,
            )
            print(f"✓ After section-based filtering: {len(filtered_keypoints)} keypoints")
        else:
            print("⚠ No valid sections found, skipping section-based filtering")

    print(f"\n✓ Using {len(filtered_keypoints)} filtered keypoints for segmentation")

    # Debug: 打印所有分割点的详细信息
    print("\n" + "-"*80)
    print("[DEBUG] Final keypoints for segmentation:")
    print("-"*80)
    print(f"{'#':>3} | {'Time':>8} | {'Type':<30} | {'Intensity':>9} | {'Norm_Int':>9} | {'Section':<15}")
    print("-"*100)
    for idx, kp in enumerate(filtered_keypoints):
        time_str = f"{kp['time']:.2f}s"
        kp_type = kp.get('type', 'Unknown')[:30]
        intensity = kp.get('intensity', 0)
        norm_intensity = kp.get('normalized_intensity', intensity)
        section = kp.get('section', '-')[:15]
        boosted = " *" if kp.get('type_boosted', False) else ""
        print(f"{idx+1:>3} | {time_str:>8} | {kp_type:<30} | {intensity:>9.4f} | {norm_intensity:>9.4f} | {section:<15}{boosted}")
    print("-"*100)
    if any(kp.get('type_boosted', False) for kp in filtered_keypoints):
        print("  * = type_boosted (权重已增强)")
    print()

    # Stage 2.6: 将 Level 1 sections 的边界吸附到最近的分割点
    print("\n" + "-"*80)
    print("[Step 2.6] Snapping Level 1 section boundaries to nearest keypoints...")
    print("-"*80)

    # 获取所有分割点时间（包括音频开头和结尾）
    keypoint_times = sorted([kp['time'] for kp in filtered_keypoints])
    snap_points = [0.0] + keypoint_times + [audio_duration] if audio_duration else [0.0] + keypoint_times
    snap_points = sorted(set(snap_points))  # 去重并排序

    def find_nearest_snap_point(t, snap_points, exclude_exact=False):
        """找到最近的吸附点"""
        if not snap_points:
            return t
        # 找到最近的点
        min_dist = float('inf')
        nearest = t
        for sp in snap_points:
            if exclude_exact and abs(sp - t) < 0.01:
                continue
            dist = abs(sp - t)
            if dist < min_dist:
                min_dist = dist
                nearest = sp
        return nearest

    # 对每个 section 的边界进行吸附
    snapped_sections = []
    for section_idx, stage1_sec in enumerate(stage1_sections):
        original_start = mmss_to_seconds(stage1_sec.get("Start_Time", "00:00"))
        original_end = mmss_to_seconds(stage1_sec.get("End_Time", "00:00"))

        # 第一个 section 的开始固定为 0
        if section_idx == 0:
            snapped_start = 0.0
        else:
            prev_snapped_end = snapped_sections[-1]['snapped_end']
            gap = original_start - snapped_sections[-1]['original_end']
            # 如果两段之间的空洞接近 min_segment_duration，插入一个 gap section
            if gap >= min_segment_duration * 0.5:
                gap_sec = {
                    'original': {
                        'name': f'Gap {section_idx}',
                        'description': '',
                        'Start_Time': seconds_to_mmss(prev_snapped_end),
                        'End_Time': seconds_to_mmss(original_start),
                    },
                    'original_start': prev_snapped_end,
                    'original_end': original_start,
                    'snapped_start': prev_snapped_end,
                    'snapped_end': original_start,
                    'is_gap': True,
                }
                snapped_sections.append(gap_sec)
                print(f"  [Gap {section_idx}] inserted gap section "
                      f"{seconds_to_mmss(prev_snapped_end)}-{seconds_to_mmss(original_start)} "
                      f"(gap={gap:.2f}s >= {min_segment_duration * 0.5:.2f}s threshold)")
                snapped_start = original_start
            else:
                # gap 太小，直接接续上一个 section
                snapped_start = prev_snapped_end

        # 最后一个 section 的结束固定为音频时长
        if section_idx == len(stage1_sections) - 1 and audio_duration:
            snapped_end = audio_duration
        else:
            # 找到最近的分割点（必须大于 snapped_start）
            valid_snap_points = [sp for sp in snap_points if sp > snapped_start + min_segment_duration]
            if valid_snap_points:
                snapped_end = find_nearest_snap_point(original_end, valid_snap_points)
            else:
                snapped_end = original_end

        snapped_sections.append({
            'original': stage1_sec,
            'original_start': original_start,
            'original_end': original_end,
            'snapped_start': snapped_start,
            'snapped_end': snapped_end,
            'is_gap': False,
        })

        print(f"  [{stage1_sec.get('name', 'Section')}] "
              f"{seconds_to_mmss(original_start)}-{seconds_to_mmss(original_end)} -> "
              f"{seconds_to_mmss(snapped_start)}-{seconds_to_mmss(snapped_end)}")

    # 更新 stage1_sections：将 gap sections 插入并更新时间
    new_stage1_sections = []
    for snapped in snapped_sections:
        sec = dict(snapped['original'])
        sec['Start_Time'] = seconds_to_mmss(snapped['snapped_start'])
        sec['End_Time'] = seconds_to_mmss(snapped['snapped_end'])
        new_stage1_sections.append(sec)
    stage1_sections = new_stage1_sections

    print(f"✓ Section boundaries snapped to {len(keypoint_times)} keypoints")

    # Stage 3: For each Level 1 section, create Level 2 sub-segments using filtered keypoints
    print("\n" + "="*80)
    print("STAGE 3: Creating Level 2 sub-segments based on filtered keypoints")
    print("="*80)
    print("\nMapping filtered keypoints to Level 1 sections...")

    # Store temporary audio files for cleanup
    temp_files = []

    # Build the final sections with two-level structure
    final_sections = []

    # Collect all sub-segments to process in batches
    all_subsegments = []  # List of (section_idx, subseg_idx, start, end)

    for section_idx, stage1_sec in enumerate(stage1_sections):
        # Parse section times
        sec_start = mmss_to_seconds(stage1_sec.get("Start_Time", "00:00"))
        sec_end = mmss_to_seconds(stage1_sec.get("End_Time", "00:00"))

        # Validate times
        if sec_end <= sec_start:
            if audio_duration:
                sec_end = audio_duration
            else:
                sec_end = sec_start + 30.0  # Default 30s section

        print(f"\n{'-'*80}")
        print(f"Level 1 Section {section_idx + 1}: {stage1_sec.get('name', 'Section')} [{sec_start:.2f}s - {sec_end:.2f}s]")
        print(f"{'-'*80}")

        # Find filtered keypoints within this section's time range
        section_keypoints = [kp for kp in filtered_keypoints
                           if sec_start <= kp['time'] < sec_end]

        print(f"  Found {len(section_keypoints)} filtered keypoints within this section")

        # Build sub-segments by greedily selecting the strongest keypoints as split
        # boundaries, guaranteeing every segment is in [min_segment_duration, max_segment_duration].
        #
        # Algorithm (two steps):
        # 1. Collect all raw keypoints in this section, sorted by intensity (desc).
        #    Greedily accept each keypoint as a split boundary only if it is at least
        #    min_segment_duration away from all already-accepted boundaries (including
        #    sec_start and sec_end).
        # 2. If any resulting interval still exceeds max_segment_duration (no keypoint
        #    was available to split it), insert evenly-spaced midpoints to cap the length.

        # Step 1: greedy keypoint selection
        section_all_kps = sorted(
            [kp for kp in keypoints if sec_start < kp['time'] < sec_end],
            key=lambda x: x.get('normalized_intensity', x.get('intensity', 0)),
            reverse=True,
        )
        accepted = [sec_start, sec_end]
        for kp in section_all_kps:
            t = kp['time']
            if all(abs(t - a) >= min_segment_duration for a in accepted):
                accepted.append(t)
        accepted.sort()

        # Step 2: insert midpoints for intervals that still exceed max_segment_duration
        boundaries = [accepted[0]]
        for t in accepted[1:]:
            gap = t - boundaries[-1]
            if max_segment_duration > 0 and gap > max_segment_duration:
                n = int(np.ceil(gap / max_segment_duration))
                step = gap / n
                for j in range(1, n):
                    boundaries.append(boundaries[-1] + step)
            boundaries.append(t)

        # Build final sub-segment list
        merged_subsegments = []
        for i in range(len(boundaries) - 1):
            s, e = boundaries[i], boundaries[i + 1]
            merged_subsegments.append({
                "start_time": s,
                "end_time": e,
                "duration": e - s,
                "relative_start": s - sec_start,
                "relative_end": e - sec_start,
            })

        if not merged_subsegments:
            merged_subsegments = [{
                "start_time": sec_start,
                "end_time": sec_end,
                "duration": sec_end - sec_start,
                "relative_start": 0.0,
                "relative_end": sec_end - sec_start,
            }]

        # Log coverage
        total_covered = sum(s['duration'] for s in merged_subsegments)
        expected = sec_end - sec_start
        if abs(total_covered - expected) > 0.1:
            print(f"  ⚠ Gap-fill incomplete: covered {total_covered:.1f}s / {expected:.1f}s")

        print(f"  Created {len(merged_subsegments)} Level 2 sub-segments")

        # Store for batch processing
        for subseg_idx, subseg in enumerate(merged_subsegments):
            all_subsegments.append((section_idx, subseg_idx, subseg))

        # Initialize the section entry
        final_sections.append({
            "name": stage1_sec.get("name", f"Section {section_idx + 1}"),
            "description": stage1_sec.get("description", ""),
            "Start_Time": stage1_sec.get("Start_Time", seconds_to_mmss(sec_start)),
            "End_Time": stage1_sec.get("End_Time", seconds_to_mmss(sec_end)),
            "detailed_analysis": {
                "summary": "",
                "sections": []
            },
            "_subsegments": merged_subsegments  # Temporary storage
        })

    # Stage 4: Extract and analyze all sub-segments in batches using AI model
    print("\n" + "="*80)
    print("STAGE 4: AI-based caption generation for sub-segments")
    print("="*80)
    print(f"\nUsing AI model to caption {len(all_subsegments)} sub-segments (between keypoints)...")

    # Step 1: Extract all audio sub-segments
    print(f"\n{'-'*80}")
    print("Step 1: Extracting audio sub-segments...")
    print(f"{'-'*80}")

    subsegment_info_list = []  # Store (section_idx, subseg_idx, subseg_dict, segment_path)

    for section_idx, subseg_idx, subseg in all_subsegments:
        start_time = subseg['start_time']
        end_time = subseg['end_time']
        duration = subseg['duration']

        try:
            # Segment the audio
            segment_path = segment_audio_file(audio_path, start_time, end_time)
            temp_files.append(segment_path)

            print(f"✓ Section {section_idx + 1}, Sub-segment {subseg_idx + 1}: {start_time:.2f}s - {end_time:.2f}s ({duration:.1f}s)")

            subsegment_info_list.append((section_idx, subseg_idx, subseg, segment_path))

        except Exception as e:
            print(f"✗ Section {section_idx + 1}, Sub-segment {subseg_idx + 1}: Error extracting - {e}")
            subsegment_info_list.append((section_idx, subseg_idx, subseg, None))

    # Step 2: Process sub-segments in batches
    print(f"\n{'-'*80}")
    valid_count = len([s for s in subsegment_info_list if s[3] is not None])
    print(f"Step 2: Processing {valid_count} sub-segments (max_workers={max_workers})...")
    print(f"{'-'*80}")

    # Create a mapping from segment_path to subsegment info
    path_to_info = {}
    valid_segment_paths = []

    for section_idx, subseg_idx, subseg, segment_path in subsegment_info_list:
        if segment_path is not None:
            path_to_info[segment_path] = (section_idx, subseg_idx, subseg)
            valid_segment_paths.append(segment_path)

    # Process all sub-segments concurrently via cloud API
    subsegment_captions = {}  # Maps segment_path to caption text

    print(f"\n{'·'*80}")
    print(f"Processing {len(valid_segment_paths)} sub-segments concurrently (max_workers={max_workers})...")
    print(f"{'·'*80}")

    # Skip AI captioning if overall analysis already failed (model doesn't support audio)
    if not overall_summary:
        print(f"  ⏭️ Skipping AI captioning — model '{_get_model_name()}' does not support audio input")
        print(f"  Using empty captions for all {len(valid_segment_paths)} sub-segments")
        caption_texts = [""] * len(valid_segment_paths)
    else:
        try:
            caption_texts = generate_audio_captions_batch(
                audio_paths=valid_segment_paths,
                prompt=AUDIO_SEG_KEYPOINT_PROMPT,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                max_workers=max_workers,
            )
        except Exception as e:
            print(f"⚠ AI sub-segment captioning failed: {e}")
            print(f"  Falling back to empty captions for all {len(valid_segment_paths)} sub-segments")
            caption_texts = [""] * len(valid_segment_paths)

    for path, caption_text in zip(valid_segment_paths, caption_texts):
        subsegment_captions[path] = caption_text

    print(f"✓ All sub-segments processed")

    # Step 3: Build the two-level structure
    print(f"\n{'-'*80}")
    print("Step 3: Building two-level structure...")
    print(f"{'-'*80}")

    for section_idx, subseg_idx, subseg, segment_path in subsegment_info_list:
        # Create sub-section entry with relative times
        sub_section = {
            "name": f"Section {subseg_idx + 1}",
            "description": "",
            "Start_Time": seconds_to_mmss(subseg['relative_start']),
            "End_Time": seconds_to_mmss(subseg['relative_end'])
        }

        if segment_path is None:
            print(f"  Section {section_idx + 1}, Sub {subseg_idx + 1}: No audio (skipped)")
            final_sections[section_idx]["detailed_analysis"]["sections"].append(sub_section)
            continue

        # Get caption for this sub-segment
        caption_text = subsegment_captions.get(segment_path)

        if caption_text is None:
            print(f"  Section {section_idx + 1}, Sub {subseg_idx + 1}: Caption generation failed")
            final_sections[section_idx]["detailed_analysis"]["sections"].append(sub_section)
            continue

        # Try to parse JSON from caption
        segment_json = extract_json_from_text(caption_text)

        if segment_json and isinstance(segment_json, dict):
            # Merge the analysis fields into sub_section
            if "summary" in segment_json:
                sub_section["description"] = segment_json["summary"]
            if "emotion" in segment_json:
                sub_section["Emotional_Tone"] = segment_json["emotion"]
            if "energy" in segment_json:
                sub_section["energy"] = segment_json["energy"]
            if "rhythm" in segment_json:
                sub_section["rhythm"] = segment_json["rhythm"]

            # Also store the raw detailed analysis if there are extra fields
            for key, value in segment_json.items():
                if key not in ["summary", "emotion", "energy", "rhythm"]:
                    sub_section[key] = value

            print(f"✓ Section {section_idx + 1}, Sub {subseg_idx + 1}: Detailed analysis added")
        else:
            sub_section["description"] = caption_text
            print(f"⚠ Section {section_idx + 1}, Sub {subseg_idx + 1}: Raw text added (JSON parsing failed)")

        final_sections[section_idx]["detailed_analysis"]["sections"].append(sub_section)

    # Stage 5: Merge into two-level output format
    print(f"\n{'='*80}")
    print("STAGE 5: Merging into two-level output format")
    print("="*80)

    # Remove temporary storage
    for section in final_sections:
        if "_subsegments" in section:
            del section["_subsegments"]

    # Cleanup temporary files
    print(f"\n{'-'*80}")
    print("Cleaning up temporary files...")
    for temp_file in temp_files:
        try:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        except Exception as e:
            print(f"⚠ Warning: Could not remove temp file {temp_file}: {e}")

    # Prepare final result in the target format
    result_data = {
        "audio_path": audio_path,
        "overall_analysis": {
            "prompt": AUDIO_OVERALL_PROMPT.strip(),
            "summary": overall_summary
        },
        "sections": final_sections,
        # Debug: 保存分割点详细信息
        "_keypoints_detail": [
            {
                "time": kp['time'],
                "time_mmss": seconds_to_mmss(kp['time']),
                "type": kp.get('type', 'Unknown'),
                "intensity": round(kp.get('intensity', 0), 4),
                "normalized_intensity": round(kp.get('normalized_intensity', kp.get('intensity', 0)), 4),
                "section": kp.get('section', None),
                "type_boosted": kp.get('type_boosted', False)
            }
            for kp in filtered_keypoints
        ],
        "_debug_config": {
            "min_segment_duration": min_segment_duration,
            "max_segment_duration": max_segment_duration,
        }
    }

    # Save to file if output_path is specified
    if output_path:
        # Create output directory if needed
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Save to file
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)

        print(f"\n{'='*80}")
        print("✓ Processing Complete!")
        print(f"  - Level 1 sections: {len(final_sections)}")
        total_subsections = sum(len(sec.get('detailed_analysis', {}).get('sections', [])) for sec in final_sections)
        print(f"  - Level 2 sub-segments: {total_subsections}")
        print(f"  - Keypoints used: {len(filtered_keypoints)}")
        print(f"  - Output saved to: {output_path}")
        print(f"  - Debug info: _keypoints_detail, _debug_config (in JSON)")
        print(f"{'='*80}")

    return result_data


# --------------------------------------------------------------------------- #
#                                    main                                     #
# --------------------------------------------------------------------------- #
def main():
    """Example usage of audio caption function with Madmom-based segmentation."""
    # Example audio file path
    audio_path = "/public_hw/home/cit_shifangzhao/zsf/VideoCuttingAgent/Dataset/Audio/Way_down_we_go/Way Down We Go-Kaleo#1NrOG.mp3"

    # Generate caption with Madmom-based segmentation
    # All parameters will use config defaults if not specified
    result = caption_audio_with_madmom_segments(
        audio_path=audio_path,
        output_path="./captioner_Way_down_we_go_caption_madmom_output.json",
        max_tokens=config.AUDIO_KEYPOINT_MAX_TOKENS,
        # All other parameters will be loaded from config automatically
    )

    print(f"\n{'='*80}")
    print("Processing Complete!")
    print(f"Total sections analyzed: {len(result.get('sections', []))}")
    print(f"{'='*80}")



if __name__ == "__main__":
    main()

# 0.11.0rc2.dev113+gf9e714813
