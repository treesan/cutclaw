import os
import json
import copy
import re
import time
import gc
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Annotated as A
from src.video.deconstruction.video_caption import (
    SYSTEM_PROMPT,
    messages as caption_messages,
)
from src.video.preprocess.video_utils import _create_decord_reader
import litellm
from src.utils.media_utils import seconds_to_hhmmss as convert_seconds_to_hhmmss, hhmmss_to_seconds as _hhmmss_to_seconds, parse_srt_to_dict, array_to_base64
from src import config
from src.func_call_shema import as_json_schema
from src.func_call_shema import doc as D
from src.Reviewer import ReviewerAgent
from src.prompt import (
    DENSE_CAPTION_PROMPT_FILM,
    EDITOR_SYSTEM_PROMPT,
    EDITOR_USER_PROMPT,
    EDITOR_FINISH_PROMPT,
    EDITOR_USE_TOOL_PROMPT,
)


class StopException(Exception):
    """
    Stop Execution by raising this exception (Signal that the task is Finished).
    """


TOOL_NAME_ALIASES = {
    "semantic_neighborhood_retrieval": "Semantic Neighborhood Retrieval",
    "fine_grained_shot_trimming": "Fine-Grained Shot Trimming",
    "commit": "Commit",
}

LEGACY_TOOL_NAME_MAP = {
    "get_related_shot": "semantic_neighborhood_retrieval",
    "trim_shot": "fine_grained_shot_trimming",
    "finish": "commit",
}


_THREAD_VIDEO_READERS = threading.local()

# ── fine_grained_shot_trimming result cache ──
# Key: (video_path, time_range) → result string
# Avoids redundant VLM calls when the Agent re-analyzes the same clip.
_TRIM_SHOT_CACHE = {}  # OrderedDict analogue via insertion order (Python 3.7+)
_TRIM_SHOT_CACHE_MAX = 50


def clear_trim_shot_cache():
    """Clear the fine_grained_shot_trimming result cache."""
    _TRIM_SHOT_CACHE.clear()


def _get_trim_shot_cache(video_path: str, time_range: str) -> str | None:
    """Return cached result or None."""
    key = (video_path, time_range)
    result = _TRIM_SHOT_CACHE.get(key)
    if result is not None:
        # Touch: re-insert to maintain LRU-ish ordering (dict preserves insertion order)
        del _TRIM_SHOT_CACHE[key]
        _TRIM_SHOT_CACHE[key] = result
    return result


def _set_trim_shot_cache(video_path: str, time_range: str, result: str):
    """Store result in cache, evicting oldest entry if at capacity."""
    key = (video_path, time_range)
    if len(_TRIM_SHOT_CACHE) >= _TRIM_SHOT_CACHE_MAX:
        # Evict the oldest entry
        oldest = next(iter(_TRIM_SHOT_CACHE))
        del _TRIM_SHOT_CACHE[oldest]
    _TRIM_SHOT_CACHE[key] = result


def _get_thread_video_reader(video_path: str):
    if not video_path:
        return None
    reader = getattr(_THREAD_VIDEO_READERS, "reader", None)
    reader_path = getattr(_THREAD_VIDEO_READERS, "video_path", None)
    if reader is None or reader_path != video_path:
        target_resolution = getattr(config, "VIDEO_RESOLUTION", None)
        reader = _create_decord_reader(video_path, target_resolution)
        _THREAD_VIDEO_READERS.reader = reader
        _THREAD_VIDEO_READERS.video_path = video_path
    return reader


def _clear_thread_video_reader():
    if hasattr(_THREAD_VIDEO_READERS, "reader"):
        _THREAD_VIDEO_READERS.reader = None
    if hasattr(_THREAD_VIDEO_READERS, "video_path"):
        _THREAD_VIDEO_READERS.video_path = None


def _normalize_video_reader(video_reader):
    if isinstance(video_reader, dict):
        video_reader = video_reader.get("video_reader")
    if video_reader is not None and not hasattr(video_reader, "get_avg_fps"):
        return None
    return video_reader


def _canonical_tool_name(name: str) -> str:
    if name in LEGACY_TOOL_NAME_MAP:
        return LEGACY_TOOL_NAME_MAP[name]
    for original_name, alias_name in TOOL_NAME_ALIASES.items():
        if name == original_name or name == alias_name:
            return original_name
    return name


def _parse_shot_time_ranges(answer: str) -> list[tuple[float, float]]:
    """Parse shot ranges in the format: [shot: HH:MM:SS to HH:MM:SS]."""
    shot_pattern = re.compile(r'\[?shot[\s_]*\d*:\s*([0-9:.]+)\s+to\s+([0-9:.]+)\]?', re.IGNORECASE)
    matches = shot_pattern.findall(answer or "")
    if not matches:
        return []

    ranges = []
    for start_time, end_time in matches:
        start_sec = _hhmmss_to_seconds(start_time, fps=getattr(config, 'VIDEO_FPS', 24) or 24)
        end_sec = _hhmmss_to_seconds(end_time, fps=getattr(config, 'VIDEO_FPS', 24) or 24)
        ranges.append((start_sec, end_sec))
    return ranges


def _ranges_overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> bool:
    return a_start < b_end and a_end > b_start


def _parse_retry_after_seconds(error_text: str, default_seconds: float = 1.0) -> float:
    match = re.search(r'after\s+(\d+(?:\.\d+)?)\s*seconds?', error_text or "", re.IGNORECASE)
    if match:
        try:
            return float(match.group(1))
        except Exception:
            return default_seconds
    return default_seconds


def _compact_json_str_for_log(s: str, max_len: int = 500) -> str:
    if len(s) <= max_len:
        return s
    return s[:max_len] + f"... [truncated {len(s) - max_len} chars]"


def commit(
    answer: A[str, D("Output the final shot time range. Must be exactly ONE continuous clip.")],
    video_path: A[str, D("Path to the source video file")] = "",
    output_path: A[str, D("Path to save the edited video")] = "",
    target_length_sec: A[float, D("Expected total length in seconds")] = 0.0,
    section_idx: A[int, D("Current section index. Auto-injected.")] = -1,
    shot_idx: A[int, D("Current shot index. Auto-injected.")] = -1,
    protagonist_frame_data: A[list, D("Frame-by-frame protagonist detection data. Auto-injected.")] = None
) -> str:
    """
    Call this function to finalize the shot selection and save the result.
    NOTE: You MUST call commit first to validate the shot before calling this function.

    IMPORTANT: Only accepts ONE continuous time range.
    Example: [shot: 00:10:00 to 00:10:06.4]

    Returns:
        str: Success message with saved result, or error message if parsing fails.
    """
    def hhmmss_to_seconds(time_str: str) -> float:
        return _hhmmss_to_seconds(time_str, fps=getattr(config, 'VIDEO_FPS', 24) or 24)

    seconds_to_hhmmss = convert_seconds_to_hhmmss

    # Parse the answer to extract shot time ranges
    shot_pattern = re.compile(r'\[?shot[\s_]*\d*:\s*([0-9:.]+)\s+to\s+([0-9:.]+)\]?', re.IGNORECASE)
    matches = shot_pattern.findall(answer)

    if not matches:
        return "Error: Could not parse shot time ranges. Please provide format: [shot: HH:MM:SS to HH:MM:SS]"

    # Support multiple shots for stitching (with constraints)
    max_shots_allowed = getattr(config, 'MAX_SHOTS_PER_CLIP', 3)
    if len(matches) > max_shots_allowed:
        return f"Error: Too many shots detected ({len(matches)}). Maximum allowed: {max_shots_allowed}"

    # Parse all time ranges and validate
    clips = []
    total_duration = 0
    for i, (start_time, end_time) in enumerate(matches):
        try:
            start_sec = hhmmss_to_seconds(start_time)
            end_sec = hhmmss_to_seconds(end_time)
            duration = end_sec - start_sec

            if duration <= 0:
                return f"Error: Shot {i+1} has invalid duration (start: {start_time}, end: {end_time})"

            clips.append({
                'start_sec': start_sec,
                'end_sec': end_sec,
                'duration': duration,
                'start_time': start_time,
                'end_time': end_time
            })
            total_duration += duration
        except Exception as e:
            return f"Error parsing shot {i+1} time range: {str(e)}"

    # Validate time continuity and gaps for multi-shot stitching
    if len(clips) > 1:
        max_gap = getattr(config, 'MAX_STITCH_GAP_SEC', 2.0)
        for i in range(len(clips) - 1):
            gap = clips[i+1]['start_sec'] - clips[i]['end_sec']
            if gap < 0:
                return f"Error: Overlapping shots detected. Shot {i+1} ends at {clips[i]['end_time']}, but shot {i+2} starts at {clips[i+1]['start_time']}"
            if gap > max_gap:
                return f"Error: Time gap ({gap:.2f}s) between shot {i+1} and {i+2} exceeds maximum ({max_gap}s). Shots must maintain visual continuity."

    # Use total duration for validation
    duration = total_duration

    # For result building, we'll use the first and last timestamps
    start_sec = clips[0]['start_sec']
    end_sec = clips[-1]['end_sec']
    start_time = clips[0]['start_time']
    end_time = clips[-1]['end_time']

    # Auto-trim if slightly over target (within 1 second)
    duration_diff = duration - target_length_sec
    if 0 < duration_diff <= 1.0:
        # Only auto-trim the last clip's end time
        clips[-1]['end_sec'] = clips[-1]['end_sec'] - duration_diff
        clips[-1]['duration'] = clips[-1]['end_sec'] - clips[-1]['start_sec']
        clips[-1]['end_time'] = seconds_to_hhmmss(clips[-1]['end_sec'])

        # Recalculate total duration
        duration = sum(c['duration'] for c in clips)

        end_sec = clips[-1]['end_sec']
        end_time = clips[-1]['end_time']
        print(f"✂️  [Trim] Auto-trimmed by {duration_diff:.2f}s. New end: {end_time}")

    # Build result data with all clips
    result_clips = []
    for i, clip in enumerate(clips):
        result_clips.append({
            "shot": i + 1,
            "start": seconds_to_hhmmss(clip['start_sec']),
            "end": seconds_to_hhmmss(clip['end_sec']),
            "duration": round(clip['duration'], 2)
        })

    result_data = {
        "status": "success",
        "section_idx": section_idx,
        "shot_idx": shot_idx,
        "total_duration": round(duration, 2),
        "target_duration": target_length_sec,
        "num_clips": len(clips),
        "is_stitched": len(clips) > 1,
        "clips": result_clips
    }

    # Add protagonist frame detection data if available
    if protagonist_frame_data:
        result_data["protagonist_detection"] = {
            "method": "vlm",
            "total_frames_analyzed": len(protagonist_frame_data),
            "frames_with_protagonist": sum(1 for f in protagonist_frame_data if f.get("protagonist_detected", False)),
            "protagonist_ratio": round(
                sum(1 for f in protagonist_frame_data if f.get("protagonist_detected", False)) / len(protagonist_frame_data),
                3
            ) if protagonist_frame_data else 0.0,
            "frame_detections": protagonist_frame_data
        }

    # Save result to output_path if provided
    if output_path:
        if os.path.exists(output_path):
            with open(output_path, 'r', encoding='utf-8') as f:
                try:
                    all_results = json.load(f)
                except json.JSONDecodeError:
                    all_results = []
        else:
            all_results = []

        all_results.append(result_data)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"💾 [Output] Result saved to {output_path}")

    success_msg = f"Successfully created edited video: {seconds_to_hhmmss(start_sec)} to {seconds_to_hhmmss(end_sec)} ({duration:.2f}s)"
    print(f"✅ [Success] {success_msg}")

    return json.dumps(result_data, ensure_ascii=False)


def semantic_neighborhood_retrieval(
        related_scenes: A[list, D("List of scene indices to search. Optional - you can specify nearby scenes within allowed range.")] = None,
        scene_folder_path: A[str, D("Path to the folder containing scene JSON files. Auto-injected.")] = None,
        recommended_scenes: A[list, D("Recommended scene indices from shot_plan. Auto-injected.")] = None
) -> str:
    """
    Retrieves shot information from specified scenes.

    You can optionally specify which scenes to search by passing a 'related_scenes' list.
    However, you can only explore scenes within ±SCENE_EXPLORATION_RANGE of the recommended scenes.

    Example:
    - If recommended scenes are [8, 12] and SCENE_EXPLORATION_RANGE=3
    - You can search scenes 5-11 (around 8) and 9-15 (around 12)
    - Searching scene 50 would be REJECTED

    If you don't specify scenes, the system will use the recommended scenes automatically.

    Returns:
        str: A formatted string containing the shot information from the requested scenes.
        IMPORTANT: Select segments within shot boundaries to avoid visual discontinuities.

    Notes:
        - If you can't find suitable shots in recommended scenes, try nearby scenes
        - Going too far from recommended scenes may result in mismatched content
    """

    # Validate scene range if agent requested specific scenes
    if related_scenes and recommended_scenes:
        from src import config
        allowed_range = getattr(config, 'SCENE_EXPLORATION_RANGE', 3)

        # Get total scene count by checking available scene files
        max_scene_idx = 0
        if scene_folder_path and os.path.isdir(scene_folder_path):
            import glob
            scene_files = glob.glob(os.path.join(scene_folder_path, "scene_*.json"))
            if scene_files:
                # Extract scene numbers from filenames
                scene_numbers = []
                for f in scene_files:
                    basename = os.path.basename(f)  # e.g., "scene_42.json"
                    try:
                        num = int(basename.replace("scene_", "").replace(".json", ""))
                        scene_numbers.append(num)
                    except ValueError:
                        continue
                max_scene_idx = max(scene_numbers) if scene_numbers else 0

        # Build allowed scene set with boundary constraints
        allowed_scenes = set()
        for rec_scene in recommended_scenes:
            for offset in range(-allowed_range, allowed_range + 1):
                scene_idx = rec_scene + offset
                # Ensure scene index is within valid range [0, max_scene_idx]
                if scene_idx >= 0 and (max_scene_idx == 0 or scene_idx <= max_scene_idx):
                    allowed_scenes.add(scene_idx)

        # Check if all requested scenes are within allowed range
        invalid_scenes = [s for s in related_scenes if s not in allowed_scenes]
        if invalid_scenes:
            return (
                f"❌ Error: Cannot search scenes {invalid_scenes} - they are outside the allowed range.\n"
                f"Recommended scenes: {recommended_scenes}\n"
                f"Allowed exploration range: ±{allowed_range} scenes\n"
                f"Valid scenes you can search: {sorted(list(allowed_scenes))}\n"
                f"Please select scenes within the allowed range or omit the 'related_scenes' parameter to use defaults."
            )

        print(f"🗺️  [Explore] Agent exploring nearby scenes: {related_scenes} (recommended: {recommended_scenes})")
    elif not related_scenes and recommended_scenes:
        # Use recommended scenes if agent didn't specify
        related_scenes = recommended_scenes
        print(f"🗺️  [Explore] Using recommended scenes: {related_scenes}")

    if not related_scenes:
        return "Error: No scenes specified and no recommended scenes available."

    all_shots_info = []

    for scene_idx in related_scenes:
        scene_file = os.path.join(scene_folder_path, f"scene_{scene_idx}.json")
        if os.path.exists(scene_file):
            try:
                with open(scene_file, 'r', encoding='utf-8') as f:
                    scene_data = json.load(f)

                scene_time_range = scene_data.get('time_range', {})
                scene_start = scene_time_range.get('start_seconds', '00:00:00')
                scene_end = scene_time_range.get('end_seconds', '00:00:00')

                all_shots_info.append(f"\n=== Scene {scene_idx} ({scene_start} - {scene_end}) ===")

                shots_data = scene_data.get('shots_data', [])
                for shot in shots_data:
                    duration = shot.get('duration', {})
                    start_time = duration.get('clip_start_time', '')
                    end_time = duration.get('clip_end_time', '')

                    action = shot.get('action_atoms', {})
                    event_summary = action.get('event_summary', '')

                    narrative = shot.get('narrative_analysis', {})
                    mood = narrative.get('mood', '')

                    shot_info = f"[{start_time} - {end_time}] {event_summary}"
                    if mood:
                        shot_info += f" (Mood: {mood})"

                    all_shots_info.append(shot_info)

            except Exception as e:
                print(f"⚠️  [Warning] Failed to load scene {scene_idx}: {e}")
                all_shots_info.append(f"Scene {scene_idx}: Failed to load - {e}")
        else:
            all_shots_info.append(f"Scene {scene_idx}: File not found")

    result = "\n".join(all_shots_info)
    return f"Here are the available shots from related scenes {related_scenes}:\n{result}"


def review_clip(
    time_range: A[str, D("The time range to check (e.g., '00:13:28 to 00:13:40').")],
    used_time_ranges: A[list, D("List of already used time ranges. Auto-injected.")] = None
) -> str:
    """
    Check if the proposed time range overlaps with any previously used clips.
    You MUST call this tool BEFORE calling finish to ensure no duplicate footage.

    Returns:
        str: A message indicating whether the time range is available or overlaps with used clips.
             If overlap is detected, you should select a different time range.
    """
    def hhmmss_to_seconds(time_str: str) -> float:
        return _hhmmss_to_seconds(time_str, fps=getattr(config, 'VIDEO_FPS', 24) or 24)

    if used_time_ranges is None:
        used_time_ranges = []

    # Parse the time range
    match = re.search(r'([\d:.]+)\s+to\s+([\d:.]+)', time_range, re.IGNORECASE)
    if not match:
        return f"Error: Could not parse time range '{time_range}'. Please use format 'HH:MM:SS to HH:MM:SS'."

    try:
        start_sec = hhmmss_to_seconds(match.group(1))
        end_sec = hhmmss_to_seconds(match.group(2))
    except Exception as e:
        return f"Error parsing time range: {e}"

    if not used_time_ranges:
        return f"✅ OK: Time range {time_range} is available. No previous clips have been used yet. You can proceed with finish."

    # Check for overlaps
    overlapping_clips = []
    for idx, (used_start, used_end) in enumerate(used_time_ranges):
        if start_sec < used_end and end_sec > used_start:
            overlap_start = max(start_sec, used_start)
            overlap_end = min(end_sec, used_end)
            overlapping_clips.append({
                "clip_idx": idx + 1,
                "used_range": f"{convert_seconds_to_hhmmss(used_start)} to {convert_seconds_to_hhmmss(used_end)}",
                "overlap": f"{convert_seconds_to_hhmmss(overlap_start)} to {convert_seconds_to_hhmmss(overlap_end)}"
            })

    if overlapping_clips:
        result = f"❌ OVERLAP DETECTED: Time range {time_range} overlaps with {len(overlapping_clips)} previously used clip(s):\n"
        for clip in overlapping_clips:
            result += f"  - Clip {clip['clip_idx']}: {clip['used_range']} (overlap: {clip['overlap']})\n"
        result += "\n⚠️ Please select a DIFFERENT time range to avoid duplicate footage. Do NOT call finish with this range."
        return result
    else:
        return f"✅ OK: Time range {time_range} does not overlap with any previously used clips. You can proceed with finish."


def fine_grained_shot_trimming(
    time_range: A[str, D("The time range to analyze ('HH:MM:SS to HH:MM:SS'). This tool will analyze the ENTIRE range and provide scene breakdowns within it.")],
    frame_path: A[str, D("The path to the video frames file.")] = "",
    transcript_path: A[str, D("Optional path to an .srt transcript file; subtitles in this range will be injected into the prompt.")] = "",
    original_shot_boundaries: A[list, D("List of original shot boundaries from source material. Auto-injected.")] = None,
) -> str:
    """
    Analyze a video clip time range and return detailed scene information and usability assessment.
        
    Returns:
        A JSON string with structure:
        {
            "analyzed_range": "HH:MM:SS to HH:MM:SS",  # The full range you requested, must longer that 3.0s
            "total_duration_sec": float,                # Total duration
            "usability_assessment": "...",              # Overall evaluation
            "recommended_usage": "...",                 # How to use this clip
            "internal_scenes": [...]                    # Scene breakdowns (for reference)
        }
        
        The "internal_scenes" are fine-grained descriptions to help you understand what's  happening INSIDE the analyzed range.
        Use them to decide whether to use the full range, a subset, or refine with another call.

    
    Args:
        time_range: String in format 'HH:MM:SS to HH:MM:SS' - the range to analyze
        frame_path: Path to the video frames directory
    """
    def hhmmss_to_seconds(time_str: str) -> float:
        return _hhmmss_to_seconds(time_str, fps=getattr(config, 'VIDEO_FPS', 24) or 24)

    def _extract_subtitles_in_range(srt_path: str, start_s: float, end_s: float, max_chars: int = 1500) -> str:
        """Reuse video_caption.parse_srt_to_dict() and only do range filtering + formatting here."""
        if not srt_path:
            return ""
        if not os.path.exists(srt_path):
            return ""
        try:
            subtitle_map = parse_srt_to_dict(srt_path)
            if not subtitle_map:
                return ""

            # parse_srt_to_dict() truncates timestamps to int seconds for keys.
            # Align the filtering to the same granularity to avoid boundary misses.
            import math
            start_i = int(start_s)
            end_i = int(math.ceil(end_s))
            if end_i <= start_i:
                end_i = start_i + 1


            picked = []  # list[tuple[int, str]]
            for key, text in subtitle_map.items():
                try:
                    s_sec, e_sec = map(int, key.split("_"))
                except Exception:
                    continue

                # Overlap check in integer-second domain (half-open interval)
                if s_sec >= end_i or e_sec <= start_i:
                    continue
                t = re.sub(r"\s+", " ", (text or "")).strip()
                if t:
                    picked.append((s_sec, t))

            if not picked:
                return ""

            picked.sort(key=lambda x: x[0])
            joined = " ".join(t for _, t in picked)
            joined = re.sub(r"\s+", " ", joined).strip()
            if len(joined) > max_chars:
                joined = joined[:max_chars].rsplit(' ', 1)[0] + "…"
            return joined
        except Exception as e:
            import traceback
            traceback.print_exc()
            return ""

    
    # Parse the time range string: 'HH:MM:SS to HH:MM:SS' or 'HH:MM:SS.s to HH:MM:SS.s'
    match = re.search(r'([\d:.]+)\s+to\s+([\d:.]+)', time_range, re.IGNORECASE)
    
    if not match:
        return f"Error: Could not parse time range '{time_range}'."
    
    start_time_str = match.group(1)
    end_time_str = match.group(2)
    
    # Convert to seconds
    start_sec = hhmmss_to_seconds(start_time_str)
    end_sec = hhmmss_to_seconds(end_time_str)
    
    # Convert seconds to HH:MM:SS format for display
    clip_start_time = convert_seconds_to_hhmmss(start_sec)
    clip_end_time = convert_seconds_to_hhmmss(end_sec)

    # ── Check cache to avoid redundant VLM calls ──
    if frame_path:
        cached = _get_trim_shot_cache(frame_path, time_range)
        if cached is not None:
            print(f"🔁 [Cache] Reusing cached trim_shot result for {time_range}")
            return cached

    subtitles_context = _extract_subtitles_in_range(transcript_path, start_sec, end_sec)
    
    # Prepare messages for VIDEO_ANALYSIS_MODEL
    # Use the same prompt template as video_caption.py
    send_messages = copy.deepcopy(caption_messages)
    send_messages[0]["content"] = SYSTEM_PROMPT

    dense_caption_prompt = DENSE_CAPTION_PROMPT_FILM.replace(
        "MAIN_CHARACTER_NAME_PLACEHOLDER",
        getattr(config, 'MAIN_CHARACTER_NAME', 'the main character')
    ).replace(
        "MIN_SEGMENT_DURATION_PLACEHOLDER",
        str(getattr(config, 'AUDIO_MIN_SEGMENT_DURATION', 3.0))
    )
    requested_duration = max(0.0, end_sec - start_sec)
    requested_end_rel = convert_seconds_to_hhmmss(requested_duration)
    dense_caption_prompt += (
        "\n\n[Clip Timing Constraints]\n"
        f"- Requested clip duration: {requested_duration:.2f}s\n"
        f"- Relative timeline MUST start at 00:00:00 and end at {requested_end_rel}\n"
        "- Segments must collectively cover the full duration without truncation\n"
        "- Keep timestamps monotonic and contiguous\n"
    )
    if subtitles_context:
        dense_caption_prompt += f"\n\n[Subtitles in this range]\n{subtitles_context}\n"
    send_messages[1]["content"] = dense_caption_prompt

    # Extract frames from the video clip and encode as base64
    # Use native video fps by default (no fixed sampling fps override).
    def _extract_clip_frames(video_path, start_s, end_s, video_reader=None):
        vr = _normalize_video_reader(video_reader)
        if vr is None:
            if not video_path:
                return []
            vr = _get_thread_video_reader(video_path)
            if vr is None:
                return []
        video_fps = float(vr.get_avg_fps())
        start_f = max(0, int(start_s * video_fps))
        end_f = min(int(end_s * video_fps), len(vr) - 1)
        if end_f < start_f:
            return []
        indices = list(range(start_f, end_f + 1))
        # Safety cap to avoid provider limit: max data-uri per request (e.g., 250 on OpenAI).
        # Keep a margin for robustness.
        max_frames = int(getattr(config, 'CORE_MAX_FRAMES', getattr(config, 'TRIM_SHOT_MAX_FRAMES', 240)))
        if max_frames > 0 and len(indices) > max_frames:
            import math
            stride = max(1, math.ceil(len(indices) / max_frames))
            indices = indices[::stride]
            # Ensure last frame is included so ending timestamp context is preserved.
            if indices[-1] != end_f:
                indices.append(end_f)
            # Hard cap after end-frame append.
            if len(indices) > max_frames:
                indices = indices[:max_frames - 1] + [end_f]

        if not indices:
            return []
        frames = vr.get_batch(indices).asnumpy()
        return [array_to_base64(frames[i]) for i in range(len(frames))]

    # Build litellm messages with base64 frames injected
    def _build_litellm_messages(base_messages, b64_frames):
        result = []
        for msg in base_messages:
            if msg["role"] == "user" and b64_frames:
                content = [{"type": "text", "text": msg["content"]}] if msg["content"] else []
                for b64 in b64_frames:
                    content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
                result.append({"role": "user", "content": content})
            else:
                result.append({"role": msg["role"], "content": msg["content"]})
        return result

    active_reader = _get_thread_video_reader(frame_path) if frame_path else None
    b64_frames = _extract_clip_frames(frame_path, start_sec, end_sec, video_reader=active_reader)
    litellm_messages = _build_litellm_messages(send_messages, b64_frames)

    try:
        # Call VIDEO_ANALYSIS_MODEL via litellm
        tries = 3
        while tries > 0:
            tries -= 1
            try:
                kwargs = dict(
                    model=config.VIDEO_ANALYSIS_MODEL,
                    messages=litellm_messages,
                    max_tokens=config.VIDEO_ANALYSIS_MODEL_MAX_TOKEN,
                    temperature=0.0,
                )
                if config.VIDEO_ANALYSIS_ENDPOINT:
                    kwargs["api_base"] = config.VIDEO_ANALYSIS_ENDPOINT
                if config.VIDEO_ANALYSIS_API_KEY:
                    kwargs["api_key"] = config.VIDEO_ANALYSIS_API_KEY
                raw = litellm.completion(**kwargs)
                content_str = raw.choices[0].message.content
            except Exception as e:
                print(f"❌ [Error] [trim_shot] litellm call failed: {e}")
                content_str = None

            if not content_str:
                if tries == 0:
                    return f"Error: Failed to generate caption for time range {time_range}."
                continue

            try:
                content = content_str.strip()
                
                # Debug: print the raw content to help diagnose issues
                if not content:
                    print(f"⚠️  [Warning] Empty content from model for time range {time_range}")
                    if tries == 0:
                        return f"Error: Empty response from model for time range {time_range}."
                    continue
                
                # Try to extract JSON from markdown code blocks if present
                # Pattern: ```json ... ``` or ``` ... ```
                json_block_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', content, re.DOTALL)
                if json_block_match:
                    content = json_block_match.group(1).strip()
                
                # Try to parse JSON
                parsed = json.loads(content)

                # DEBUG: Print first segment to diagnose timestamp issues
                # Handle "segments" format (from DENSE_CAPTION_PROMPT)
                if isinstance(parsed, dict) and "segments" in parsed:
                    result = {
                        "analyzed_range": f"{clip_start_time} to {clip_end_time}",
                        "total_duration_sec": end_sec - start_sec,
                        "usability_assessment": "See segment details with quality scores and emotions.",
                        "internal_scenes": []
                    }

                    for seg in parsed["segments"]:
                        # Build comprehensive description from new format
                        desc_parts = []

                        # Cut type
                        if seg.get("cut_type"):
                            desc_parts.append(f"[{seg['cut_type'].upper()}]")

                        # Content description
                        if seg.get("content_description"):
                            desc_parts.append(seg["content_description"])

                        # Visual quality info
                        visual_quality = seg.get("visual_quality", {})
                        quality_score = visual_quality.get("score", "N/A")
                        quality_notes = visual_quality.get("notes", "")

                        # Emotion info
                        emotion = seg.get("emotion", {})
                        mood = emotion.get("mood", "")
                        intensity = emotion.get("intensity", "")
                        narrative_func = emotion.get("narrative_function", "")

                        # Editor recommendation
                        editor_rec = seg.get("editor_recommendation", "")

                        # Get base character_presence from VLM scene analysis
                        character_presence = seg.get("character_presence", {})

                        scene = {
                            "scene_time": seg.get("timestamp", ""),
                            "description": " ".join(desc_parts),
                            "cut_type": seg.get("cut_type", ""),
                            "visual_quality": {
                                "score": quality_score,
                                "notes": quality_notes
                            },
                            "emotion": {
                                "mood": mood,
                                "intensity": intensity,
                                "narrative_function": narrative_func
                            },
                            "character_presence": character_presence,
                            "editor_recommendation": editor_rec,
                            "duration_sec": 0
                        }

                        # Calculate absolute timestamps and duration
                        seg_start_sec = None
                        seg_end_sec = None
                        if "timestamp" in seg:
                            range_match = re.search(r'([0-9:.]+)\s+to\s+([0-9:.]+)', seg["timestamp"], re.IGNORECASE)
                            if range_match:
                                try:
                                    # Timestamps from model are relative to the clip start (00:00:00)
                                    # We need to convert them to absolute timestamps
                                    s_rel = hhmmss_to_seconds(range_match.group(1))
                                    e_rel = hhmmss_to_seconds(range_match.group(2))

                                    s_abs = start_sec + s_rel
                                    e_abs = start_sec + e_rel

                                    seg_start_sec = s_abs
                                    seg_end_sec = e_abs

                                    scene["scene_time"] = f"{convert_seconds_to_hhmmss(s_abs)} to {convert_seconds_to_hhmmss(e_abs)}"
                                    scene["duration_sec"] = round(e_abs - s_abs, 2)
                                except ValueError:
                                    pass

                        result["internal_scenes"].append(scene)


                    # Validate that internal_scenes cover the requested time range
                    total_requested_duration = end_sec - start_sec
                    covered_duration = sum(scene.get("duration_sec", 0) for scene in result["internal_scenes"])
                    coverage_ratio = covered_duration / total_requested_duration if total_requested_duration > 0 else 0

                    min_coverage_ratio = 0.5  # Require at least 50% coverage
                    if coverage_ratio < min_coverage_ratio:
                        print(f"⚠️ trim_shot output validation failed:")
                        print(f"   Requested: {total_requested_duration:.2f}s ({clip_start_time} to {clip_end_time})")
                        print(f"   Covered: {covered_duration:.2f}s (ratio: {coverage_ratio:.1%})")
                        print(f"   Scenes returned: {len(result['internal_scenes'])}")

                        # Print scene details for debugging
                        for i, scene in enumerate(result["internal_scenes"]):
                            scene_time = scene.get("scene_time", "unknown")
                            scene_dur = scene.get("duration_sec", 0)
                            print(f"   Scene {i+1}: {scene_time} ({scene_dur:.2f}s)")

                        if tries > 0:
                            print(f"   Retrying... ({tries} attempts remaining)")
                            continue
                        else:
                            print(f"   ⚠️ Max retries reached. Returning partial result.")
                            # Add warning to result
                            result["usability_assessment"] = (
                                f"⚠️ WARNING: Model only provided {coverage_ratio:.0%} coverage of requested range. "
                                f"Scenes may be incomplete or improperly segmented. Consider calling trim_shot with a different time range."
                            )

                    result_json = json.dumps(result, indent=4, ensure_ascii=False)
                    # Cache the successful result
                    if frame_path:
                        _set_trim_shot_cache(frame_path, time_range, result_json)
                    return result_json
                
            except json.JSONDecodeError as e:
                print(f"❌ [Error] JSON decode error for time range {time_range}: {e}")
                print(f"📄 [Data] Raw content (first 500 chars): {content_str[:500]}")
                if tries == 0:
                    return f"Error: Failed to parse model response for time range {time_range}. Content: {content_str[:200]}"
                continue
            except Exception as e:
                print(f"❌ [Error] Unexpected error processing response for time range {time_range}: {e}")
                if tries == 0:
                    return f"Error: Unexpected error processing response: {str(e)}"
                continue

        return f"Error: Failed to generate caption for time range {time_range} after multiple attempts."
    finally:
        b64_frames = None
        litellm_messages = None
        gc.collect()


class EditorCoreAgent:
    def __init__(self, video_caption_path, video_scene_path, audio_caption_path, output_path, max_iterations, video_path=None, video_reader=None, frame_folder_path=None, transcript_path: str = None):
        self.tools = [semantic_neighborhood_retrieval, fine_grained_shot_trimming, review_clip, commit]
        self.name_to_function_map = {tool.__name__: tool for tool in self.tools}
        for original_name, alias_name in TOOL_NAME_ALIASES.items():
            if original_name in self.name_to_function_map:
                self.name_to_function_map[alias_name] = self.name_to_function_map[original_name]

        self.function_schemas = []
        for func in self.tools:
            schema = as_json_schema(func)
            schema["name"] = func.__name__
            display_name = TOOL_NAME_ALIASES.get(func.__name__)
            if display_name:
                original_desc = schema.get("description", "")
                schema["description"] = f"Display name: {display_name}.\n{original_desc}".strip()
            self.function_schemas.append({"function": schema, "type": "function"})
        self.video_caption_path = video_caption_path
        self.video_scene_path = video_scene_path
        self.audio_db = json.load(open(audio_caption_path, 'r', encoding='utf-8'))
        self.max_iterations = max_iterations
        self.frame_folder_path = frame_folder_path
        self.video_path = video_path
        self.video_reader = _normalize_video_reader(video_reader)
        self.transcript_path = transcript_path
        self.output_path = output_path
        self.current_target_length = None  # Will be set during run()
        self.messages = self._construct_messages()
        # Track used time ranges to avoid duplicate clip selection
        self.used_time_ranges = []  # List of (start_sec, end_sec) tuples
        self.current_section_idx = None
        self.current_shot_idx = None
        self.current_related_scenes = []  # Will be set during run() for each shot
        self.attempted_time_ranges = set()  # Track attempted trim_shot time ranges to avoid duplicate calls
        self.duplicate_call_count = 0  # Count consecutive duplicate calls
        self.max_duplicate_calls = 3  # Max duplicates before restart
        self.forbidden_time_ranges = []  # Global avoid ranges injected by orchestrator
        self.guidance_text = None
        self.last_commit_result = None
        self.last_commit_raw = None

        # Initialize ReviewerAgent for finish validation
        self.reviewer = ReviewerAgent(
            frame_folder_path=frame_folder_path,
            video_path=video_path
        )
        # Current shot context for reviewer
        self.current_shot_context = {}

    def _load_progress(self):
        """
        Load progress from existing output file to support resume functionality.
        Returns a set of completed (section_idx, shot_idx) tuples.
        """
        if not self.output_path or not os.path.exists(self.output_path):
            return set()

        try:
            with open(self.output_path, 'r', encoding='utf-8') as f:
                results = json.load(f)

            completed = set()
            for result in results:
                if result.get('status') == 'success':
                    sec_idx = result.get('section_idx')
                    shot_idx = result.get('shot_idx')
                    if sec_idx is not None and shot_idx is not None:
                        completed.add((sec_idx, shot_idx))

            if completed:
                print(f"📋 Found {len(completed)} completed shots in existing output file")
                print(f"   Completed: {sorted(completed)}")

            return completed
        except Exception as e:
            print(f"⚠️ Error loading progress: {e}")
            return set()

    def _construct_messages(self):
        user_prompt = (
            EDITOR_USER_PROMPT
            .replace("SCENE_EXPLORATION_RANGE_PLACEHOLDER", str(getattr(config, 'SCENE_EXPLORATION_RANGE', 3)))
            .replace("MIN_PROTAGONIST_RATIO_PLACEHOLDER", f"{config.MIN_PROTAGONIST_RATIO * 100:.0f}")
            .replace("MIN_ACCEPTABLE_SHOT_DURATION_PLACEHOLDER", str(getattr(config, 'MIN_ACCEPTABLE_SHOT_DURATION', 2.0)))
            .replace("ALLOW_DURATION_TOLERANCE_PLACEHOLDER", str(getattr(config, 'ALLOW_DURATION_TOLERANCE', 1.0)))
        )
        messages = [
            {"role": "system", "content": EDITOR_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        return messages

    def _build_audio_section_info(self, audio_section, shot_idx):
        detailed_analysis = audio_section.get('detailed_analysis', {})
        audio_info_parts = []

        if 'name' in audio_section:
            audio_info_parts.append(f"Section: {audio_section['name']}")
        if 'description' in audio_section:
            audio_info_parts.append(f"Description: {audio_section['description']}")

        if isinstance(detailed_analysis, dict) and 'summary' in detailed_analysis:
            audio_info_parts.append(f"Summary: {detailed_analysis['summary']}")

        if isinstance(detailed_analysis, dict) and 'sections' in detailed_analysis:
            sections_list = detailed_analysis['sections']
            if isinstance(sections_list, list) and shot_idx < len(sections_list):
                audio_info_parts.append(f"Shot caption: {sections_list[shot_idx]}")
            elif isinstance(sections_list, dict) and str(shot_idx) in sections_list:
                audio_info_parts.append(f"Shot caption: {sections_list[str(shot_idx)]}")

        return "\n".join(audio_info_parts) if audio_info_parts else "No audio information available"

    def _prepare_shot_messages(self, shot, audio_section_info, related_scene_value, guidance_text=None, forbidden_time_ranges=None):
        msgs = copy.deepcopy(self.messages)
        msgs[-1]["content"] = msgs[-1]["content"].replace("VIDEO_LENGTH_PLACEHOLDER", str(shot['time_duration']))
        msgs[-1]["content"] = msgs[-1]["content"].replace("CURRENT_VIDEO_CONTENT_PLACEHOLDER", shot['content']).replace("CURRENT_VIDEO_EMOTION_PLACEHOLDER", shot['emotion'])
        msgs[-1]["content"] = msgs[-1]["content"].replace("BACKGROUND_MUSIC_PLACEHOLDER", audio_section_info)

        recommended_scenes_str = str(related_scene_value) if related_scene_value else "None specified"
        msgs[-1]["content"] = msgs[-1]["content"].replace("RECOMMENDED_SCENES_PLACEHOLDER", recommended_scenes_str)

        if guidance_text or forbidden_time_ranges:
            avoid_msg = []
            if forbidden_time_ranges:
                formatted = []
                for start_sec, end_sec in forbidden_time_ranges:
                    formatted.append(f"{convert_seconds_to_hhmmss(start_sec)} to {convert_seconds_to_hhmmss(end_sec)}")
                avoid_msg.append("Avoid time ranges: " + "; ".join(formatted))
            if guidance_text:
                avoid_msg.append("Guidance: " + guidance_text)
            msgs.append({
                "role": "user",
                "content": "\n".join(avoid_msg)
            })

        return msgs

    def _run_shot_loop(self, msgs, max_iterations=None):
        if max_iterations is None:
            max_iterations = self.max_iterations

        should_restart = False
        section_completed = False

        for i in range(max_iterations):
            if i == max_iterations - 1:
                msgs.append(
                    {
                        "role": "user",
                        "content": EDITOR_FINISH_PROMPT,
                    }
                )

            max_model_retries = getattr(config, "AGENT_MODEL_MAX_RETRIES", 2)
            max_tool_retries = 2
            tool_execution_success = False

            for tool_retry in range(max_tool_retries):
                msgs_snapshot_len = len(msgs)  # track length for rollback instead of deepcopy

                response = None
                context_length_error = False
                for model_retry in range(max_model_retries):
                    tool_calls_raw = None
                    try:
                        kwargs = dict(
                            model=config.AGENT_LITELLM_MODEL,
                            messages=msgs,
                            temperature=1.0,
                            max_tokens=config.AGENT_MODEL_MAX_TOKEN,
                            tools=self.function_schemas,
                            tool_choice="auto",
                        )
                        if config.AGENT_LITELLM_URL:
                            kwargs["api_base"] = config.AGENT_LITELLM_URL
                        if config.AGENT_LITELLM_API_KEY:
                            kwargs["api_key"] = config.AGENT_LITELLM_API_KEY
                        raw = litellm.completion(**kwargs)
                        msg = raw.choices[0].message
                        tool_calls_raw = getattr(msg, "tool_calls", None)
                        response = {
                            "role": msg.role or "assistant",
                            "content": msg.content,
                            "tool_calls": [
                                {
                                    "id": tc.id,
                                    "type": tc.type,
                                    "function": {
                                        "name": tc.function.name,
                                        "arguments": tc.function.arguments,
                                    },
                                }
                                for tc in (tool_calls_raw or [])
                            ] or None,
                        }
                        reasoning = getattr(msg, "reasoning_content", None)
                        if reasoning:
                            response["reasoning_content"] = reasoning
                        if response is not None:
                            break
                        else:
                            print(f"🔄 [Retry] Model returned None, retrying ({model_retry + 1}/{max_model_retries})...")
                    except Exception as e:
                        error_msg = str(e).lower()
                        if "context length" in error_msg or "too large" in error_msg or "max_tokens" in error_msg:
                            print(f"❌ [Error] Context length exceeded: {e}")
                            context_length_error = True
                            break
                        is_rate_limited = (
                            "ratelimit" in error_msg
                            or "max organization concurrency" in error_msg
                            or "too many requests" in error_msg
                            or " 429" in error_msg
                        )
                        if is_rate_limited:
                            base_wait = _parse_retry_after_seconds(
                                str(e),
                                default_seconds=getattr(config, "AGENT_RATE_LIMIT_BACKOFF_BASE", 1.0),
                            )
                            max_backoff = getattr(config, "AGENT_RATE_LIMIT_MAX_BACKOFF", 8.0)
                            wait_seconds = min(max_backoff, base_wait * (2 ** model_retry))
                            print(
                                f"Rate limit encountered, sleeping {wait_seconds:.1f}s "
                                f"before retry ({model_retry + 1}/{max_model_retries})..."
                            )
                            time.sleep(wait_seconds)
                        print(f"🔄 [Retry] Model call failed: {e}, retrying ({model_retry + 1}/{max_model_retries})...")
                        if model_retry == max_model_retries - 1:
                            raise

                if context_length_error:
                    print("🔄 [Restart] Triggering restart due to context overflow...")
                    should_restart = True
                    break

                if response is None:
                    print(f"❌ [Error] Model call failed after {max_model_retries} retries.")
                    msgs[:] = msgs_snapshot
                    break

                response.setdefault("role", "assistant")
                msgs.append(response)
                print("#### Iteration: ", i, f"(Tool retry: {tool_retry + 1}/{max_tool_retries})" if tool_retry > 0 else "")
                print(response)

                tool_execution_failed = False

                try:
                    tool_calls = response.get("tool_calls", [])
                    if tool_calls is None:
                        tool_calls = []

                    if not tool_calls:
                        content = response.get("content", "")
                        final_shot_pattern = re.search(r'\[shot:\s*[\d:.]+\s+to\s+[\d:.]+\s*\]', content, re.IGNORECASE)
                        is_short_response = len(content) < 500
                        is_final_answer = final_shot_pattern and (is_short_response or content.strip().endswith(']'))

                        if is_final_answer:
                            print("✅ [Agent] Model returned final answer. Task completed.")
                            section_completed = True
                            tool_execution_success = True
                            break
                        else:
                            print("⚠️  [Agent] Model did not call any tool. Adding prompt to use tools...")
                            msgs.append({
                                "role": "user",
                                "content": EDITOR_USE_TOOL_PROMPT
                            })

                    for tool_call in tool_calls:
                        is_finished = self._exec_tool(tool_call, msgs)
                        if is_finished == "RESTART":
                            should_restart = True
                            break
                        if is_finished:
                            section_completed = True
                            break

                    if should_restart:
                        print("🔄 [Restart] Restarting conversation for current shot...")
                        break

                    tool_execution_success = True

                except StopException:
                    return True, False
                except Exception as e:
                    print(f"❌ [Error] Error executing tool calls: {e}")
                    import traceback
                    traceback.print_exc()
                    for tc in (response.get("tool_calls") or []):
                        self._append_tool_msg(
                            tc["id"],
                            tc["function"]["name"],
                            f"Tool execution error: {e}",
                            msgs,
                        )
                    tool_execution_success = True
                    tool_execution_failed = False

                if tool_execution_success or tool_retry == max_tool_retries - 1:
                    break

                if tool_execution_failed:
                    print("🔄 [Retry] Rolling back messages and retrying...")
                    del msgs[msgs_snapshot_len:]
                    continue

            if should_restart:
                break

            if section_completed:
                print(f"⏭️  [Progress] Shot {self.current_shot_idx + 1} completed. Moving to next shot...")
                break

        return section_completed, should_restart

    # ------------------------------------------------------------------ #
    # Helper methods
    # ------------------------------------------------------------------ #
    def _append_tool_msg(self, tool_call_id, name, content, msgs):
        msgs.append(
            {
                "tool_call_id": tool_call_id,
                "role": "tool",
                "name": name,
                "content": content,
            }
        )

    def _exec_tool(self, tool_call, msgs):
        name = tool_call["function"]["name"]
        canonical_name = _canonical_tool_name(name)
        if canonical_name not in self.name_to_function_map:
            self._append_tool_msg(tool_call["id"], name, f"Invalid function name: {name!r}", msgs)
            return False

        # Parse arguments
        try:
            args = json.loads(tool_call["function"]["arguments"])
        except json.JSONDecodeError as exc:
            raise StopException(f"Error decoding arguments: {exc!s}")

        # Inject system-provided parameters
        if "topk" in args:
            if config.OVERWRITE_CLIP_SEARCH_TOPK > 0:
                args["topk"] = config.OVERWRITE_CLIP_SEARCH_TOPK

        # For semantic_neighborhood_retrieval, inject scene_folder_path and recommended_scenes
        if canonical_name == "semantic_neighborhood_retrieval":
            agent_requested_scenes = args.get("related_scenes", [])
            if agent_requested_scenes and isinstance(agent_requested_scenes, list):
                # Agent explicitly requested specific scenes - will be validated in function
                print(f"📍 Agent requested scenes: {agent_requested_scenes}")
            else:
                # No agent request, will use default recommended related scenes in function
                print(f"📍 No specific scenes requested, will use recommended: {self.current_related_scenes}")

            # Inject both scene_folder_path and recommended_scenes for validation
            args["scene_folder_path"] = self.video_scene_path
            args["recommended_scenes"] = self.current_related_scenes

        # For fine_grained_shot_trimming, inject video/transcript parameters and check for duplicate calls
        if canonical_name == "fine_grained_shot_trimming":
            if self.video_path:
                args["frame_path"] = self.video_path
            elif self.video_reader is None:
                self._append_tool_msg(
                    tool_call["id"],
                    name,
                    "Error: neither video_reader nor video_path is configured in agent.",
                    msgs
                )
                return False

            if self.transcript_path:
                args["transcript_path"] = self.transcript_path

            # Check for duplicate time range calls to prevent infinite loops
            time_range = args.get("time_range", "")
            # Normalize the time range for comparison (remove extra spaces)
            normalized_range = " ".join(time_range.split())

            if normalized_range in self.attempted_time_ranges:
                self.duplicate_call_count += 1
                print(f"⚠️ Duplicate call detected ({self.duplicate_call_count}/{self.max_duplicate_calls}): {normalized_range}")

                if self.duplicate_call_count >= self.max_duplicate_calls:
                    print(f"🔄 Max duplicate calls reached. Restarting conversation for this shot...")
                    return "RESTART"  # Signal to restart the conversation

                # Return a helpful message instead of calling the tool again
                self._append_tool_msg(
                    tool_call["id"],
                    name,
                    f"Warning: You have already analyzed '{time_range}'. "
                    f"Duplicate call {self.duplicate_call_count}/{self.max_duplicate_calls}. "
                    f"Call 'Commit' NOW with your best selection, or conversation will restart.",
                    msgs
                )
                return False

            # Reset duplicate counter on new time range
            self.duplicate_call_count = 0
            # Record this time range as attempted
            self.attempted_time_ranges.add(normalized_range)
        
        # For review_clip, inject used_time_ranges
        if canonical_name == "review_clip":
            args["used_time_ranges"] = self.used_time_ranges + (self.forbidden_time_ranges or [])
            print(f"📍 Checking overlap against {len(self.used_time_ranges)} used clips")

        # For commit, first call ReviewerAgent to validate
        if canonical_name == "commit":
            args["video_path"] = self.video_path or ""
            args["output_path"] = self.output_path or ""
            args["target_length_sec"] = self.current_target_length or 0.0
            args["section_idx"] = self.current_section_idx if self.current_section_idx is not None else -1
            args["shot_idx"] = self.current_shot_idx if self.current_shot_idx is not None else -1
            # Note: protagonist_frame_data will be set after face quality check

            # Enforce forbidden time ranges (from parallel orchestrator guidance)
            if self.forbidden_time_ranges:
                proposed_ranges = _parse_shot_time_ranges(args.get("answer", ""))
                if not proposed_ranges:
                    self._append_tool_msg(
                        tool_call["id"],
                        name,
                        "Error: Could not parse shot time range for overlap checks. Please use format: [shot: HH:MM:SS to HH:MM:SS]",
                        msgs
                    )
                    return False
                for p_start, p_end in proposed_ranges:
                    for f_start, f_end in self.forbidden_time_ranges:
                        if _ranges_overlap(p_start, p_end, f_start, f_end):
                            self._append_tool_msg(
                                tool_call["id"],
                                name,
                                "Overlap detected with forbidden ranges. Please select a different time range.",
                                msgs
                            )
                            return False

            # Call ReviewerAgent to validate before executing finish
            if config.ENABLE_REVIEWER:
                shot_proposal = {
                    "answer": args.get("answer", ""),
                    "target_length_sec": self.current_target_length or 0.0
                }

                # Face quality check (optional, controlled by config.ENABLE_FACE_QUALITY_CHECK)
                if config.ENABLE_FACE_QUALITY_CHECK:
                    time_match = re.search(r'\[?shot[\s_]*\d*:\s*([0-9:.]+)\s+to\s+([0-9:.]+)\]?', shot_proposal["answer"], re.IGNORECASE)
                    if time_match:
                        time_range = f"{time_match.group(1)} to {time_match.group(2)}"

                        face_check_method = getattr(config, 'FACE_QUALITY_CHECK_METHOD', 'vlm')
                        if face_check_method != 'vlm':
                            print("⚠️  FACE_QUALITY_CHECK_METHOD is not 'vlm'; falling back to 'vlm' (face_recognition removed).")
                            face_check_method = 'vlm'

                        print("🎬 Using VLM face quality check method...")
                        check_result = self.reviewer.check_face_quality_vlm(
                            video_path=args.get("video_path", ""),
                            time_range=time_range,
                            main_character_name=getattr(config, 'MAIN_CHARACTER_NAME', 'the main character'),
                            min_protagonist_ratio=getattr(config, 'MIN_PROTAGONIST_RATIO', 0.7),
                            min_box_size=getattr(config, 'VLM_MIN_BOX_SIZE', 50),
                            return_frame_data=True,
                        )

                        # check_face_quality_vlm returns str on error, tuple[str, list] on success
                        if isinstance(check_result, str):
                            face_check = check_result
                            protagonist_frame_data = []
                        else:
                            face_check, protagonist_frame_data = check_result

                        # Store for debugging/trace
                        self.current_shot_context["face_quality"] = face_check
                        self.current_shot_context["face_quality_method"] = face_check_method

                        if "❌" in face_check or "FAILED" in face_check:
                            self._append_tool_msg(
                                tool_call["id"],
                                name,
                                f"Review Failed - Face quality check ({face_check_method}) did not pass:\n{face_check}",
                                msgs
                            )
                            return False

                        if not protagonist_frame_data:
                            self.current_shot_context["protagonist_frame_data"] = None
                            args["protagonist_frame_data"] = None
                        else:
                            self.current_shot_context["protagonist_frame_data"] = protagonist_frame_data
                            args["protagonist_frame_data"] = protagonist_frame_data

                # Set protagonist_frame_data if not already set
                if "protagonist_frame_data" not in args or args.get("protagonist_frame_data") is None:
                    context_data = self.current_shot_context.get("protagonist_frame_data", None)
                    if context_data:
                        args["protagonist_frame_data"] = context_data
                        print(f"✅ Set protagonist_frame_data from context: {len(context_data)} detections")
                    else:
                        args["protagonist_frame_data"] = None

                review_result = self.reviewer.review(
                    shot_proposal=shot_proposal,
                    context=self.current_shot_context,
                    used_time_ranges=self.used_time_ranges
                )

                self.current_shot_context["review_result"] = review_result

                if not review_result["approved"]:
                    self._append_tool_msg(
                        tool_call["id"],
                        name,
                        f"Review Failed - Please adjust your selection:\n{review_result['feedback']}",
                        msgs
                    )
                    return False  # Continue iteration

        # Call the tool
        try:
            result = self.name_to_function_map[canonical_name](**args)
            tool_result_for_msg = result
            if canonical_name == "commit":
                # commit payload can include large frame-level detections; avoid flooding logs/context
                tool_result_for_msg = _compact_json_str_for_log(result)
            elif canonical_name == "fine_grained_shot_trimming":
                # Truncate internal_scenes list to avoid msgs growing unboundedly across iterations
                try:
                    parsed = json.loads(result)
                    scenes = parsed.get("internal_scenes", [])
                    max_scenes = getattr(config, "TRIM_SHOT_MAX_SCENES_IN_HISTORY", 8)
                    if len(scenes) > max_scenes:
                        parsed["internal_scenes"] = scenes[:max_scenes]
                        parsed["_scenes_truncated"] = f"{len(scenes) - max_scenes} more scenes omitted from context"
                    tool_result_for_msg = json.dumps(parsed, ensure_ascii=False)
                except Exception:
                    pass
            self._append_tool_msg(tool_call["id"], name, tool_result_for_msg, msgs)

            # Check if commit was successful
            if canonical_name == "commit":
                # Parse result as JSON and check status
                try:
                    result_data = json.loads(result)
                    if result_data.get("status") == "success":
                        self.last_commit_result = result_data
                        self.last_commit_raw = result
                        # Record used time ranges to prevent duplicate selection
                        clips = result_data.get("clips", [])
                        for clip in clips:
                            start_str = clip.get("start", "")
                            end_str = clip.get("end", "")
                            if start_str and end_str:
                                # Convert to seconds for comparison
                                def hhmmss_to_sec(t):
                                    parts = t.strip().split(':')
                                    if len(parts) == 3:
                                        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
                                    elif len(parts) == 2:
                                        return int(parts[0]) * 60 + float(parts[1])
                                    return float(parts[0])
                                start_sec = hhmmss_to_sec(start_str)
                                end_sec = hhmmss_to_sec(end_str)
                                self.used_time_ranges.append((start_sec, end_sec))
                        return True  # Signal to break the current section loop
                except json.JSONDecodeError:
                    # If not JSON, check for success message in string
                    if "Successfully validated shot selection" in result:
                        return True
            
            return False
        except StopException as exc:  # graceful stop
            raise

    # ------------------------------------------------------------------ #
    # Main loop
    # ------------------------------------------------------------------ #
    def run(self, shot_plan_path: str) -> list[dict]:
        """
        Run the ReAct-style loop with OpenAI Function Calling.

        Args:
            shot_plan_path: Path to a pre-generated shot_plan.json file.
        """

        # Load shot plan from file
        print(f"📂 [Init] Loading shot plan from: {shot_plan_path}")
        with open(shot_plan_path, 'r', encoding='utf-8') as f:
            structure_proposal = json.load(f)

        # Load progress from existing output file (for resume functionality)
        completed_shots = self._load_progress()

        # Store original output path and create section-specific paths
        original_output_path = self.output_path
        print("📄 [Data] structure_proposal: ", structure_proposal)
        for sec_idx, sec_cur in enumerate(structure_proposal['video_structure']):
            print(f"\n{'='*60}")
            print(f"Processing Section {sec_idx + 1}/{len(structure_proposal['video_structure'])}")
            print(f"{'='*60}\n")
            # Set current section for reporting
            self.current_section_idx = sec_idx
            
            # Load shot_plan from sec_cur (loaded from file)
            shot_plan = sec_cur.get('shot_plan')
            if not shot_plan:
                print(f"❌ [Error] No shot_plan found for section {sec_idx}")
                continue
            print("Using shot plan from file")
            for idx, shot in enumerate(shot_plan['shots']):
                # Check if this shot is already completed (resume functionality)
                if (sec_idx, idx) in completed_shots:
                    print(f"\n⏭️  Skipping Shot {idx + 1}/{len(shot_plan['shots'])} - Already completed")
                    continue

                max_shot_restarts = 3  # Max restarts per shot
                for restart_attempt in range(max_shot_restarts):
                    if restart_attempt > 0:
                        print(f"\nRestart attempt {restart_attempt + 1}/{max_shot_restarts} for Shot {idx + 1}")

                    print(f"\n{'='*60}")
                    print(f"Processing Shot {idx + 1}/{len(shot_plan['shots'])}")
                    print(f"{'='*60}\n")
                    print("shot plan: ", shot)

                    self.output_path = original_output_path
                    print(f"Output path: {self.output_path}")
                    self.current_shot_idx = idx
                    self.attempted_time_ranges = set()
                    self.duplicate_call_count = 0

                    audio_section = self.audio_db['sections'][sec_idx]
                    audio_section_info = self._build_audio_section_info(audio_section, idx)

                    self.current_target_length = shot['time_duration']
                    self.current_shot_context = {
                        "content": shot.get('content', ''),
                        "emotion": shot.get('emotion', ''),
                        "section_idx": sec_idx,
                        "shot_idx": idx,
                        "time_duration": shot.get('time_duration', 0)
                    }

                    related_scene_value = shot.get('related_scene', [])
                    if isinstance(related_scene_value, int):
                        self.current_related_scenes = [related_scene_value]
                    elif isinstance(related_scene_value, list):
                        self.current_related_scenes = related_scene_value
                    else:
                        self.current_related_scenes = []

                    msgs = self._prepare_shot_messages(
                        shot=shot,
                        audio_section_info=audio_section_info,
                        related_scene_value=related_scene_value,
                    )

                    section_completed, should_restart = self._run_shot_loop(msgs, max_iterations=self.max_iterations)

                    if section_completed:
                        break
                    if not should_restart:
                        print(f"Max iterations reached for Shot {idx + 1}. Moving to next shot.")
                        break

                # End restart loop

            # End of shot loop
            print(f"\nSection {sec_idx + 1} completed. All shots processed.")

        return msgs

    def run_single_shot(self, shot, sec_idx: int, shot_idx: int, guidance_text: str = None, forbidden_time_ranges: list = None, max_shot_restarts: int = 2, max_iterations: int = None):
        """Run a single shot selection loop and return the commit result dict on success."""
        if max_iterations is None:
            max_iterations = self.max_iterations

        self.output_path = ""
        self.current_section_idx = sec_idx
        self.current_shot_idx = shot_idx
        self.forbidden_time_ranges = forbidden_time_ranges or []
        self.guidance_text = guidance_text
        self.last_commit_result = None
        self.last_commit_raw = None

        audio_section = self.audio_db['sections'][sec_idx]
        audio_section_info = self._build_audio_section_info(audio_section, shot_idx)

        self.current_target_length = shot['time_duration']
        self.current_shot_context = {
            "content": shot.get('content', ''),
            "emotion": shot.get('emotion', ''),
            "section_idx": sec_idx,
            "shot_idx": shot_idx,
            "time_duration": shot.get('time_duration', 0)
        }

        related_scene_value = shot.get('related_scene', [])
        if isinstance(related_scene_value, int):
            self.current_related_scenes = [related_scene_value]
        elif isinstance(related_scene_value, list):
            self.current_related_scenes = related_scene_value
        else:
            self.current_related_scenes = []

        for restart_attempt in range(max_shot_restarts):
            if restart_attempt > 0:
                print(f"Restart attempt {restart_attempt + 1}/{max_shot_restarts} for Shot {shot_idx + 1}")

            self.attempted_time_ranges = set()
            self.duplicate_call_count = 0

            msgs = self._prepare_shot_messages(
                shot=shot,
                audio_section_info=audio_section_info,
                related_scene_value=related_scene_value,
                guidance_text=guidance_text,
                forbidden_time_ranges=forbidden_time_ranges,
            )

            section_completed, should_restart = self._run_shot_loop(msgs, max_iterations=max_iterations)

            if section_completed and self.last_commit_result:
                return self.last_commit_result
            if not should_restart:
                break

        return None

    def cleanup(self):
        # Release large references to help GC after each subagent run.
        self.messages = []
        self.audio_db = {}
        self.used_time_ranges = []
        self.current_related_scenes = []
        self.attempted_time_ranges = set()
        self.current_shot_context = {}
        self.last_commit_result = None
        self.last_commit_raw = None
        clear_trim_shot_cache()
        self.reviewer = None
        self.video_reader = None
        _clear_thread_video_reader()
        _clear_thread_video_reader()
        gc.collect()


class ParallelShotOrchestrator:
    def __init__(self, video_caption_path, video_scene_path, audio_caption_path, output_path, max_iterations, video_path=None, frame_folder_path=None, transcript_path: str = None, max_workers: int = None, max_reruns: int = None):
        self.video_caption_path = video_caption_path
        self.video_scene_path = video_scene_path
        self.audio_caption_path = audio_caption_path
        self.output_path = output_path
        self.max_iterations = max_iterations
        self.video_path = video_path
        self.frame_folder_path = frame_folder_path
        self.transcript_path = transcript_path

        self.max_workers = max_workers or getattr(config, 'PARALLEL_SHOT_MAX_WORKERS', 4)
        self.max_reruns = max_reruns if max_reruns is not None else getattr(config, 'PARALLEL_SHOT_MAX_RERUNS', 2)
        self._output_lock = threading.Lock()

    def _compute_quality_score(self, result_data: dict) -> float:
        if not result_data:
            return 0.0
        protagonist_ratio = 0.0
        if "protagonist_detection" in result_data:
            protagonist_ratio = result_data["protagonist_detection"].get("protagonist_ratio", 0.0)
        total_duration = result_data.get("total_duration", 0.0)
        target_duration = result_data.get("target_duration", 0.0)
        if target_duration <= 0:
            duration_score = 0.0
        else:
            duration_score = 1.0 - min(1.0, abs(total_duration - target_duration) / max(target_duration, 1.0))
        return 0.6 * protagonist_ratio + 0.4 * duration_score

    def _result_ranges(self, result_data: dict) -> list[tuple[float, float]]:
        ranges = []
        if not result_data:
            return ranges
        clips = result_data.get("clips", [])
        for clip in clips:
            start = clip.get("start")
            end = clip.get("end")
            if start and end:
                start_sec = _hhmmss_to_seconds(start, fps=getattr(config, 'VIDEO_FPS', 24) or 24)
                end_sec = _hhmmss_to_seconds(end, fps=getattr(config, 'VIDEO_FPS', 24) or 24)
                ranges.append((start_sec, end_sec))
        return ranges

    def _detect_conflicts(self, results: dict, keep_ranges: list) -> dict:
        """Return a dict of losers keyed by (sec_idx, shot_idx) with guidance text."""
        losers = {}
        items = list(results.items())

        # Conflicts with already kept ranges (from prior sections or winners)
        for key, result in items:
            ranges = self._result_ranges(result)
            for r_start, r_end in ranges:
                for k_start, k_end in keep_ranges:
                    if _ranges_overlap(r_start, r_end, k_start, k_end):
                        losers[key] = "Overlap with already selected clips. Please choose a different time range."
                        break
                if key in losers:
                    break

        # Pairwise conflicts in the current batch
        for i in range(len(items)):
            key_i, res_i = items[i]
            if key_i in losers:
                continue
            ranges_i = self._result_ranges(res_i)
            for j in range(i + 1, len(items)):
                key_j, res_j = items[j]
                if key_j in losers:
                    continue
                ranges_j = self._result_ranges(res_j)
                overlap = False
                for a_start, a_end in ranges_i:
                    for b_start, b_end in ranges_j:
                        if _ranges_overlap(a_start, a_end, b_start, b_end):
                            overlap = True
                            break
                    if overlap:
                        break
                if overlap:
                    score_i = self._compute_quality_score(res_i)
                    score_j = self._compute_quality_score(res_j)
                    if score_i >= score_j:
                        losers[key_j] = f"Overlap with shot {key_i[1] + 1}. Please choose a different time range."
                    else:
                        losers[key_i] = f"Overlap with shot {key_j[1] + 1}. Please choose a different time range."
                        break

        return losers

    def _run_worker(self, shot, sec_idx, shot_idx, guidance_text=None, forbidden_time_ranges=None):
        mode = "rerun" if guidance_text else "initial"
        print(f"[SubAgent S{sec_idx + 1} Shot{shot_idx + 1}] start ({mode})")
        agent = EditorCoreAgent(
            self.video_caption_path,
            self.video_scene_path,
            self.audio_caption_path,
            output_path="",
            max_iterations=self.max_iterations,
            video_path=self.video_path,
            frame_folder_path=self.frame_folder_path,
            transcript_path=self.transcript_path
        )
        try:
            result = agent.run_single_shot(
                shot=shot,
                sec_idx=sec_idx,
                shot_idx=shot_idx,
                guidance_text=guidance_text,
                forbidden_time_ranges=forbidden_time_ranges
            )
            if result:
                print(f"[SubAgent S{sec_idx + 1} Shot{shot_idx + 1}] success")
                self._append_result_to_output((sec_idx, shot_idx), result)
            else:
                print(f"[SubAgent S{sec_idx + 1} Shot{shot_idx + 1}] no-result")
            return result
        finally:
            agent.cleanup()
            # Explicitly clear thread-local video reader in this worker thread
            _clear_thread_video_reader()
            gc.collect()

    def _merge_results(self, existing_list: list, new_results: dict) -> list:
        result_map = {}
        for item in existing_list or []:
            if item.get("status") == "success":
                key = (item.get("section_idx"), item.get("shot_idx"))
                result_map[key] = item

        for key, result in new_results.items():
            if result:
                result_map[key] = result

        merged = list(result_map.values())
        merged.sort(key=lambda x: (x.get("section_idx", 0), x.get("shot_idx", 0)))
        return merged

    def _save_checkpoint(self, existing_list: list, new_results: dict):
        if not self.output_path:
            return
        merged = self._merge_results(existing_list, new_results)
        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(merged, f, ensure_ascii=False, indent=2)
        print(f"[Parallel] checkpoint saved: {len(merged)} shots")

    def _append_result_to_output(self, key: tuple, result: dict):
        if not self.output_path or not result:
            return
        with self._output_lock:
            existing = []
            if os.path.exists(self.output_path):
                try:
                    with open(self.output_path, 'r', encoding='utf-8') as f:
                        existing = json.load(f)
                except Exception:
                    existing = []
            merged = self._merge_results(existing, {key: result})
            with open(self.output_path, 'w', encoding='utf-8') as f:
                json.dump(merged, f, ensure_ascii=False, indent=2)
            print(f"[Parallel] subagent saved: section={key[0]} shot={key[1]}")

    def run_parallel(self, shot_plan_path: str):
        with open(shot_plan_path, 'r', encoding='utf-8') as f:
            structure_proposal = json.load(f)

        global_keep_ranges = []
        final_results = {}
        existing = []
        completed_shots = set()

        if self.output_path and os.path.exists(self.output_path):
            try:
                with open(self.output_path, 'r', encoding='utf-8') as f:
                    existing = json.load(f)
            except Exception:
                existing = []

        for item in existing:
            if item.get("status") != "success":
                continue
            sec_idx = item.get("section_idx")
            shot_idx = item.get("shot_idx")
            if sec_idx is None or shot_idx is None:
                continue
            completed_shots.add((sec_idx, shot_idx))
            for r_start, r_end in self._result_ranges(item):
                global_keep_ranges.append((r_start, r_end))

        if completed_shots:
            print(f"📋 [Parallel] Found {len(completed_shots)} completed shots in existing output file")
            print(f"   Completed: {sorted(completed_shots)}")

        for sec_idx, sec_cur in enumerate(structure_proposal['video_structure']):
            shot_plan = sec_cur.get('shot_plan')
            if not shot_plan:
                print(f"❌ [Error] No shot_plan found for section {sec_idx}")
                continue
            shots = shot_plan['shots']
            print(f"\n[Parallel] Processing Section {sec_idx + 1}/{len(structure_proposal['video_structure'])}")

            pending = {}
            skipped_in_section = 0
            for idx, shot in enumerate(shots):
                key = (sec_idx, idx)
                if key in completed_shots:
                    skipped_in_section += 1
                    continue
                pending[key] = shot

            if skipped_in_section:
                print(f"[Parallel][Section {sec_idx + 1}] skipped {skipped_in_section} completed shots")
            if not pending:
                print(f"[Parallel][Section {sec_idx + 1}] all shots already completed")
                continue

            pending_guidance = {key: None for key in pending}
            section_keep_ranges = []
            rerun_count = 0
            round_idx = 0

            while pending:
                round_idx += 1
                results = {}
                combined_keep_ranges = global_keep_ranges + section_keep_ranges
                print(
                    f"[Parallel][Section {sec_idx + 1}][Round {round_idx}] "
                    f"pending={len(pending)} rerun_count={rerun_count}/{self.max_reruns}"
                )

                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    futures = {}
                    for (s_idx, shot_idx), shot in pending.items():
                        key = (s_idx, shot_idx)
                        futures[executor.submit(
                            self._run_worker,
                            shot,
                            s_idx,
                            shot_idx,
                            guidance_text=pending_guidance.get(key),
                            forbidden_time_ranges=combined_keep_ranges
                        )] = key

                    for future in as_completed(futures):
                        key = futures[future]
                        try:
                            results[key] = future.result()
                        except Exception as e:
                            print(f"Worker failed for shot {key}: {e}")
                            results[key] = None

                losers = self._detect_conflicts(results, combined_keep_ranges)
                print(
                    f"[Parallel][Section {sec_idx + 1}][Round {round_idx}] "
                    f"conflicts={len(losers)} winners={len(results) - len(losers)}"
                )

                # Keep winners
                round_has_updates = False
                for key, result in results.items():
                    if key in losers:
                        continue
                    if result:
                        final_results[key] = result
                        round_has_updates = True
                        for r_start, r_end in self._result_ranges(result):
                            section_keep_ranges.append((r_start, r_end))

                if round_has_updates:
                    self._save_checkpoint(existing, final_results)

                if not losers:
                    print(f"[Parallel][Section {sec_idx + 1}] no conflicts remaining")
                    break

                if rerun_count >= self.max_reruns:
                    print(
                        f"[Parallel][Section {sec_idx + 1}] reached max reruns "
                        f"({self.max_reruns}), stop rerunning unresolved shots"
                    )
                    break

                pending = {key: shots[key[1]] for key in losers}
                pending_guidance = losers
                rerun_count += 1

            global_keep_ranges.extend(section_keep_ranges)

        # Save merged results
        merged = self._merge_results(existing, final_results)
        if self.output_path:
            with open(self.output_path, 'w', encoding='utf-8') as f:
                json.dump(merged, f, ensure_ascii=False, indent=2)
        return merged
