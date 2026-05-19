import os
import json
import copy
import re
import numpy as np
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from typing import Annotated as A, Optional, Tuple, List
import cv2
import litellm
from src.utils.media_utils import seconds_to_hhmmss as convert_seconds_to_hhmmss, array_to_base64
from src.utils.time_format_convert import hhmmss_to_seconds, seconds_to_hhmmss
from src import config
from src.prompt import VLM_AESTHETIC_ANALYSIS_PROMPT, VLM_PROTAGONIST_DETECTION_PROMPT
from src.func_call_shema import doc as D
from src.video.preprocess.video_utils import _create_decord_reader

class StopException(Exception):
    """
    Stop Execution by raising this exception (Signal that the task is Finished).
    """


_THREAD_VIDEO_READERS = threading.local()


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

def Review_timeline(timeline, ):
    """
    Review Execution by raising this exception (Signal that the task is Finished).
    """

def Review_audio_video_alignment(alignment, ):
    """
    Review Execution by raising this exception (Signal that the task is Finished).
    """


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
    if used_time_ranges is None:
        used_time_ranges = []

    # Parse the time range
    match = re.search(r'([\d:.]+)\s+to\s+([\d:.]+)', time_range, re.IGNORECASE)
    if not match:
        return f"Error: Could not parse time range '{time_range}'. Please use format 'HH:MM:SS to HH:MM:SS'."

    try:
        fps = getattr(config, "VIDEO_FPS", 24) or 24
        start_sec = hhmmss_to_seconds(match.group(1), fps=fps)
        end_sec = hhmmss_to_seconds(match.group(2), fps=fps)
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


def review_finish(
    answer: A[str, D("Output the final shot time range. Must be exactly ONE continuous clip.")],
    target_length_sec: A[float, D("Expected total length in seconds")] = 0.0,
) -> str:
    """
    Review and validate the proposed shot selection before finishing.
    Validates that exactly ONE shot is provided and its duration matches the target.
    You MUST call this tool BEFORE calling finish to ensure the shot is valid.

    IMPORTANT: Only accepts ONE continuous time range. Multiple shots will be rejected.
    Example: [shot: 00:10:00 to 00:10:03.4]

    Returns:
        str: Success message if validation passes, or error message if validation fails.
             If validation fails, you should adjust your shot selection.
    """
    # Parse the answer to extract shot time ranges
    # Expected formats: "[shot: 00:10:00 to 00:10:05]" or "shot 1: 00:10:00 to 00:10:05"
    shot_pattern = re.compile(r'\[?shot[\s_]*\d*:\s*([0-9:.]+)\s+to\s+([0-9:.]+)\]?', re.IGNORECASE)
    matches = shot_pattern.findall(answer)

    if not matches:
        return "❌ Error: Could not parse shot time ranges from the answer. Please provide time range(s) in the format: [shot: HH:MM:SS to HH:MM:SS]"

    # Allow multiple shots for stitching (with reasonable limit)
    from src import config
    max_shots_allowed = getattr(config, 'MAX_SHOTS_PER_CLIP', 3)
    if len(matches) > max_shots_allowed:
        return (
            f"❌ Error: You provided {len(matches)} shots, but maximum allowed is {max_shots_allowed}. "
            f"Please reduce the number of stitched shots or combine them into fewer segments."
        )

    # Calculate total duration and collect clips
    clips = []
    total_duration = 0

    fps = getattr(config, "VIDEO_FPS", 24) or 24
    for i, (start_time, end_time) in enumerate(matches, 1):
        try:
            start_sec = hhmmss_to_seconds(start_time, fps=fps)
            end_sec = hhmmss_to_seconds(end_time, fps=fps)
            duration = end_sec - start_sec

            if duration <= 0:
                return f"❌ Error: Shot {i} has invalid duration (start: {start_time}, end: {end_time}). End time must be greater than start time."

            clips.append({
                'start': start_time,
                'end': end_time,
                'start_sec': start_sec,
                'end_sec': end_sec,
                'duration': duration
            })
            total_duration += duration
        except Exception as e:
            return f"❌ Error parsing shot {i} time range ({start_time} to {end_time}): {str(e)}"

    # Validate continuity for multi-shot stitching
    if len(clips) > 1:
        max_gap = getattr(config, 'MAX_STITCH_GAP_SEC', 2.0)
        for i in range(len(clips) - 1):
            gap = clips[i+1]['start_sec'] - clips[i]['end_sec']
            if gap < 0:
                return f"❌ Error: Overlapping shots. Shot {i+1} ends at {clips[i]['end']}, but shot {i+2} starts at {clips[i+1]['start']}"
            if gap > max_gap:
                return (
                    f"❌ Error: Time gap ({gap:.2f}s) between shot {i+1} and {i+2} exceeds maximum ({max_gap}s).\n"
                    f"Stitched shots must maintain visual continuity. Please select closer shots or use a single continuous clip."
                )

    # Check if total duration matches target length (allow tolerance)
    duration_diff = total_duration - target_length_sec

    # Prepare duration summary
    if len(clips) == 1:
        duration_line = f"shot: {clips[0]['start']} to {clips[0]['end']} ({clips[0]['duration']:.2f}s)"
    else:
        duration_line = f"{len(clips)} stitched shots (total {total_duration:.2f}s):\n"
        for i, clip in enumerate(clips, 1):
            duration_line += f"  Shot {i}: {clip['start']} to {clip['end']} ({clip['duration']:.2f}s)\n"

    # Check for very short clips
    min_acceptable = getattr(config, 'MIN_ACCEPTABLE_SHOT_DURATION', 2.0)
    short_clips = [c for c in clips if c['duration'] < min_acceptable]
    short_warning = ""
    if short_clips:
        short_warning = f"\n⚠️ Warning: {len(short_clips)} shot(s) shorter than {min_acceptable}s - consider using longer clips if possible."

    # Allow flexible tolerance
    tolerance = getattr(config, 'ALLOW_DURATION_TOLERANCE', 1.0)
    if abs(duration_diff) > tolerance:
        if duration_diff > 0:
            action = "shorten"
            suggestion = f"Try trimming {duration_diff:.2f}s from the end."
        else:
            action = "extend"
            suggestion = f"Try adding {abs(duration_diff):.2f}s more footage."

        return (
            f"❌ Error: Duration mismatch! Your total duration is {total_duration:.2f}s but target is {target_length_sec:.2f}s.\n"
            f"Current selection:\n{duration_line}"
            f"Difference: {abs(duration_diff):.2f}s ({action} needed)\n"
            f"Suggestion: {suggestion}{short_warning}\n"
            f"⚠️ Please adjust your shot selection before calling finish."
        )

    # If duration exceeds target by small amount, provide trimming suggestion
    tolerance = getattr(config, 'ALLOW_DURATION_TOLERANCE', 1.0)
    if 0 < duration_diff <= tolerance:
        new_end_sec = clips[-1]['end_sec'] - duration_diff
        new_end = seconds_to_hhmmss(new_end_sec)
        return (
            f"✅ OK: Shot validation passed (will auto-trim {duration_diff:.2f}s from last clip).\n"
            f"Current selection:\n{duration_line}"
            f"Target duration: {target_length_sec:.2f}s\n"
            f"Auto-adjusted end time: {new_end}\n"
            f"You can proceed with finish.{short_warning}"
        )

    # Validation passed
    status_msg = "✅ OK: Shot validation passed.\n"
    if len(clips) > 1:
        status_msg += f"✓ {len(clips)} shots stitched successfully with proper continuity\n"

    return (
        f"{status_msg}"
        f"Current selection:\n{duration_line}"
        f"Target duration: {target_length_sec:.2f}s\n"
        f"Duration match: ✓{short_warning}\n"
        f"You can proceed with finish."
    )



class ReviewerAgent:
    """
    ReviewerAgent reviews shot selections generated by DVDCoreAgent.
    The Core should pass review before calling finish.
    """

    def __init__(self, frame_folder_path=None, video_path=None):
        """
        Initialize ReviewerAgent.

        Args:
            frame_folder_path: Path to extracted video frames for review.
            video_path: Path to the video file.
        """
        self.frame_folder_path = frame_folder_path
        self.video_path = video_path

    def cleanup(self):
        _clear_thread_video_reader()
        gc.collect()

    def _compute_frame_indices(self, start_sec: float, end_sec: float, fps: float, max_frames: Optional[int] = None) -> list:
        """Compute frame indices for a time range using native fps with a hard cap."""
        if fps <= 0:
            return []
        start_f = max(0, int(start_sec * fps))
        end_f = max(0, int(end_sec * fps))
        if end_f < start_f:
            return []
        indices = list(range(start_f, end_f + 1))
        if max_frames and len(indices) > max_frames:
            import math
            stride = max(1, math.ceil(len(indices) / max_frames))
            indices = indices[::stride]
            if indices and indices[-1] != end_f:
                indices.append(end_f)
            if len(indices) > max_frames:
                indices = indices[:max_frames - 1] + [end_f]
        return indices

    def _call_video_analysis_model(self, messages: list) -> Optional[str]:
        """Call VIDEO_ANALYSIS_MODEL using the same settings as fine_grained_shot_trimming."""
        tries = 3
        while tries > 0:
            tries -= 1
            try:
                kwargs = dict(
                    model=config.VIDEO_ANALYSIS_MODEL,
                    messages=messages,
                    max_tokens=getattr(config, "VIDEO_ANALYSIS_MODEL_MAX_TOKEN", 2048),
                    temperature=0.0,
                )
                if config.VIDEO_ANALYSIS_ENDPOINT:
                    kwargs["api_base"] = config.VIDEO_ANALYSIS_ENDPOINT
                if config.VIDEO_ANALYSIS_API_KEY:
                    kwargs["api_key"] = config.VIDEO_ANALYSIS_API_KEY
                raw = litellm.completion(**kwargs)
                return raw.choices[0].message.content
            except Exception as e:
                print(f"❌ [Reviewer] video analysis call failed: {e}")
                if tries == 0:
                    return None
        return None


    def check_face_quality_vlm(
        self,
        video_path: A[str, D("Path to the video file.")],
        time_range: A[str, D("The time range to check (e.g., '00:13:28 to 00:13:40').")],
        main_character_name: A[str, D("Name of the main character/protagonist to look for. Default: 'the main character'")] = "the main character",
        min_protagonist_ratio: A[float, D("Minimum required ratio of frames where protagonist is the main focus (0.0-1.0). Default: 0.7 (70%)")] = 0.7,
        min_box_size: A[int, D("Minimum bounding box size in pixels. Default: 50")] = 50,
        return_frame_data: A[bool, D("Whether to return frame-level protagonist data from the same VLM call.")] = False,
    ) -> str | tuple[str, list]:
        """
        Check face quality using VLM frame-by-frame detection (same logic as check_face_quality but using VLM).

        This function loops through frames in the time range, calls VLM to detect the protagonist
        in each frame and get bounding box coordinates, then calculates break_ratio based on detection results.

        Args:
            video_path: Path to the video file
            time_range: Time range in format "HH:MM:SS to HH:MM:SS" or "MM:SS to MM:SS"
            main_character_name: Name of the main character to detect (default: "the main character")
            min_protagonist_ratio: Minimum required ratio of non-break frames (default: 0.7 = 70%)
            min_box_size: Minimum bounding box size in pixels (default: 50)

        Returns:
            str | tuple[str, list]: Success message if protagonist ratio is acceptable, or error message with details.
            If return_frame_data is True, returns (message, frame_data).

        Example:
            >>> check_face_quality_vlm("/path/to/video.mp4", "00:10:00 to 00:10:10", "Bruce Wayne", min_protagonist_ratio=0.7)
        """
        # Parse time range
        match = re.search(r'([\d:.]+)\s+to\s+([\d:.]+)', time_range, re.IGNORECASE)
        if not match:
            return f"❌ Error: Could not parse time range '{time_range}'. Please use format 'HH:MM:SS to HH:MM:SS'."

        try:
            fps = getattr(config, "VIDEO_FPS", 24) or 24
            start_sec = hhmmss_to_seconds(match.group(1), fps=fps)
            end_sec = hhmmss_to_seconds(match.group(2), fps=fps)
            duration_sec = end_sec - start_sec

            if duration_sec <= 0:
                return f"❌ Error: Invalid time range. End time must be greater than start time."
        except Exception as e:
            return f"❌ Error parsing time range: {e}"

        if not os.path.exists(video_path):
            return f"❌ Error: Video file not found: {video_path}"

        print(f"🔍 [Reviewer: VLM] Analyzing {time_range} ({duration_sec:.2f}s)...")

        try:
            break_frames = 0
            total_sampled_frames = 0
            detection_details = []
            frame_results = []
            verbose_frame_log = bool(getattr(config, "VLM_FACE_LOG_EACH_FRAME", False))

            max_frames = int(getattr(config, "VLM_FACE_MAX_FRAMES", 24))
            vr = _get_thread_video_reader(video_path)
            if vr is None:
                return "❌ Error: Unable to initialize video reader."
            video_fps = float(vr.get_avg_fps())
            frame_indices = self._compute_frame_indices(start_sec, end_sec, video_fps, max_frames=max_frames)
            if not frame_indices:
                return f"❌ Error: No frames to process in the specified time range."
            # Clamp frame indices to valid range (fix: decord reader may have fewer frames)
            num_frames = len(vr)
            frame_indices = [idx for idx in frame_indices if idx < num_frames]
            if not frame_indices:
                return f"❌ Error: All computed frame indices exceed video length ({num_frames} frames)."
            if verbose_frame_log:
                print(f"🎞️  [Reviewer: VLM] {len(frame_indices)} in-bounds frames (video: {num_frames} total)")

            if verbose_frame_log:
                print(f"🎞️  [Reviewer: VLM] Decoding video; processing {len(frame_indices)} frames...")

            frames = vr.get_batch(frame_indices).asnumpy()
            frame_items = list(zip(frame_indices, frames))
            batch_size = int(getattr(config, "VLM_FACE_BATCH_SIZE", 8))
            batch_concurrency = int(getattr(config, "VLM_FACE_BATCH_CONCURRENCY", 16))

            batches = [
                frame_items[batch_start:batch_start + batch_size]
                for batch_start in range(0, len(frame_items), batch_size)
            ]
            batch_results_list = [None] * len(batches)

            with ThreadPoolExecutor(max_workers=min(batch_concurrency, len(batches) or 1)) as executor:
                future_to_idx = {}
                for idx, batch in enumerate(batches):
                    batch_indices = [frame_idx for frame_idx, _ in batch]
                    batch_frames = [frame for _, frame in batch]
                    future = executor.submit(
                        self._detect_protagonist_in_frames_vlm,
                        frame_arrays=batch_frames,
                        frame_indices=batch_indices,
                        main_character_name=main_character_name,
                        min_box_size=min_box_size,
                    )
                    future_to_idx[future] = idx

                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        batch_results_list[idx] = future.result()
                    except Exception:
                        batch_results_list[idx] = None

            for batch, batch_results in zip(batches, batch_results_list):
                if not batch_results or len(batch_results) != len(batch):
                    return "❌ Error: VLM batch detection failed or returned mismatched results."

                for (frame_idx, _), detection_result in zip(batch, batch_results):
                    total_sampled_frames += 1
                    is_break_frame = detection_result["is_break"]
                    reason = detection_result["reason"]
                    status = "❌" if is_break_frame else "✅"

                    time_at_frame = frame_idx / video_fps
                    if verbose_frame_log:
                        print(f"  Frame {frame_idx:6d} | Time: {time_at_frame:7.2f}s | {status:15s} | {reason}")

                    frame_results.append({
                        "frame_idx": frame_idx,
                        "time_sec": time_at_frame,
                        "protagonist_detected": not is_break_frame,
                        "bounding_box": detection_result.get("bounding_box"),
                        "confidence": detection_result.get("confidence", 0.0),
                        "reason": reason
                    })

                    if is_break_frame:
                        break_frames += 1
                        detection_details.append({
                            "frame": frame_idx,
                            "time": time_at_frame,
                            "reason": reason
                        })

            # Calculate break ratio
            if total_sampled_frames == 0:
                return f"❌ Error: No frames were successfully processed."

            break_ratio = break_frames / total_sampled_frames
            non_break_ratio = 1.0 - break_ratio

            # Prepare result message
            result_msg = f"\n[VLM Face Quality Check Results (Frame-by-Frame)]\n"
            result_msg += f"Time range: {time_range} ({duration_sec:.2f}s)\n"
            result_msg += f"Character: {main_character_name}\n"
            result_msg += f"Sampled frames: {total_sampled_frames}\n"
            result_msg += f"Break frames: {break_frames}/{total_sampled_frames}\n"
            result_msg += f"Protagonist ratio: {non_break_ratio * 100:.1f}%\n"
            result_msg += f"Required ratio: {min_protagonist_ratio * 100:.1f}%\n"

            # Check if ratio meets threshold
            if non_break_ratio < min_protagonist_ratio:
                result_msg += f"\n❌ FAILED: Protagonist ratio ({non_break_ratio * 100:.1f}%) is below minimum threshold ({min_protagonist_ratio * 100:.1f}%)\n"

                if detection_details:
                    result_msg += f"\nBreak frame examples (first 5):\n"
                    for detail in detection_details[:5]:
                        result_msg += f"  - Frame {detail['frame']} ({detail['time']:.2f}s): {detail['reason']}\n"

                result_msg += f"\n⚠️ This shot does not maintain sufficient focus on {main_character_name}. Please select a different shot."
                if return_frame_data:
                    return result_msg, frame_results
                return result_msg
            else:
                result_msg += f"\n✅ PASSED: Protagonist ratio ({non_break_ratio * 100:.1f}%) meets the minimum threshold.\n"
                result_msg += f"Shot maintains good focus on {main_character_name}. You can proceed with this shot."
                if return_frame_data:
                    return result_msg, frame_results
                return result_msg

        except Exception as e:
            import traceback
            traceback.print_exc()
            if return_frame_data:
                return f"❌ Error during VLM frame-by-frame detection: {str(e)}", []
            return f"❌ Error during VLM frame-by-frame detection: {str(e)}"
        finally:
            frames = None
            frame_items = None
            batches = None
            batch_results_list = None
            gc.collect()


    def _evaluate_protagonist_detection(self, detection: dict, min_box_size: int) -> dict:
        """Evaluate a raw VLM detection result and apply size/role rules."""
        protagonist_detected = detection.get("protagonist_detected", False)
        is_minor_character = detection.get("is_minor_character", False)
        bounding_box = detection.get("bounding_box", None)
        confidence = detection.get("confidence", 0.0)
        reason_text = detection.get("reason", "")

        if is_minor_character:
            return {
                "is_break": True,
                "reason": f"minor_character_detected ({reason_text})",
                "bounding_box": None,
                "confidence": confidence
            }

        if not protagonist_detected:
            return {
                "is_break": True,
                "reason": f"no_protagonist ({reason_text})",
                "bounding_box": None,
                "confidence": confidence
            }

        if bounding_box is None:
            return {
                "is_break": True,
                "reason": "no_bounding_box",
                "bounding_box": None,
                "confidence": confidence
            }

        box_width = bounding_box.get("width", 0)
        box_height = bounding_box.get("height", 0)
        box_size = min(box_width, box_height)
        relaxed_min_size = max(30, min_box_size // 2)

        if box_size < relaxed_min_size:
            return {
                "is_break": True,
                "reason": f"protagonist_too_small ({box_size}px < {relaxed_min_size}px)",
                "bounding_box": bounding_box,
                "confidence": confidence
            }

        return {
            "is_break": False,
            "reason": f"protagonist_ok (size={box_size}px, conf={confidence:.2f})",
            "bounding_box": bounding_box,
            "confidence": confidence
        }


    def _detect_protagonist_in_frames_vlm(
        self,
        frame_arrays: List[np.ndarray],
        frame_indices: List[int],
        main_character_name: str,
        min_box_size: int
    ) -> Optional[List[dict]]:
        """Call VLM once for multiple frames and return per-frame results in order."""
        if not frame_arrays or not frame_indices or len(frame_arrays) != len(frame_indices):
            return None

        prompt = VLM_PROTAGONIST_DETECTION_PROMPT.format(
            main_character_name=main_character_name,
            frame_count=len(frame_indices),
            frame_indices=frame_indices,
        )

        user_content = None
        messages = None
        content_str = None
        content = None
        detections = None
        try:
            user_content = [{"type": "text", "text": prompt}]
            for frame in frame_arrays:
                b64 = array_to_base64(frame)
                if b64:
                    user_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})

            messages = [
                {"role": "system", "content": "You are an expert at character detection and localization in video frames."},
                {"role": "user", "content": user_content}
            ]

            content_str = self._call_video_analysis_model(messages)
            if not content_str:
                return None

            content = content_str.strip()
            json_block_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', content, re.DOTALL)
            if json_block_match:
                content = json_block_match.group(1).strip()

            detections = json.loads(content)
            if not isinstance(detections, list):
                return None
            if len(detections) != len(frame_indices):
                return None

            results = []
            for detection in detections:
                results.append(self._evaluate_protagonist_detection(detection, min_box_size))
            return results
        except Exception:
            return None
        finally:
            user_content = None
            messages = None
            content_str = None
            content = None
            detections = None
            gc.collect()


    def get_protagonist_frame_data(
        self,
        video_path: str,
        time_range: str,
        main_character_name: str = "the main character",
        min_box_size: int = 50,
    ) -> list:
        """
        Get frame-level protagonist detection data for a time range.
        Returns structured data instead of a summary string.

        Args:
            video_path: Path to the video file
            time_range: Time range in format "HH:MM:SS to HH:MM:SS" or "MM:SS to MM:SS"
            main_character_name: Name of the main character to detect
            min_box_size: Minimum bounding box size in pixels (default: 50)

        Returns:
            list: List of frame detection results, each containing:
                {
                    "frame_idx": int,
                    "time_sec": float,
                    "protagonist_detected": bool,
                    "bounding_box": dict or None,  # {"x": int, "y": int, "width": int, "height": int}
                    "confidence": float,
                    "reason": str
                }
        """
        # Parse time range
        match = re.search(r'([\d:.]+)\s+to\s+([\d:.]+)', time_range, re.IGNORECASE)
        if not match:
            print(f"❌ [Reviewer] Error: Could not parse time range \'{time_range}\'")
            return []

        try:
            fps = getattr(config, "VIDEO_FPS", 24) or 24
            start_sec = hhmmss_to_seconds(match.group(1), fps=fps)
            end_sec = hhmmss_to_seconds(match.group(2), fps=fps)
            duration_sec = end_sec - start_sec

            if duration_sec <= 0:
                print(f"❌ [Reviewer] Error: Invalid time range")
                return []
        except Exception as e:
            print(f"❌ [Reviewer] Error parsing time range: {e}")
            return []

        frame_results = []

        vr = None
        frames = None
        frame_items = None
        batches = None
        batch_results_list = None
        try:
            if not os.path.exists(video_path):
                print(f"❌ [Reviewer] Error: Video file not found: {video_path}")
                return []

            vr = _get_thread_video_reader(video_path)
            if vr is None:
                print("❌ [Reviewer] Error: Unable to initialize video reader.")
                return []
            video_fps = float(vr.get_avg_fps())
            max_frames = int(getattr(config, "CORE_MAX_FRAMES", getattr(config, "TRIM_SHOT_MAX_FRAMES", 240)))
            frame_indices = self._compute_frame_indices(start_sec, end_sec, video_fps, max_frames=max_frames)
            if not frame_indices:
                return []

            frames = vr.get_batch(frame_indices).asnumpy()
            frame_items = list(zip(frame_indices, frames))
            batch_size = int(getattr(config, "VLM_FACE_BATCH_SIZE", 8))
            batch_concurrency = int(getattr(config, "VLM_FACE_BATCH_CONCURRENCY", 16))

            batches = [
                frame_items[batch_start:batch_start + batch_size]
                for batch_start in range(0, len(frame_items), batch_size)
            ]
            batch_results_list = [None] * len(batches)

            with ThreadPoolExecutor(max_workers=min(batch_concurrency, len(batches) or 1)) as executor:
                future_to_idx = {}
                for idx, batch in enumerate(batches):
                    batch_indices = [frame_idx for frame_idx, _ in batch]
                    batch_frames = [frame for _, frame in batch]
                    future = executor.submit(
                        self._detect_protagonist_in_frames_vlm,
                        frame_arrays=batch_frames,
                        frame_indices=batch_indices,
                        main_character_name=main_character_name,
                        min_box_size=min_box_size,
                    )
                    future_to_idx[future] = idx

                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        batch_results_list[idx] = future.result()
                    except Exception:
                        batch_results_list[idx] = None

            for batch, batch_results in zip(batches, batch_results_list):
                if not batch_results or len(batch_results) != len(batch):
                    print("❌ [Reviewer] Error: VLM batch detection failed or returned mismatched results.")
                    return []

                for (frame_idx, _), detection_result in zip(batch, batch_results):
                    time_at_frame = frame_idx / video_fps
                    frame_data = {
                        "frame_idx": frame_idx,
                        "time_sec": time_at_frame,
                        "protagonist_detected": not detection_result["is_break"],
                        "bounding_box": detection_result.get("bounding_box"),
                        "confidence": detection_result.get("confidence", 0.0),
                        "reason": detection_result["reason"]
                    }
                    frame_results.append(frame_data)

        except Exception as e:
            print(f"❌ [Reviewer] Error during protagonist detection: {str(e)}")
            import traceback
            traceback.print_exc()
            return []
        finally:
            frames = None
            frame_items = None
            batches = None
            batch_results_list = None
            gc.collect()

        return frame_results


    def check_aesthetic_quality(
        self,
        video_path: A[str, D("Path to the video file.")],
        time_range: A[str, D("The time range to check (e.g., '00:13:28 to 00:13:40').")],
        min_aesthetic_score: A[float, D("Minimum required aesthetic score (1-5 scale). Default: 3.0")] = 3.0,
        sample_fps: A[float, D("Sampling frame rate for analysis. Default: 2.0")] = 2.0,
    ) -> str:
        """
        Check aesthetic quality of a video clip using VLM analysis.
        Analyzes visual appeal, lighting, composition, colors, and cinematography.

        Args:
            video_path: Path to the video file
            time_range: Time range in format "HH:MM:SS to HH:MM:SS" or "MM:SS to MM:SS"
            min_aesthetic_score: Minimum required aesthetic score (default: 3.0 on 1-5 scale)
            sample_fps: Frame sampling rate for analysis (default: 2.0 fps)

        Returns:
            str: Success message if aesthetic quality meets requirements, or error message with details.

        Example:
            >>> check_aesthetic_quality("/path/to/video.mp4", "00:10:00 to 00:10:10", min_aesthetic_score=3.5)
        """
        match = re.search(r'([\d:.]+)\s+to\s+([\d:.]+)', time_range, re.IGNORECASE)
        if not match:
            return f"❌ Error: Could not parse time range '{time_range}'. Please use format 'HH:MM:SS to HH:MM:SS'."

        try:
            fps = getattr(config, "VIDEO_FPS", 24) or 24
            start_sec = hhmmss_to_seconds(match.group(1), fps=fps)
            end_sec = hhmmss_to_seconds(match.group(2), fps=fps)
            duration_sec = end_sec - start_sec

            if duration_sec <= 0:
                return "❌ Error: Invalid time range. End time must be greater than start time."
        except Exception as e:
            return f"❌ Error parsing time range: {e}"

        if not os.path.exists(video_path):
            return f"❌ Error: Video file not found: {video_path}"

        print(f"✨ [Reviewer: Aesthetics] Analyzing {time_range} ({duration_sec:.2f}s)...")

        vr = None
        b64_frames = None
        user_content = None
        litellm_messages = None
        content_str = None
        response = None
        try:
            vr = _get_thread_video_reader(video_path)
            if vr is None:
                return "❌ Error: Unable to initialize video reader."
            video_native_fps = vr.get_avg_fps()
            max_frames = int(getattr(config, "CORE_MAX_FRAMES", getattr(config, "TRIM_SHOT_MAX_FRAMES", 240)))
            indices = self._compute_frame_indices(start_sec, end_sec, video_native_fps, max_frames=max_frames)
            if indices:
                indices = [i for i in indices if i < len(vr)]
            b64_frames = [array_to_base64(vr.get_batch([i]).asnumpy()[0]) for i in indices] if indices else []
            user_content = [{"type": "text", "text": VLM_AESTHETIC_ANALYSIS_PROMPT}]
            for b64 in b64_frames:
                user_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
            litellm_messages = [
                {"role": "system", "content": "You are an expert cinematographer and visual aesthetics analyst."},
                {"role": "user", "content": user_content},
            ]
            content_str = self._call_video_analysis_model(litellm_messages)
            response = {"content": content_str} if content_str else None

            if response is None or response.get("content") is None:
                return "⚠️ WARNING: VLM returned no response for aesthetic analysis. Proceeding without validation."

            content = response["content"].strip()
            json_block_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', content, re.DOTALL)
            if json_block_match:
                content = json_block_match.group(1).strip()

            analysis = json.loads(content)

            overall_score = analysis.get("overall_aesthetic_score", 0.0)
            lighting_score = analysis.get("lighting_score", 0.0)
            color_score = analysis.get("color_score", 0.0)
            composition_score = analysis.get("composition_score", 0.0)
            camera_work_score = analysis.get("camera_work_score", 0.0)
            visual_interest_score = analysis.get("visual_interest_score", 0.0)
            strengths = analysis.get("strengths", [])
            weaknesses = analysis.get("weaknesses", [])
            recommendation = analysis.get("recommendation", "UNKNOWN")
            detailed_analysis = analysis.get("detailed_analysis", "")

            result_msg = "\n[Aesthetic Quality Check Results]\n"
            result_msg += f"Time range: {time_range} ({duration_sec:.2f}s)\n"
            result_msg += f"Overall Aesthetic Score: {overall_score:.2f}/5.0\n"
            result_msg += f"  • Lighting: {lighting_score:.2f}/5.0\n"
            result_msg += f"  • Color: {color_score:.2f}/5.0\n"
            result_msg += f"  • Composition: {composition_score:.2f}/5.0\n"
            result_msg += f"  • Camera Work: {camera_work_score:.2f}/5.0\n"
            result_msg += f"  • Visual Interest: {visual_interest_score:.2f}/5.0\n"
            result_msg += f"Recommendation: {recommendation}\n"
            result_msg += f"Minimum Required: {min_aesthetic_score:.2f}/5.0\n"

            if strengths:
                result_msg += "\nStrengths:\n"
                for strength in strengths[:3]:
                    result_msg += f"  ✓ {strength}\n"

            if weaknesses:
                result_msg += "\nWeaknesses:\n"
                for weakness in weaknesses[:3]:
                    result_msg += f"  ✗ {weakness}\n"

            if detailed_analysis:
                result_msg += f"\nAnalysis: {detailed_analysis}\n"

            if overall_score < min_aesthetic_score:
                result_msg += (
                    f"\n❌ FAILED: Aesthetic score ({overall_score:.2f}/5.0) is below minimum "
                    f"threshold ({min_aesthetic_score:.2f}/5.0)\n"
                )
                result_msg += "\n⚠️ This shot does not meet the aesthetic quality requirements. Please select a shot with:\n"
                result_msg += "  • Better lighting (natural light preferred)\n"
                result_msg += "  • Improved composition (well-framed, balanced)\n"
                result_msg += "  • More vibrant colors and good contrast\n"
                result_msg += "  • Stable camera work\n"
                result_msg += "  • More visually interesting content\n"
                return result_msg

            result_msg += (
                f"\n✅ PASSED: Aesthetic score ({overall_score:.2f}/5.0) meets the minimum threshold.\n"
            )
            if overall_score >= 4.0:
                result_msg += "⭐ Excellent visual quality! This shot is highly recommended for the final edit.\n"
            result_msg += "You can proceed with this shot."
            return result_msg

        except json.JSONDecodeError as e:
            return f"⚠️ WARNING: Could not parse VLM response for aesthetic analysis: {e}\n\nProceeding without validation."
        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"⚠️ WARNING: Error during aesthetic analysis: {str(e)}\n\nProceeding without validation."
        finally:
            b64_frames = None
            user_content = None
            litellm_messages = None
            content_str = None
            response = None
            gc.collect()


    def review(self, shot_proposal: dict, context: dict, used_time_ranges: list = None) -> dict:
        """
        Review whether the shot selection meets requirements.

        Args:
            shot_proposal: Shot selection info
                - answer: str, selected time range (e.g., "[shot: 00:10:00 to 00:10:07]")
                - target_length_sec: float, target duration
            context: Current shot context
                - content: str, target content description
                - emotion: str, target emotion
                - section_idx: int, current section index
                - shot_idx: int, current shot index
            used_time_ranges: List of used time ranges [(start_sec, end_sec), ...]

        Returns:
            dict: {
                "approved": bool,  # Whether the review passed
                "feedback": str,   # Feedback message
                "issues": list,    # Issues found
                "suggestions": list  # Improvement suggestions
            }
        """
        if used_time_ranges is None:
            used_time_ranges = []

        answer = shot_proposal.get("answer", "")
        target_length_sec = shot_proposal.get("target_length_sec", 0.0)

        issues = []
        suggestions = []

        # 1. Validate time range format and duration
        finish_review = review_finish(answer, target_length_sec)
        if "❌" in finish_review:
            issues.append(finish_review)

        # 2. Check for overlap with already used clips
        # Extract time range from answer
        match = re.search(r'\[?shot[\s_]*\d*:\s*([0-9:.]+)\s+to\s+([0-9:.]+)\]?', answer, re.IGNORECASE)
        if match:
            time_range = f"{match.group(1)} to {match.group(2)}"
            overlap_review = review_clip(time_range, used_time_ranges)
            if "❌" in overlap_review:
                issues.append(overlap_review)

        # 3. Content match checks (LLM can be used for deeper review)
        # TODO: Add more checks here, e.g.:
        # - Whether the selected clip matches target content/emotion
        # - Visual quality checks
        # - Narrative coherence

        # Build feedback
        if issues:
            feedback = "❌ Review failed with the following issues:\n"
            for i, issue in enumerate(issues, 1):
                feedback += f"\nIssue {i}:\n{issue}\n"

            suggestions.append("Adjust your shot selection based on the issues above.")
            if "Duration mismatch" in str(issues) or "duration" in str(issues).lower():
                suggestions.append("Adjust start/end times to match the target duration.")
            if "OVERLAP" in str(issues) or "overlap" in str(issues).lower():
                suggestions.append("Choose a time range that does not overlap with previously used clips.")

            feedback += "\nSuggestions:\n" + "\n".join(f"- {s}" for s in suggestions)

            return {
                "approved": False,
                "feedback": feedback,
                "issues": issues,
                "suggestions": suggestions
            }
        else:
            return {
                "approved": True,
                "feedback": f"✅ Review passed! Shot selection meets requirements.\n{finish_review}",
                "issues": [],
                "suggestions": []
            }
