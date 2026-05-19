"""
ASR (Automatic Speech Recognition) module supporting multiple backends:
  - whisper: Local Whisper model with pyannote speaker diarization
  - whisper_cpp: whisper.cpp via pywhispercpp (fast, quantized GGML models)
  - litellm: LiteLLM API (e.g., Gemini, GPT-4o) for cloud-based transcription
"""

import os
import base64
import subprocess
from typing import Optional, List, Dict, Any
from src.utils.time_format_convert import format_srt_timestamp

from src import config

# 异步音频 API 导入
from src.audio.mimo_audio_api import MimoAudioAPI


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _parse_srt_content(srt_text: str) -> List[Dict[str, Any]]:
    """
    Parse SRT format text into list of segments.
    Returns: [{"text": str, "start_s": float, "end_s": float}, ...]
    """
    import re

    segments = []
    # Split by double newline to get individual subtitle blocks
    blocks = re.split(r'\n\s*\n', srt_text.strip())

    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) < 2:
            continue

        # Line 0: sequence number (skip)
        # Line 1: timestamp line (HH:MM:SS,mmm --> HH:MM:SS,mmm)
        # Line 2+: text content

        timestamp_line = lines[1] if len(lines) > 1 else ""
        text_lines = lines[2:] if len(lines) > 2 else []

        # Parse timestamp: "00:08:02,400 --> 00:08:06,100"
        match = re.match(r'(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2}),(\d{3})', timestamp_line)
        if not match:
            continue

        h1, m1, s1, ms1, h2, m2, s2, ms2 = match.groups()
        start_s = int(h1) * 3600 + int(m1) * 60 + int(s1) + int(ms1) / 1000.0
        end_s = int(h2) * 3600 + int(m2) * 60 + int(s2) + int(ms2) / 1000.0

        text = ' '.join(text_lines).strip()
        if text:
            segments.append({
                "text": text,
                "start_s": start_s,
                "end_s": end_s,
            })

    return segments




def write_srt_from_sentence_info(
    sentence_info: List[Dict[str, Any]],
    srt_path: str,
    include_speaker: bool = True
) -> None:
    """
    Write SRT file from sentence_info list.
    Each item: {'text': str, 'timestamp': [[word, start_ms, end_ms], ...], 'speaker': str|None}
    """
    _ensure_dir(os.path.dirname(srt_path) or ".")
    with open(srt_path, "w", encoding="utf-8") as f:
        for idx, sent in enumerate(sentence_info):
            text = sent.get('text', '')
            timestamp = sent.get('timestamp', [])
            speaker = sent.get('speaker', None)

            if not timestamp:
                continue

            start_ms = int(timestamp[0][1]) if len(timestamp[0]) >= 2 else 0
            end_ms = int(timestamp[-1][2]) if len(timestamp[-1]) >= 3 else start_ms

            f.write(f"{idx + 1}\n")
            f.write(f"{format_srt_timestamp(start_ms)} --> {format_srt_timestamp(end_ms)}\n")
            if include_speaker and speaker:
                f.write(f"[{speaker}] {text}\n\n")
            else:
                f.write(f"{text}\n\n")


def extract_audio_mp3_16k(
    video_path: str,
    audio_path: str,
    start_sec: Optional[float] = None,
    end_sec: Optional[float] = None,
    bitrate: str = "32k",
) -> None:
    """Extract mono 16k MP3 from video using ffmpeg."""
    cmd: List[str] = ["ffmpeg", "-y"]
    if start_sec is not None:
        cmd += ["-ss", str(float(start_sec))]
    cmd += ["-i", video_path]
    if end_sec is not None:
        cmd += ["-to", str(float(end_sec))]
    cmd += ["-vn", "-ac", "1", "-ar", "16000", "-ab", bitrate, audio_path]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# ─────────────────────────────────────────────────────────────────────────────
# Backend: whisper.cpp via pywhispercpp
# ─────────────────────────────────────────────────────────────────────────────

def _transcribe_whisper_cpp(
    audio_path: str,
    model_name: str,
    device: str = "cuda:0",
    language: Optional[str] = None,
    n_threads: int = 4,
    enable_diarization: Optional[bool] = None,
    diarization_model_path: Optional[str] = None,
    merge_same_speaker: Optional[bool] = None,
    merge_gap: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Run whisper.cpp ASR via pywhispercpp with optional pyannote speaker diarization.

    Returns a dict with keys:
      - text: full transcription
      - sentence_info: list of sentence dicts with 'text', 'timestamp', and 'speaker' fields
      - segments: raw segments with speaker information

    Args:
        model_name: Model name (e.g. "base.en", "large-v3") or path to local ggml model file.
        n_threads: Number of CPU threads for whisper.cpp inference.
    """
    from pywhispercpp.model import Model

    # Use config defaults if not specified
    if enable_diarization is None:
        enable_diarization = getattr(config, 'ASR_ENABLE_DIARIZATION', False)
    if merge_same_speaker is None:
        merge_same_speaker = getattr(config, 'ASR_MERGE_SAME_SPEAKER', True)
    if merge_gap is None:
        merge_gap = getattr(config, 'ASR_MERGE_GAP', 1.0)
    if diarization_model_path is None:
        diarization_model_path = getattr(config, 'ASR_DIARIZATION_MODEL_PATH',
            "../HF/hub/models--pyannote--speaker-diarization-community-1/snapshots/3533c8cf8e369892e6b79ff1bf80f7b0286a54ee/")

    model_kwargs = {"n_threads": n_threads}
    if language:
        model_kwargs["language"] = language

    print(f"[ASR/WhisperCPP] Loading model: {model_name}")
    model = Model(model_name, **model_kwargs)

    print(f"[ASR/WhisperCPP] Transcribing: {audio_path}")
    raw_segments = model.transcribe(audio_path)

    # Optional: Perform speaker diarization
    diarization_tracks = None
    if enable_diarization:
        print("[ASR/WhisperCPP] Performing speaker diarization with pyannote...")
        diarization_tracks = _run_pyannote_diarization(audio_path, diarization_model_path, device)
        print(f"[ASR/WhisperCPP] Diarization complete: {len(diarization_tracks)} speaker turns detected")

    # Convert segments: t0/t1 are in 10ms units → seconds
    segments_with_speakers = []
    for seg in raw_segments:
        start_s = seg.t0 * 0.01
        end_s = seg.t1 * 0.01
        text = seg.text.strip()
        if not text:
            continue
        speaker = None
        if diarization_tracks:
            speaker = _get_speaker_at_time(diarization_tracks, start_s, end_s)
        segments_with_speakers.append({
            "start": start_s,
            "end": end_s,
            "speaker": speaker,
            "text": text,
        })

    if enable_diarization and merge_same_speaker:
        segments_with_speakers = _merge_same_speaker_segments(segments_with_speakers, max_gap=merge_gap)

    full_text = " ".join(s["text"] for s in segments_with_speakers)

    sentence_info = []
    for seg in segments_with_speakers:
        start_ms = int(seg["start"] * 1000)
        end_ms = int(seg["end"] * 1000)
        sentence_info.append({
            "text": seg["text"],
            "speaker": seg.get("speaker"),
            "timestamp": [[seg["text"], start_ms, end_ms]],
        })

    return {"text": full_text, "sentence_info": sentence_info, "segments": segments_with_speakers}

def _load_audio_for_pyannote(audio_path: str) -> dict:
    """
    Pre-load audio as waveform dict to bypass torchcodec/AudioDecoder incompatibility in pyannote 4.x.
    Returns {'waveform': (1, time) torch.Tensor, 'sample_rate': int}.
    Uses soundfile to avoid torchaudio's torchcodec dependency.
    """
    import torch
    import soundfile as sf
    import numpy as np
    data, sample_rate = sf.read(audio_path, dtype='float32')
    waveform = torch.from_numpy(data)
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)  # (time,) -> (1, time)
    elif waveform.dim() == 2:
        waveform = waveform.T  # (time, channels) -> (channels, time)
    return {"waveform": waveform, "sample_rate": sample_rate}


def _run_pyannote_diarization(audio_path: str, model_path: str, device: str) -> list:
    """
    Run pyannote speaker diarization on audio file.
    Returns list of (segment, track, speaker) tuples.
    """
    import torch
    from pyannote.audio import Pipeline

    diarization_pipeline = Pipeline.from_pretrained(model_path)
    diarization_pipeline = diarization_pipeline.to(torch.device(device))

    # Pre-load audio to bypass torchcodec/AudioDecoder incompatibility in pyannote 4.x
    audio_data = _load_audio_for_pyannote(audio_path)
    diarization = diarization_pipeline(audio_data)

    # Convert diarization output to a list of tracks
    if hasattr(diarization, 'itertracks'):
        diarization_tracks = list(diarization.itertracks(yield_label=True))
    elif hasattr(diarization, 'speaker_diarization'):
        diarization_tracks = list(diarization.speaker_diarization.itertracks(yield_label=True))
    else:
        raise TypeError(f"Unknown diarization output type: {type(diarization)}")

    return diarization_tracks


# ─────────────────────────────────────────────────────────────────────────────
# Backend: LiteLLM API (Gemini or other multimodal models)
# ─────────────────────────────────────────────────────────────────────────────


def _get_speaker_at_time(diarization_tracks: list, start: float, end: float) -> str:
    """
    Get the dominant speaker in a given time segment.

    Args:
        diarization_tracks: List of (segment, track, speaker) tuples
        start: Start time in seconds
        end: End time in seconds
    """
    from pyannote.core import Segment

    segment = Segment(start, end)
    speakers = {}

    for turn, _, speaker in diarization_tracks:
        overlap = segment & turn
        if overlap:
            duration = overlap.duration
            speakers[speaker] = speakers.get(speaker, 0) + duration

    if not speakers:
        return "UNKNOWN"

    return max(speakers, key=speakers.get)


def _merge_same_speaker_segments(
    segments: List[Dict[str, Any]],
    max_gap: float = 1.0
) -> List[Dict[str, Any]]:
    """
    Merge consecutive segments from the same speaker.

    Args:
        segments: Original segment list
        max_gap: Maximum time gap allowed for merging (seconds)
    """
    if not segments:
        return []

    merged = [segments[0].copy()]

    for seg in segments[1:]:
        last = merged[-1]
        if (seg.get("speaker") == last.get("speaker") and
            seg["start"] - last["end"] <= max_gap):
            last["end"] = seg["end"]
            last["text"] = last["text"] + " " + seg["text"]
        else:
            merged.append(seg.copy())

    return merged


# ─────────────────────────────────────────────────────────────────────────────
# Backend: LiteLLM API (Gemini or other multimodal models)
# ─────────────────────────────────────────────────────────────────────────────

def _transcribe_litellm(
    audio_path: str,
    model: str,
    api_key: Optional[str],
    language: Optional[str],
    api_base: Optional[str] = None,
    max_segment_size_mb: float = 30.0,
    batch_size: int = 8,
    debug_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Transcribe audio via LiteLLM using a multimodal model (e.g. Gemini).

    Audio is automatically split into chunks no larger than max_segment_size_mb.
    Each chunk is base64-encoded and sent via litellm.batch_completion in batches.
    The model is prompted to return SRT-format transcription with speaker labels.
    Results are merged with correct absolute timestamps.

    Args:
        audio_path: Path to the audio file.
        model: LiteLLM model ID (e.g., "gemini/gemini-2.0-flash").
        api_key: API key for LiteLLM provider.
        language: Language code ("en", "zh", ...). None = auto-detect.
        api_base: Optional API base URL.
        max_segment_size_mb: Maximum size per segment in MB (default: 30MB).
        batch_size: Number of segments to send per batch_completion call (default: 8).
        debug_dir: If set, save each segment's raw SRT (with absolute timestamps) to this directory.

    Returns:
        dict with keys: text, sentence_info, segments
    """
    import json
    import re
    import tempfile
    import litellm
    import soundfile as sf

    print(f"[ASR/LiteLLM] Using model: {model}")

    # Get audio info
    audio_array, audio_sr = sf.read(audio_path, dtype="float32", always_2d=False)
    total_duration = len(audio_array) / audio_sr

    # Calculate number of segments based on estimated output MP3 size (32k mono)
    # Use the actual output bitrate (32kbps) rather than the input file size,
    # since the input may be a high-bitrate or uncompressed file.
    estimated_mp3_mb = (32000 * total_duration) / (8 * 1024 * 1024)
    num_segments = max(1, int(estimated_mp3_mb // max_segment_size_mb) + (1 if estimated_mp3_mb % max_segment_size_mb > 0 else 0))
    segment_duration = total_duration / num_segments

    print(f"[ASR/LiteLLM] Audio: {total_duration:.1f}s, estimated output MP3: {estimated_mp3_mb:.1f}MB")
    print(f"[ASR/LiteLLM] Splitting into {num_segments} segments (~{segment_duration:.1f}s each) for ≤{max_segment_size_mb}MB per chunk")

    prompt = (
        "Transcribe the speech in SRT format. Each subtitle block must follow this exact format:\n\n"
        "1\n"
        "00:00:01,000 --> 00:00:03,500\n"
        "[Speaker Name] Dialogue text here\n\n"
        "Rules:\n"
        "- Timestamp format: HH:MM:SS,mmm --> HH:MM:SS,mmm\n"
        "- Always prefix each line with [Speaker Name] using the actual speaker's name or role\n"
        "- Output only the SRT content, no extra commentary"
    )

    if api_key:
        litellm.api_key = api_key

    def _build_segment_message(i: int):
        """Encode one audio segment to base64 and return (message_list, segment_start_time, tmp_path)."""
        start_sample = int(i * segment_duration * audio_sr)
        end_sample = int((i + 1) * segment_duration * audio_sr) if i < num_segments - 1 else len(audio_array)
        segment_audio = audio_array[start_sample:end_sample]
        segment_start_time = i * segment_duration

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav_f:
            tmp_wav_path = tmp_wav_f.name
            sf.write(tmp_wav_path, segment_audio, audio_sr)
        tmp_path = tmp_wav_path.replace(".wav", ".mp3")
        subprocess.run(
            ["ffmpeg", "-y", "-i", tmp_wav_path, "-ac", "1", "-ar", "16000", "-ab", "32k", tmp_path],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        os.unlink(tmp_wav_path)

        with open(tmp_path, "rb") as f:
            audio_b64 = base64.b64encode(f.read()).decode("utf-8")

        # 使用正确的音频输入格式：Qwen3-Omni-Captioner 用 input_audio，旧模型用 image_url
        is_qwen = "qwen3-omni" in model.lower() or "qwen3_omni" in model.lower()
        if is_qwen and api_base and "dashscope" in api_base:
            msg = [{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "input_audio", "input_audio": {"data": f"data:audio/mp3;base64,{audio_b64}"}},
            ]}]
        else:
            msg = [{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:audio/mp3;base64,{audio_b64}"}},
            ]}]
        return msg, segment_start_time, tmp_path

    def _parse_response(content, segment_idx, segment_start_time):
        """Parse SRT response and adjust timestamps."""
        if not content:
            print(f"[ASR/LiteLLM] Warning: empty response for segment {segment_idx + 1}")
            return []
        content_clean = re.sub(r"```(?:srt)?\s*", "", content)
        content_clean = re.sub(r"```", "", content_clean).strip()
        try:
            chunks = _parse_srt_content(content_clean)
            if not chunks:
                print(f"[ASR/LiteLLM] Warning: no valid SRT entries parsed for segment {segment_idx + 1}")
        except Exception as e:
            print(f"[ASR/LiteLLM] Error parsing SRT for segment {segment_idx + 1}: {e}")
            return []

        results = []
        for chunk in chunks:
            text = chunk.get("text", "").strip()
            if not text:
                continue
            results.append({
                "text": text,
                "start_s": float(chunk.get("start_s", 0.0)) + segment_start_time,
                "end_s": float(chunk.get("end_s", 0.0)) + segment_start_time,
            })
        return results

    # Build all segment messages upfront
    all_meta = []  # (segment_idx, segment_start_time, tmp_path)
    all_messages = []
    for i in range(num_segments):
        msg, seg_start, tmp_path = _build_segment_message(i)
        all_messages.append(msg)
        all_meta.append((i, seg_start, tmp_path))

    # Send in batches via batch_completion
    all_results = [None] * num_segments
    kwargs_base = dict(
        model=model,
        timeout=60,
        **({"api_base": api_base} if api_base else {}),
    )
    for batch_start in range(0, num_segments, batch_size):
        batch_indices = list(range(batch_start, min(batch_start + batch_size, num_segments)))
        batch_messages = [all_messages[i] for i in batch_indices]
        print(f"[ASR/LiteLLM] Sending batch segments {batch_indices[0]+1}-{batch_indices[-1]+1}/{num_segments}...")
        try:
            responses = litellm.batch_completion(messages=batch_messages, **kwargs_base)
        except Exception as e:
            print(f"[ASR/LiteLLM] batch_completion error: {e}")
            responses = [None] * len(batch_indices)

        for idx, resp in zip(batch_indices, responses):
            seg_idx, seg_start, _ = all_meta[idx]
            seg_end = seg_start + segment_duration
            try:
                content = resp.choices[0].message.content if resp else None
            except Exception:
                content = None
            parsed = _parse_response(content, seg_idx, seg_start)
            all_results[idx] = {"segment_idx": seg_idx, "results": parsed}

            # Save per-segment SRT with absolute timestamps for debugging
            if debug_dir:
                os.makedirs(debug_dir, exist_ok=True)
                seg_label = f"seg{seg_idx+1:03d}_{int(seg_start//60):02d}m{int(seg_start%60):02d}s-{int(seg_end//60):02d}m{int(seg_end%60):02d}s"
                seg_srt_path = os.path.join(debug_dir, f"{seg_label}.srt")
                with open(seg_srt_path, "w", encoding="utf-8") as sf_out:
                    for k, item in enumerate(parsed):
                        sf_out.write(f"{k+1}\n")
                        sf_out.write(f"{format_srt_timestamp(int(item['start_s']*1000))} --> {format_srt_timestamp(int(item['end_s']*1000))}\n")
                        sf_out.write(f"{item['text']}\n\n")
                print(f"[ASR/LiteLLM] Saved segment SRT -> {seg_srt_path}")

    # Cleanup temp files
    for _, _, tmp_path in all_meta:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    # Merge results in segment order
    all_sentence_info = []
    all_segments = []
    full_text_parts = []

    for seg_result in all_results:
        if not seg_result:
            continue
        for item in seg_result["results"]:
            text, start_s, end_s = item["text"], item["start_s"], item["end_s"]
            full_text_parts.append(text)
            all_sentence_info.append({
                "text": text,
                "speaker": None,
                "timestamp": [[text, int(start_s * 1000), int(end_s * 1000)]],
            })
            all_segments.append({"start": start_s, "end": end_s, "text": text})

    # Sort by start time
    combined = list(zip(all_sentence_info, all_segments))
    combined.sort(key=lambda x: x[1]["start"])
    all_sentence_info, all_segments = zip(*combined) if combined else ([], [])
    all_sentence_info = list(all_sentence_info)
    all_segments = list(all_segments)

    print(f"[ASR/LiteLLM] Transcription complete: {len(all_sentence_info)} segments")

    return {
        "text": " ".join(full_text_parts),
        "sentence_info": all_sentence_info,
        "segments": all_segments,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Unified transcribe entry point
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Qwen3-Omni-Captioner ASR (原生 input_audio 协议)
# ─────────────────────────────────────────────────────────────────────────────

def _transcribe_qwen(
    audio_path: str,
    model_raw: str,
    api_key: Optional[str],
    api_base: Optional[str],
    language: Optional[str],
    max_segment_size_mb: float = 1.0,
    batch_size: int = 8,
    debug_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    使用 Qwen-Omni 系列模型做语音识别。
    Qwen3.5-Omni-Plus 支持 text + input_audio 同时输入，
    可以指定 prompt 要求输出纯文本转写结果。
    通过 call_qwen_audio_batch 并发调用。
    """
    import soundfile as sf
    import tempfile
    import re

    from src.audio.qwen_audio_api import call_qwen_audio_batch

    # 去掉 litellm 前缀
    model = model_raw
    if model.startswith("openai/"):
        model = model[len("openai/"):]

    print(f"[ASR/Qwen] Model: {model}")

    # 读取音频
    audio_array, audio_sr = sf.read(audio_path, dtype="float32", always_2d=False)
    total_duration = len(audio_array) / audio_sr

    # 按文件大小 + 最大时长 8s 分片（确保每条字幕在 10-20s 范围内）
    estimated_mp3_mb = (32000 * total_duration) / (8 * 1024 * 1024)
    num_segments_by_size = max(1, int(estimated_mp3_mb // max_segment_size_mb) +
                        (1 if estimated_mp3_mb % max_segment_size_mb > 0 else 0))
    num_segments_by_duration = max(1, int(total_duration // 8) +
                                  (1 if total_duration % 8 > 0 else 0))
    num_segments = max(num_segments_by_size, num_segments_by_duration)
    segment_duration = total_duration / num_segments

    print(f"[ASR/Qwen] Audio: {total_duration:.1f}s, splitting into {num_segments} segments (by_size={num_segments_by_size}, by_duration={num_segments_by_duration})")

    # ASR prompt（Qwen3.5-Omni 支持 text+audio，prompt 不会被忽略！）
    prompt = (
        "Transcribe the speech in this audio segment. "
        "Output ONLY the raw transcribed text, no commentary, no descriptions. "
        "If there is no speech, output '[no speech]'. "
    )
    if language:
        lang_hint = "Chinese" if language in ("zh", "Chinese") else language
        prompt += f"The audio language is {lang_hint}. "
    prompt = prompt.strip()

    # 提取每个 segment 的音频文件
    segment_paths = []
    segment_meta = []  # (idx, start_time, end_time)
    for i in range(num_segments):
        start_sample = int(i * segment_duration * audio_sr)
        end_sample = int((i + 1) * segment_duration * audio_sr) if i < num_segments - 1 else len(audio_array)
        segment_audio = audio_array[start_sample:end_sample]
        start_time = i * segment_duration
        end_time = min((i + 1) * segment_duration, total_duration)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            seg_path = tmp.name
            sf.write(seg_path, segment_audio, audio_sr)

        segment_paths.append(seg_path)
        segment_meta.append((i, start_time, end_time))

    print(f"[ASR/Qwen] Sending {len(segment_paths)} segments via Qwen-Omni...")

    # Qwen3.5-Omni 支持 prompt+audio，传入 ASR 指示
    texts = call_qwen_audio_batch(
        audio_paths=segment_paths,
        prompt=prompt,
        max_tokens=2048,
        max_workers=batch_size,
        model=model,
        api_key=api_key,
        base_url=api_base,
    )

    # 清理临时文件
    for p in segment_paths:
        try:
            os.unlink(p)
        except OSError:
            pass

    # 解析结果（Qwen3.5-Omni 会按 prompt 要求输出纯文本）
    all_sentence_info = []
    all_segments = []
    full_text_parts = []

    for idx, text in enumerate(texts):
        seg_idx, start_time, end_time = segment_meta[idx]

        if not text or not text.strip() or text.strip() == "[no speech]":
            continue

        # 保存调试文件
        if debug_dir:
            os.makedirs(debug_dir, exist_ok=True)
            seg_label = f"qseg{seg_idx+1:03d}_{int(start_time//60):02d}m{int(start_time%60):02d}s-{int(end_time//60):02d}m{int(end_time%60):02d}s"
            seg_srt_path = os.path.join(debug_dir, f"{seg_label}.txt")
            with open(seg_srt_path, "w", encoding="utf-8") as f:
                f.write(text)

        raw_text = text.strip()

        full_text_parts.append(raw_text)
        all_sentence_info.append({
            "text": raw_text,
            "speaker": None,
            "timestamp": [[raw_text, int(start_time * 1000), int(end_time * 1000)]],
        })
        all_segments.append({"start": start_time, "end": end_time, "text": raw_text})

    print(f"[ASR/Qwen] Transcription complete: {len(all_sentence_info)} segments")

    return {
        "text": " ".join(full_text_parts),
        "sentence_info": all_sentence_info,
        "segments": all_segments,
    }


# ─────────────────────────────────────────────────────────────────────────────
# MiMo Audio ASR（使用 MimoAudioAPI，原生 input_audio 协议）
# ─────────────────────────────────────────────────────────────────────────────

def _transcribe_mimo(
    audio_path: str,
    model_raw: str,
    api_key: Optional[str],
    api_base: Optional[str],
    language: Optional[str],
    max_segment_size_mb: float = 1.0,
    batch_size: int = 3,
    debug_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    使用 MiMo 系列模型做语音识别。
    通过 MimoAudioAPI 并发调用，原生 input_audio 协议。
    """
    import soundfile as sf
    import tempfile
    import re

    # 去掉 litellm 前缀
    model = model_raw
    if model.startswith("openai/"):
        model = model[len("openai/"):]

    print(f"[ASR/MiMo] Model: {model}")

    # 读取音频
    audio_array, audio_sr = sf.read(audio_path, dtype="float32", always_2d=False)
    total_duration = len(audio_array) / audio_sr

    # 按文件大小 + 最大时长 8s 分片（确保每条字幕在 10-20s 范围内，Screenwriter 需要）
    estimated_mp3_mb = (32000 * total_duration) / (8 * 1024 * 1024)
    num_segments_by_size = max(1, int(estimated_mp3_mb // max_segment_size_mb) +
                        (1 if estimated_mp3_mb % max_segment_size_mb > 0 else 0))
    num_segments_by_duration = max(1, int(total_duration // 8) +
                                  (1 if total_duration % 8 > 0 else 0))
    num_segments = max(num_segments_by_size, num_segments_by_duration)
    segment_duration = total_duration / num_segments

    print(f"[ASR/MiMo] Audio: {total_duration:.1f}s, splitting into {num_segments} segments (by_size={num_segments_by_size}, by_duration={num_segments_by_duration})")

    # ASR prompt（MiMo 支持 text+audio）
    prompt = (
        "Transcribe the speech in this audio segment. "
        "Output ONLY the raw transcribed text, no commentary, no descriptions. "
        "If there is no speech, output '[no speech]'. "
    )
    if language:
        lang_hint = "Chinese" if language in ("zh", "Chinese") else language
        prompt += f"The audio language is {lang_hint}. "
    prompt = prompt.strip()

    # 提取每个 segment 的音频文件
    segment_paths = []
    segment_meta = []  # (idx, start_time, end_time)
    for i in range(num_segments):
        start_sample = int(i * segment_duration * audio_sr)
        end_sample = int((i + 1) * segment_duration * audio_sr) if i < num_segments - 1 else len(audio_array)
        segment_audio = audio_array[start_sample:end_sample]
        start_time = i * segment_duration
        end_time = min((i + 1) * segment_duration, total_duration)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            seg_path = tmp.name
            sf.write(seg_path, segment_audio, audio_sr)

        segment_paths.append(seg_path)
        segment_meta.append((i, start_time, end_time))

    print(f"[ASR/MiMo] Sending {len(segment_paths)} segments via MimoAudioAPI (max_concurrent={batch_size})...")

    api = MimoAudioAPI(
        model=model,
        api_key=api_key,
        base_url=api_base,
        use_tp="token-plan" in (api_base or "").lower(),
        max_retries=5,
    )
    texts = api.call_batch(
        audio_paths=segment_paths,
        prompt=prompt,
        max_tokens=2048,
        max_workers=batch_size,
    )

    # 清理临时文件
    for p in segment_paths:
        try:
            os.unlink(p)
        except OSError:
            pass

    # 解析结果
    all_sentence_info = []
    all_segments = []
    full_text_parts = []

    for idx, text in enumerate(texts):
        seg_idx, start_time, end_time = segment_meta[idx]

        if not text or not text.strip() or text.strip() == "[no speech]":
            continue

        # 保存调试文件
        if debug_dir:
            os.makedirs(debug_dir, exist_ok=True)
            seg_label = f"mimo_seg{seg_idx+1:03d}_{int(start_time//60):02d}m{int(start_time%60):02d}s-{int(end_time//60):02d}m{int(end_time%60):02d}s"
            seg_srt_path = os.path.join(debug_dir, f"{seg_label}.txt")
            with open(seg_srt_path, "w", encoding="utf-8") as f:
                f.write(text)

        raw_text = text.strip()

        full_text_parts.append(raw_text)
        all_sentence_info.append({
            "text": raw_text,
            "speaker": None,
            "timestamp": [[raw_text, int(start_time * 1000), int(end_time * 1000)]],
        })
        all_segments.append({"start": start_time, "end": end_time, "text": raw_text})

    print(f"[ASR/MiMo] Transcription complete: {len(all_sentence_info)} segments")

    return {
        "text": " ".join(full_text_parts),
        "sentence_info": all_sentence_info,
        "segments": all_segments,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Unified ASR entry
# ─────────────────────────────────────────────────────────────────────────────

def transcribe_audio(
    audio_path: str,
    backend: Optional[str] = None,
    device: str = "cuda:0",
    language: Optional[str] = None,
    # whisper.cpp
    whisper_cpp_model_name: Optional[str] = None,
    whisper_cpp_n_threads: int = 4,
    enable_diarization: Optional[bool] = None,
    diarization_model_path: Optional[str] = None,
    merge_same_speaker: Optional[bool] = None,
    merge_gap: Optional[float] = None,
    # LiteLLM
    litellm_model: Optional[str] = None,
    litellm_api_key: Optional[str] = None,
    litellm_api_base: Optional[str] = None,
    litellm_max_segment_mb: float = 30.0,
    litellm_batch_size: int = 8,
    litellm_debug_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Unified ASR entry point.

    backend: "whisper_cpp" | "litellm"
             Defaults to config.ASR_BACKEND or "whisper_cpp".
    """
    if backend is None:
        backend = getattr(config, 'ASR_BACKEND', 'whisper_cpp')

    if backend == "whisper_cpp":
        model_name = whisper_cpp_model_name or getattr(config, 'ASR_WHISPER_CPP_MODEL', 'base.en')
        n_threads = whisper_cpp_n_threads or getattr(config, 'ASR_WHISPER_CPP_N_THREADS', 4)
        dev = device or getattr(config, 'ASR_DEVICE', 'cuda:0')
        lang = language or getattr(config, 'ASR_LANGUAGE', None)
        return _transcribe_whisper_cpp(
            audio_path, model_name, dev, lang, n_threads,
            enable_diarization=enable_diarization,
            diarization_model_path=diarization_model_path,
            merge_same_speaker=merge_same_speaker,
            merge_gap=merge_gap,
        )

    elif backend == "litellm":
        model = litellm_model or getattr(config, 'AUDIO_LITELLM_MODEL')
        api_key = litellm_api_key or getattr(config, 'AUDIO_LITELLM_API_KEY', None)
        api_base = litellm_api_base or getattr(config, 'AUDIO_LITELLM_BASE_URL', None)
        lang = language or getattr(config, 'ASR_LANGUAGE', None)

        # 路由：MiMo → 原生 input_audio 协议
        _model_lower = model.lower()
        if "mimo" in _model_lower:
            return _transcribe_mimo(
                audio_path, model, api_key, api_base, lang,
                max_segment_size_mb=litellm_max_segment_mb,
                batch_size=litellm_batch_size,
                debug_dir=litellm_debug_dir,
            )

        # 路由：Qwen3-Omni-Captioner → 原生 input_audio 协议
        if "qwen3-omni" in _model_lower or "qwen3_omni" in _model_lower:
            from src.audio.qwen_audio_api import call_qwen_audio_batch
            return _transcribe_qwen(
                audio_path, model, api_key, api_base, lang,
                max_segment_size_mb=litellm_max_segment_mb,
                batch_size=litellm_batch_size,
                debug_dir=litellm_debug_dir,
            )

        # 回退：旧 litellm 模型
        return _transcribe_litellm(
            audio_path, model, api_key, lang, api_base,
            max_segment_size_mb=litellm_max_segment_mb,
            batch_size=litellm_batch_size,
            debug_dir=litellm_debug_dir,
        )

    else:
        raise ValueError(f"Unknown ASR backend: {backend!r}. Choose 'whisper_cpp' or 'litellm'.")


# ─────────────────────────────────────────────────────────────────────────────
# High-level run_asr
# ─────────────────────────────────────────────────────────────────────────────

def assign_speakers_to_srt(
    srt_path: str,
    audio_path: str,
    output_srt_path: str,
    device: str = "cuda:0",
    diarization_model_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Parse external SRT and assign speakers via diarization only (no transcription).

    Args:
        srt_path: Path to existing SRT file.
        audio_path: Path to audio file (16k mono MP3) for diarization.
        output_srt_path: Path to write the speaker-labeled SRT.
        device: CUDA device string.
        diarization_model_path: Path to pyannote model (defaults to config value).

    Returns:
        sentence_info list with speaker labels assigned.
    """
    if diarization_model_path is None:
        diarization_model_path = getattr(
            config, 'ASR_DIARIZATION_MODEL_PATH',
            "../HF/hub/models--pyannote--speaker-diarization-community-1/snapshots/3533c8cf8e369892e6b79ff1bf80f7b0286a54ee/"
        )

    with open(srt_path, 'r', encoding='utf-8') as f:
        srt_text = f.read()
    segments = _parse_srt_content(srt_text)

    print(f"[ASR/SRT] Running diarization on {audio_path}...")
    diarization_tracks = _run_pyannote_diarization(audio_path, diarization_model_path, device)
    print(f"[ASR/SRT] Diarization complete: {len(diarization_tracks)} speaker turns")

    sentence_info = []
    for seg in segments:
        start_s = seg['start_s']
        end_s = seg['end_s']
        speaker = _get_speaker_at_time(diarization_tracks, start_s, end_s)
        start_ms = int(start_s * 1000)
        end_ms = int(end_s * 1000)
        sentence_info.append({
            'text': seg['text'],
            'speaker': speaker,
            'timestamp': [[seg['text'], start_ms, end_ms]],
        })

    write_srt_from_sentence_info(sentence_info, output_srt_path)
    print(f"[ASR/SRT] Speaker-labeled SRT saved -> {output_srt_path}")
    return sentence_info


def run_asr(
    video_path: str,
    output_dir: str,
    srt_path: Optional[str] = None,
    backend: Optional[str] = None,
    asr_device: str = "cuda:0",
    asr_language: Optional[str] = None,
    # whisper.cpp
    whisper_cpp_model_name: Optional[str] = None,
    whisper_cpp_n_threads: int = 4,
    enable_diarization: Optional[bool] = None,
    diarization_model_path: Optional[str] = None,
    merge_same_speaker: Optional[bool] = None,
    merge_gap: Optional[float] = None,
    # LiteLLM
    litellm_model: Optional[str] = None,
    litellm_api_key: Optional[str] = None,
    litellm_api_base: Optional[str] = None,
    litellm_max_segment_mb: float = 30.0,
    litellm_batch_size: int = 8,
    litellm_debug_dir: Optional[str] = None,
    # Common
    start_sec: Optional[float] = None,
    end_sec: Optional[float] = None,
    keep_extracted_audio: bool = False,
    include_speaker: bool = True,
) -> Dict[str, Any]:
    """
    High-level ASR: extract audio from video, transcribe, write SRT.

    Args:
        video_path: Input video file.
        output_dir: Directory for temporary files.
        srt_path: Output SRT path (defaults to video path + .srt).
        backend: "whisper_cpp" or "litellm".
        asr_device: CUDA device string (e.g. "cuda:0").
        asr_language: Language code ("en", "zh", ...). None = auto-detect.
        whisper_cpp_model_name: whisper.cpp model name (e.g. "base.en", "large-v3") or path to ggml model.
        whisper_cpp_n_threads: Number of CPU threads for whisper.cpp inference.
        enable_diarization: Enable speaker diarization with pyannote.
        diarization_model_path: Path to pyannote diarization model.
        merge_same_speaker: Merge consecutive segments from same speaker.
        merge_gap: Max time gap for merging same-speaker segments (seconds).
        litellm_model: LiteLLM model ID (e.g. "gemini/gemini-2.5-flash").
        litellm_api_key: API key for LiteLLM provider.
        litellm_max_segment_mb: Max segment size in MB for LiteLLM auto-splitting (default: 30MB).
        litellm_batch_size: Number of segments per batch_completion call for LiteLLM (default: 8).
        start_sec / end_sec: Optional clip range in the video.
        keep_extracted_audio: Keep intermediate audio file if True.
        include_speaker: Include speaker labels in SRT output.

    Returns:
        dict with keys: srt_path, sentence_info, segments, text
    """
    _ensure_dir(output_dir)

    final_srt_path = srt_path or (os.path.splitext(video_path)[0] + ".srt")

    if os.path.exists(final_srt_path):
        print(f"[Skip] Found existing SRT: {final_srt_path}")
        return {"srt_path": final_srt_path}

    audio_wav_path = os.path.join(output_dir, "audio_16k_mono.mp3")
    print(f"[ASR] Extracting audio -> {audio_wav_path}")
    extract_audio_mp3_16k(video_path, audio_wav_path, start_sec, end_sec)

    asr_output = transcribe_audio(
        audio_wav_path,
        backend=backend,
        device=asr_device,
        language=asr_language,
        whisper_cpp_model_name=whisper_cpp_model_name,
        whisper_cpp_n_threads=whisper_cpp_n_threads,
        enable_diarization=enable_diarization,
        diarization_model_path=diarization_model_path,
        merge_same_speaker=merge_same_speaker,
        merge_gap=merge_gap,
        litellm_model=litellm_model,
        litellm_api_key=litellm_api_key,
        litellm_api_base=litellm_api_base,
        litellm_max_segment_mb=litellm_max_segment_mb,
        litellm_batch_size=litellm_batch_size,
        litellm_debug_dir=litellm_debug_dir,
    )

    sentence_info = asr_output.get("sentence_info", [])

    # Offset timestamps for clipped audio
    if start_sec is not None and start_sec > 0:
        offset_ms = int(float(start_sec) * 1000)
        sentence_info = [
            {
                **sent,
                "timestamp": [
                    [ts[0], ts[1] + offset_ms, ts[2] + offset_ms]
                    if isinstance(ts, (list, tuple)) and len(ts) >= 3 else ts
                    for ts in sent.get("timestamp", [])
                ],
            }
            for sent in sentence_info
        ]

    write_srt_from_sentence_info(sentence_info, final_srt_path, include_speaker=include_speaker)
    print(f"[ASR] SRT saved -> {final_srt_path}")

    result: Dict[str, Any] = {
        "srt_path": final_srt_path,
        "sentence_info": sentence_info,
        "text": asr_output.get("text", ""),
    }
    if "segments" in asr_output:
        result["segments"] = asr_output["segments"]

    if not keep_extracted_audio:
        try:
            os.remove(audio_wav_path)
        except OSError:
            pass

    return result

