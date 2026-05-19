import streamlit as st
import os
import re
import hashlib
import importlib
import subprocess
import threading
import signal
import queue
import logging
logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context").setLevel(logging.ERROR)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "src", "config.py")
MIN_TARGET_SHOT_LENGTH_SEC = 0.2
MIN_SEGMENT_DURATION_FLOOR_SEC = 0.1
SHOT_LENGTH_RANGE_CAP_SEC = 1.0


def _read_config() -> dict:
    """Read key=value pairs from config.py as strings."""
    vals = {}
    with open(CONFIG_PATH, "r") as f:
        for line in f:
            m = re.match(r'^([A-Z_][A-Z0-9_]*)\s*=\s*(.+)', line)
            if m:
                vals[m.group(1)] = m.group(2).strip()
    return vals


def _cfg(key: str, fallback: str) -> str:
    """Get a config value as a plain string (strips quotes)."""
    raw = _read_config().get(key, fallback)
    # strip surrounding quotes if present
    if (raw.startswith('"') and raw.endswith('"')) or \
       (raw.startswith("'") and raw.endswith("'")):
        return raw[1:-1]
    return raw


def save_config(key: str, value: str):
    """Overwrite a single key in config.py."""
    with open(CONFIG_PATH, "r") as f:
        content = f.read()
    # Determine whether to quote: if original had quotes or value is not numeric
    try:
        float(value)
        new_val = value
    except ValueError:
        new_val = f'"{value}"'
    content = re.sub(
        rf'^({re.escape(key)}\s*=\s*).*',
        rf'\g<1>{new_val}',
        content,
        flags=re.MULTILINE,
    )
    with open(CONFIG_PATH, "w") as f:
        f.write(content)


def _persist_target_output_length(target_length: float):
    """Persist target output length by materializing the derived min/max config values."""
    min_duration = max(5.0, target_length - 5.0)
    max_duration = target_length + 5.0
    save_config("AUDIO_SEGMENT_MIN_DURATION_SEC", str(min_duration))
    save_config("AUDIO_SEGMENT_MAX_DURATION_SEC", str(max_duration))


def _derive_shot_duration_bounds(shot_length: float) -> tuple[float, float]:
    """Derive segment duration bounds from the desired target shot length."""
    shot_length = max(MIN_TARGET_SHOT_LENGTH_SEC, float(shot_length))
    range_radius = min(SHOT_LENGTH_RANGE_CAP_SEC, max(MIN_SEGMENT_DURATION_FLOOR_SEC, shot_length))
    min_seg_duration = max(MIN_SEGMENT_DURATION_FLOOR_SEC, shot_length - range_radius)
    max_seg_duration = shot_length + range_radius
    return round(min_seg_duration, 3), round(max_seg_duration, 3)


def _derive_target_shot_length_from_config() -> float:
    """Reconstruct the target shot length from persisted min/max segment durations."""
    min_seg_duration = float(_cfg("AUDIO_MIN_SEGMENT_DURATION", "3.0"))
    max_seg_duration = float(_cfg("AUDIO_MAX_SEGMENT_DURATION", "5.0"))
    return round(max(MIN_TARGET_SHOT_LENGTH_SEC, (min_seg_duration + max_seg_duration) / 2.0), 3)


def _persist_target_shot_length(shot_length: float):
    """Persist target shot length by materializing the derived min/max config values."""
    min_seg_duration, max_seg_duration = _derive_shot_duration_bounds(shot_length)
    save_config("AUDIO_MIN_SEGMENT_DURATION", str(min_seg_duration))
    save_config("AUDIO_MAX_SEGMENT_DURATION", str(max_seg_duration))

st.set_page_config(
    page_title="CutClaw",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Reset & base ── */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "SF Pro Display", sans-serif !important;
}

/* ── Buttons ── */
div.stButton > button {
    border-radius: 12px;
    font-weight: 600;
    font-size: 0.95rem;
    padding: 0.6rem 1.4rem;
    border: 1px solid rgba(150, 150, 150, 0.2);
    transition: all 0.2s ease;
    box-shadow: 0 2px 4px rgba(0,0,0,0.02);
}
div.stButton > button:first-child {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
    color: white !important;
    border: none;
    box-shadow: 0 4px 10px rgba(99, 102, 241, 0.3);
}
div.stButton > button:first-child:hover {
    opacity: 0.95;
    transform: translateY(-1px);
    box-shadow: 0 6px 14px rgba(99, 102, 241, 0.4);
}
div.stButton > button:disabled { opacity: 0.5; cursor: not-allowed; transform: none; box-shadow: none; }

/* ── Cards ── */
.vca-card {
    border-radius: 16px;
    padding: 1.8rem;
    background: rgba(150, 150, 150, 0.05);
    box-shadow: 0 4px 20px rgba(0,0,0,0.03);
    margin-bottom: 1.5rem;
    border: 1px solid rgba(150, 150, 150, 0.15);
    backdrop-filter: blur(10px);
}

/* ── Log terminal ── */
.vca-log {
    background: #0f172a;
    color: #f8fafc;
    font-family: "JetBrains Mono", "SF Mono", monospace;
    font-size: 0.82rem;
    line-height: 1.65;
    border-radius: 12px;
    padding: 1.5rem;
    height: 480px;
    overflow-y: auto;
    white-space: pre-wrap;
    word-break: break-all;
    border: 1px solid #1e293b;
    box-shadow: inset 0 2px 8px rgba(0,0,0,0.2);
}
.vca-stage { color: #818cf8; font-weight: 600; text-shadow: 0 0 8px rgba(129,140,248,0.2); }
.vca-error { color: #f87171; font-weight: 500; }
.vca-success { color: #34d399; font-weight: 500; }

/* ── Status badge ── */
.vca-badge {
    display: inline-block;
    border-radius: 9999px;
    padding: 0.25rem 0.85rem;
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.vca-badge-idle    { background: #f1f5f9; color: #475569; border: 1px solid #e2e8f0; }
.vca-badge-running { background: #fef3c7; color: #b45309; border: 1px solid #fde68a; animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite; }
.vca-badge-done    { background: #dcfce7; color: #047857; border: 1px solid #a7f3d0; }
.vca-badge-error   { background: #fee2e2; color: #b91c1c; border: 1px solid #fecaca; }

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.6; }
}
</style>
""", unsafe_allow_html=True)
# ── Session state ──────────────────────────────────────────────
_STAGE_NAMES = ["shot_detection", "asr", "video_captioning", "audio_analysis", "screenwriter", "editor"]

_DEFAULTS = {
    "running": False,
    "process": None,
    "log_lines": [],
    "log_queue": None,
    "result_shot_json": None,
    "pipeline_failed": False,
    "start_error": None,
    "stage_status": {s: "pending" for s in _STAGE_NAMES},
    "stage_times": {},
}
for _k, _v in _DEFAULTS.items():
    st.session_state.setdefault(_k, _v)

# ── Sidebar ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎬 CutClaw")
    st.markdown("---")

    # ── Video file picker ──
    _VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v"}
    _AUDIO_EXTS = {".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a"}

    def _scan_files(folder: str, exts: set) -> list:
        base = os.path.join(PROJECT_ROOT, folder)
        if not os.path.isdir(base):
            return []
        return sorted(
            os.path.join(folder, f)
            for f in os.listdir(base)
            if os.path.splitext(f)[1].lower() in exts
        )

    _video_files = _scan_files("resource/video", _VIDEO_EXTS)
    _saved_video = _cfg("VIDEO_PATH", "")
    if _saved_video and _saved_video not in _video_files:
        _video_files = [_saved_video] + _video_files

    if _video_files:
        _vi = _video_files.index(_saved_video) if _saved_video in _video_files else 0
        video_path = st.selectbox("Video", _video_files, index=_vi, key="si_video_path")
    else:
        video_path = st.text_input(
            "Video Path",
            value=_saved_video,
            placeholder="/path/to/video.mkv",
            key="si_video_path",
        )
    if video_path != _saved_video:
        save_config("VIDEO_PATH", video_path)

    with st.expander("📝 Provide SRT Path (Optional)", expanded=False):
        _SRT_EXTS = {".srt", ".vtt", ".ass", ".ssa"}
        _srt_files = _scan_files("resource/subtitle", _SRT_EXTS)
        _saved_srt = _cfg("SRT_PATH", "")
        if _saved_srt and _saved_srt not in _srt_files:
            _srt_files = [_saved_srt] + _srt_files

        if _srt_files:
            _srt_options = [""] + _srt_files
            _sri = _srt_options.index(_saved_srt) if _saved_srt in _srt_options else 0
            srt_path = st.selectbox(
                "SRT File",
                _srt_options,
                index=_sri,
                help="Select an existing SRT file. Skips ASR transcription; diarization still runs to assign speakers.",
                key="si_srt_path",
            )
        else:
            srt_path = st.text_input(
                "SRT Path",
                value=_saved_srt,
                placeholder="/path/to/subtitles.srt",
                help="Path to existing SRT file. Skips ASR transcription; diarization still runs to assign speakers.",
                key="si_srt_path",
            )
        if srt_path != _saved_srt:
            save_config("SRT_PATH", srt_path)

    # ── Audio file picker ──
    _audio_files = _scan_files("resource/audio", _AUDIO_EXTS)
    _saved_audio = _cfg("AUDIO_PATH", "")
    if _saved_audio and _saved_audio not in _audio_files:
        _audio_files = [_saved_audio] + _audio_files

    if _audio_files:
        _ai = _audio_files.index(_saved_audio) if _saved_audio in _audio_files else 0
        audio_path = st.selectbox("Audio", _audio_files, index=_ai, key="si_audio_path")
    else:
        audio_path = st.text_input(
            "Audio Path",
            value=_saved_audio,
            placeholder="/path/to/audio.mp3",
            key="si_audio_path",
        )
    if audio_path != _saved_audio:
        save_config("AUDIO_PATH", audio_path)

    instruction = st.text_area(
        "Instruction",
        value=_cfg("INSTRUCTION", ""),
        placeholder="Describe the edit you want...",
        height=100,
        key="si_instruction",
    )
    if instruction != _cfg("INSTRUCTION", ""):
        save_config("INSTRUCTION", instruction)

    video_type = "film"

    main_character = st.text_input(
        "Main Character Name",
        value=_cfg("MAIN_CHARACTER_NAME", ""),
        placeholder="e.g. Batman",
        key="si_main_character",
    )
    if main_character != _cfg("MAIN_CHARACTER_NAME", ""):
        save_config("MAIN_CHARACTER_NAME", main_character)

    target_length = st.number_input(
        "Target Output Length (seconds)",
        min_value=10.0, max_value=300.0,
        value=float(_cfg("AUDIO_SEGMENT_MAX_DURATION_SEC", "35.0")) - 5.0,
        step=5.0,
        help="Sets AUDIO_SEGMENT_MIN/MAX_DURATION_SEC to Target ± 5 seconds.",
        key="si_target_length",
    )
    if target_length != float(_cfg("AUDIO_SEGMENT_MAX_DURATION_SEC", "35.0")) - 5.0:
        _persist_target_output_length(target_length)

    shot_length = st.number_input(
        "Target Shot Length (seconds)",
        min_value=MIN_TARGET_SHOT_LENGTH_SEC, max_value=30.0,
        value=_derive_target_shot_length_from_config(),
        step=0.1,
        help="Allows rapid montage pacing. For >=1s, uses Target ±1s; for sub-second targets, the range narrows automatically.",
        key="si_shot_length",
    )
    if shot_length != _derive_target_shot_length_from_config():
        _persist_target_shot_length(shot_length)

    with st.expander("⚙️ Model Settings", expanded=False):
        st.markdown("**1. Video Analysis Model (Vision)**")
        st.caption("Used for dense video captioning and scene understanding.")
        video_analysis_model = st.text_input("VIDEO_ANALYSIS_MODEL", value=_cfg("VIDEO_ANALYSIS_MODEL", "openai/qwen3.5-plus"), key="si_va_model")
        if video_analysis_model != _cfg("VIDEO_ANALYSIS_MODEL", ""):
            save_config("VIDEO_ANALYSIS_MODEL", video_analysis_model)
        video_analysis_endpoint = st.text_input("VIDEO_ANALYSIS_ENDPOINT", value=_cfg("VIDEO_ANALYSIS_ENDPOINT", "https://coding.dashscope.aliyuncs.com/v1"), key="si_va_ep")
        if video_analysis_endpoint != _cfg("VIDEO_ANALYSIS_ENDPOINT", ""):
            save_config("VIDEO_ANALYSIS_ENDPOINT", video_analysis_endpoint)
        video_analysis_api_key = st.text_input("VIDEO_ANALYSIS_API_KEY", value=_cfg("VIDEO_ANALYSIS_API_KEY", ""), type="password", key="si_va_key")
        if video_analysis_api_key != _cfg("VIDEO_ANALYSIS_API_KEY", ""):
            save_config("VIDEO_ANALYSIS_API_KEY", video_analysis_api_key)

        st.markdown("**2. Audio Analysis Model (Language)**")
        st.caption("Used for structure analysis and capturing musical emotions/beats.")
        audio_litellm_model = st.text_input("AUDIO_LITELLM_MODEL", value=_cfg("AUDIO_LITELLM_MODEL", "openai/gemini-3-flash-preview-nothinking"), key="si_al_model")
        if audio_litellm_model != _cfg("AUDIO_LITELLM_MODEL", ""):
            save_config("AUDIO_LITELLM_MODEL", audio_litellm_model)
        audio_litellm_base_url = st.text_input("AUDIO_LITELLM_BASE_URL", value=_cfg("AUDIO_LITELLM_BASE_URL", "https://api.gpt.ge/v1"), key="si_al_url")
        if audio_litellm_base_url != _cfg("AUDIO_LITELLM_BASE_URL", ""):
            save_config("AUDIO_LITELLM_BASE_URL", audio_litellm_base_url)
        audio_litellm_api_key = st.text_input("AUDIO_LITELLM_API_KEY", value=_cfg("AUDIO_LITELLM_API_KEY", ""), type="password", key="si_al_key")
        if audio_litellm_api_key != _cfg("AUDIO_LITELLM_API_KEY", ""):
            save_config("AUDIO_LITELLM_API_KEY", audio_litellm_api_key)

        st.markdown("**3. Editing Agent Model (Reasoning)**")
        st.caption("Used by Screenwriter, Editor, and Reviewer agents to select and review shots.")
        agent_litellm_model = st.text_input("AGENT_LITELLM_MODEL", value=_cfg("AGENT_LITELLM_MODEL", "openai/MiniMax-M2.5"), key="si_ag_model")
        if agent_litellm_model != _cfg("AGENT_LITELLM_MODEL", ""):
            save_config("AGENT_LITELLM_MODEL", agent_litellm_model)
        agent_litellm_url = st.text_input("AGENT_LITELLM_URL", value=_cfg("AGENT_LITELLM_URL", "https://api.minimaxi.com/v1"), key="si_ag_url")
        if agent_litellm_url != _cfg("AGENT_LITELLM_URL", ""):
            save_config("AGENT_LITELLM_URL", agent_litellm_url)
        agent_litellm_api_key = st.text_input("AGENT_LITELLM_API_KEY", value=_cfg("AGENT_LITELLM_API_KEY", ""), type="password", key="si_ag_key")
        if agent_litellm_api_key != _cfg("AGENT_LITELLM_API_KEY", ""):
            save_config("AGENT_LITELLM_API_KEY", agent_litellm_api_key)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        run_clicked = st.button(
            "▶ Run",
            disabled=st.session_state.running,
            use_container_width=True,
        )
    with col2:
        stop_clicked = st.button(
            "■ Stop",
            disabled=not st.session_state.running,
            use_container_width=True,
        )
# ── Helpers ────────────────────────────────────────────────────
def derive_shot_point_path(video_path: str, audio_path: str, instruction: str) -> str:
    import src.config as config
    video_id = os.path.splitext(os.path.basename(video_path))[0].replace('.', '_').replace(' ', '_')
    audio_id = os.path.splitext(os.path.basename(audio_path))[0].replace('.', '_').replace(' ', '_')
    instruction_hash = hashlib.md5(instruction.encode('utf-8')).hexdigest()[:8]
    instruction_safe = re.sub(r'[^\w\s-]', '', instruction)[:50].strip().replace(' ', '_')
    instruction_id = f"{instruction_safe}_{instruction_hash}" if instruction_safe else f"instruction_{instruction_hash}"
    return os.path.join(
        config.VIDEO_DATABASE_FOLDER, 'Output',
        f"{video_id}_{audio_id}",
        f"shot_point_{instruction_id}.json"
    )


def derive_shot_plan_path(video_path: str, audio_path: str, instruction: str) -> str:
    return derive_shot_point_path(video_path, audio_path, instruction).replace("shot_point_", "shot_plan_")


def _resolve_path(path: str) -> str:
    if not path:
        return ""
    if os.path.isabs(path):
        return path
    return os.path.join(PROJECT_ROOT, path)


def resolve_hook_subtitle_path(video_path: str, srt_path: str) -> str:
    import src.config as config

    video_id = os.path.splitext(os.path.basename(video_path))[0].replace('.', '_').replace(' ', '_')
    video_dir = os.path.join(config.VIDEO_DATABASE_FOLDER, 'Video', video_id)
    candidates = [
        os.path.join(video_dir, "subtitles_with_characters.srt"),
        os.path.join(video_dir, "subtitles.srt"),
        _resolve_path(srt_path.strip()) if srt_path.strip() else "",
    ]

    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    return ""


def rerun_hook_dialogue_selection(
    shot_plan_path: str,
    video_path: str,
    instruction: str,
    main_character: str,
    srt_path: str,
) -> str:
    subtitle_path = resolve_hook_subtitle_path(video_path, srt_path)
    if not subtitle_path:
        raise FileNotFoundError("No subtitle file is available for hook dialogue selection.")
    if not os.path.exists(shot_plan_path):
        raise FileNotFoundError(f"Shot plan file not found: {shot_plan_path}")

    import src.config as runtime_config
    runtime_config = importlib.reload(runtime_config)

    import src.Screenwriter_scene_short as screenwriter_module
    screenwriter_module = importlib.reload(screenwriter_module)

    screenwriter_module.refresh_hook_dialogue_in_shot_plan(
        shot_plan_path=shot_plan_path,
        subtitle_path=subtitle_path,
        instruction=instruction,
        main_character=main_character.strip() or None,
        prompt_window_mode="random_window",
        random_window_attempts=4,
    )
    return subtitle_path


def _read_stdout(proc, q):
    """Read process stdout line-by-line into queue. Sentinel None signals EOF."""
    try:
        for line in proc.stdout:
            q.put(line.rstrip())
    finally:
        q.put(None)


def start_pipeline(video_path, audio_path, instruction, video_type, main_character, srt_path, target_length, shot_length,
                   video_analysis_model, video_analysis_endpoint, video_analysis_api_key,
                   audio_litellm_model, audio_litellm_base_url, audio_litellm_api_key,
                   agent_litellm_model, agent_litellm_url, agent_litellm_api_key):
    # Calculate min and max duration based on target length
    min_duration = max(5.0, target_length - 5.0)
    max_duration = target_length + 5.0

    # Calculate min and max segment duration based on shot length
    min_seg_duration, max_seg_duration = _derive_shot_duration_bounds(shot_length)
    
    cmd = [
        "python", "local_run.py",
        "--Video_Path", video_path,
        "--Audio_Path", audio_path,
        "--Instruction", instruction,
        "--type", video_type,
        "--instruction_type", "object",
        "--config.AUDIO_SEGMENT_MIN_DURATION_SEC", str(min_duration),
        "--config.AUDIO_SEGMENT_MAX_DURATION_SEC", str(max_duration),
        "--config.AUDIO_MIN_SEGMENT_DURATION", str(min_seg_duration),
        "--config.AUDIO_MAX_SEGMENT_DURATION", str(max_seg_duration),
        "--config.VIDEO_ANALYSIS_MODEL", video_analysis_model,
        "--config.VIDEO_ANALYSIS_ENDPOINT", video_analysis_endpoint,
        "--config.VIDEO_ANALYSIS_API_KEY", video_analysis_api_key,
        "--config.AUDIO_LITELLM_MODEL", audio_litellm_model,
        "--config.AUDIO_LITELLM_BASE_URL", audio_litellm_base_url,
        "--config.AUDIO_LITELLM_API_KEY", audio_litellm_api_key,
        "--config.AGENT_LITELLM_MODEL", agent_litellm_model,
        "--config.AGENT_LITELLM_URL", agent_litellm_url,
        "--config.AGENT_LITELLM_API_KEY", agent_litellm_api_key,
    ]
    if main_character.strip():
        cmd += ["--config.MAIN_CHARACTER_NAME", main_character.strip()]
    if srt_path.strip():
        cmd += ["--SRT_Path", srt_path.strip()]

    # Set PYTHONUNBUFFERED to force immediate stdout flushing
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=PROJECT_ROOT,
            env=env,
            start_new_session=True,
        )
    except Exception as e:
        return str(e)

    q = queue.Queue()
    t = threading.Thread(target=_read_stdout, args=(proc, q), daemon=True)
    t.start()

    st.session_state.process = proc
    st.session_state.log_queue = q
    st.session_state.log_lines = []
    st.session_state.running = True
    st.session_state.pipeline_failed = False
    st.session_state.stage_status = {s: "pending" for s in _STAGE_NAMES}
    st.session_state.stage_times = {}
    st.session_state.result_shot_json = derive_shot_point_path(video_path, audio_path, instruction)
    return None


def stop_pipeline():
    proc = st.session_state.process
    if proc and proc.poll() is None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
    st.session_state.running = False


_STAGE_STARTS = {
    "shot_detection": "[Step 1] Extracting video frames",
    "asr":            "[Thread A: ASR]",
    "video_captioning": "[Thread B: Video]",
    "audio_analysis": "[Thread C: Audio]",
    "screenwriter":   "Running Screenwriter",
    "editor":         "Running EditorCoreAgent",
}
_STAGE_ENDS = {
    "shot_detection": "[Step 1] Shot detection completed",
    "asr":            "[Thread A] ✨ Completed",
    "video_captioning": "[Thread B] ✨ Completed",
    "audio_analysis": "[Thread C] ✨ Completed",
    "screenwriter":   "Shot plan generated successfully",
    "editor":         "Video clip selection completed",
}
_STAGE_ERRORS = {
    "asr":            "[Thread A] ❌",
    "video_captioning": "[Thread B] ❌",
    "audio_analysis": "[Thread C] ❌",
}
_TIME_RE = re.compile(r"[Cc]ompleted in ([\d.]+)s")


def parse_stage_from_line(line: str, stage_status: dict, stage_times: dict):
    for stage, kw in _STAGE_STARTS.items():
        if kw in line and stage_status.get(stage) == "pending":
            stage_status[stage] = "running"
    for stage, kw in _STAGE_ENDS.items():
        if kw in line and stage_status.get(stage) == "running":
            stage_status[stage] = "done"
            m = _TIME_RE.search(line)
            if m:
                stage_times[stage] = float(m.group(1))
    for stage, kw in _STAGE_ERRORS.items():
        if kw in line:
            stage_status[stage] = "error"
    if "❌ Pipeline stage" in line:
        for stage, status in stage_status.items():
            if status == "running":
                stage_status[stage] = "error"


def build_graph_html(stage_status: dict, stage_times: dict) -> str:
    done_any = any(v in ("done", "running") for v in stage_status.values())
    input_state  = "done" if done_any else "pending"
    output_state = "done" if stage_status.get("editor") == "done" else "pending"

    def node(stage, label, state_override=None):
        state = state_override or stage_status.get(stage, "pending")
        t = stage_times.get(stage)
        time_html = f'<div class="nt">{int(t)}s</div>' if t else ""
        badge = {"done": "✓", "error": "✗", "running": "●"}.get(state, "")
        badge_html = f'<div class="nb">{badge}</div>' if badge else ""
        return (
            f'<div class="node node-{state}">'
            f'  <div class="ni">{badge_html}<div class="nl">{label}</div>{time_html}</div>'
            f'</div>'
        )

    def arrow():
        return '<div class="arr">→</div>'

    # parallel branch: three nodes stacked vertically, connected by fork/join lines
    par_nodes = (
        f'<div class="par-col">'
        f'  {node("asr", "ASR")}'
        f'  {node("video_captioning", "Video")}'
        f'  {node("audio_analysis", "Audio")}'
        f'</div>'
    )

    css = """
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:transparent}
.pipeline{
  display:flex;align-items:center;justify-content:center;
  font-family:"Inter","SF Pro Display",-apple-system,BlinkMacSystemFont,sans-serif;
  padding:20px 12px;gap:0;flex-wrap:nowrap;
}
.node{
  display:flex;align-items:center;justify-content:center;
  width:88px;height:72px;border-radius:12px;
  border:1px solid rgba(255,255,255,0.1);
  background:rgba(255,255,255,0.05);
  transition:all 0.3s ease;flex-shrink:0;
}
.ni{
  display:flex;flex-direction:column;align-items:center;justify-content:center;
  gap:2px;width:100%;padding:0 6px;
}
.nb{font-size:0.75rem;line-height:1;opacity:0.9}
.nl{font-size:0.68rem;font-weight:600;text-align:center;line-height:1.35;letter-spacing:0.02em;color:inherit}
.nt{font-size:0.58rem;opacity:0.5;font-weight:400;font-variant-numeric:tabular-nums;letter-spacing:0.03em}
.node-pending{border-color:rgba(255,255,255,0.08);background:rgba(255,255,255,0.03);color:rgba(255,255,255,0.25)}
.node-running{
  border-color:rgba(129,140,248,0.6);background:rgba(99,102,241,0.15);color:#a5b4fc;
  box-shadow:0 0 12px rgba(99,102,241,0.25);
  animation:gpulse 1.6s ease-in-out infinite
}
.node-done{border-color:rgba(52,211,153,0.5);background:rgba(16,185,129,0.15);color:#6ee7b7}
.node-error{border-color:rgba(248,113,113,0.5);background:rgba(239,68,68,0.15);color:#fca5a5}
.arr{color:rgba(255,255,255,0.15);font-size:1rem;padding:0 6px;flex-shrink:0;align-self:center;line-height:1}
.par-wrap{display:flex;align-items:center;flex-shrink:0}
.par-col{display:flex;flex-direction:column;gap:6px;flex-shrink:0}
@keyframes gpulse{
  0%,100%{box-shadow:0 0 8px rgba(99,102,241,0.2)}
  50%{box-shadow:0 0 18px rgba(99,102,241,0.35)}
}
</style>
"""

    # Build fork/join connectors with branching lines
    fork_join = f"""
<div class="par-wrap">
  <svg width="28" height="120" viewBox="0 0 28 120" fill="none" xmlns="http://www.w3.org/2000/svg">
    <line x1="0" y1="60" x2="14" y2="60" stroke="rgba(255,255,255,0.2)" stroke-width="1.5"/>
    <line x1="14" y1="20" x2="14" y2="100" stroke="rgba(255,255,255,0.2)" stroke-width="1.5"/>
    <line x1="14" y1="20" x2="28" y2="20" stroke="rgba(255,255,255,0.2)" stroke-width="1.5"/>
    <line x1="14" y1="60" x2="28" y2="60" stroke="rgba(255,255,255,0.2)" stroke-width="1.5"/>
    <line x1="14" y1="100" x2="28" y2="100" stroke="rgba(255,255,255,0.2)" stroke-width="1.5"/>
  </svg>
  {par_nodes}
  <svg width="28" height="120" viewBox="0 0 28 120" fill="none" xmlns="http://www.w3.org/2000/svg">
    <line x1="28" y1="60" x2="14" y2="60" stroke="rgba(255,255,255,0.2)" stroke-width="1.5"/>
    <line x1="14" y1="20" x2="14" y2="100" stroke="rgba(255,255,255,0.2)" stroke-width="1.5"/>
    <line x1="14" y1="20" x2="0" y2="20" stroke="rgba(255,255,255,0.2)" stroke-width="1.5"/>
    <line x1="14" y1="60" x2="0" y2="60" stroke="rgba(255,255,255,0.2)" stroke-width="1.5"/>
    <line x1="14" y1="100" x2="0" y2="100" stroke="rgba(255,255,255,0.2)" stroke-width="1.5"/>
  </svg>
</div>
"""

    html = css + '<div class="pipeline">'
    html += node("_input",       "Input",        state_override=input_state)
    html += arrow()
    html += node("shot_detection", "Shot Detect")
    html += fork_join
    html += node("screenwriter", "Screenwriter")
    html += arrow()
    html += node("editor",       "Editor")
    html += arrow()
    html += node("_output",      "Output",       state_override=output_state)
    html += '</div>'
    return html


# ── Button handlers ────────────────────────────────────────────
if run_clicked and not st.session_state.running:
    err = start_pipeline(
        video_path, audio_path, instruction, video_type, main_character, srt_path, target_length, shot_length,
        video_analysis_model, video_analysis_endpoint, video_analysis_api_key,
        audio_litellm_model, audio_litellm_base_url, audio_litellm_api_key,
        agent_litellm_model, agent_litellm_url, agent_litellm_api_key
    )
    if err:
        st.session_state["start_error"] = err
    st.rerun()

if stop_clicked and st.session_state.running:
    stop_pipeline()
    st.rerun()
# ── Status badge ───────────────────────────────────────────────
def status_badge() -> str:
    if st.session_state.pipeline_failed:
        return '<span class="vca-badge vca-badge-error">Failed</span>'
    if st.session_state.running:
        return '<span class="vca-badge vca-badge-running">Running…</span>'
    if st.session_state.log_lines:
        return '<span class="vca-badge vca-badge-done">Done</span>'
    return '<span class="vca-badge vca-badge-idle">Idle</span>'


# ── Log formatter ──────────────────────────────────────────────
STAGE_KEYWORDS = [
    "[Thread A]", "[Thread B]", "[Thread C]",
    "Shot detection", "ASR", "Captioning", "Scene",
    "Audio", "Screenwriter", "Editor", "Processing",
]

def format_log_line(line: str) -> str:
    escaped = line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    lower = line.lower()
    if any(k.lower() in lower for k in STAGE_KEYWORDS):
        return f'<span class="vca-stage">{escaped}</span>'
    if "error" in lower or "traceback" in lower or "exception" in lower:
        return f'<span class="vca-error">{escaped}</span>'
    if "complete" in lower or "done" in lower or "finished" in lower or "success" in lower:
        return f'<span class="vca-success">{escaped}</span>'
    return escaped


# ── Main area ──────────────────────────────────────────────────
st.markdown(
    f'<h1 style="font-size:2.2rem;font-weight:800;letter-spacing:-0.02em;margin-bottom:0.2rem">'
    f'🎬 CutClaw &nbsp; {status_badge()}</h1>',
    unsafe_allow_html=True,
)
st.markdown("---")

if st.session_state.get("start_error"):
    st.error(f"Failed to start pipeline: {st.session_state.start_error}")
    st.session_state.start_error = None

graph_placeholder = st.empty()

with st.expander("🖥️ Pipeline Logs", expanded=False):
    log_placeholder = st.empty()

def render_graph():
    graph_placeholder.markdown(
        build_graph_html(st.session_state.stage_status, st.session_state.stage_times),
        unsafe_allow_html=True,
    )

def render_log():
    lines_html = "<br>".join(format_log_line(l) for l in st.session_state.log_lines[-500:])
    log_placeholder.markdown(
        f'<div class="vca-log" id="vca-log-box" style="display: flex; flex-direction: column-reverse;">'
        f'<div>{lines_html}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

@st.fragment(run_every=0.5)
def pipeline_monitor():
    if st.session_state.running:
        q = st.session_state.log_queue
        proc = st.session_state.process
        while True:
            try:
                line = q.get_nowait()
            except queue.Empty:
                break
            if line is None:
                proc.wait()
                st.session_state.running = False
                if proc.returncode != 0:
                    st.session_state.pipeline_failed = True
                st.rerun()
                break
            st.session_state.log_lines.append(line)
            parse_stage_from_line(line, st.session_state.stage_status, st.session_state.stage_times)
    render_log()
    render_graph()

pipeline_monitor()
# ── Result section ─────────────────────────────────────────────
st.markdown("---")

if st.session_state.pipeline_failed:
    st.error("Pipeline failed. Check the log above for details.")

elif not st.session_state.running and st.session_state.result_shot_json:
    shot_json = st.session_state.result_shot_json
    abs_shot_json = os.path.join(PROJECT_ROOT, shot_json)

    st.markdown("### ✅ Pipeline Complete")

    RATIOS = ["9:16", "16:9", "1:1"]

    # Derive shot_plan path from shot_point path (same dir, same suffix)
    abs_shot_plan = abs_shot_json.replace("shot_point_", "shot_plan_")
    output_dir = os.path.dirname(abs_shot_json)

    if os.path.exists(abs_shot_plan):
        subtitle_path_for_hook = resolve_hook_subtitle_path(video_path, srt_path)
        action_cols = st.columns([2, 1])
        with action_cols[0]:
            st.caption(
                "Use the button below to manually re-pick the hook dialogue. "
                "Manual re-picks use a random subtitle window instead of always starting from the end."
            )
        with action_cols[1]:
            if st.button(
                "Re-select Hook Dialogue",
                key="reselect_hook_dialogue",
                use_container_width=True,
                disabled=not bool(subtitle_path_for_hook),
            ):
                try:
                    with st.spinner("Re-selecting hook dialogue from a random subtitle window..."):
                        used_subtitle_path = rerun_hook_dialogue_selection(
                            shot_plan_path=abs_shot_plan,
                            video_path=video_path,
                            instruction=instruction,
                            main_character=main_character,
                            srt_path=srt_path,
                        )
                    st.success(f"Hook dialogue updated using subtitles from {used_subtitle_path}")
                except Exception as exc:
                    st.error(f"Failed to re-select hook dialogue: {exc}")
        if not subtitle_path_for_hook:
            st.warning("No subtitle file was found for hook dialogue re-selection.")

    def output_path(ratio: str) -> str:
        safe = ratio.replace(":", "x")
        return os.path.join(output_dir, f"output_{safe}.mp4")

    ending_video = os.path.join(PROJECT_ROOT, "resource", "ending", "ending.mp4")
    dialogue_font = os.path.join(PROJECT_ROOT, "resource", "font", "Pulp Fiction Italic M54.ttf")

    def run_render(ratio: str) -> bool:
        out = output_path(ratio)
        cmd = [
            "python", "render/render_video.py",
            "--shot-plan", abs_shot_plan,
            "--shot-json", abs_shot_json,
            "--video", video_path,
            "--audio", audio_path,
            "--output", out,
            "--crop-ratio", ratio,
            "--no-labels",
            "--render-hook-dialogue",
        ]
        if os.path.exists(ending_video):
            cmd += ["--ending-video", ending_video]
        if os.path.exists(dialogue_font):
            cmd += ["--dialogue-font", dialogue_font]
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        result = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, capture_output=True, text=True)
        if result.returncode != 0:
            st.error(f"❌ Render {ratio} failed. Check terminal logs for details:\n{result.stderr[-1000:]}")
            return False
        return True

    # Render buttons
    cols = st.columns(3)
    for i, ratio in enumerate(RATIOS):
        with cols[i]:
            if st.button(f"▶ Render {ratio}", key=f"render_{ratio}", use_container_width=True):
                with st.spinner(f"Rendering {ratio}…"):
                    ok = run_render(ratio)
                if ok:
                    st.rerun()

    # Video previews
    rendered = [(r, output_path(r)) for r in RATIOS if os.path.exists(output_path(r))]
    if rendered:
        st.markdown("#### Preview")
        # Width hints per ratio: 9:16 narrow, 16:9 wide, 1:1 square
        _w = {"9:16": 1, "16:9": 3, "1:1": 2}
        # Center narrow videos by padding with empty columns
        for ratio, path in rendered:
            w = _w.get(ratio, 2)
            pad = (6 - w) // 2
            cols = st.columns([pad, w, pad] if pad > 0 else [w])
            with cols[1] if pad > 0 else cols[0]:
                st.caption(ratio)
                st.video(path)
