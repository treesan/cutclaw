import os

# ==============================================================================
# CutClaw global configuration (beginner-friendly version)
# ------------------------------------------------------------------------------
# This file controls the full editing pipeline:
# 1) Video preprocessing (frame sampling, shot detection, scene analysis)
# 2) Audio analysis (beat/energy/structure)
# 3) Agent generation (Screenwriter + Editor + Reviewer)
#
# Recommended usage:
# - Prefer CLI overrides for experiments instead of changing defaults directly:
#   python local_run.py ... --config.PARAM_NAME VALUE
# - Example: --config.VIDEO_FPS 1 --config.AUDIO_TOTAL_SHOTS 80
# ==============================================================================

# ------------------ UI Remembered Inputs ------------------ #
# These are saved automatically by the app when you change sidebar fields.

VIDEO_PATH = ""
AUDIO_PATH = ""
INSTRUCTION = ""
SRT_PATH = ""


# ------------------ Video Preprocess ------------------ #
# These parameters control how video is sampled, where outputs are written,
# and how much content is processed.


VIDEO_DATABASE_FOLDER = "./Output/"
# Root folder for all intermediate artifacts and output JSON files.
# Changing this affects read/write paths across the whole project.

VIDEO_RESOLUTION = 240
# Target short-side resolution during frame extraction (aspect ratio preserved).
# - Smaller: faster and lower memory usage, but less detail
# - Larger: better detail, but slower

VIDEO_FPS = 2 
# Frame sampling rate during preprocessing (frames per second).
# Typical range: 1~3. Higher FPS gives finer analysis but increases cost/time.

VIDEO_MAX_MINUTES = None
# Maximum video duration to process (minutes). None means full video.
# For quick debugging, setting this to 3~10 is often helpful.

VIDEO_MAX_FRAMES = None if VIDEO_MAX_MINUTES is None else int(VIDEO_MAX_MINUTES * 60 * VIDEO_FPS)
# Auto-computed frame cap when VIDEO_MAX_MINUTES is set.

VIDEO_SAVE_DEBUG_FRAMES = False
# If True, sampled debug frames are written to disk for troubleshooting.
# This increases disk I/O.


# ------------------ Shot Detection ------------------ #
# These parameters determine where shot boundaries are detected.

SHOT_DETECTION_FPS = 2.0
# Sampling FPS specifically for shot detection (independent from VIDEO_FPS).

VIDEO_TYPE = "vlog"
# Global video type: "film" or "vlog".
# Note: local_run.py overrides this with CLI --type.

if VIDEO_TYPE == "film":
    SHOT_DETECTION_THRESHOLD = 3.0
    SHOT_DETECTION_MIN_SCENE_LEN = 3
elif VIDEO_TYPE == "vlog":
    SHOT_DETECTION_THRESHOLD = 0.5       # ↓ 从 1.5 改为 0.5 — DJI 航拍缓慢运动检测
    SHOT_DETECTION_MIN_SCENE_LEN = 15     # ↓ 从 45 改为 15 — 42s 短片装不下 22.5s 最小场景
else:
    # Fallback defaults: use film settings.
    SHOT_DETECTION_THRESHOLD = 3.0
    # For scenedetect: lower = more cuts, higher = more conservative.
    SHOT_DETECTION_MIN_SCENE_LEN = 3
    # Minimum shot length (in frames), used by some detectors.

# Frame skip for shot detection (sampling step).
# 0 = every frame, 1 = every 2nd frame, 29 = every 30th frame.
# Lower values give finer detection but slower processing.
# For 60fps DJI: 14 → 4 帧/秒采样（vs 默认 29 → 2 帧/秒）,
# 可捕捉航拍缓慢运动中的微镜头变化。
FRAME_SKIP = 14   # ↓ 从 29 改为 14 — 提高航拍素材的镜头检测精度

SHOT_DETECTION_SCENES_PATH = "shot_scenes.txt"
# Output filename for shot boundary results (usually no need to change).

SHOT_DETECTION_MODEL = "scenedetect"
# Options: "autoshot", "transnetv2", "Qwen3VL", "scenedetect".
# Beginners should start with the default: scenedetect.



CLIP_SECS = 30
# Maximum length (seconds) of a single candidate clip.

MERGE_SHORT_SCENES = True
# If True, consecutive short scenes are merged to reduce fragmentation.

SCENE_MERGE_METHOD = "min_length"
# Scene merging strategy:
# - min_length: prioritize avoiding overly short scenes
# - max_length: prioritize not exceeding max scene duration

SCENE_MIN_LENGTH_SECS = 3
# Minimum scene length in seconds when SCENE_MERGE_METHOD="min_length".

SCENE_SIMILARITY_THRESHOLD = 0.5
# Similarity threshold for scene segmentation:
# - Lower: harder to split (longer scenes)
# - Higher: easier to split (more fragmented scenes)

MAX_SCENE_DURATION_SECS = 300
# Maximum allowed scene duration before forced split.

MIN_SCENE_DURATION_SECS = 30.0
# Scenes shorter than this are candidates for merging into neighboring scenes.

WHOLE_VIDEO_SUMMARY_BATCH_SIZE = 50
# Number of clips per batch for whole-video summarization.
# Affects parallelism and throughput.


# ═══════════════════════════════════════════════════════════════════════════════
# ASR (Speech Recognition)
# Converts dialogue to subtitles and can optionally run speaker diarization.
# Recommendation: use one backend at a time (local whisper_cpp or cloud litellm).
# ═══════════════════════════════════════════════════════════════════════════════

ASR_BACKEND = "litellm"
# Options: "whisper_cpp" (local) | "litellm" (cloud).
# - Local: lower cost/offline, speed depends on hardware
# - Cloud: easier setup, but incurs API cost

ASR_LANGUAGE = "zh"
# Recognition language. Example values: "English", "Chinese", "en", "zh".
# Set None for auto-detection.

# ───────────────────────────────────────────────────────────────────────────────
# Option 1: whisper.cpp (local)
# ───────────────────────────────────────────────────────────────────────────────

ASR_DEVICE = "cuda:0" if __import__('torch').cuda.is_available() else "cpu"
# Device for local ASR. Uses cuda:0 when available, otherwise cpu.

ASR_WHISPER_CPP_MODEL = "base"  # "base" 支持中文，"base.en" 只支持英文
# whisper.cpp model name or local ggml model path (e.g., "base.en", "large-v3").

ASR_WHISPER_CPP_N_THREADS = 8
# CPU inference thread count (relevant mainly for CPU runs).

ASR_ENABLE_DIARIZATION = True
# Enable speaker diarization (who is speaking).
# Improves dialogue understanding for films but adds runtime.

ASR_DIARIZATION_MODEL_PATH = "pyannote/speaker-diarization-community-1"
# HuggingFace model name/path for diarization.

ASR_MERGE_SAME_SPEAKER = True
# Merge adjacent subtitle segments from the same speaker.

ASR_MERGE_GAP = 1.0
# Maximum gap (seconds) to merge adjacent segments from same speaker.

# ───────────────────────────────────────────────────────────────────────────────
# Option 2: LiteLLM (cloud ASR)
# ───────────────────────────────────────────────────────────────────────────────

ASR_LITELLM_MAX_SEGMENT_MB = 1.0
# Max size (MB) per uploaded audio segment for cloud ASR.
# Helps avoid oversized requests.

ASR_LITELLM_BATCH_SIZE = 128
# Number of audio segments per request batch.
# Larger batches improve throughput but may increase rate-limit risk.

# ------------------ Video Understanding Model ------------------ #

SCENE_PROMPT_TYPE = VIDEO_TYPE
# Prompt style for scene analysis. Usually kept consistent with VIDEO_TYPE.

VIDEO_ANALYSIS_MODEL_MAX_TOKEN = 16384 
# Max output token count for the video analysis model.

VIDEO_ANALYSIS_MODEL = "openai/modelname"
# Video semantic analysis model name (called via OpenAI-compatible endpoint).

VIDEO_ANALYSIS_ENDPOINT = ""
# API base URL for the video analysis model.

VIDEO_ANALYSIS_API_KEY = os.environ.get("VIDEO_ANALYSIS_API_KEY", "")
# API key for the video analysis model. Set as env var VIDEO_ANALYSIS_API_KEY.

CAPTION_BATCH_SIZE = 64
# Batch size for parallel clip captioning/analysis.

SCENE_ANALYSIS_MIN_FRAMES = 6
# Minimum number of sampled frames per scene.
# Higher values may improve stability but increase runtime.


# ------------------ Audio Model ------------------ #
# Analyzes musical beat/energy/structure and outputs editing keypoints.

AUDIO_LITELLM_MODEL = "openai/modelname"
# Cloud model used for audio captioning and structure analysis.

AUDIO_LITELLM_API_KEY = os.environ.get("AUDIO_LITELLM_API_KEY", "")
# API key for the audio model. Set as env var AUDIO_LITELLM_API_KEY.

AUDIO_LITELLM_BASE_URL = ""
# API base URL for the audio model.

AUDIO_DETECTION_METHODS = ["downbeat", "pitch", "mel_energy"]
# Keypoint detection methods (single or combined):
# - downbeat: rhythm/beat structure
# - pitch: melodic variation
# - mel_energy: energy peaks

# ----- Downbeat -----
AUDIO_BEATS_PER_BAR = 4
# Beats per bar (e.g., 4 for 4/4 time).

AUDIO_DBN_THRESHOLD = 0.05
# Activation threshold for downbeat tracking.

AUDIO_MIN_BPM = 55.0
AUDIO_MAX_BPM = 215.0
# BPM search range preset (may be weakly used in some paths).

# ----- Pitch -----
AUDIO_PITCH_TOLERANCE = 0.8
# Pitch matching tolerance.

AUDIO_PITCH_THRESHOLD = 0.8
# Confidence threshold for keeping pitch keypoint candidates.
# Higher value = stricter filtering.

AUDIO_PITCH_MIN_DISTANCE = 0.3
# Minimum spacing (seconds) between pitch keypoints.

AUDIO_PITCH_NMS_METHOD = "basic"
# Non-maximum suppression strategy: "basic" / "adaptive" / "window".

AUDIO_PITCH_MAX_POINTS = 50
# Maximum number of retained pitch keypoints.

# ----- Mel Energy -----
AUDIO_MEL_WIN_S = 512
# FFT window size.

AUDIO_MEL_N_FILTERS = 40
# Number of mel filters.

AUDIO_MEL_THRESHOLD_RATIO = 0.3
# Threshold ratio for energy peak detection.

AUDIO_MEL_MIN_DISTANCE = 0.3
# Minimum spacing (seconds) between mel-energy keypoints.

AUDIO_MEL_NMS_METHOD = "basic"
# Non-maximum suppression strategy: "basic" / "adaptive" / "window".

AUDIO_MEL_MAX_POINTS = 50
# Maximum number of retained mel-energy keypoints.

# ----- Keypoint post-processing (denoising/sparsification) -----
AUDIO_MERGE_CLOSE = 0
# [May be unused in main path] Merge very close keypoints.

AUDIO_MIN_INTERVAL = 0
# Global minimum keypoint interval (seconds).

AUDIO_TOP_K = 0
# Keep only top-K strongest keypoints. 0 means unlimited.

AUDIO_ENERGY_PERCENTILE = 0
# Keep only keypoints above this energy percentile (0~100).

AUDIO_SILENCE_THRESHOLD_DB = -45.0
# Silence filtering threshold (dB).
# Segments below this level are treated as too quiet and filtered.

# ----- Audio segment duration constraints (frequently tuned) -----
AUDIO_MIN_SEGMENT_DURATION = 1.8
# Minimum segment duration (seconds). 调高到 0.5 避免 LLM 对超短片段返回 "can't hear"

AUDIO_MAX_SEGMENT_DURATION = 3.8
# Maximum segment duration (seconds). Larger values create slower pacing.

# ----- Music structure analysis (Level-1) -----
AUDIO_STRUCTURE_TEMPERATURE = 0.7
AUDIO_STRUCTURE_TOP_P = 0.95
AUDIO_STRUCTURE_MAX_TOKENS = 4096
# Controls generation style and token limit for structure analysis.

# ----- Section-aware shot allocation -----
AUDIO_USE_STAGE1_SECTIONS = True
# Use Level-1 structural sections to guide keypoint filtering.

AUDIO_SECTION_MIN_INTERVAL = AUDIO_MIN_SEGMENT_DURATION
# Global minimum keypoint interval across sections.

AUDIO_TOTAL_SHOTS = 30
# 目标关键点总数。减小数值 → 段落更长 → LLM 识别率更高。
# 原值 200 时 90s 音乐每段仅 0.45s，LLM 成功率 ~29%。若改为 35 后每段 ~2.6s，成功率明显提升。
# Target total shot count, allocated proportionally by sections.
# For quick debugging, try reducing this to 30~80.

# ----- Multi-feature fusion weights -----
AUDIO_WEIGHT_DOWNBEAT = 1.0
AUDIO_WEIGHT_PITCH = 1.0
AUDIO_WEIGHT_MEL_ENERGY = 1.0
# Fusion weights for downbeat/pitch/mel-energy signals.
# Defaults use equal weighting.

# ----- Keypoint caption analysis (Level-2) -----
AUDIO_BATCH_SIZE = 3
# Batch size for parallel audio-segment inference.
# Larger values increase throughput but use more resources.

AUDIO_KEYPOINT_TEMPERATURE = 0.7
AUDIO_KEYPOINT_TOP_P = 0.95
AUDIO_KEYPOINT_MAX_TOKENS = 4096
# Controls style and maximum length for keypoint caption generation.

# ------------------ Agent Runtime ------------------ #

AGENT_MODEL_MAX_TOKEN = 8192
# Maximum generated tokens per agent response (not total context size).

AGENT_MODEL_MAX_RETRIES = 4
# Max retries per agent step when model calls fail.

AGENT_RATE_LIMIT_BACKOFF_BASE = 1.0
AGENT_RATE_LIMIT_MAX_BACKOFF = 8.0
# Backoff timing (seconds) when rate limits occur.

AUDIO_SEGMENT_MIN_DURATION_SEC = 10.0
AUDIO_SEGMENT_MAX_DURATION_SEC = 20.0
# Allowed music-span duration range for short-video planning.

AUDIO_SEGMENT_SELECTION_MAX_RETRIES = 3
# Retry count when selected music span is invalid.

AUDIO_SEGMENT_TIME_TOLERANCE_SEC = 0.25
# Allowed timestamp drift (seconds) when validating selected spans.

ENABLE_TRIM_SHOT_CHARACTER_ANALYSIS = True
# Enable VLM character analysis during trim_shot.

CORE_MAX_FRAMES = 30
# Maximum sampled frames per clip for core + reviewer analysis.

VLM_FACE_MAX_FRAMES = 24
# Maximum frames sampled during VLM face quality check.
# Can be lower than CORE_MAX_FRAMES since face detection only needs ~2-3 FPS.

EDITOR_MAX_ITERATIONS = 10
# Maximum Agent loop iterations per shot for the EditorCoreAgent.
# The model typically converges in 5-8 rounds; 10 provides ample headroom.

AGENT_LITELLM_URL = ""
# API base URL for the agent LLM.

AGENT_LITELLM_API_KEY = os.environ.get("AGENT_LITELLM_API_KEY", "")
# API key for the agent LLM. Set as env var AGENT_LITELLM_API_KEY.

AGENT_LITELLM_MODEL = "openai/modelname"
# Primary model for the agent.DirectShotSelector 使用的模型

# ───────────────────────────────────────────────────────────────────────────────
# DirectShotSelector (直接推理替代 EditorCoreAgent Agent 循环)
# ───────────────────────────────────────────────────────────────────────────────

DIRECT_SHOT_SELECTOR_ENABLED = True
# 主开关：True 用 DirectShotSelector 直推，False 用原有的 EditorCoreAgent 循环。

DIRECT_SHOT_SELECTOR_MAX_TOKENS = 8192
# 直推输出的最大 token 数。

DIRECT_SHOT_SELECTOR_TEMPERATURE = 0.3
# 直推温度，低温度保一致性。

PARALLEL_SHOT_ENABLED = True
# Whether to enable parallel shot selection (ParallelShotOrchestrator) in film mode.

PARALLEL_SHOT_MAX_WORKERS = 4
# Number of parallel workers.

PARALLEL_SHOT_MAX_RERUNS = 2
# Maximum rerun rounds for conflicted shots.



# ------------------ Reviewer (Quality Checks) ------------------ #

ENABLE_REVIEWER = True
# Master switch for the Reviewer agent. Set to False to skip all review checks (face quality, duplicate detection, etc.).

# ═══════════════════════════════════════════════════════════════════════════════
# Quality Validation (face / protagonist / aesthetic checks)
# ═══════════════════════════════════════════════════════════════════════════════

FACE_QUALITY_CHECK_MODE = "auto"
# 验证模式 (quality validation mode):
#   "auto"    - 自动采样 scene_summaries 判断内容类型
#              有人物场景 → strict，无人物场景 → relaxed
#   "strict"  - 严格模式：主角出镜率 ≥ MIN_PROTAGONIST_RATIO，主角框 ≥ VLM_MIN_BOX_SIZE
#   "relaxed" - 跳过 face check，只做 shot 不重叠检查
#   "off"     - 跳过所有质量验证（face + aesthetic + overlap）
# 旅行/Vlog/航拍推荐 "relaxed" 或 "auto"

# --- 以下配置仅 strict 模式生效 ---

VLM_FACE_LOG_EACH_FRAME = False
# Print per-frame protagonist detection logs (very verbose; debug only).

ENABLE_AESTHETIC_QUALITY_CHECK = True
# Toggle aesthetic quality checks for vlog mode (may not be fully implemented).

FACE_QUALITY_CHECK_METHOD = "vlm"
# Quality check method (currently only "vlm" is supported).

# ------------------ Protagonist Presence Constraints ------------------ #

MAIN_CHARACTER_NAME = ""
# Main character / target subject name (comma-separated for multiple roles).
# Auto-detected from system home directory (e.g. /Users/tree -> "Tree").
# This is one of the highest-impact parameters in object mode.

MIN_PROTAGONIST_RATIO = 0.7
# Minimum ratio (0~1) of frames where the protagonist should be the main focus.

VLM_MIN_BOX_SIZE = 100
# Minimum protagonist bounding-box size in pixels. Smaller detections are ignored.


# ------------------ Shot Selection Constraints ------------------ #
# These parameters define fallback behavior when perfect matches are unavailable.

MIN_ACCEPTABLE_SHOT_DURATION = 2.0
# Minimum acceptable final shot duration (seconds).
# Smaller values increase match rate but may produce more fragmented edits.

ALLOW_DURATION_TOLERANCE = 1.0
# Allow duration deviation of ±N seconds from target.

ALLOW_CONTENT_MISMATCH = True
# Allow semantically similar (not exact) content matches.

ENABLE_FALLBACK_STRATEGY = True
# Enable multi-level fallback strategy to reduce hard failures.

SCENE_EXPLORATION_RANGE = 3
# Extra exploration range around recommended scenes (±N scenes).
# Example: if recommended scene is 8 and range=3, search scene 5~11.
# Set to 0 to strictly limit selection to recommended scenes only.

# ── 色彩空间配置 ────────────────────────────────────────────────
# 设备默认模式: DJI(Normal) = BT.709, Nikon(Z6ii) = BT.709, iPhone(SDR) = BT.709
# HDR 模式:     DJI(HLG) = BT.2020+HLG, iPhone(HDR) = BT.2020+PQ
# 社交媒体平台大多不支持 HDR，建议统一到 BT.709

TARGET_COLOR_SPACE = "bt709"
# 目标色彩空间: "bt709" (SDR, 通用兼容) 或 "bt2020_hlg" (HDR, B站等支持)
# bt709: 所有 HDR 内容自动转换为 SDR，零质量损失于原生 BT.709 素材
# bt2020_hlg: 所有 SDR 内容上转换为 HDR，适合 HDR 显示设备

COLOR_SPACE_CONVERSION = "auto"
# 色彩空间转换策略:
# "auto" — 检测源文件色彩空间，自动转换到 TARGET_COLOR_SPACE
# "skip" — 不做色彩空间转换（保持源文件原始色彩空间）

# ── 结尾视频配置 ───────────────────────────────────────────────
# 用于在成片末尾自动拼接一段固定片尾（logo/版权/订阅提示等）

ENDING_VIDEO_PATH = "resource/ending/ending.mp4"
# 默认结尾视频路径。文件不存在时自动跳过，不会中断渲染。
# 设为空字符串 "" 禁用结尾拼接。

ENDING_VIDEO_DURATION = 0.0
# 截取结尾视频的时长（秒）。0 = 使用结尾视频的完整时长。
# 例如设 3.0 表示只截取前 3 秒。

ENDING_FADE_DURATION = 0.5
# 主体视频与结尾视频之间的交叉淡入淡出时长（秒）。

# ── 媒体 Profile 覆盖 ─────────────────────────────────────────────
# 修正 (优化 #9): 在 config_my.py 中设置可覆盖内置 profile 的 preset/crf，
# 用于生产画质（慢预设 + 低 crf）vs 快速测试（ultrafast + 高 crf）。

# MEDIA_PROFILE_PRESET_OVERRIDE = "slow"     # 覆盖 preset
# MEDIA_PROFILE_CRF_OVERRIDE = 16            # 覆盖 crf
MEDIA_PROFILE_PRESET_OVERRIDE = None   # None = 使用 profile 内置值
MEDIA_PROFILE_CRF_OVERRIDE = None      # None = 使用 profile 内置值


# ───────────────────────────────────────────────────────────────────────────────
# 本地覆写层（config_my.py）
# 如果存在 config_my.py，其中的值会覆盖上方所有默认值。
#
# 优先级链：config_my.py > os.environ.get() > config.py 默认值
#
# 使用方式：
#   cp src/config_my.example.py src/config_my.py
#   然后填入你的 API Key 和自定义配置。
#   config_my.py 已在 .gitignore 中，不会被提交。
# ───────────────────────────────────────────────────────────────────────────────
try:
    from .config_my import *  # type: ignore[import-untyped]  # noqa: F401, F403
except (ImportError, ModuleNotFoundError):
    pass


