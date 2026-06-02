# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Is

vcutclaw is an AI-powered video editing pipeline that takes raw long-form footage + BGM music and produces a cinematic short video. It supports both single-video and **batch multi-clip project** workflows. It uses a multi-agent architecture (Screenwriter, Editor, Reviewer) orchestrated via LiteLLM to plan shots, select precise clip timestamps, and render a final music-synced edit.

## Commands

### Environment Setup
```bash
conda activate CutClaw
# Requires: ffmpeg in PATH, torch 2.8.0, decord, litellm, opencv, sentence-transformers, madmom
```

### CLI — Single Video Mode
```bash
# UI (Streamlit)
streamlit run app.py

# Full pipeline
python local_run.py --Video_Path "resource/video/sample.MOV" --Audio_Path "resource/audio/bgm.mp3" --Instruction "editing instruction"

# Preprocess only (no creative generation)
python local_run.py --Video_Path "resource/video/sample.MOV" --type vlog --preprocess-only

# Override any config.py parameter at runtime
python local_run.py ... --config.VIDEO_FPS 2 --config.AUDIO_TOTAL_SHOTS 50
```

### CLI — Batch Project Mode (7-step workflow)
```bash
PROJECT="Output/Projects/<id>/project.json"

# Step 1: Create project from video directory
python local_run.py --project create --video-dir "/path/to/videos" --project-name "My Trip"

# Step 2: Review source media consistency
python local_run.py --project review-sources --project-path "$PROJECT"

# Step 3: Batch preprocess all clips (parallel, checkpoint resume)
python local_run.py --project preprocess --project-path "$PROJECT" --type vlog --max-workers 2

# Step 4: Build global material index
python local_run.py --project build-index --project-path "$PROJECT"

# Step 5: Generate shot plan (auto BGM analysis + LLM planning)
python local_run.py --project plan --project-path "$PROJECT" --profile bilibili_1080p --strategy "travel vlog"

# Step 6: Generate shot points (per-clip LLM timestamp selection)
python local_run.py --project edit --project-path "$PROJECT" --profile bilibili_1080p

# Step 7: Render final video (multi-source renderer)
python local_run.py --project render --project-path "$PROJECT" --profile bilibili_1080p --extract-timeout 600

# Check project status at any time
python local_run.py --project status --project-path "$PROJECT"
```

### Rendering (single-video)
```bash
python render/render_video.py \
  --shot-json "Output/.../shot_point_xxx.json" \
  --shot-plan "Output/.../shot_plan_xxx.json" \
  --video "resource/video/sample.MOV" \
  --audio "resource/audio/bgm.mp3" \
  --output "output/final.mp4" --crop-ratio "9:16"
```

### No Test Suite
There is no test framework (no pytest, no test files). Validate changes by running the pipeline end-to-end on sample media.

## Architecture

### Pipeline Flow
```
Source Video + BGM
    │
    ├─ Preprocessing (local_run.py --preprocess-only / --project preprocess)
    │   ├─ decode_video_to_frames → shot detection (PySceneDetect)
    │   ├─ ASR (whisper_cpp or litellm) — film mode only
    │   ├─ VLM video captioning → per-clip captions
    │   ├─ scene_merge (sentence-transformer similarity)
    │   └─ scene_analysis → scene_summaries_video/
    │
    ├─ Audio Analysis (audio_caption_madmom.py)
    │   └─ madmom beat/keypoint detection → LLM structure analysis → captions.json
    │
    ├─ Screenwriting (Screenwriter_scene_short.py)
    │   └─ select_audio_segment → structure_proposal → shot_plan → hook_dialogue → shot_plan.json
    │
    ├─ Shot Selection (direct_shot_selector.py or core.py)
    │   └─ shot_plan.json → shot_point.json
    │
    └─ Rendering (render_video.py / multi_source_renderer.py)
        └─ shot_point.json + source video + BGM → final .mp4
```

### Three-Agent Roles
- **PlannerAgent** (`src/planner_agent.py`) — "Producer": orchestrates Screenwriter, generates `shot_plan.json`
- **ShortVideoEditor** (`src/short_video_editor.py`) — "Editor": generates `shot_point.json` via DirectShotSelector (single LLM call, default) or EditorCoreAgent (agent loop with tools)
- **ReviewerAgent** (`src/Reviewer.py`) — "Reviewer": validates face quality, aesthetic quality, overlap; used inside the EditorCoreAgent loop

### Core Module Map
| Module | Role |
|--------|------|
| `src/config.py` | All global config (~500 lines). Override via `config_my.py` (imported at end) or `--config.PARAM` CLI |
| `src/core.py` | EditorCoreAgent + ParallelShotOrchestrator + tool functions |
| `src/direct_shot_selector.py` | Single-LLM-call alternative to the agent loop (default) |
| `src/prompt.py` | All LLM prompt templates |
| `src/Screenwriter_scene_short.py` | Multi-step creative pipeline: audio selection → structure proposal → shot plan |
| `render/render_video.py` | ffmpeg-based clip extraction, concatenation, crop, subtitle overlay |
| `render/multi_source_renderer.py` | Multi-source renderer: validate → extract → stitch → BGM → subtitles → ending |
| `render/multi_profile_renderer.py` | Multi-profile priority rendering |

### Batch Editing Modules
| Module | Role |
|--------|------|
| `src/project/project.py` | Project/Clip/Day dataclasses + ProjectManager (scan dir, ffprobe metadata, group by day) |
| `src/project/media_profiles.py` | MediaProfile frozen dataclass + built-in platform profiles (bilibili_4k, douyin, etc.) |
| `src/batch/batch_preprocess.py` | BatchPreprocessor — parallel per-clip preprocessing with checkpoint resume |
| `src/batch/checkpoint.py` | CheckpointManager — per-clip JSON checkpoints for resumable stages |
| `src/batch/material_index.py` | MaterialIndex — flat global scene index across all clips for the planner |
| `src/batch/quality_filter.py` | Clip-level quality scoring (blur, exposure, duration) |
| `src/batch/source_media_review.py` | Project-wide codec/resolution/fps/colorspace consistency audit |
| `src/batch/scene_adapter.py` | SceneEntry → virtual scene JSON for Screenwriter compatibility |
| `src/utils/paths.py` | Shared path utilities (derive_artifact_path, discover_shot_point) |

### Key Data Artifacts
| File | Produced by | Contents |
|------|-------------|----------|
| `shot_plan.json` | Screenwriter | Creative structure, audio segments, per-section shot descriptions |
| `shot_point.json` | Editor | Precise per-shot start/end seconds + reasoning + clip_file_path |
| `captions.json` | audio_caption_madmom | BGM sections with BPM, structure, sub-segments |
| `scene_summaries_video/scene_N.json` | SceneVideoAnalyzer | Per-scene visual description, environment, dialogue |
| `project.json` | ProjectManager | Batch project metadata, clip list, preprocessing progress |
| `material_index.json` | batch/material_index | Flat global scene index across all clips |

### Configuration Override Chain
Priority: `config_my.py` (gitignored local overrides) > `os.environ` > `config.py` defaults > CLI `--config.PARAM_NAME`

### Rendering Path (Multi-Source)
`MultiSourceRenderer` implements: validate → extract → stitch → BGM mix → subtitles → ending video.
- Supports HDR→SDR tonemap (zscale + hable), NTSC frame rate handling, color space detection
- Subtitle overlay via ffmpeg drawtext filter (burn-in)
- BGM looping via stream_loop -1 with volume expression fade
- Checkpoint-based render skip for completed profiles

## 四个原则

### 1. 编码前思考
**不要假设。不要隐藏困惑。呈现权衡。**
- 明确说明假设，呈现多种解释，适时提出异议，困惑时停下来

### 2. 简洁优先
**用最少的代码解决问题。不要过度推测。**
- 不要添加要求之外的功能，不要为一次性代码创建抽象
- 检验标准：资深工程师会觉得这过于复杂吗？如果是，简化。

### 3. 精准修改
**只碰必须碰的。只清理自己造成的混乱。**
- 不要"改进"相邻的代码，不要重构没坏的东西，匹配现有风格
- 检验标准：每一行修改都应该能直接追溯到用户的请求。

### 4. 目标驱动执行
**定义成功标准。循环验证直到达成。**
- 将指令式任务转化为可验证的目标
- 对于多步骤任务，说明一个简短的计划
