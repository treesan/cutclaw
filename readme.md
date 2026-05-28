<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="asset/CutClaw_dark.png" />
  <source media="(prefers-color-scheme: light)" srcset="asset/CutClaw_light.png" />
  <img src="asset/Cutclaw_light.png" alt="CutClaw teaser" width="50%" />
</picture>

## 🦞CutClaw-Batch: Agentic Batch Long-Form Video Editing System with Music Synchronization

#### <a href='https://zzsf11.github.io/'>Shifang Zhao</a> · <a href='https://scholar.google.com/citations?hl=zh-CN&user=UP2IgWIAAAAJ'>Yihan Hu</a> · <a href='https://scholar.google.com/citations?user=4oXBp9UAAAAJ&hl=zh-CN'>Ying Shan</a> · <a href='https://scholar.google.com/citations?user=qL9Csv0AAAAJ&hl=zh-CN'>Yunchao Wei</a> · <a href='http://vinthony.github.io/'>Xiaodong Cun</a> 
###### Beijing Jiaotong University · GVC Lab, Great Bay University · Tencent ARC Lab

**🎬 Your personal editor for turning hours of footage into cinematic montages.**

<p align="center">
  <img src="https://img.shields.io/badge/🎞️_Hours--Long_Footage-1f6feb?style=flat-square" alt="Hours-Long Footage" />
  <img src="https://img.shields.io/badge/🎵_Music--Beat_Sync-00b894?style=flat-square" alt="Music Beat Sync" />
  <img src="https://img.shields.io/badge/✍️_Instruction--Following-f59f00?style=flat-square" alt="Instruction Following" />
  <img src="https://img.shields.io/badge/🖱️_One--Click_Editing-e17055?style=flat-square" alt="One-Click Editing" />
  <img src="https://img.shields.io/badge/🔌_LiteLLM--Powered-6c5ce7?style=flat-square" alt="LiteLLM Powered" />
</p>

<p>
	<a href="readme.md"><img src="https://img.shields.io/badge/English-1a1a2e"></a>
    <a href="readme_zh.md"><img src="https://img.shields.io/badge/中文版-1a1a2e"></a>
	<a href="https://arxiv.org/abs/2603.29664"><img src="https://img.shields.io/badge/arXiv-paper-b31b1b.svg"></a>
	<a href="https://github.com/GVCLab/CutClaw"><img src="https://img.shields.io/github/stars/GVCLab/CutClaw?style=social"></a>
</p>

[Overview](#-overview) • [Roadmap](#-roadmap) • [Features](#-key-features) • [Gallery](#️-gallery) • [Quick Start](#-quick-start) • [CLI Reference](#-cli-reference) • [Troubleshooting](#️-troubleshooting) • [Citation](#-citation) • [Star History](#-star-history)

</div>

---

## 💡 Overview

CutClaw is an end-to-end editing system for long-form footage + music.

It first deconstructs raw video/audio into structured captions, then uses a multi-agent pipeline to plan shots (`shot_plan`), select clip timestamps (`shot_point`), and validate final quality before rendering.

![CutClaw Pipeline](asset/method.png)

---

## 🗺️ Roadmap

> We warmly welcome new issues and ideas from the community. If you have suggestions, please open an issue. Your feedback will help shape our future plans and be the fuel that helps this project take off. 🔥

### Short-Term Goals

> What we're building next for faster, cheaper, and more expressive video editing.

- [ ] 🧩 **ARC-Chapter Integration**  
  Bring in [ARC-Chapter](https://github.com/TencentARC/ARC-Chapter) to reduce the cost of long-form footage deconstruction.
- [ ] 💸 **Low-Cost Mode**  
  Add a budget-friendly mode that proactively reads only relevant footage instead of fully processing all source material.
- [ ] 🎙️ **Talking-Head + Visual Mixing**  
  Introduce hybrid editing logic that coordinates narration-driven clips with supporting visual footage.

### Long-Term Goals

> Broader product and ecosystem directions for the next stage of CutClaw.

- [ ] ✍️ **Playwriter Upgrade**  
  Expand the Playwriter with richer editing patterns and more diverse visual storytelling methods.
- [ ] 🔌 **Claude Code MCP Support**  
  Adapt CutClaw to work smoothly within Claude Code MCP workflows.
- [ ] 🌐 **Online Service Interface**  
  Build a web-based service interface for easier access and deployment.

---

## ✨ Key Features

<table align="center" width="100%" style="border: none; table-layout: fixed;">
<tr>
<td width="25%" align="center" valign="top" style="padding: 16px;">

### 🎬 **One-Click Deconstruction**

<img src="https://img.shields.io/badge/LONG--FORM%20PROCESSING-4c6ef5?style=for-the-badge" alt="Long-Form Processing" />

Effortlessly transforms hours-long raw video and audio into structured, searchable assets with a single click.

</td>
<td width="25%" align="center" valign="top" style="padding: 16px;">

### 🎯 **Instruction Control**

<img src="https://img.shields.io/badge/TEXT%20TO%20EDIT-f59f00?style=for-the-badge" alt="Text to Edit" />

Requires only one text instruction to steer the editing style—easily generating fast-paced character montages or slow-paced emotional narratives.

</td>
<td width="25%" align="center" valign="top" style="padding: 16px;">

### 📱 **Smart Auto-Cropping**

<img src="https://img.shields.io/badge/SMART%20ADAPTATION-12b886?style=for-the-badge" alt="Smart Adaptation" />

Content-aware cropping automatically identifies core subjects and adjusts aspect ratios to fit various social platforms.

</td>
<td width="25%" align="center" valign="top" style="padding: 16px;">

### 🎵 **Music-Aware Sync**

<img src="https://img.shields.io/badge/AUDIO%20SYNC-e64980?style=for-the-badge" alt="Audio Sync" />

Extracts musical beats and energy signals to build rhythm-aware cuts that perfectly match the music's pacing.

</td>
</tr>
</table>

---


## 🖼️ Gallery（remember to turn on the audio）
……
----

## 🚀 Quick Start

### 1. Install

```bash
git clone https://github.com/GVCLab/CutClaw.git
cd CutClaw
conda create -n CutClaw python=3.12
conda activate CutClaw
pip install -r requirements.txt
```

> We strongly recommend the GPU-accelerated Decord/NVDEC build for faster video decoding. Build from [source](https://github.com/dmlc/decord?tab=readme-ov-file#install-from-source).

### 2. Add your files

```
resource/
├── video/      ← put your .mp4 / .mkv here
├── audio/      ← put your .mp3 / .wav here
└── subtitle/   ← optional .srt (skips ASR, saves time)
```

### 3. Run

**UI (recommended)**

```bash
streamlit run app.py
```
Then open `http://localhost:8501` in your browser. (*If `http://localhost:8501` does not work well, try `http://127.0.0.1:8501`)

![CutClaw UI demo](asset/UI.png)

> Place your footage in the paths above, then you can directly select those files in the UI.

Model selection guidance:

- **Video model**
  - **Role**: shot/scene understanding and visual captioning.
  - **Recommended**: Gemini-3, Qwen3.5, GPT-5.3

- **Audio model**
  - **Role**: ASR plus music-structure parsing (beat/downbeat, pitch, energy) for music-aware segmentation.
  - **Recommended**: Gemini-3

- **Agent model**
  - **Role**: drives the Screenwriter + Editor + Reviewer loop to generate `shot_plan` and `shot_point`.
  - **Recommended**: MiniMax-2.7, Kimi-2.5, Claude-4.5

We leverage `LiteLLM` as the api manager gateway, the typical Model name is e.g. 'openai/MiniMax-2.7' which means using openai protocol to call the given model, more information see [LiteLLM documents](https://github.com/BerriAI/litellm).


<details>
<summary><strong>CLI (advanced)</strong></summary>

```bash
python local_run.py \
  --Video_Path "resource/video/xxxx.mp4" \
  --Audio_Path "resource/audio/xxxx.mp3" \
  --Instruction "xxxx"
```

<details>
<summary>Common config overrides</summary>

Any `src/config.py` parameter can be overridden with `--config.PARAM_NAME VALUE`.

| Parameter | Default | Effect |
|---|---|---|
| `VIDEO_PATH` | `"resource/video/The_Dark_Knight.mkv"` | Default input video path used by UI remembered inputs |
| `AUDIO_PATH` | `"resource/audio/Way_Down_We_Go.mp3"` | Default input audio path used by UI remembered inputs |
| `INSTRUCTION` | `"Joker's crazy that want to change the world."` | Default editing instruction prompt |
| `ASR_BACKEND` | `"litellm"` | ASR engine (`litellm` cloud or `whisper_cpp` local) |
| `VIDEO_FPS` | `2` | Sampling FPS for preprocessing |
| `MAIN_CHARACTER_NAME` | `"Joker"` | Protagonist name for character-focused edits |
| `AUDIO_MIN_SEGMENT_DURATION` | `3.0` | Minimum beat segment duration (seconds) |
| `AUDIO_MAX_SEGMENT_DURATION` | `5.0` | Maximum beat segment duration (seconds) |
| `AUDIO_DETECTION_METHODS` | `["downbeat", "pitch", "mel_energy"]` | Audio keypoint detection methods |
| `PARALLEL_SHOT_MAX_WORKERS` | `4` | Parallel shot selection workers |

Example:

```bash
python local_run.py \
  --Video_Path "resource/video/xxxx.mp4" \
  --Audio_Path "resource/audio/xxxx.mp3" \
  --Instruction "xxxx" \
  --config.MAIN_CHARACTER_NAME "Batman" \
  --config.VIDEO_FPS 2 \
  --config.AUDIO_TOTAL_SHOTS 50
```

</details>



Then render manually:

```bash
python render/render_video.py \
  --shot-plan  "Output/<video_audio>/shot_plan_*.json" \
  --shot-json  "Output/<video_audio>/shot_point_*.json" \
  --video  "resource/video/xxxx.mp4" \
  --audio  "resource/audio/xxxx.mp3" \
  --output "output/final.mp4" \
  --crop-ratio "9:16" \
  --no-labels --render-hook-dialogue
```

</details>

---

## 🚀 CLI Quick Reference

All commands must be run from the CutClaw project directory with the correct conda environment:

```bash
cd ~/Develop/CutClaw
conda activate CutClaw
```

### 1. Video Preprocessing 

Analyze video without BGM (used when BGM is not yet selected):

```bash
python local_run.py \
  --Video_Path "resource/video/sample.MOV" \
  --Instruction "video analysis only" \
  --type vlog \
  --preprocess-only
```

**Output:** `Output/Video/{VIDEO_ID}/captions/scene_summaries_video/` + `shot_scenes.txt`

### 2. BGM Rhythm Analysis 

Analyze BGM structure (after the content strategist has downloaded the BGM):

```bash
python -c "
from src.audio.audio_caption_madmom import caption_audio_with_madmom_segments

caption_audio_with_madmom_segments(
    audio_path='resource/audio/bgm.mp3',
    output_path='Output/Audio/{BGM_ID}/captions/captions.json',
)
"
```

**Output:** `Output/Audio/{BGM_ID}/captions/captions.json` (BPM, structure segments, keypoints)

### 3. Combined Video + BGM Preprocessing

Run both video and BGM analysis together (preprocess only, no creative generation):

```bash
python local_run.py \
  --Video_Path "resource/video/sample.MOV" \
  --Audio_Path "resource/audio/bgm.mp3" \
  --Instruction "preprocess only" \
  --type vlog \
  --preprocess-only
```

### 4. BGM Download (Pixabay)

Search and download BGM from Pixabay (free commercial use, no API key required):

```bash
# Search
python3 ~/.openclaw/skills/pixabay-music-skill/scripts/pixabay_music.py \
  search "upbeat travel vlog" --max-duration 120

# Download
python3 ~/.openclaw/skills/pixabay-music-skill/scripts/pixabay_music.py \
  download "upbeat travel vlog" \
  -o ~/Develop/CutClaw/resource/audio/bgm.mp3
```

### 5. Generate Shot Plan (shot_plan)

Based on scene analysis + BGM structure, the content strategist generates a shot plan:

```bash
python src/planner_agent.py \
  --video "resource/video/sample.MOV" \
  --scene-summaries "Output/Video/{VIDEO_ID}/captions/scene_summaries_video" \
  --audio-captions "Output/Audio/{BGM_ID}/captions/captions.json" \
  --subtitle "Output/Video/{VIDEO_ID}/subtitles_with_characters.srt" \
  --bgm-name "bgm.mp3" \
  --output-dir "Output/Output/{VIDEO_ID}_{BGM_ID}" \
  --strategy "fast cuts in first 4s, warm interaction in middle 6s, emotional climax in last 5s" \
  --action shot_plan
```

### 6. Generate Shot Point 

Generate precise clip timestamps from the confirmed shot plan:

```bash
python src/short_video_editor.py \
  --video "resource/video/sample.MOV" \
  --shot-plan "Output/Output/{VIDEO_ID}_{BGM_ID}/shot_plan_xxx.json" \
  --scene-summaries "Output/Video/{VIDEO_ID}/captions/scene_summaries_video" \
  --audio-captions "Output/Audio/{BGM_ID}/captions/captions.json" \
  --scene-cuts "Output/Video/{VIDEO_ID}/frames/shot_scenes.txt" \
  --instruction "warm family outing, 15s beat-sync short video" \
  --shot-point-context "prioritize shots with children laughing" \
  --action shot_point
```

### 7. Preview Shot Point (dry-run)

Preview the generated composition without rendering:

```bash
python src/short_video_editor.py ... --action dry_run
```

### 8. Render Final Video 

Once shot points are confirmed, render the final video:

```bash
python src/short_video_editor.py \
  --video "resource/video/sample.MOV" \
  --shot-plan "Output/Output/{VIDEO_ID}_{BGM_ID}/shot_plan_xxx.json" \
  --scene-summaries "Output/Video/{VIDEO_ID}/captions/scene_summaries_video" \
  --audio-captions "Output/Audio/{BGM_ID}/captions/captions.json" \
  --action render
```

### 9. Batch Editing (Multi-Clip Project)

For projects with multiple source clips (e.g. a trip with 40+ DJI drone videos), use the `--project` commands to batch-process all clips and prepare them for cross-clip editing.

**Create a project from a video directory:**

```bash
python local_run.py --project create \
  --video-dir "/path/to/your/videos" \
  --project-name "My Trip"
```

This scans all `.mp4`/`.mov` files, extracts metadata via ffprobe, and groups clips by recording date.

**Review source media consistency:**

```bash
python local_run.py --project review-sources \
  --project-path "Output/Projects/<project_id>/project.json"
```

Checks codec, resolution, fps, and colorspace across all clips. Flags issues and reports whether normalization is needed during rendering.

**Batch preprocess all clips:**

```bash
python local_run.py --project preprocess \
  --project-path "Output/Projects/<project_id>/project.json" \
  --type vlog \
  --max-workers 2
```

Runs shot detection, captioning, scene merge, and scene analysis for every clip in parallel. Supports checkpoint-based resume — if interrupted, rerun the same command to skip completed clips.

**Build global material index:**

```bash
python local_run.py --project build-index \
  --project-path "Output/Projects/<project_id>/project.json"
```

Aggregates all clip scene summaries into a flat `material_index.json` for the planner agent to select shots across the entire project.

**Check project status:**

```bash
python local_run.py --project status \
  --project-path "Output/Projects/<project_id>/project.json"
```

### 10. Key Config Overrides

Common runtime configuration overrides:

```bash
python local_run.py ... \
  --config.VIDEO_FPS 2 \
  --config.AUDIO_TOTAL_SHOTS 50 \
  --config.MAIN_CHARACTER_NAME "Tree" \
  --config.MIN_PROTAGONIST_RATIO 0.7 \
  --config.AUDIO_MIN_SEGMENT_DURATION 1.8 \
  --config.AUDIO_MAX_SEGMENT_DURATION 3.8
```

### Output Files

| Operation | Output Path | Description |
|-----------|-------------|-------------|
| Video Analysis | `Output/Video/{ID}/captions/scene_summaries_video/` | Per-scene descriptions |
| Scene Cuts | `Output/Video/{ID}/frames/shot_scenes.txt` | Shot boundaries |
| BGM Analysis | `Output/Audio/{ID}/captions/captions.json` | Rhythm structure + captions |
| ASR Subtitles | `Output/Video/{ID}/subtitles.srt` | Speech-to-text |
| Shot Plan | `Output/Output/{ID}/{BGM}/shot_plan_xxx.json` | Creative plan |
| Shot Point | `Output/Output/{ID}/{BGM}/shot_point_xxx.json` | Precise timestamps |
| Final Video | `Output/Output/{ID}/{BGM}/output_9x16.mp4` | Rendered video |

#### Batch Editing Outputs

| Operation | Output Path | Description |
|-----------|-------------|-------------|
| Project | `Output/Projects/{ID}/project.json` | Project metadata + clip list |
| Source Review | `Output/Projects/{ID}/source_review.json` | Codec/resolution/fps audit |
| Clip Preprocess | `Output/Projects/{ID}/Clips/{clip_id}/` | Per-clip scene analysis |
| Checkpoints | `Output/Projects/{ID}/checkpoints/` | Resumable stage state |
| Material Index | `Output/Projects/{ID}/material_index.json` | Global scene index for planner |

---

## 🛠️ Troubleshooting

**Very slow runtime**

1. **API latency** — the pipeline sends a large number of concurrent requests to vision/language APIs. Speed is heavily dependent on your API provider's response time and rate limits.
2. **First-run Footage Deconstruction** — the first time you process a video, shot detection, captioning, ASR, and scene analysis all run from scratch. This is a one-time cost per video; subsequent edits with the same footage reuse the cached results and are much faster.
3. **GPU acceleration** — a CUDA-capable GPU significantly speeds up video decoding and encoding. We recommend building Decord with NVDEC support (see Install section).
4. **Video codec compatibility** — if the pipeline appears to hang during video-related steps, the source video's encoding may be the cause. In our testing, videos encoded with `libx264` worked reliably.





## ⭐ Citation
If you find CutClaw useful for your research, welcome to cite our work using the following BibTeX:
 ```bibtex
@article{cutclaw,
  title={CutClaw: Agentic Hours-Long Video Editing via Music Synchronization},
  author={Shifang Zhao, Yihan Hu, Ying Shan, Yunchao Wei, Xiaodong Cun},
  journal={arXiv preprint arXiv:2603.29664},
  year={2026}
}
``` 

---

## 📜 License & Attribution

This project is a **derivative work** of [GVCLab/CutClaw](https://github.com/GVCLab/CutClaw), the original academic research project by Shifang Zhao, Yihan Hu, Ying Shan, Yunchao Wei, and Xiaodong Cun from Beijing Jiaotong University, Great Bay University, and Tencent ARC Lab.

- The original codebase and research are (c) GVCLab and its authors.
- New features, modifications, and extensions by [@treesan](https://github.com/treesan) are released under the **MIT License** (see [LICENSE](LICENSE)).
- Please cite the original [CutClaw paper](https://arxiv.org/abs/2603.29664) if you use this work in your research.

---

## 📈 Star History

<p align="center">
  <a href="https://www.star-history.com/#GVCLab/CutClaw&Date">
    <img src="https://api.star-history.com/svg?repos=GVCLab/CutClaw&type=Date" alt="Star History Chart" width="100%" />
  </a>
</p>
