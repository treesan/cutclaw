# VCutClaw 批量剪辑架构设计文档

> 基于原始设计文档 + CutClaw 源码深度分析 + OpenMontage 架构借鉴 + 实际素材调研
> 日期: 2026-05-28
> 目标素材: 42 个视频, DJI 4K HEVC 60fps

---

## 1. 现状分析

### 1.1 源素材概况

| 维度 | 数据 |
|------|------|
| 文件数 | 42 个视频文件 (40 DJI + 2 Nikon DSC) |
| 日期分布 | 10/02(1), 10/03(5), 10/04(2), 10/05(9), 10/06(11) |
| 编码 | HEVC Main 10 (H.265) |
| 分辨率 | 3840x2160 (4K) |
| 帧率 | 59.94fps |
| 码率 | ~11 Mbps |
| 单文件大小 | ~30MB/16s, 估算总计 ~30-50GB |
| 设备 | 大疆无人机 (主体) + 尼康相机 |

### 1.2 现有架构瓶颈

通过源码分析 (`core.py`, `planner_agent.py`, `short_video_editor.py`)，确认以下单点限制：

1. **单视频入口**: `EditorCoreAgent.__init__` 只接受一个 `video_path`，`PlannerAgent` 同理
2. **预处理绑定单文件**: `local_run.py` 的 `decode_video_to_frames` → `process_video` → `scene_merge` 全链路面向单文件
3. **render_video.py 单源拼接**: 只从一个源视频中提取时间范围，不支持跨文件
4. **shot_point 结构**: 当前 shot_point 只有 `start_sec/end_sec`，无 `clip_id` 字段标识来源文件
5. **无废片过滤**: 所有场景一视同仁送入锦书，噪音大
6. **无渲染后验证**: 成片质量靠人工目视

### 1.3 OpenMontage 可借鉴的设计

通过深度阅读 OpenMontage 源码，以下设计模式可直接移植到 CutClaw：

| OpenMontage 模块 | 借鉴价值 | 对应 CutClaw 改造点 |
|-----------------|---------|-------------------|
| `lib/media_profiles.py` MediaProfile + `ffmpeg_output_args()` | 平台配置标准化, 自动生成 ffmpeg 参数 | 替代现有的散落配置 |
| `tools/video/video_stitch.py` validate + stitch | ffprobe 探测兼容性 → 自动归一化 → concat demuxer | 新增 MultiSourceRenderer |
| `tools/audio/audio_mixer.py` segmented_music | 帧精确 volume expression + 分段 BGM + EBU R128 响度标准化 | 长视频多段 BGM |
| `tools/video/video_compose.py` post-render self-review | ffprobe 技术探针 + 黑帧检测 + 音量检测 | 渲染后质量门禁 |
| `lib/checkpoint.py` per-stage checkpoint | 每阶段写 checkpoint JSON, 中断后自动续跑 | 批量预处理续跑 |
| `tools/video/clip_cache.py` LRU cache | 硬链接 + 文件锁 + LRU 驱逐 | 预处理结果缓存 |
| `schemas/artifacts/edit_decisions.schema.json` cuts 结构 | source + in/out + speed + transition | shot_point v2 格式 |
| `lib/slideshow_risk.py` 6 维质量评分 | 重复度/运动弱/镜头意图缺失 | 锦书策划质量检查 |
| `lib/source_media_review.py` 源素材审查 | 逐文件 ffprobe + 问题标记 | 废片检测 |

---

## 2. 整体架构

### 2.1 分层架构

```
┌──────────────────────────────────────────────────────────────────────┐
│                          CLI / Web UI                                │
│  local_run.py --project create/preprocess/plan/edit/render          │
│  app.py (Streamlit)                                                  │
└──────────────┬───────────────────────────────────┬───────────────────┘
               │                                   │
               ▼                                   ▼
┌──────────────────────────┐      ┌──────────────────────────────────┐
│   Phase 1-2: 素材准备     │      │   Phase 3-5: 剪辑流水线           │
│                          │      │                                  │
│  ProjectManager          │      │  PlannerAgent.from_project()    │
│  ├── scan_directory()    │      │  ShortVideoEditor.from_project()│
│  ├── group_by_day()      │      │  MultiSourceRenderer            │
│  └── project.json        │      │  MultiProfileRenderer           │
│                          │      │  RenderValidator                 │
│  BatchPreprocessor       │      │                                  │
│  ├── preprocess_clip()   │      └──────────────────────────────────┘
│  ├── QualityFilter       │
│  ├── CheckpointManager   │
│  └── MaterialIndex       │
└──────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────────────────┐
│                        核心引擎层 (现有, 基本不改)                     │
│  Screenwriter │ EditorCoreAgent │ DirectShotSelector │ ReviewerAgent │
│  scene_analysis_video │ scene_merge │ asr │ audio_caption_madmom    │
└──────────────────────────────────────────────────────────────────────┘
```

### 2.2 核心设计原则

| 原则 | 说明 |
|------|------|
| **向后兼容** | `local_run.py` 单视频模式完全不变，`--project` 是纯增量 |
| **复用引擎** | 预处理核心 (scene_analysis, scene_merge, asr) 逐 Clip 复用，不重写 |
| **增量处理** | 每阶段写 checkpoint JSON，中断后自动续跑 (借鉴 OpenMontage `checkpoint.py`) |
| **数据驱动** | `project.json` + `material_index.json` 是全链路的数据契约 |
| **ffmpeg 无损** | 渲染阶段先 validate → 尽量 `-c copy`，仅必要时重编码 |
| **渲染后验证** | 每次渲染后自动跑 ffprobe 技术检查 + 黑帧检测 + 音量检测 (借鉴 OpenMontage post-render review) |

---

## 3. 数据结构详细设计

### 3.1 Project

```python
@dataclass
class Project:
    """一次旅行/活动的顶层容器"""
    project_id: str                     # "20251001_青甘小环线"
    name: str                           # "青甘小环线自驾游"
    base_dir: str                       # 源视频目录
    output_dir: str                     # "Output/Projects/20251001_青甘小环线"
    created_at: str                     # ISO 8601

    clips: list[Clip]                   # 所有原始视频
    days: list[Day]                     # 按天分组

    output_profiles: list[str]          # ["bilibili_4k", "douyin"]
    bgm_config: BGMConfig               # BGM 配置

    # 运行时状态
    preprocess_progress: dict           # {clip_id: "done"/"pending"/"failed"}

    metadata_path: str                  # project.json 路径
```

### 3.2 Clip (增强版)

```python
@dataclass
class Clip:
    clip_id: str                        # "DJI_20251002182137_0005_D"
    file_path: str                      # 完整路径
    file_size_mb: float
    duration_sec: float
    width: int                          # 3840
    height: int                         # 2160
    fps: float                          # 59.94
    codec: str                          # "hevc"
    bitrate_kbps: int                   # 11002
    creation_time: str                  # ISO 8601, from ffprobe/EXIF
    device: str                         # "dji_mavic3" / "nikon_z8"

    # 预处理结果路径 (相对于 output_dir)
    scene_summaries_dir: str | None
    shot_scenes_path: str | None
    asr_path: str | None
    captions_path: str | None

    # 质量评估
    quality_score: float | None         # 0-100
    quality_flags: list[str]
    is_valid: bool
```

### 3.3 Day

```python
@dataclass
class Day:
    day_idx: int                        # 1-based
    date: str                           # "2025-10-02"
    title: str                          # "西宁 → 青海湖" (用户可编辑)
    clip_ids: list[str]
    summary: str | None                 # 锦书生成的天级摘要
```

### 3.4 MaterialIndex (关键新增)

```python
@dataclass
class MaterialIndex:
    """全局素材索引 — 锦书策划的唯一数据源"""
    project_id: str
    build_time: str
    total_clips: int
    valid_clips: int
    total_scenes: int
    total_duration_sec: float
    scenes: list[SceneEntry]

@dataclass
class SceneEntry:
    """全局唯一场景条目 — 打平了 clip 边界"""
    scene_id: str                       # "S001_003" (全局唯一)
    clip_id: str                        # 来源 Clip
    clip_file_path: str                 # 渲染时需要的完整路径
    scene_idx_in_clip: int

    start_sec: float
    end_sec: float
    duration_sec: float

    caption: str
    location: str
    emotion: str
    shot_type: str                      # "drone_aerial" / "WS" / "MS" / "CU"
    has_protagonist: bool
    has_dialogue: bool
    quality_score: float

    day_idx: int
    tags: list[str]                     # ["航拍", "盐湖", "日落", "公路"]
```

### 3.5 MediaProfile (借鉴 OpenMontage)

直接借鉴 OpenMontage 的 `MediaProfile` 设计，取代原有的 `OutputProfile`:

```python
# 文件: src/project/media_profiles.py
# 借鉴自: OpenMontage/lib/media_profiles.py

from dataclasses import dataclass
from enum import Enum
from typing import Optional

class AspectRatio(str, Enum):
    LANDSCAPE_16_9 = "16:9"
    PORTRAIT_9_16 = "9:16"
    SQUARE_1_1 = "1:1"
    STANDARD_4_3 = "4:3"
    PORTRAIT_3_4 = "3:4"
    CINEMATIC_21_9 = "21:9"

@dataclass(frozen=True)
class MediaProfile:
    name: str
    width: int
    height: int
    aspect_ratio: AspectRatio
    fps: int
    codec: str                          # ffmpeg encoder name: "libx265" / "libx264"
    audio_codec: str
    crf: int
    pixel_format: str = "yuv420p"
    preset: str = "medium"
    max_file_size_mb: Optional[float] = None
    max_duration_seconds: Optional[float] = None
    notes: str = ""

# ---- 内置 Profiles ----

BILIBILI_4K = MediaProfile(
    name="bilibili_4k",
    width=3840, height=2160,
    aspect_ratio=AspectRatio.LANDSCAPE_16_9,
    fps=60, codec="libx265", audio_codec="aac", crf=16,
    pixel_format="yuv420p10le", preset="slow",
    max_duration_seconds=900,
    notes="B站超高清, HEVC源获超高清标签流量加持",
)

BILIBILI_1080P = MediaProfile(
    name="bilibili_1080p",
    width=1920, height=1080,
    aspect_ratio=AspectRatio.LANDSCAPE_16_9,
    fps=60, codec="libx264", audio_codec="aac", crf=20,
    max_duration_seconds=900,
    notes="B站高清, 兼容性最好",
)

DOUYIN = MediaProfile(
    name="douyin",
    width=1080, height=1920,
    aspect_ratio=AspectRatio.PORTRAIT_9_16,
    fps=30, codec="libx264", audio_codec="aac", crf=22,
    max_duration_seconds=60,
    notes="抖音短视频",
)

XIAOHONGSHU = MediaProfile(
    name="xiaohongshu",
    width=1080, height=1440,
    aspect_ratio=AspectRatio.PORTRAIT_3_4,  # 3:4 竖版
    fps=30, codec="libx264", audio_codec="aac", crf=22,
    max_duration_seconds=300,
    notes="小红书图文视频",
)

ALL_PROFILES: dict[str, MediaProfile] = {
    p.name: p for p in [BILIBILI_4K, BILIBILI_1080P, DOUYIN, XIAOHONGSHU]
}

def get_profile(name: str) -> MediaProfile:
    if name not in ALL_PROFILES:
        available = ", ".join(ALL_PROFILES.keys())
        raise ValueError(f"Unknown profile {name!r}. Available: {available}")
    return ALL_PROFILES[name]

def ffmpeg_output_args(profile: MediaProfile) -> list[str]:
    """生成 ffmpeg 输出参数列表 (借鉴 OpenMontage)"""
    args = [
        "-c:v", profile.codec,
        "-c:a", profile.audio_codec,
        "-crf", str(profile.crf),
        "-preset", profile.preset,
        "-pix_fmt", profile.pixel_format,
        "-r", str(profile.fps),
    ]
    return args
```

### 3.6 BGMConfig

```python
@dataclass
class BGMConfig:
    strategy: str                       # "multi_segment" / "single"
    segments: list[BGMSegment]
    short_video_bgm: str | None

@dataclass
class BGMSegment:
    segment_id: str
    audio_path: str
    day_idx: int | None
    start_sec: float
    duration_sec: float
    fade_in_sec: float = 1.0
    fade_out_sec: float = 2.0
```

### 3.7 Checkpoint (借鉴 OpenMontage)

```python
# 文件: src/batch/checkpoint.py
# 借鉴自: OpenMontage/lib/checkpoint.py

@dataclass
class StageCheckpoint:
    """每阶段的状态快照, 支持中断续跑"""
    project_id: str
    stage: str                          # "preprocess" / "build_index" / "shot_plan" / "shot_point" / "render"
    status: str                         # "completed" / "failed" / "in_progress"
    clip_id: str | None                 # preprocess 阶段: 单个 clip 的 checkpoint
    artifacts: list[str]                # 产出文件路径
    error: str | None
    timestamp: str

class CheckpointManager:
    """管理项目级 checkpoint, 支持断点续跑"""

    def __init__(self, project_dir: str, project_id: str = ""):
        self.project_id = project_id
        self.checkpoint_dir = os.path.join(project_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def write(self, checkpoint: StageCheckpoint):
        """写入或覆盖 checkpoint JSON, 自动设置 timestamp"""
        ...

    def read(self, stage: str, clip_id: str | None = None) -> StageCheckpoint | None:
        """读取 checkpoint, 不存在返回 None"""
        ...

    def mark_completed(self, stage: str, clip_id: str | None = None, artifacts: list[str] | None = None):
        """便捷方法: 标记阶段完成"""
        ...

    def mark_failed(self, stage: str, clip_id: str | None = None, error: str = ""):
        """便捷方法: 标记阶段失败"""
        ...

    def get_completed_clips(self, stage: str = "preprocess") -> set[str]:
        """获取已完成指定阶段的 clip_id 集合"""
        ...

    def get_next_stage(self, stages: list[str]) -> str | None:
        """返回第一个未完成的阶段, 全部完成返回 None"""
        ...

    def is_clip_done(self, stage: str, clip_id: str) -> bool:
        """检查指定 clip 是否已完成指定阶段"""
        ...

    def cleanup(self, valid_clip_ids: set[str], stage: str = "preprocess"):
        """清理不属于当前项目的过期 checkpoint 文件"""
        ...
```

---

## 4. 模块详细设计

### 4.1 ProjectManager (`src/project/project.py`)

```python
class ProjectManager:
    @staticmethod
    def create_from_directory(
        video_dir: str,
        project_name: str = "",
        group_by: str = "day",
        output_root: str = "Output/Projects",
    ) -> Project:
        """
        扫描目录 → 创建 Project

        流程:
        1. 递归扫描 .mp4/.mov/.MP4/.MOV (排除 ._ 开头的 macOS 元数据文件)
        2. ffprobe 读取每个文件的元数据 (codec, resolution, fps, duration, creation_time)
        3. 按 creation_time 的日期分组为 Day[]
        4. 为每个文件创建 Clip 对象
        5. 创建输出目录结构
        6. 写入 project.json

        实际素材注意:
        - DJI 文件名格式: DJI_YYYYMMDDHHMMSS_XXXX_D.MP4
        - Nikon 文件名格式: DSC_XXXX.MP4
        - 存在 _CUT 后缀的裁切版本
        - 需排除 macOS 隐藏文件 (._DJI_...)
        """
        ...

    @staticmethod
    def load(project_path: str) -> Project:
        ...

    @staticmethod
    def save(project: Project):
        ...
```

### 4.2 BatchPreprocessor (`src/batch/batch_preprocess.py`)

```python
class BatchPreprocessor:
    """
    批量预处理编排器

    设计要点:
    - 并行处理多个 Clip (ThreadPoolExecutor)
    - 每阶段写 checkpoint, 中断后可续跑 (借鉴 OpenMontage checkpoint 模式)
    - 复用现有的 scene_analysis_video + scene_merge + asr 全链路
    - 新增 QualityFilter 废片检测
    - 线程安全: 使用 threading.Lock 保护共享状态
    """

    def __init__(self, project: Project, max_workers: int = 2):
        self.project = project
        self.max_workers = max_workers
        self.quality_filter = QualityFilter()
        self.checkpoint = CheckpointManager(project.output_dir, project_id=project.project_id)
        self._lock = threading.Lock()

    def run(self, clip_ids: list[str] | None = None) -> dict:
        """
        并行预处理所有 (或指定的) Clip

        增量策略 (借鉴 OpenMontage skip_existing 模式):
        - skip_existing=True 时, 检查 checkpoint 中已完成的 clip_id, 跳过
        - 每个 clip 完成后立即写 checkpoint, 不等全部完成

        返回: {"total": N, "skipped": N, "success": N, "failed": N, "errors": [...]}
        """
        ...

    def _process_one_clip(self, clip: Clip, results: dict):
        """
        单个 Clip 的完整预处理流程 (6 步):

        Step 1: decode_video_to_frames — 解码视频 + 镜头检测
                返回 decord.VideoReader 对象 (后续步骤复用)
        Step 2: ASR (仅 film 模式) — 语音识别 + 说话人识别
        Step 3: process_video — 视频描述生成 (使用 Step 1 的 vr)
        Step 4: OptimizedSceneSegmenter — 场景合并 (基于语义相似度)
        Step 5: SceneVideoAnalyzer — 场景分析 (使用 Step 1 的 vr)
        Step 6: QualityFilter.evaluate_clip — 快速质量评分

        每步完成后更新 checkpoint。异常时标记 failed 并继续。
        所有对 results dict 的修改通过 self._lock 保护线程安全。
        """
        ...
```

> 注: `build_material_index` 已移至独立模块 `src/batch/material_index.py` 的 `build_material_index(project)` 函数，由 CLI `--project build-index` 直接调用。这更符合单一职责原则。

### 4.3 QualityFilter (`src/batch/quality_filter.py`)

```python
class QualityFilter:
    """
    废片检测器 (Clip 级快速评估)

    已实现的检测维度:
    1. 模糊度: Laplacian 方差 (cv2.Laplacian), 阈值 50.0
    2. 曝光: 亮度均值, 暗阈值 30.0 / 亮阈值 225.0
    3. 时长过短: < 1秒

    未来扩展 (需额外依赖, 暂不实现):
    4. 稳定性: 相邻帧光流差异 (需要 cv2.calcOpticalFlowFarneback, 4K 素材开销大)
    5. 有效内容比例: 纯地面/纯天空占比 (需要语义分割模型)

    分层策略:
    - Clip 级 (已实现): 快速采样 10 帧, 判断整个文件是否废片
    - Scene 级 (未来扩展): 在预处理后对每个场景单独打分 (更精确)

    降级策略: 无 cv2 时使用 PIL 解码图片, 仅计算曝光指标
    """

    def __init__(
        self,
        clip_threshold: int = 30,
        blur_threshold: float = 50.0,
        dark_threshold: float = 30.0,
        bright_threshold: float = 225.0,
        min_duration_sec: float = 1.0,
    ):
        ...

    def evaluate_clip(
        self, clip_path: str, clip_id: str = "",
        duration_sec: float = 0.0, sample_frames: int = 10,
    ) -> QualityResult:
        """评估单个 Clip, 返回 0-100 评分和 flags"""
        ...
```

### 4.4 SourceMediaReview (借鉴 OpenMontage, 新增)

```python
# 文件: src/batch/source_media_review.py
# 借鉴自: OpenMontage/lib/source_media_review.py

class SourceMediaReview:
    """
    源素材标准化审查 — 在 Project 创建后、预处理前运行

    对每个 Clip 执行:
    1. ffprobe 完整探针 (编码/分辨率/帧率/色彩空间/音频)
    2. 标记问题: 低分辨率(<720p)、单声道音频、超短片段(<3s)、HDR/SDR 不一致
    3. 生成审查报告, 为后续渲染的归一化策略提供依据
    """

    def review_project(self, project: Project) -> SourceReviewReport:
        ...

@dataclass
class SourceReviewReport:
    total_clips: int
    issues: list[dict]              # [{clip_id, severity, message}]
    resolution_variants: list[str]  # ["3840x2160", "1920x1080"]
    codec_variants: list[str]       # ["hevc", "h264"]
    fps_variants: list[float]       # [59.94, 30.0]
    color_spaces: list[str]         # ["bt2020", "bt709"]
    needs_normalization: bool       # 渲染时是否需要重编码
```

### 4.5 PlannerAgent 升级

```python
class PlannerAgent:
    # ====== 原有代码完全保留 ======

    @classmethod
    def from_project(
        cls,
        project: Project,
        material_index: MaterialIndex,
        profile: MediaProfile,
        bgm_segments: list[BGMSegment] | None = None,
    ) -> "PlannerAgent":
        """
        从 Project 创建 PlannerAgent (工厂方法, 不改动 __init__)

        关键变化:
        1. scene_summaries_dir 指向合并后的全局场景目录
        2. instruction 自动根据 MediaProfile 和 Day 分组生成
        3. bgm_path 支持多段配置

        实现: 创建临时 "virtual scene_summaries_dir",
        将 MaterialIndex 中的 SceneEntry 转换为现有 scene_*.json 格式,
        这样 Screenwriter 完全不需要修改
        """
        ...

    def generate_project_shot_plan(
        self,
        strategy_context: str = "",
        narrative_structure: str = "day_linear",
    ) -> dict:
        """
        生成项目级 shot_plan

        shot_plan 结构变化 (新增 source_clip 和 scene_id):
        {
          "video_structure": [
            {
              "section_name": "Day 1: 西宁 → 青海湖",
              "day_idx": 1,
              "bgm_segment": "bgm_day1",
              "shot_plan": {
                "shots": [
                  {
                    "id": 0,
                    "time_duration": 4.0,
                    "content": "aerial drone shot over Qinghai Lake",
                    "source_clip": "DJI_20251002182137_0005_D",
                    "scene_id": "S001_003",
                    "related_scene": [3]
                  }
                ]
              }
            }
          ]
        }
        """
        ...
```

### 4.6 core.py 升级 (多源选片)

```python
# core.py 改动: 工具函数新增 clip_file_path 参数

def semantic_neighborhood_retrieval(
    related_scenes: list[int] = None,
    scene_folder_path: str = None,
    recommended_scenes: list[int] = None,
    material_index_path: str = None,     # 🆕
) -> str:
    """升级: 支持从 MaterialIndex 检索跨 Clip 的场景"""
    ...

def fine_grained_shot_trimming(
    time_range: str,
    frame_path: str = "",
    clip_file_path: str = "",           # 🆕 明确指定来源文件
    transcript_path: str = "",
    original_shot_boundaries: list = None,
) -> str:
    """升级: clip_file_path 优先于 frame_path"""
    ...

def commit(
    answer: str,
    video_path: str = "",
    clip_file_path: str = "",           # 🆕
    output_path: str = "",
    target_length_sec: float = 0.0,
    section_idx: int = -1,
    shot_idx: int = -1,
    protagonist_frame_data: list = None,
) -> str:
    """升级: shot_point 中新增 clip_file_path 字段"""
    ...
```

---

## 5. 渲染管线详细设计

### 5.1 MultiSourceRenderer (`render/multi_source_renderer.py`)

借鉴 OpenMontage 的 `VideoStitch` 设计模式: **validate → normalize → stitch → mix audio → review**

```python
class MultiSourceRenderer:
    """
    多源视频渲染器

    渲染流程 (借鉴 OpenMontage VideoStitch + VideoCompose):
    1. validate: ffprobe 探测所有源片段的编码/分辨率/帧率, 检查兼容性
    2. extract: ffmpeg -ss -to -c copy 无损提取 (如果编码一致)
               或 ffmpeg -ss -to -vf scale 重编码 (如果不一致)
    3. stitch: concat demuxer 拼接 (如果编码一致)
              或 filter_complex concat (如果不一致)
    4. audio_mix: 叠加 BGM (借鉴 OpenMontage segmented_music 模式)
    5. subtitles: 叠加字幕 (可选)
    6. review: 渲染后自动验证 (借鉴 OpenMontage post-render self-review)
    """

    def __init__(
        self,
        shot_point_path: str,
        profile: MediaProfile,
        bgm_config: BGMConfig | None = None,
        subtitle_path: str | None = None,
    ):
        ...

    def render(self, output_path: str) -> RenderResult:
        """执行渲染, 返回渲染报告"""
        ...

    # ---- Step 1: Validate (借鉴 OpenMontage VideoStitch._validate) ----

    def _validate_segments(self, shots: list[dict]) -> ValidationResult:
        """
        ffprobe 探测所有源片段, 检查兼容性

        检查项 (与 OpenMontage VideoStitch 一致):
        - width, height
        - fps
        - video_codec
        - pixel_format
        - audio_codec, sample_rate, channels

        返回:
        - compatible: bool (是否可以直接 concat demuxer)
        - mismatches: 不兼容的字段列表
        - reference: 参考片段的属性
        """
        ...

    # ---- Step 2: Extract segments ----

    def _extract_segments(
        self,
        shots: list[dict],
        output_dir: str,
        use_copy: bool = True,
    ) -> list[str]:
        """
        从源视频提取每个 shot 的片段

        use_copy=True 且编码一致时:
          ffmpeg -ss {start} -to {end} -i {source} -c copy -avoid_negative_ts make_zero temp_N.mp4

        use_copy=False 或编码不一致时:
          ffmpeg -ss {start} -to {end} -i {source} \
            -vf "scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,setsar=1" \
            -r {fps} -c:v {codec} -crf {crf} -preset {preset} -pix_fmt {pix_fmt} \
            -c:a {audio_codec} -ar 48000 -ac 2 temp_N.mp4
        """
        ...

    # ---- Step 3: Stitch (借鉴 OpenMontage VideoStitch._stitch) ----

    def _concat_segments(
        self,
        segment_paths: list[str],
        output_path: str,
        compatible: bool,
        profile: MediaProfile,
    ) -> str:
        """
        拼接所有片段

        compatible=True: concat demuxer (最快, 无损)
          ffmpeg -f concat -safe 0 -i filelist.txt -c copy output.mp4

        compatible=False: filter_complex concat
          ffmpeg -i seg1.mp4 -i seg2.mp4 ... \
            -filter_complex "[0:v][0:a][1:v][1:a]concat=n=N:v=1:a=1" \
            -c:v {codec} -crf {crf} output.mp4
        """
        ...

    # ---- Step 4: Audio Mix (借鉴 OpenMontage AudioMixer._segmented_music) ----

    def _add_bgm_segmented(
        self,
        video_path: str,
        bgm_config: BGMConfig,
        output_path: str,
        video_duration: float,
    ) -> str:
        """
        分段 BGM 混合 (借鉴 OpenMontage segmented_music)

        核心: FFmpeg volume expression 实现帧精确的分段音量控制

        对每个 BGM segment 生成 volume 表达式:
          if(lt(t,<start>),0,
            if(lt(t,<fade_in_end>),<vol>*(t-<start>)/<fade_dur>,
              if(lt(t,<fade_out_start>),<vol>,
                if(lt(t,<end>),<vol>*(<end>-t)/<fade_dur>,
                  0))))

        多段用 "+" 连接, 然后:
          -stream_loop -1 循环 BGM (如果比视频短)
          -vf volume='{vol_expr}':eval=frame
          原视频音量降低, amix 混合
          loudnorm=I=-16:LRA=11:TP=-1.5 响度标准化 (EBU R128)
        """
        ...

    # ---- Step 5: Subtitles ----

    def _add_subtitles(
        self,
        video_path: str,
        subtitle_path: str,
        output_path: str,
    ) -> str:
        """叠加字幕"""
        ...

    # ---- Step 6: Post-render Review (借鉴 OpenMontage VideoCompose._run_final_review) ----

    def _validate_render(self, output_path: str, profile: MediaProfile) -> RenderResult:
        """
        渲染后自动验证 (借鉴 OpenMontage post-render self-review)

        检查项:
        1. 技术探针 (ffprobe): 容器、时长、分辨率、帧率、音频流是否存在
        2. 黑帧检测: 采样 4 帧 (10%/35%/65%/90%), 检测纯黑帧
        3. 音量检测: volumedetect, 静音(< -60dB) / 削波(> -0.5dB)
        4. 时长漂移: 与目标时长对比, 偏差 > 25% 标记警告
        """
        ...
```

### 5.2 RenderResult (渲染报告, 借鉴 OpenMontage render_report)

```python
@dataclass
class RenderResult:
    """渲染报告 (借鉴 OpenMontage render_report schema)"""
    status: str                         # "pass" / "revise" / "fail"
    output_path: str
    duration_sec: float
    file_size_mb: float
    resolution: str                     # "3840x2160"
    fps: float
    codec: str
    audio_codec: str
    profile_name: str

    # 验证结果
    technical_probe: dict               # ffprobe 结果
    black_frame_check: dict             # 黑帧检测
    audio_check: dict                   # 音量检测
    warnings: list[str]
    errors: list[str]
```

### 5.3 MultiProfileRenderer (`render/multi_profile_renderer.py`)

```python
class MultiProfileRenderer:
    """
    多平台输出渲染器

    策略:
    1. 先渲染长视频 (bilibili_4k) — 完整叙事
    2. 从长视频切片短视频 (douyin/xiaohongshu) — 精华摘取
    """

    def render_all(
        self,
        project: Project,
        shot_points: dict[str, str],    # {profile_id: shot_point_path}
    ) -> dict[str, RenderResult]:
        """按顺序渲染所有 Profile, 返回每个的渲染报告"""
        ...

    def _slice_from_long_video(
        self,
        long_video_path: str,
        time_ranges: list[tuple[float, float]],
        output_path: str,
        profile: MediaProfile,
    ) -> str:
        """从已渲染的长视频切片"""
        ...
```

### 5.4 色彩空间统一

```python
def normalize_colorspace(input_path: str, output_path: str, target: str = "bt709"):
    """
    统一色彩空间

    DJI 拍摄通常是 BT.2020 + HLG (HDR)
    Nikon 可能是 BT.709 (SDR)

    策略:
    - bilibili_4k: 保持 BT.2020 (B站支持 HDR)
    - 其他 Profile: 转为 BT.709 (兼容性)
    """
    import subprocess
    ffmpeg_cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-vf", f"colorspace=all={target}:iall=bt2020:fast=1",
        "-c:v", "libx265", "-crf", "16", "-preset", "slow",
        "-c:a", "copy",
        output_path,
    ]
    result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg colorspace normalization failed: {result.stderr}")
    return output_path
```

---

## 6. shot_point.json v2.0 格式

借鉴 OpenMontage `edit_decisions.schema.json` 的 `cuts` 结构:

```json
{
  "version": "2.0",
  "project_id": "20251001_青甘小环线",
  "profile": "bilibili_4k",
  "total_duration_target": 600,

  "shots": [
    {
      "section_idx": 0,
      "shot_idx": 0,
      "start_sec": 13.400,
      "end_sec": 17.800,
      "duration": 4.40,
      "clip_file_path": "/Volumes/Data/Media/.../DJI_0005_D.MP4",
      "clip_id": "DJI_20251002182137_0005_D",
      "scene_id": "S001_003",
      "speed": 1.0,
      "transition": "cut",
      "reasoning": "Aerial drone ascending over Qinghai Lake at sunset"
    }
  ],

  "bgm_segments": [
    {
      "segment_id": "bgm_day1",
      "audio_path": "Output/Projects/.../BGM/bgm_day1.mp3",
      "start_sec": 0,
      "duration_sec": 180,
      "fade_in_sec": 1.0,
      "fade_out_sec": 2.0
    }
  ]
}
```

向后兼容: 渲染器检测 shot 中是否有 `clip_file_path`, 没有则回退到全局 `video_path`。

---

## 7. CLI 设计

### 7.1 新增命令 (纯增量)

```bash
# ═══════════════════════════════════════════
# 项目管理
# ═══════════════════════════════════════════

python local_run.py --project create \
  --video-dir "/Volumes/Data/Media/20251001青甘小环线/视频" \
  --project-name "青甘小环线自驾游"

python local_run.py --project status \
  --project-path "Output/Projects/20251001_青甘小环线/project.json"

# ═══════════════════════════════════════════
# 源素材审查 (新增, 借鉴 OpenMontage source_media_review)
# ═══════════════════════════════════════════

python local_run.py --project review-sources \
  --project-path "..."

# ═══════════════════════════════════════════
# 批量预处理
# ═══════════════════════════════════════════

python local_run.py --project preprocess \
  --project-path "..." --max-workers 2 --skip-existing

python local_run.py --project build-index \
  --project-path "..."

# ═══════════════════════════════════════════
# 重置/清理 (运维)
# ═══════════════════════════════════════════

# 重跑单个 clip 的预处理 (删除其 checkpoint 后重新 preprocess)
python local_run.py --project reset-preprocess \
  --project-path "..." --clip-id "DJI_20251002182137_0005_D"

# 清理所有中间文件, 保留 project.json 和源视频
python local_run.py --project clean \
  --project-path "..."

# ═══════════════════════════════════════════
# 内容策划 (锦书)
# ═══════════════════════════════════════════

python src/planner_agent.py --action shot_plan \
  --project-path "..." --profile bilibili_4k \
  --strategy "第一天用航拍大景开场..."

# ═══════════════════════════════════════════
# 剪辑点生成 (短影)
# ═══════════════════════════════════════════

python src/short_video_editor.py --action shot_point \
  --project-path "..." --shot-plan "..." --profile bilibili_4k

# ═══════════════════════════════════════════
# 渲染输出
# ═══════════════════════════════════════════

python render/multi_source_renderer.py \
  --shot-point "..." --profile bilibili_4k \
  --output "Output/Projects/.../output/bilibili_4k.mp4"

python render/multi_profile_renderer.py \
  --project-path "..." --shot-points-dir "Output/Projects/.../shot_points/"
```

### 7.2 现有命令完全不变

```bash
python local_run.py --Video_Path "resource/video/sample.MOV" --Audio_Path "resource/audio/bgm.mp3"
streamlit run app.py
```

---

## 8. 输出目录结构

```
Output/Projects/20251001_青甘小环线/
├── project.json
├── source_review.json                       # 🆕 源素材审查报告
├── checkpoints/                             # 🆕 阶段 checkpoint
│   ├── preprocess_DJI_0005.json
│   ├── preprocess_DJI_0007.json
│   ├── build_index.json
│   └── render_bilibili_4k.json
│
├── Clips/
│   ├── DJI_20251002182137_0005_D/
│   │   ├── scene_summaries_video/
│   │   ├── captions/
│   │   ├── shot_scenes.txt
│   │   ├── subtitles.srt
│   │   └── quality.json
│   └── ...
│
├── material_index.json
├── BGM/
├── shot_plans/
│   └── shot_plan_bilibili_4k.json
├── shot_points/
│   └── shot_point_bilibili_4k.json
├── temp/                                    # 渲染中间文件
│   ├── segment_000.mp4
│   ├── segment_001.mp4
│   └── concat_raw.mp4
├── renders/                                 # 🆕 渲染报告
│   └── render_report_bilibili_4k.json
└── output/
    ├── bilibili_4k.mp4
    ├── bilibili_1080p.mp4
    ├── douyin_01.mp4
    └── xiaohongshu_01.mp4
```

---

## 9. 实现路线图

### Phase 1: 项目基础设施 (1-2 天)

- [ ] `src/project/project.py` — Project / Clip / Day 数据类 + ProjectManager
- [ ] `src/project/media_profiles.py` — MediaProfile (借鉴 OpenMontage) + 内置 Profiles
- [ ] `src/batch/checkpoint.py` — CheckpointManager (借鉴 OpenMontage)
- [ ] `project.json` 读写 + ffprobe 元数据提取
- [ ] CLI: `--project create` / `--project status`

### Phase 2: 批量预处理 (2-3 天)

- [ ] `src/batch/batch_preprocess.py` — BatchPreprocessor (并行 + checkpoint)
- [ ] `src/batch/quality_filter.py` — QualityFilter
- [ ] `src/batch/source_media_review.py` — SourceMediaReview (借鉴 OpenMontage)
- [ ] `material_index.json` 构建
- [ ] CLI: `--project preprocess` / `--project build-index` / `--project review-sources`

### Phase 3: 策划升级 (1-2 天)

- [ ] `PlannerAgent.from_project()`
- [ ] Screenwriter prompt 调整 — 支持跨 Clip 选片
- [ ] shot_plan 新增 `source_clip` / `scene_id` 字段
- [ ] 多 BGM 段落配置

### Phase 4: 剪辑点升级 (1-2 天)

- [ ] `ShortVideoEditor.from_project()`
- [ ] `DirectShotSelector` / `EditorCoreAgent` 工具升级
- [ ] shot_point v2.0 格式

### Phase 5: 渲染引擎 (2-3 天)

- [ ] `render/multi_source_renderer.py` — validate → extract → stitch → audio_mix → review
- [ ] `render/multi_profile_renderer.py` — 长视频优先 + 短视频切片
- [ ] 色彩空间统一
- [ ] 渲染后验证 (ffprobe + 黑帧 + 音量)

### Phase 6: 集成测试 (1 天)

- [ ] 用青甘小环线素材端到端测试
- [ ] 验证 B站 4K 输出画质
- [ ] 验证向后兼容

**总计预估: 8-13 天**

---

## 10. 关键设计决策

| 决策 | 选择 | 原因 |
|------|------|------|
| MediaProfile 设计 | 借鉴 OpenMontage `frozen=True` dataclass | 不可变配置, `ffmpeg_output_args()` 自动生成参数 |
| 渲染流程 | validate → extract → stitch → audio → review | 借鉴 OpenMontage VideoStitch 的标准化流程 |
| BGM 混合 | segmented_music volume expression | 借鉴 OpenMontage, 帧精确分段控制 |
| 渲染后验证 | ffprobe + 黑帧 + 音量 | 借鉴 OpenMontage post-render self-review |
| Checkpoint | per-clip JSON checkpoint | 借鉴 OpenMontage, 支持中断续跑 |
| 废片过滤 | Clip 级快速 + Scene 级精细 | Clip 级省预处理时间, Scene 级更精确 |
| 画质 | HEVC + CRF 16 + 4K/60fps | B站 HEVC 源获超高清标签 |
| shot_point 格式 | v2.0 (新增 clip_file_path + transition) | 借鉴 OpenMontage edit_decisions, 向后兼容 v1.0 |
| 叙事策略 | 长视频为主, 短视频为切片 | 省素材分析成本, 叙事完整 |

---

## 11. 端到端优化清单 (2026-06-02 补充)

> 基于川西小环线 (16 个视频) 真实 LLM 端到端测试后整理
> 测试 LLM: openai/deepseek-v4-flash (ark.cn-beijing.volces.com)

### A. 内容质量 (影响成片观感)

| # | 优化点 | 问题 | 改进方案 |
|---|--------|------|---------|
| **#1** | PlannerAgent 主题与 strategy 脱节 | LLM 输出的 `overall_theme` 是 "Short video for audio segment" 而非 strategy 关键词 | 强制主题必须包含 strategy 核心关键词 |
| **#2** | 虚构场景内容 | LLM 生成 "Main character stands on a grassy hilltop" 但素材中无此画面 | 强制注入 material_index 的 caption/location 到 prompt |
| **#3** | DJI 航拍被说成有人 | 12 个 DJI 航拍 shot 全部说"main character stands on..." | 注入 clip_device 标签，明确 DJI=无人机航拍无人物 |
| **#4** | BGM 节奏未真正分析 | BGM_path 直接传但没跑 madmom 分析 | `from_project()` 中检测 BGM 时自动调用 `analyze_bgm()`，把 BPM/sections 注入 instruction |

### B. 阶段衔接 (影响工程效率)

| # | 优化点 | 问题 | 改进方案 |
|---|--------|------|---------|
| **#5** | CLI `--project render` 必须显式指定 `--shot-point` | 用户得知道 shot_point 路径 | 加智能默认：自动找 `<output_dir>/shot_points/shot_point_<profile>.json` |
| **#6** | shot_point 跨阶段 checkpoint 缺失 | render 中断后必须从头重跑 | 复用 `CheckpointManager` 记录 `render_<profile>.json` 状态 |
| **#7** | Source Review 没强制 | Step 2 输出 needs_normalization 但用户可能跳过 | render 前自动校验 source_review.json 并 warn |
| **#8** | `render_project()` 未被 CLI 调用 | 走的是 MultiSourceRenderer 直接路径而非统一入口 | 后续可让 --project render 优先调用 editor.render_project() |

### C. 渲染质量 (影响最终画质)

| # | 优化点 | 问题 | 改进方案 |
|---|--------|------|---------|
| **#9** | 生产画质未走慢预设 | 测试用 ultrafast (crf 20)，生产应 slow (crf 16) | config_my.py 加 `MEDIA_PROFILE_PRESET_OVERRIDE` 覆盖机制 |
| **#10** | 缺字幕/标题叠加 | 批量渲染没集成 subtitles | 复用 render_video.py 的 subtitle_lines 逻辑到 MultiSourceRenderer |
| **#11** | 色彩空间转换质量 | HDR→SDR 用 npl=1000/100 通用值 | 读取源 color_transfer 字段智能选择 npl |

### D. 架构清理 (影响可维护性)

| # | 优化点 | 问题 | 改进方案 |
|---|--------|------|---------|
| **#12** | 代码重复 | shot_point 路径生成、临时目录管理、ffmpeg cmd 构建在多处重复 | 抽取 `src/utils/render_utils.py` 共享 |

### E. 端到端流程架构 (优化后)

```
┌──────────────────────────────────────────────────────────┐
│  CLI 智能默认 + 强制检查 (优化 #5, #7)                    │
│  --project render 自动找 shot_point_<profile>.json          │
│  --with_ending 启用结尾视频拼接                          │
│  render 前自动 warn source_review needs_normalization      │
└──────────────┬───────────────────────────────────────────┘
               ▼
┌──────────────────────────────────────────────────────────┐
│  PlannerAgent.from_project() (优化 #1-3)                  │
│  instruction 注入:                                        │
│    - [实际素材内容] (50 个 scene caption/location)        │
│    - [素材设备] (DJI/Nikon 分布 + 创作提示)              │
│    - [强制主题要求] (strategy 关键词约束)                │
└──────────────┬───────────────────────────────────────────┘
               ▼
┌──────────────────────────────────────────────────────────┐
│  ShortVideoEditor.generate_shot_point_project() (优化 #8)  │
│  按 clip 分组 → DirectShotSelector (LLM) → 合并 v2.0      │
│  BUG-RT-07/08 修复: 空 audio_caption 占位 + start_sec 匹配│
└──────────────┬───────────────────────────────────────────┘
               ▼
┌──────────────────────────────────────────────────────────┐
│  MultiSourceRenderer (优化 #6, #8, #10)                    │
│  validate → extract (BUG-RT-05/15/16)                   │
│  → re-check compat (BUG-RT-10) → concat (BUG-RT-06)     │
│  → BGM mix (BUG-RT-17) → subtitles (#10) → ending       │
│  → CheckpointManager.mark_completed()                    │
└──────────────────────────────────────────────────────────┘
```

### F. 端到端测试验证结果

| 阶段 | 状态 | 备注 |
|------|------|------|
| Step 1-2 (create + review-sources) | ✅ | 16 clips, 4 days, BUG-02/03/08 修复 |
| Step 3-4 (preprocess + build-index) | ✅ | 14/14 成功 (BUG-20 过滤 2 个极短) |
| Phase 3 (plan) | ✅ | 14 shots, 143.7s, 跨 14 clip 选片 |
| Phase 4 (edit) | ✅ | 14 shots, 30.5s, 3/14 scene_id 填充 (LLM 选新窗口) |
| Phase 5 (render) | ✅ | 7.52s 视频, 10.3MB, 5.5s 渲染 |


---

## 12. 优化实施汇总 (2026-06-02 补充)

> 本轮实施全部 12 项优化 

### 12.1 实施清单

| # | 优化点 | 文件 | 关键改动 |
|---|--------|------|----------|
| **#1** | strategy 强制主题 | `src/planner_agent.py` | from_project() 加 `strategy` 参数，注入 `[强制主题要求]` 段 |
| **#2** | 实际素材内容 | `src/planner_agent.py` | 注入 material_index 前 50 个 scene 的 caption/location |
| **#3** | 设备感知 | `src/planner_agent.py` | 注入 DJI/Nikon 设备分布和创作提示 |
| **#5** | CLI 智能默认 | `local_run.py` | --shot-point 未指定时自动找 `shot_point_<profile>.json` |
| **#6** | render checkpoint | `local_run.py` | 复用 CheckpointManager 写 `render_<profile>.json` |
| **#7** | source review 强制 | `local_run.py` | render 前自动 warn source_review needs_normalization |
| **#8** | 结尾视频配置 | `src/config.py` + renderer | ENDING_VIDEO_PATH/DURATION/FADE_DURATION + --with-ending CLI |
| **#9** | 生产 preset 覆盖 | `src/config.py` + renderer | MEDIA_PROFILE_PRESET_OVERRIDE/CRF_OVERRIDE |
| **#11** | npl 智能选择 | `render/multi_source_renderer.py` | `_select_tone_map_npl()` 按 hdr_type 选 1000 nits |
| **#4** | BGM 节奏分析 | `src/planner_agent.py` | `from_project()` 自动调用 `caption_audio_with_madmom_segments()` 生成 `captions.json`（Level 1 sections + Level 2 sub-segments），替代空占位符；分析失败降级为占位符；结果缓存到 `bgm_captions/` |
| **#10** | 字幕叠加 | `render/multi_source_renderer.py` | `_collect_subtitles()` + `_add_subtitles()` + `_find_font()`；shot entry 支持 `subtitle_lines`/`overlay_text` 字段，drawtext 烧录 |
| **#12** | 架构抽取 | `src/utils/paths.py` + `local_run.py` + `app.py` | 新增 `derive_artifact_path()` + `discover_shot_point()` 共享函数；local_run.py 死代码修复（auto-discovery 生效） |

