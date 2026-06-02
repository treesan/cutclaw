#!/usr/bin/env python3
"""
ShortVideoEditor — 短影 ✂️ 剪辑师引擎
=======================================

短影的职责：
1. 接收锦书已确认的 shot_plan
2. 调用 DirectShotSelector / EditorCoreAgent 生成精确剪辑点 (shot_point)
3. 提供 --dry-run 文本摘要让 Tree 预览
4. 调用 Render 合成最终视频

用法（锦书 @短影 后，短影在群里直接调用）：
    from src.short_video_editor import ShortVideoEditor

    editor = ShortVideoEditor(
        video_path="path/to/video.MOV",
        shot_plan_path="path/to/shot_plan.json",
        scene_summary_dir="path/to/scene_summaries",
        audio_caption_path="path/to/captions.json",
    )

    # 生成剪辑点
    point = editor.generate_shot_point()
    print(editor.summarize_shot_point())  # dry-run 文本摘要

    # Tree 确认后
    editor.render(crop_ratio="9:16")
"""

import json
import os
import subprocess
import tempfile
import time
from typing import Any, TYPE_CHECKING

from src import config

if TYPE_CHECKING:
    from src.project.project import Project, BGMSegment
    from src.project.media_profiles import MediaProfile
    from src.batch.material_index import MaterialIndex, SceneEntry


# ──────────────────────────────────────────────────────────────
# 短影主类
# ──────────────────────────────────────────────────────────────

class ShortVideoEditor:
    """短影 ✂️ — 根据 shot_plan 生成 shot_point 并渲染成片。"""

    def __init__(
        self,
        video_path: str,
        shot_plan_path: str,
        scene_summary_dir: str,
        audio_caption_path: str,
        scene_cuts_path: str = "",
        instruction: str = "",
        main_character: str = "",
        output_dir: str = "",
        subtitle_path: str = "",
        bgm_path: str = "",
    ):
        self.video_path = video_path
        self.shot_plan_path = shot_plan_path
        self.scene_summary_dir = scene_summary_dir
        self.audio_caption_path = audio_caption_path
        self.scene_cuts_path = scene_cuts_path or ""
        self.instruction = instruction
        self.main_character = main_character or config.MAIN_CHARACTER_NAME
        self.subtitle_path = subtitle_path
        self.bgm_path = bgm_path or audio_caption_path

        # Build output paths
        if not output_dir:
            video_id = os.path.splitext(os.path.basename(video_path))[0].replace(".", "_").replace(" ", "_")
            # 从 audio_caption_path 提取 BGM 目录名，而非文件名
            audio_dir = os.path.basename(os.path.dirname(os.path.dirname(audio_caption_path))) if audio_caption_path else "no_audio"
            audio_id = audio_dir.replace(".", "_").replace(" ", "_")
            output_dir = os.path.join(config.VIDEO_DATABASE_FOLDER, "Output", f"{video_id}_{audio_id}")
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self._shot_point_path = os.path.join(self.output_dir, "shot_point.json")
        self._result = None

    # ══════════════════════════════════════════════════════════ #
    #  工厂方法：从项目创建 ShortVideoEditor（批量模式）          #
    # ══════════════════════════════════════════════════════════ #

    @classmethod
    def from_project(
        cls,
        project: "Project",
        material_index: "MaterialIndex",
        profile: "MediaProfile",
        shot_plan_path: str,
        video_type: str = "vlog",
        main_character: str = "",
    ) -> "ShortVideoEditor":
        """
        从项目创建 ShortVideoEditor（工厂方法）。

        按 clip_file_path 对 shot_plan 中的 shots 分组，
        每组创建独立的 DirectShotSelector 调用，最后合并结果。

        Args:
            project: 批量项目
            material_index: 全局素材索引
            profile: 输出平台配置
            shot_plan_path: Phase 3 生成的 shot_plan 路径
            video_type: "film" 或 "vlog"
            main_character: 主角名字

        Returns:
            ShortVideoEditor 实例（批量模式）
        """
        # 从 BGM 配置推导 audio_caption_path
        audio_caption_path = ""
        if project.bgm_config.segments:
            first_seg = project.bgm_config.segments[0]
            potential_caption = os.path.join(
                os.path.dirname(first_seg.audio_path), "captions.json"
            )
            if os.path.exists(potential_caption):
                audio_caption_path = potential_caption

        # 创建实例（使用项目基础路径作为占位）
        editor = cls(
            video_path=project.base_dir,
            shot_plan_path=shot_plan_path,
            scene_summary_dir=os.path.join(project.output_dir, "virtual_scenes"),
            audio_caption_path=audio_caption_path,
            output_dir=os.path.join(project.output_dir, "shot_points"),
            instruction="",
            main_character=main_character,
        )
        editor._project = project
        editor._material_index = material_index
        editor._profile = profile
        editor._video_type = video_type
        return editor

    def generate_shot_point_project(
        self,
        shot_point_context: str = "",
    ) -> dict | None:
        """
        批量模式：按 clip 分组生成 shot_point，合并为 v2.0 格式。

        流程：
        1. 加载 shot_plan，按 clip_file_path 分组 shots
        2. 对每组创建过滤版 shot_plan + 虚拟 scene 目录
        3. 调用 DirectShotSelector 处理每组
        4. 合并结果，注入 v2.0 元数据
        """
        from src.batch.scene_adapter import write_virtual_scene_dir
        from src.direct_shot_selector import DirectShotSelector

        if not os.path.exists(self.shot_plan_path):
            print(f"❌ [ShortVideoEditor] shot_plan 不存在: {self.shot_plan_path}")
            return None

        with open(self.shot_plan_path, "r", encoding="utf-8") as f:
            shot_plan = json.load(f)

        # 按 clip_file_path 分组 shots
        clip_groups = self._group_shots_by_clip(shot_plan)
        if not clip_groups:
            print("❌ [ShortVideoEditor] shot_plan 中没有带 clip_file_path 的 shots")
            return None

        print(f"\n{'='*60}")
        print(f"✂️  [ShortVideoEditor] 批量模式：{len(clip_groups)} 个 clip 待处理")
        print(f"{'='*60}\n")

        all_shots = []
        scene_mapping = getattr(self, '_material_index', None)

        for clip_path, group in clip_groups.items():
            clip_id = group["clip_id"]
            shots = group["shots"]
            print(f"\n🎬 处理 clip: {clip_id} ({len(shots)} 个 shots)")

            # 创建该 clip 的过滤版 shot_plan
            filtered_plan = self._build_filtered_shot_plan(shot_plan, clip_path, shots)

            # 创建该 clip 的虚拟 scene 目录
            clip_scenes = [s for s in (scene_mapping.scenes if scene_mapping else [])
                          if s.clip_file_path == clip_path]
            virtual_dir = None
            if clip_scenes:
                from src.batch.material_index import MaterialIndex as MI
                clip_index = MI(project_id="", scenes=clip_scenes)
                virtual_dir = tempfile.mkdtemp(prefix=f"cutclaw_{clip_id}_")
                write_virtual_scene_dir(clip_index, virtual_dir)

            try:
                # 创建 DirectShotSelector
                filtered_plan_path = os.path.join(self.output_dir, f"shot_plan_{clip_id}.json")
                with open(filtered_plan_path, "w", encoding="utf-8") as f:
                    json.dump(filtered_plan, f, ensure_ascii=False, indent=2)

                shot_point_path = os.path.join(self.output_dir, f"shot_point_{clip_id}.json")
                # 添加跨 clip 上下文到 prompt，让 LLM 知道这是项目级别的一小部分
                cross_clip_context = (
                    f"[批量模式] 这是 clip {clip_id} 的镜头选择。\n"
                    f"该 clip 有 {len(shots)} 个 shot 来自整个项目的 shot_plan。\n"
                    f"请专注于本 clip 内的镜头选择，不需要考虑跨 clip 编排（由 PlannerAgent 完成）。\n"
                )
                full_context = (cross_clip_context + shot_point_context) if shot_point_context else cross_clip_context

                # 修正 (BUG-RT-07): audio_caption_path 为空时，从 BGM 配置推导或生成占位
                actual_audio_caption = self.audio_caption_path
                if not actual_audio_caption and hasattr(self, '_project') and self._project.bgm_config.segments:
                    seg = self._project.bgm_config.segments[0]
                    potential = os.path.join(os.path.dirname(seg.audio_path), "captions.json")
                    if os.path.exists(potential):
                        actual_audio_caption = potential
                if not actual_audio_caption and hasattr(self, '_project') and self._project.bgm_config.segments:
                    seg = self._project.bgm_config.segments[0]
                    actual_audio_caption = os.path.join(self.output_dir, "BGM_captions_placeholder.json")
                    if not os.path.exists(actual_audio_caption):
                        placeholder = {
                            "metadata": {"bpm": 0, "section_count": 0, "audio_duration": seg.duration_sec},
                            "sections": [],
                            "sub_segments": [],
                        }
                        with open(actual_audio_caption, "w", encoding="utf-8") as f:
                            json.dump(placeholder, f, ensure_ascii=False, indent=2)
                if not actual_audio_caption:
                    actual_audio_caption = ""  # 仍可能为空，DirectShotSelector 需自行防御

                selector = DirectShotSelector(
                    video_path=clip_path,
                    shot_plan_path=filtered_plan_path,
                    scene_summary_dir=virtual_dir or self.scene_summary_dir,
                    audio_caption_path=actual_audio_caption,
                    instruction=self.instruction,
                    main_character=self.main_character,
                    output_path=shot_point_path,
                    shot_point_context=full_context,
                )
                # 注入 clip 元数据，selector.run() 会自动添加到每个 shot
                # 修正 (BUG-RT-08): 用 start_sec 匹配（比 id/shot_idx 更稳定，
                # 因为不同 LLM 输出可能用不同字段名）
                shot_scene_ids = {}
                if scene_mapping:
                    # 构建该 clip 的 scene_id 集合
                    clip_scene_id_set = {s.scene_id for s in scene_mapping.scenes
                                         if s.clip_id == clip_id}
                    for s in shots:
                        related = s.get("related_scene")
                        start_sec = round(s.get("start_sec", 0), 2)
                        # related_scene 可能是整数下标或字符串 ID
                        if isinstance(related, int) and 0 <= related < len(scene_mapping.scenes):
                            entry = scene_mapping.scenes[related]
                            if entry.clip_id == clip_id:
                                shot_scene_ids[start_sec] = entry.scene_id
                        elif isinstance(related, str) and related in clip_scene_id_set:
                            shot_scene_ids[start_sec] = related

                selector._clip_metadata = {
                    "clip_file_path": clip_path,
                    "clip_id": clip_id,
                    "shot_scene_ids": shot_scene_ids,
                }

                result = selector.run()

                if result and isinstance(result, dict) and "shots" in result:
                    all_shots.extend(result["shots"])
                    print(f"  ✅ {clip_id}: {len(result['shots'])} shots")
                else:
                    print(f"  ⚠️ {clip_id}: 无结果")
            finally:
                # 清理临时目录（无论成功或失败）
                if virtual_dir:
                    import shutil
                    shutil.rmtree(virtual_dir, ignore_errors=True)

        # 合并结果
        final_result = {
            "version": "2.0",
            "project_id": self._project.project_id if hasattr(self, '_project') else "",
            "profile": self._profile.name if hasattr(self, '_profile') else "",
            "shots": all_shots,
            "hook_dialogue_used": None,
        }

        # 注入 v2.0 元数据
        self._inject_v2_metadata(final_result)

        # 保存
        output_path = os.path.join(self.output_dir, f"shot_point_{getattr(self, '_profile', type('', (), {'name': 'project'})()).name}.json")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(final_result, f, ensure_ascii=False, indent=2)
        self._shot_point_path = output_path
        self._result = final_result

        print(f"\n{'='*60}")
        print(f"✅ [ShortVideoEditor] 批量 shot_point 生成完成：{len(all_shots)} shots")
        print(f"💾 保存到: {output_path}")
        print(f"{'='*60}")

        return final_result

    @staticmethod
    def _group_shots_by_clip(shot_plan: dict) -> dict[str, dict]:
        """按 clip_file_path 对 shot_plan 中的 shots 分组。

        返回: {clip_path: {"clip_id": str, "shots": [shot_dict, ...]}}
        """
        groups: dict[str, dict] = {}
        for section in shot_plan.get("video_structure", []):
            sp = section.get("shot_plan", {})
            for shot in sp.get("shots", []):
                clip_path = shot.get("clip_file_path", "")
                if not clip_path:
                    continue
                if clip_path not in groups:
                    groups[clip_path] = {
                        "clip_id": shot.get("clip_id", os.path.splitext(os.path.basename(clip_path))[0]),
                        "shots": [],
                    }
                groups[clip_path]["shots"].append(shot)
        return groups

    @staticmethod
    def _build_filtered_shot_plan(
        original_plan: dict,
        clip_path: str,
        shots: list[dict],
    ) -> dict:
        """从完整 shot_plan 中筛选出属于指定 clip 的 shots，构建过滤版。

        保持 section 结构，但只包含目标 clip 的 shots。
        """
        import copy
        filtered = copy.deepcopy(original_plan)

        for section in filtered.get("video_structure", []):
            sp = section.get("shot_plan", {})
            original_shots = sp.get("shots", [])
            # 只保留属于当前 clip 的 shots
            sp["shots"] = [s for s in original_shots if s.get("clip_file_path") == clip_path]

        # 移除空 section
        filtered["video_structure"] = [
            s for s in filtered.get("video_structure", [])
            if s.get("shot_plan", {}).get("shots")
        ]

        return filtered

    @staticmethod
    def _inject_v2_metadata(shot_point: dict):
        """为 shot_point 中的每个 shot 注入 v2.0 缺失字段。"""
        for shot in shot_point.get("shots", []):
            shot.setdefault("speed", 1.0)
            shot.setdefault("transition", "cut")
            shot.setdefault("clip_file_path", "")
            shot.setdefault("clip_id", "")
            shot.setdefault("scene_id", "")

    # ══════════════════════════════════════════════════════════ #
    #  生成 shot_point（精确剪辑点）                             #
    # ══════════════════════════════════════════════════════════ #

    def generate_shot_point(
        self,
        shot_point_context: str = "",
        iteration_note: str = "",
    ) -> dict | None:
        """
        根据 shot_plan 生成 shot_point。

        参数：
            shot_point_context: Tree/锦书给的额外提示词（如"孩子大笑镜头放慢"）
            iteration_note:     修改要求（多轮迭代时用）

        DIRECT_SHOT_SELECTOR_ENABLED=True → DirectShotSelector（直推）
        False → EditorCoreAgent（原 Agent 循环）
        """
        if not os.path.exists(self.shot_plan_path):
            print(f"❌ [ShortVideoEditor] shot_plan 不存在: {self.shot_plan_path}")
            return None

        prompt = shot_point_context
        if iteration_note:
            prompt += f"\n\n【修改要求 - 第 ? 轮】\n{iteration_note}"

        use_direct = getattr(config, "DIRECT_SHOT_SELECTOR_ENABLED", True)

        print(f"\n{'='*60}")
        print(f"✂️  [ShortVideoEditor] Generating shot point...")
        print(f"     mode: {'DirectShotSelector' if use_direct else 'EditorCoreAgent'}")
        print(f"{'='*60}\n")

        t0 = time.time()

        if use_direct:
            from src.direct_shot_selector import DirectShotSelector

            selector = DirectShotSelector(
                video_path=self.video_path,
                shot_plan_path=self.shot_plan_path,
                scene_summary_dir=self.scene_summary_dir,
                audio_caption_path=self.audio_caption_path,
                scene_cuts_path=self.scene_cuts_path if os.path.exists(self.scene_cuts_path) else None,
                instruction=self.instruction,
                main_character=self.main_character,
                output_path=self._shot_point_path,
                subtitle_path=self.subtitle_path if os.path.exists(self.subtitle_path) else None,
                shot_point_context=prompt,
            )
            result = selector.run()

        else:
            from src.core import EditorCoreAgent, ParallelShotOrchestrator

            frames_dir = os.path.join(os.path.dirname(self.output_dir), "Video",
                                       os.path.splitext(os.path.basename(self.video_path))[0], "frames")
            caption_file = os.path.join(os.path.dirname(self.output_dir), "Video",
                                         os.path.splitext(os.path.basename(self.video_path))[0],
                                         "captions", "captions.json")

            use_parallel = config.VIDEO_TYPE == "film" and getattr(config, "PARALLEL_SHOT_ENABLED", True)

            if use_parallel:
                orchestrator = ParallelShotOrchestrator(
                    video_caption_path=caption_file,
                    video_scene_path=self.scene_summary_dir,
                    audio_caption_path=self.audio_caption_path,
                    output_path=self._shot_point_path,
                    max_iterations=getattr(config, "EDITOR_MAX_ITERATIONS", 10),
                    video_path=self.video_path,
                    frame_folder_path=frames_dir,
                    transcript_path=self.subtitle_path if os.path.exists(self.subtitle_path) else None,
                    max_workers=getattr(config, "PARALLEL_SHOT_MAX_WORKERS", 4),
                    max_reruns=getattr(config, "PARALLEL_SHOT_MAX_RERUNS", 2),
                )
                result = orchestrator.run_parallel(shot_plan_path=self.shot_plan_path)
            else:
                editor_agent = EditorCoreAgent(
                    video_caption_path=caption_file,
                    video_scene_path=self.scene_summary_dir,
                    audio_caption_path=self.audio_caption_path,
                    output_path=self._shot_point_path,
                    max_iterations=getattr(config, "EDITOR_MAX_ITERATIONS", 10),
                    video_path=self.video_path,
                    frame_folder_path=frames_dir,
                    transcript_path=self.subtitle_path if os.path.exists(self.subtitle_path) else None,
                )
                result = editor_agent.run(shot_plan_path=self.shot_plan_path)

        elapsed = time.time() - t0
        self._result = result

        if result is None:
            print(f"❌ [ShortVideoEditor] Shot point generation failed ({elapsed:.1f}s)")
        else:
            n_shots = len(result.get("shots", result)) if isinstance(result, dict) else len(result) if isinstance(result, list) else 0
            print(f"✅ [ShortVideoEditor] Shot point generated: {n_shots} shots in {elapsed:.1f}s")
            print(f"💾 Saved to: {self._shot_point_path}")

        return result

    # ══════════════════════════════════════════════════════════ #
    #  预览：dry-run 文本摘要                                    #
    # ══════════════════════════════════════════════════════════ #

    def summarize_shot_point(self, shot_point_path: str = "") -> str:
        """
        从 shot_point.json 读取并生成可读摘要，供 Tree 预览。
        等价于 render --dry-run 的输出。
        """
        path = shot_point_path or self._shot_point_path
        if not os.path.exists(path):
            return "⚠️ shot_point 尚未生成"

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        shots = data.get("shots", data) if isinstance(data, dict) else data
        if not isinstance(shots, list):
            shots = [shots]

        lines = []
        lines.append("=" * 60)
        lines.append("📋 shot_point 预览 — ShortVideoEditor ✂️")
        lines.append("=" * 60)
        lines.append("")

        total_dur = 0.0
        current_section = -1

        for i, clip in enumerate(shots):
            section_idx = clip.get("section_idx", clip.get("section", -1))
            shot_idx = clip.get("shot_idx", clip.get("id", i))

            if isinstance(clip, dict) and "clips" in clip:
                sub_clips = clip["clips"]
            else:
                sub_clips = [clip]

            for sc in sub_clips:
                start = sc.get("start_sec", sc.get("start", 0))
                end = sc.get("end_sec", sc.get("end", 0))
                dur = end - start if end > start else sc.get("duration", 0)
                total_dur += dur

                # Section header
                if section_idx != current_section:
                    lines.append(f"\n[Section {section_idx}]")
                    current_section = section_idx

                start_str = sc.get("start_str", f"{start:.2f}s")
                end_str = sc.get("end_str", f"{end:.2f}s")
                reason = sc.get("reasoning", clip.get("reasoning", ""))
                reason_str = f"  ← {reason[:50]}" if reason else ""

                lines.append(f"  Shot {shot_idx}: {start_str} → {end_str} ({dur:.2f}s){reason_str}")

        lines.append("")
        lines.append("-" * 60)
        lines.append(f"Total clips: {len(shots)}")
        lines.append(f"Total duration: {total_dur:.2f}s ({total_dur/60:.2f} min)")
        lines.append("=" * 60)

        return "\n".join(lines)

    def dry_run(self, shot_point_path: str = "", crop_ratio: str = "") -> str:
        """
        调用 render_video.py --dry-run 获取标准摘要。
        比 summarize_shot_point 更详细（含 hook_dialogue 信息）。
        """
        path = shot_point_path or self._shot_point_path
        if not os.path.exists(path):
            return "⚠️ shot_point 尚未生成"

        shot_plan_dir = os.path.dirname(self.shot_plan_path)
        shot_plan_name = os.path.basename(self.shot_plan_path).replace("shot_plan_", "shot_point_")

        cmd = [
            "python", "render/render_video.py",
            "--shot-json", path,
            "--shot-plan", self.shot_plan_path,
            "--video", self.video_path,
            "--audio", self.audio_caption_path,
            "--output", "/dev/null",
            "--dry-run",
            "--no-labels",
        ]
        if crop_ratio:
            cmd += ["--crop-ratio", crop_ratio]

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=os.path.join(os.path.dirname(__file__), ".."),
                env=env,
            )
            output = result.stdout or result.stderr or ""
            if result.returncode != 0:
                return f"⚠️ dry-run 运行失败:\n{output[-2000:]}"
            return output
        except Exception as e:
            return f"⚠️ dry-run 异常: {e}"

    # ══════════════════════════════════════════════════════════ #
    #  渲染成片                                                  #
    # ══════════════════════════════════════════════════════════ #

    def render(
        self,
        crop_ratio: str = "9:16",
        shot_point_path: str = "",
        output_path: str = "",
    ) -> str | None:
        """
        调用 render_video.py 合成最终视频。

        参数：
            crop_ratio: "9:16" / "16:9" / "1:1"
            shot_point_path: 可选，指定 shot_point 文件
            output_path: 可选，指定输出路径

        返回：
            输出路径，失败返回 None
        """
        sp_path = shot_point_path or self._shot_point_path
        if not os.path.exists(sp_path):
            print(f"❌ [ShortVideoEditor] shot_point 不存在: {sp_path}")
            return None

        out = output_path or os.path.join(self.output_dir, f"output_{crop_ratio.replace(':', 'x')}.mp4")

        ending_video = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                     "resource", "ending", "ending.mp4")
        dialogue_font = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                      "resource", "font", "Pulp Fiction Italic M54.ttf")

        cmd = [
            "python", "render/render_video.py",
            "--shot-json", sp_path,
            "--shot-plan", self.shot_plan_path,
            "--video", self.video_path,
            "--audio", self.audio_caption_path,
            "--output", out,
            "--crop-ratio", crop_ratio,
            "--no-labels",
            "--render-hook-dialogue",
        ]
        if os.path.exists(ending_video):
            cmd += ["--ending-video", ending_video]
        if os.path.exists(dialogue_font):
            cmd += ["--dialogue-font", dialogue_font]

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"

        print(f"\n{'='*60}")
        print(f"🎥 [ShortVideoEditor] Rendering video...")
        print(f"   crop: {crop_ratio}")
        print(f"   output: {out}")
        print(f"{'='*60}\n")

        t0 = time.time()
        project_root = os.path.join(os.path.dirname(__file__), "..")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=project_root,
            env=env,
        )

        elapsed = time.time() - t0

        if result.returncode != 0:
            print(f"❌ [ShortVideoEditor] Render failed ({elapsed:.1f}s)")
            if result.stderr:
                print(result.stderr[-1000:])
            return None

        print(f"✅ [ShortVideoEditor] Render completed in {elapsed:.1f}s")
        print(f"💾 Output: {out}")
        return out

    def render_project(
        self,
        shot_point_path: str = "",
        crop_ratio: str = "9:16",
        extract_timeout: int = 600,
        ending_video_path: str = "",
        ending_duration: float = 0.0,
        ending_fade_duration: float = 0.5,
    ) -> dict:
        """
        批量模式渲染：使用 MultiSourceRenderer 从多源视频渲染。

        Args:
            shot_point_path: shot_point v2.0 路径（默认使用最近生成的）
            crop_ratio: 裁切比例
            extract_timeout: 单个片段提取超时（秒，默认 600）
            ending_video_path: 结尾视频路径（修正 #8 集成）
            ending_duration: 截取结尾视频的时长
            ending_fade_duration: 交叉淡入淡出

        Returns:
            渲染结果 dict
        """
        from render.multi_source_renderer import MultiSourceRenderer
        from src.project.media_profiles import get_profile

        sp_path = shot_point_path or self._shot_point_path
        if not os.path.exists(sp_path):
            print(f"❌ [短影] shot_point 不存在: {sp_path}")
            return {"status": "fail", "error": "shot_point 不存在"}

        profile = getattr(self, '_profile', get_profile("bilibili_4k"))

        # 确定 BGM 路径
        bgm_path = self.bgm_path or ""
        bgm_segments = []
        if hasattr(self, '_project') and self._project.bgm_config.segments:
            bgm_segments = self._project.bgm_config.segments
            if bgm_segments:
                bgm_path = bgm_segments[0].audio_path

        # 修正 (BUG-RT-13): 成片应输出到 <project.output_dir>/output/ 而非 shot_points/
        if hasattr(self, '_project') and self._project and self._project.output_dir:
            output_dir = os.path.join(self._project.output_dir, "output")
        else:
            output_dir = self.output_dir
        output_path = os.path.join(output_dir, f"output_{profile.name}.mp4")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        try:
            renderer = MultiSourceRenderer(
                shot_point_path=sp_path,
                profile=profile,
                bgm_path=bgm_path,
                bgm_start_sec=bgm_segments[0].start_sec if bgm_segments else 0.0,
                bgm_duration_sec=bgm_segments[0].duration_sec if bgm_segments else 0.0,
                extract_timeout=extract_timeout,
                ending_video_path=ending_video_path,
                ending_duration=ending_duration,
                ending_fade_duration=ending_fade_duration,
            )
            result = renderer.render(output_path)
            return {
                "status": result.status,
                "output_path": result.output_path,
                "duration_sec": result.duration_sec,
                "file_size_mb": result.file_size_mb,
                "warnings": result.warnings,
                "errors": result.errors,
            }
        except Exception as e:
            return {"status": "fail", "output_path": output_path, "error": str(e)}


# ──────────────────────────────────────────────────────────────
# 可作为独立脚本运行
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ShortVideoEditor — 短影剪辑师引擎")
    parser.add_argument("--shot-plan", required=True, help="shot_plan.json 路径")
    parser.add_argument("--scene-summaries", required=True, help="scene_summaries_video 目录")
    parser.add_argument("--audio-captions", required=True, help="audio/captions.json 路径")
    parser.add_argument("--video", required=True, help="原始视频路径")
    parser.add_argument("--scene-cuts", default="", help="shot_scenes.txt 路径")
    parser.add_argument("--instruction", default="", help="剪辑指令")
    parser.add_argument("--output-dir", default="", help="输出目录")
    parser.add_argument("--shot-point-context", default="", help="锦书额外提示词")
    parser.add_argument("--action", default="shot_point", choices=["shot_point", "dry_run", "render"],
                        help="执行动作")

    args = parser.parse_args()

    editor = ShortVideoEditor(
        video_path=args.video,
        shot_plan_path=args.shot_plan,
        scene_summary_dir=args.scene_summaries,
        audio_caption_path=args.audio_captions,
        scene_cuts_path=args.scene_cuts,
        instruction=args.instruction,
        output_dir=args.output_dir,
    )

    if args.action == "shot_point":
        result = editor.generate_shot_point(shot_point_context=args.shot_point_context)
        if result:
            print(f"\n=== 预览 ===")
            print(editor.summarize_shot_point())

    elif args.action == "dry_run":
        text = editor.dry_run()
        print(text)

    elif args.action == "render":
        editor.generate_shot_point(shot_point_context=args.shot_point_context)
        editor.render()
