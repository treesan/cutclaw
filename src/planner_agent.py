"""
PlannerAgent — 锦书的策划执行引擎
===================================

锦书（制片人）的工作流：
1. 编写 strategy_context（详细剪辑策略）
2. 调用 PlannerAgent 生成 shot_plan
3. 审查 shot_plan，如果不满意 → 修改 strategy → 重跑
4. 满意后将 shot_plan 传给短影生成 shot_point
5. 短影渲染成片 → 发给 Tree 确认

用法：
    from src.planner_agent import PlannerAgent

    planner = PlannerAgent(
        video_path="path/to/video.MOV",
        scene_summaries_dir="path/to/scene_summaries",
        audio_caption_path="path/to/audio_caption.json",
        subtitle_path="path/to/subtitles.srt",
        bgm_name="bgm.mp3",
        output_dir="path/to/output",
    )

    # 锦书写剪辑策略
    strategy = \"\"\"
    重点保留孩子画画的镜头（00:33-00:40），作为情感高潮。
    前5秒用快切建立节奏，后10秒放缓展现温馨互动。
    副歌部分切到爸爸开车的侧脸 + 阳光。
    \"\"\"

    # Step 1: 生成 shot_plan
    plan = planner.generate_shot_plan(strategy_context=strategy)
    print(planner.summarize_shot_plan())  # 锦书审查

    # Step 2: 生成给 Tree 的汇报
    report = planner.generate_report()
    print(report)
"""

import os
import json
import time
import hashlib
import re
from typing import Optional, Dict, Any, List, TYPE_CHECKING

if TYPE_CHECKING:
    from src.project.project import Project
    from src.project.media_profiles import MediaProfile
    from src.project.project import BGMSegment
    from src.batch.material_index import MaterialIndex, SceneEntry

try:
    from . import config as project_config
except ImportError:
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import src.config as project_config
    config = project_config


class PlannerAgent:
    """
    锦书策划引擎 — 生成 shot_plan，并产出给 Tree 的汇报。

    调用流程（锦书视角）：
    1. 初始化 PlannerAgent（传入短影产出的素材路径）
    2. 写 strategy_context → generate_shot_plan() → 审查 → 迭代
    3. 审查通过后将 shot_plan 传给短影（由短影生成 shot_point + 渲染）
    4. generate_report() → 发给 Tree 确认
    """

    def __init__(
        self,
        video_path: str,
        scene_summaries_dir: str,
        audio_caption_path: str = "",
        subtitle_path: Optional[str] = None,
        bgm_path: Optional[str] = None,
        bgm_name: Optional[str] = None,
        output_dir: str = "Output/Output",
        video_type: str = "film",
        main_character: Optional[str] = None,
        video_duration: Optional[float] = None,
    ):
        """
        Args:
            video_path: 视频文件路径
            scene_summaries_dir: scene_summaries_video 目录
            audio_caption_path: audio/captions.json 路径（BGM 分析结果，锦书可直接传已有的）
            subtitle_path: 字幕 SRT 路径
            bgm_path: BGM 音频文件路径（锦书传入后，PlannerAgent 自动调用 madmom 分析）
            bgm_name: BGM 名称，不传则从 bgm_path 或 audio_caption_path 推导
            output_dir: 输出目录
            video_type: "film" 或 "vlog"
            main_character: 主角名字
            video_duration: 视频时长（秒），可选
        """
        self.video_path = video_path
        self.scene_summaries_dir = scene_summaries_dir
        self.audio_caption_path = audio_caption_path
        self.bgm_path = bgm_path
        self.subtitle_path = subtitle_path
        self.output_dir = output_dir
        self.video_type = video_type
        self.main_character = main_character
        self.video_duration = video_duration  # 可选，用于报告

        # 如果 bgm_name 没传，自动从 bgm_path 或 audio_caption_path 推导
        if not bgm_name:
            if bgm_path:
                bgm_name = os.path.basename(bgm_path)
            elif audio_caption_path:
                bgm_name = os.path.basename(audio_caption_path)
            else:
                bgm_name = "无配乐"
        self.bgm_name = bgm_name

        # 记录每次运行的历史（锦书迭代用）
        self._shot_plan_history: List[Dict] = []

        # 构建输出路径
        video_id = os.path.splitext(os.path.basename(video_path))[0].replace(".", "_").replace(" ", "_")
        bgm_id = os.path.splitext(os.path.basename(os.path.splitext(bgm_name)[0] if bgm_name else "no_audio"))[0].replace(".", "_").replace(" ", "_")
        self._shot_plan_path = os.path.join(output_dir, f"shot_plan_{video_id}_{bgm_id}.json")

    # ═══════════════════════════════════════════════════════════════ #
    #  工厂方法：从 Project 创建 PlannerAgent（批量模式）
    # ═══════════════════════════════════════════════════════════════ #

    @classmethod
    def from_project(
        cls,
        project: "Project",
        material_index: "MaterialIndex",
        profile: "MediaProfile",
        bgm_segments: Optional[List["BGMSegment"]] = None,
        video_type: str = "vlog",
        main_character: Optional[str] = None,
        strategy: str = "",
    ) -> tuple["PlannerAgent", dict[int, "SceneEntry"]]:
        """
        从 Project 创建 PlannerAgent（工厂方法，不改动 __init__）。

        实现策略：创建临时 "virtual scene_summaries_dir"，
        将 MaterialIndex 中的 SceneEntry 转换为现有 scene_*.json 格式，
        这样 Screenwriter 完全不需要修改。

        Args:
            project: 批量项目
            material_index: 全局素材索引
            profile: 输出平台配置
            bgm_segments: BGM 分段配置（可选）
            video_type: "film" 或 "vlog"
            main_character: 主角名字

        Returns:
            (planner_agent, scene_mapping) 元组
            scene_mapping: {global_idx: SceneEntry} 用于后处理注入 source_clip/scene_id
        """
        from src.batch.scene_adapter import write_virtual_scene_dir

        # 1. 创建虚拟 scene_summaries 目录
        virtual_scene_dir = os.path.join(project.output_dir, "virtual_scenes")
        scene_mapping = write_virtual_scene_dir(material_index, virtual_scene_dir)

        # 2. 生成 instruction
        day_summary = project.get_day_summary()
        instruction = (
            f"这是一次旅行的批量剪辑项目。\n\n"
            f"{day_summary}\n\n"
            f"输出平台: {profile.name} ({profile.width}x{profile.height}, {profile.fps}fps)\n"
            f"请根据素材内容，生成一个完整的分镜方案。"
        )

        # 修正 (优化 #1-3): 注入实际素材信息，避免 LLM 虚构场景
        # 1. 素材内容摘要：把 material_index 的 caption/location/device 注入 prompt
        if material_index.scenes:
            scene_summaries = []
            for entry in material_index.scenes[:50]:  # 限制 50 个避免 prompt 过长
                loc = entry.location or "未知地点"
                cap = entry.caption or "无描述"
                scene_summaries.append(f"  - {entry.scene_id} ({entry.clip_id[:20]}): {loc} | {cap[:60]}")
            instruction += "\n\n[实际素材内容]\n" + "\n".join(scene_summaries)

        # 2. 设备分布统计
        device_counts: dict[str, int] = {}
        for entry in material_index.scenes:
            clip = next((c for c in project.clips if c.clip_id == entry.clip_id), None)
            dev = clip.device if clip else "unknown"
            device_counts[dev] = device_counts.get(dev, 0) + 1
        if device_counts:
            dev_str = ", ".join(f"{k}:{v}" for k, v in device_counts.items())
            instruction += f"\n\n[素材设备] {dev_str}"
            # 修正 (优化 #3): 明确告知设备类型，避免 LLM 给航拍虚构人物
            device_tips = {
                "dji": "DJI 无人机航拍，无人物出现，描述应为 'aerial shot of...' 或 'drone view of...'",
                "nikon": "尼康相机，可能有或无人物，需根据 caption 描述",
                "unknown": "来源未明，根据 caption 描述",
            }
            for dev, tip in device_tips.items():
                if dev in device_counts:
                    instruction += f"\n  - {tip}"
                    break

        # 3. 强制 strategy 主题（修正 #1）
        if strategy:
            instruction += (
                f"\n\n[强制主题要求]\n"
                f"用户指定的 strategy/策略: '{strategy}'\n"
                f"⚠️ overall_theme 必须包含 strategy 的核心关键词或直接表达 strategy 的意图。\n"
                f"⚠️ 每个 shot 的 content 描述必须基于实际素材的 caption/location，禁止虚构人物或场景。"
            )

        # 修正：注入 BGM 节奏信息，让 shot_plan 考虑音画同步
        if bgm_segments:
            bgm_info_lines = ["\n[BGM 信息]"]
            for i, seg in enumerate(bgm_segments):
                seg_label = f"  - {seg.segment_id} (Day {seg.day_idx or '?'}): {seg.start_sec:.0f}s-{seg.start_sec + seg.duration_sec:.0f}s, 淡入{seg.fade_in_sec}s/淡出{seg.fade_out_sec}s"
                bgm_info_lines.append(seg_label)
            bgm_info_lines.append("\n请让 shot 的节奏与 BGM 段落匹配（高潮段落用大景/慢镜头，安静段落用细节/快切）")
            instruction += "\n".join(bgm_info_lines)

        # 3. 处理 BGM — 修正 (优化 #4): 自动分析 BGM 节奏，生成 captions.json
        bgm_path = None
        audio_caption_path = ""
        if bgm_segments:
            first_seg = bgm_segments[0]
            bgm_path = first_seg.audio_path
            # 查找已有的 captions.json
            potential_caption = os.path.join(
                os.path.dirname(first_seg.audio_path), "captions.json"
            )
            if os.path.exists(potential_caption):
                audio_caption_path = potential_caption
                print(f"🎵 使用已有 BGM 分析: {potential_caption}")
            elif os.path.exists(bgm_path):
                # 修正 (优化 #4): 运行 madmom 节奏分析生成 captions.json
                # 替代之前的空占位符，让 Screenwriter 获得真实 BPM/sections 数据
                from src.audio.audio_caption_madmom import caption_audio_with_madmom_segments
                bgm_id = os.path.splitext(os.path.basename(bgm_path))[0].replace(".", "_").replace(" ", "_")
                caption_dir = os.path.join(project.output_dir, "bgm_captions")
                os.makedirs(caption_dir, exist_ok=True)
                audio_caption_path = os.path.join(caption_dir, f"captions_{bgm_id}.json")
                if not os.path.exists(audio_caption_path):
                    print(f"🎵 分析 BGM 节奏: {bgm_path}")
                    try:
                        caption_audio_with_madmom_segments(
                            audio_path=bgm_path,
                            output_path=audio_caption_path,
                        )
                        print(f"✅ BGM 分析完成: {audio_caption_path}")
                    except Exception as e:
                        print(f"⚠️ BGM 分析失败: {e}，使用占位符")
                        # 降级：生成占位符让 Screenwriter 不崩溃
                        audio_caption_path = os.path.join(caption_dir, f"placeholder_{bgm_id}.json")
                        if not os.path.exists(audio_caption_path):
                            placeholder = {
                                "metadata": {"bpm": 0, "section_count": 0, "audio_duration": first_seg.duration_sec},
                                "sections": [],
                                "sub_segments": [],
                            }
                            with open(audio_caption_path, "w", encoding="utf-8") as f:
                                json.dump(placeholder, f, ensure_ascii=False, indent=2)
                else:
                    print(f"🎵 使用缓存 BGM 分析: {audio_caption_path}")
            else:
                print(f"⚠️ BGM 文件不存在: {bgm_path}")

        # 4. 设置输出目录
        shot_plans_dir = os.path.join(project.output_dir, "shot_plans")
        os.makedirs(shot_plans_dir, exist_ok=True)

        # 5. 创建 PlannerAgent
        agent = cls(
            video_path=project.base_dir,  # 用项目目录代替单个视频
            scene_summaries_dir=virtual_scene_dir,
            audio_caption_path=audio_caption_path,
            bgm_path=bgm_path,
            output_dir=shot_plans_dir,
            video_type=video_type,
            main_character=main_character,
        )

        # 附加批量模式元数据
        agent._project = project
        agent._scene_mapping = scene_mapping
        agent._profile = profile
        agent._instruction = instruction

        # 覆盖 shot_plan 路径为项目级命名
        agent._shot_plan_path = os.path.join(
            shot_plans_dir, f"shot_plan_{profile.name}.json"
        )

        return agent, scene_mapping

    # ═══════════════════════════════════════════════════════════════ #
    #  Step 0: 锦书自分析 BGM 节奏（新增 — 不再依赖短影）
    # ═══════════════════════════════════════════════════════════════ #

    def analyze_bgm(
        self,
        bgm_path: Optional[str] = None,
        output_subdir: Optional[str] = None,
        **madmom_kwargs,
    ) -> str:
        """
        锦书调用：直接对 BGM 做 madmom 节奏分析，无需短影中转。

        参数：
            bgm_path: BGM 文件路径。不传则用 self.bgm_path
            output_subdir: 分析结果输出子目录（默认自动生成）
            **madmom_kwargs: 透传给 caption_audio_with_madmom_segments 的参数

        返回：
            captions.json 文件路径

        用法：
            planner = PlannerAgent(video_path=..., scene_summaries_dir=...)
            caption_path = planner.analyze_bgm("path/to/bgm.mp3")
            # 此时 audio_caption_path 已自动更新
            plan = planner.generate_shot_plan(strategy_context="...")
        """
        if bgm_path:
            self.bgm_path = bgm_path
        if not self.bgm_path or not os.path.exists(self.bgm_path):
            raise FileNotFoundError(f"BGM 文件不存在: {self.bgm_path}")

        from src.audio.audio_caption_madmom import caption_audio_with_madmom_segments

        # 生成输出路径
        bgm_id = os.path.splitext(os.path.basename(self.bgm_path))[0].replace(".", "_").replace(" ", "_")
        if output_subdir:
            out_dir = os.path.join(output_subdir, "captions")
        else:
            out_dir = os.path.join("Output", "Audio", bgm_id, "captions")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "captions.json")

        print(f"[PlannerAgent] 🎵  Analyzing BGM via madmom: {self.bgm_path}")
        print(f"[PlannerAgent]    Output -> {out_path}")

        result = caption_audio_with_madmom_segments(
            audio_path=self.bgm_path,
            output_path=out_path,
            **madmom_kwargs,
        )

        # 自动更新 audio_caption_path，后续步骤直接使用
        self.audio_caption_path = out_path

        print(f"[PlannerAgent] [ok] BGM analysis complete")
        print(f"[PlannerAgent]    BPM: {result.get('metadata', {}).get('bpm', '?')}")
        print(f"[PlannerAgent]    Sections: {result.get('metadata', {}).get('section_count', '?')}")

        return out_path

    # ═══════════════════════════════════════════════════════════════ #
    #  Step 1: 生成 shot_plan（创意分镜方案）
    # ═══════════════════════════════════════════════════════════════ #

    def generate_shot_plan(
        self,
        strategy_context: str = "",
        iteration_note: str = "",
    ) -> Dict[str, Any]:
        """
        锦书调用：生成 shot_plan。

        参数：
            strategy_context: 锦书写的详细剪辑策略（控制剧情走向、节奏、重点镜头等）
            iteration_note:   锦书审查后不满意时的修改要求（留空=首次生成）

        返回：
            shot_plan JSON 内容 + 路径
        """
        from src.Screenwriter_scene_short import Screenwriter

        os.makedirs(os.path.dirname(self._shot_plan_path), exist_ok=True)

        subtitle_actual = None
        if self.video_type == "film" and self.subtitle_path and os.path.exists(self.subtitle_path):
            subtitle_actual = self.subtitle_path

        # 锦书的策略注入 → 合并到 instruction
        # Screenwriter 的 prompt 里有 INSTRUCTION_PLACEHOLDER，
        # 我们把 strategy_context 作为主要的 instruction 传进去
        default_instruction = getattr(self, '_instruction', '')
        instruction = strategy_context if strategy_context else default_instruction

        # 如果有过往迭代记录，追加修改要求
        if iteration_note:
            instruction += f"\n\n【锦书修改要求 - 第 {len(self._shot_plan_history) + 1} 轮】\n{iteration_note}"

        screenwriter = Screenwriter(
            video_scene_path=self.scene_summaries_dir,
            audio_caption_path=self.audio_caption_path,
            output_path=self._shot_plan_path,
            video_path=self.video_path,
            subtitle_path=subtitle_actual,
            main_character=self.main_character,
            max_iterations=getattr(project_config, "EDITOR_MAX_ITERATIONS", 10),
        )

        print(f"[PlannerAgent] 🖊️  Generating shot plan (iteration {len(self._shot_plan_history) + 1})...")
        t0 = time.time()
        shot_plan = screenwriter.run(instruction)
        elapsed = time.time() - t0

        record = {
            "iteration": len(self._shot_plan_history) + 1,
            "strategy": strategy_context[:200] if strategy_context else "",
            "iteration_note": iteration_note[:200] if iteration_note else "",
            "elapsed": round(elapsed, 1),
            "path": self._shot_plan_path,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        self._shot_plan_history.append(record)

        print(f"[PlannerAgent] [ok] Shot plan generated in {elapsed:.1f}s")
        print(f"[PlannerAgent] 💾 Saved to: {self._shot_plan_path}")

        return {"shot_plan": shot_plan, "path": self._shot_plan_path, "elapsed": elapsed, "iteration": len(self._shot_plan_history)}

    # ═══════════════════════════════════════════════════════════════ #
    #  批量模式：生成项目级 shot_plan（注入跨 Clip 元数据）
    # ═══════════════════════════════════════════════════════════════ #

    def generate_project_shot_plan(
        self,
        scene_mapping: dict[int, "SceneEntry"],
        strategy_context: str = "",
    ) -> Dict[str, Any]:
        """
        批量模式：生成 shot_plan 并注入 source_clip / scene_id。

        流程：
        1. 调用 generate_shot_plan() 生成基础 shot_plan
        2. 遍历 shots，用 scene_mapping 注入跨 Clip 元数据
        3. 保存增强版 shot_plan

        Args:
            scene_mapping: {global_idx: SceneEntry} 映射（来自 from_project()）
            strategy_context: 锦书写的剪辑策略

        Returns:
            与 generate_shot_plan() 相同格式
        """
        # Step 1: 生成基础 shot_plan
        result = self.generate_shot_plan(strategy_context=strategy_context)

        # Step 2: 注入 source_clip / scene_id
        shot_plan = result["shot_plan"]
        if not shot_plan:
            return result

        injected_count = 0
        video_structure = shot_plan.get("video_structure", [])
        for section in video_structure:
            shot_plan_data = section.get("shot_plan", {})
            shots = shot_plan_data.get("shots", [])
            for shot in shots:
                related_scene = shot.get("related_scene")
                if related_scene is not None and isinstance(related_scene, int):
                    entry = scene_mapping.get(related_scene)
                    if entry:
                        shot["source_clip"] = entry.clip_id
                        shot["clip_file_path"] = entry.clip_file_path
                        shot["scene_id"] = entry.scene_id
                        shot["day_idx"] = entry.day_idx
                        injected_count += 1

        # Step 3: 保存增强版 shot_plan
        with open(self._shot_plan_path, "w", encoding="utf-8") as f:
            json.dump(shot_plan, f, ensure_ascii=False, indent=2)

        print(f"[PlannerAgent] 🔗 Injected source_clip/scene_id into {injected_count} shots")
        result["shot_plan"] = shot_plan
        return result

    def summarize_shot_plan(self) -> str:
        """
        锦书审查用：从 shot_plan JSON 提取可读摘要。
        """
        if not os.path.exists(self._shot_plan_path):
            return "⚠️ shot_plan 尚未生成"

        with open(self._shot_plan_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        lines = []
        lines.append("[clip] 分镜方案审查")
        lines.append("")

        # 整体主题
        theme = data.get("overall_theme", data.get("metadata", {}).get("overall_theme", "未指定"))
        lines.append(f"🎯 主题：{theme}")

        # 配乐段落
        meta = data.get("metadata", {})
        audio_start = meta.get("selected_audio_start", "?")
        audio_end = meta.get("selected_audio_end", "?")
        lines.append(f"[music] 配乐段落：{audio_start} → {audio_end}")

        # Hook 对话
        hook = data.get("hook_dialogue", {})
        if hook:
            hook_text = hook.get("reason", "")
            hook_dur = hook.get("duration_seconds", 0)
            lines.append(f"💬 Hook 对话：{hook_text[:80]}（{hook_dur:.1f}s）")
            for tl in hook.get("timed_lines", []):
                lines.append(f"   · 「{tl.get('text', '')[:40]}」{tl.get('start','')}→{tl.get('end','')}")

        # 分镜列表
        video_struct = data.get("video_structure", [])
        if video_struct:
            lines.append("")
            lines.append("[film] 分镜列表：")
            for sec in video_struct:
                shot_plan = sec.get("shot_plan", {})
                shots = shot_plan.get("shots", [])
                lines.append(f"   {sec.get('overall_theme','')[:60]}")
                lines.append(f"   时长：{sec.get('start_time','?')}s → {sec.get('end_time','?')}s | {len(shots)} 个镜头")
                for s in shots[:8]:  # 最多显示 8 个
                    sid = s.get("id", "?")
                    dur = s.get("time_duration", "?")
                    content = s.get("content", "")[:50]
                    lines.append(f"     #{sid} [{dur}s] {content}")
                if len(shots) > 8:
                    lines.append(f"     ... 还有 {len(shots) - 8} 个镜头")
        else:
            lines.append("   （无分镜信息）")

        lines.append("")
        total_dur = sum(
            s.get("shot_plan", {}).get("shots", [{}])[-1].get("time_duration", 0)
            if s.get("shot_plan", {}).get("shots") else 0
            for s in video_struct
        )
        total_shots = sum(len(s.get("shot_plan", {}).get("shots", [])) for s in video_struct)
        lines.append(f"📊 统计：{total_shots} 个镜头 | 总时长约 {total_dur:.0f}s")

        return "\n".join(lines)

    # ═══════════════════════════════════════════════════════════════ #
    #  Step 2: 生成给 Tree 的汇报
    # ═══════════════════════════════════════════════════════════════ #

    def generate_report(self) -> str:
        """
        锦书调用：生成结构化的汇报，发给 Tree 确认。

        输出格式包含：
        - 素材概览
        - 分镜方案
        - 配乐信息
        """
        video_name = os.path.basename(self.video_path)

        # 获取视频时长
        if self.video_duration:
            dur_str = f"{self.video_duration:.0f}s"
        else:
            dur_str = "?"

        lines = [
            f"[report] 锦书策划汇报",
            "",
            f"素材：{video_name} ({dur_str})",
            f"BGM：{self.bgm_name}",
        ]

        # 读取 shot_plan
        if os.path.exists(self._shot_plan_path):
            with open(self._shot_plan_path, "r", encoding="utf-8") as f:
                plan_data = json.load(f)
            theme = plan_data.get("overall_theme", plan_data.get("metadata", {}).get("overall_theme", "?"))
            meta = plan_data.get("metadata", {})
            audio_start = meta.get("selected_audio_start", "?")
            audio_end = meta.get("selected_audio_end", "?")
            lines.append(f"分镜主题：{theme}")
            lines.append(f"配乐段落：{audio_start} → {audio_end}")

            # 分镜列表
            video_struct = plan_data.get("video_structure", [])
            if video_struct:
                lines.append("")
                lines.append("[film] 分镜列表：")
                for sec in video_struct:
                    shot_plan = sec.get("shot_plan", {})
                    shots = shot_plan.get("shots", [])
                    lines.append(f"  {sec.get('overall_theme', '')[:60]} | {len(shots)} 个镜头")
        else:
            lines.append("（shot_plan 尚未生成）")

        return "\n".join(lines)

    # ═══════════════════════════════════════════════════════════════ #
    #  便利方法：一键全流程（适合快速测试）
    # ═══════════════════════════════════════════════════════════════ #

    def run(
        self,
        strategy_context: str = "",
    ) -> Dict[str, Any]:
        """生成 shot_plan（锦书用）。"""
        result = {
            "video_path": self.video_path,
            "shot_plan_path": self._shot_plan_path,
        }

        r1 = self.generate_shot_plan(strategy_context=strategy_context)
        result["shot_plan_elapsed"] = r1["elapsed"]

        return result


# ──────────────────────────────────────────────────────────────
# CLI 入口
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PlannerAgent — 锦书策划引擎 CLI")
    parser.add_argument("--video", required=True, help="视频路径")
    parser.add_argument("--scene-summaries", required=True, help="scene_summaries_video 目录")
    parser.add_argument("--audio-captions", default="", help="audio/captions.json 路径（已有分析结果时传入）")
    parser.add_argument("--bgm-path", default="", help="BGM 音频文件路径（锦书传入后自动分析）")
    parser.add_argument("--subtitle", default="", help="字幕 SRT 路径")
    parser.add_argument("--bgm-name", default="", help="BGM 名称")
    parser.add_argument("--output-dir", default="", help="输出目录")
    parser.add_argument("--strategy", default="", help="剪辑策略（strategy_context）")
    parser.add_argument("--action", default="shot_plan", choices=["shot_plan", "analyze_bgm", "report"],
                        help="执行动作")

    args = parser.parse_args()

    planner = PlannerAgent(
        video_path=args.video,
        scene_summaries_dir=args.scene_summaries,
        audio_caption_path=args.audio_captions or "",
        bgm_path=args.bgm_path or None,
        subtitle_path=args.subtitle or None,
        bgm_name=args.bgm_name or None,
        output_dir=args.output_dir or "Output/Output",
    )

    if args.action == "analyze_bgm":
        if not args.bgm_path:
            print("❌ 请指定 --bgm-path")
            sys.exit(1)
        caption_path = planner.analyze_bgm(bgm_path=args.bgm_path)
        print(f"\n🎵 BGM 分析完成: {caption_path}")
        # 打印摘要
        import json
        with open(caption_path) as f:
            data = json.load(f)
        meta = data.get("metadata", {})
        print(f"    BPM: {meta.get('bpm', '?')}")
        print(f"    分段: {meta.get('section_count', '?')} 段")
        print(f"    总时长: {meta.get('audio_duration', '?'):.1f}s")
        sections = data.get("sections", [])
        for sec in sections:
            print(f"    · {sec.get('section_name', '?'):12s}  {sec.get('start','?'):>8s} → {sec.get('end','?'):>8s}")

    elif args.action == "shot_plan":
        result = planner.generate_shot_plan(strategy_context=args.strategy)
        print(f"\n=== 审查 shot_plan ===")
        print(planner.summarize_shot_plan())

    elif args.action == "report":
        print(planner.generate_report())
