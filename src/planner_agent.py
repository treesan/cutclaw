"""
PlannerAgent — 锦书的策划执行引擎
===================================

锦书（制片人）的工作流：
1. 编写 strategy_context（详细剪辑策略）
2. 调用 PlannerAgent 生成 shot_plan
3. 审查 shot_plan，如果不满意 → 修改 strategy → 重跑
4. 满意后调用 PlannerAgent 生成 shot_point
5. 审查 shot_point，如果不满意 → 重跑
6. 满意后生成汇报 → 发给 Tree 确认

用法：
    from src.planner_agent import PlannerAgent

    planner = PlannerAgent(
        video_path="resource/video/IMG_5747.MOV",
        scene_summaries_dir="Output/Video/IMG_5747/captions/scene_summaries_video",
        audio_caption_path="Output/Audio/nastelbom/captions/captions.json",
        subtitle_path="Output/Video/IMG_5747/subtitles_with_characters.srt",
        bgm_name="nastelbom.mp3",
        output_dir="Output/Output/IMG_5747_nastelbom",
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

    # Step 2: 满意后生成 shot_point
    point = planner.generate_shot_point()
    print(planner.summarize_shot_point())  # 锦书审查

    # Step 3: 生成给 Tree 的汇报
    report = planner.generate_report()
    print(report)
"""

import os
import json
import time
import hashlib
import re
from typing import Optional, Dict, Any, List

try:
    from . import config as project_config
except ImportError:
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import src.config as project_config
    config = project_config


class PlannerAgent:
    """
    锦书策划引擎 — 生成 shot_plan、shot_point，并产出给 Tree 的汇报。

    调用流程（锦书视角）：
    1. 初始化 PlannerAgent（传入短影产出的素材路径）
    2. 写 strategy_context → generate_shot_plan() → 审查 → 迭代
    3. generate_shot_point() → 审查 → 迭代
    4. generate_report() → 发给 Tree 确认
    """

    def __init__(
        self,
        video_path: str,
        scene_summaries_dir: str,
        audio_caption_path: str,
        subtitle_path: Optional[str] = None,
        bgm_name: Optional[str] = None,
        output_dir: str = "Output/Output",
        video_type: str = "film",
        main_character: Optional[str] = None,
        video_duration: Optional[float] = None,
    ):
        self.video_path = video_path
        self.scene_summaries_dir = scene_summaries_dir
        self.audio_caption_path = audio_caption_path
        self.subtitle_path = subtitle_path
        self.bgm_name = bgm_name or os.path.basename(audio_caption_path) if audio_caption_path else "无配乐"
        self.output_dir = output_dir
        self.video_type = video_type
        self.main_character = main_character
        self.video_duration = video_duration  # 可选，用于报告

        # 记录每次运行的历史（锦书迭代用）
        self._shot_plan_history: List[Dict] = []
        self._shot_point_history: List[Dict] = []

        # 构建输出路径
        video_id = os.path.splitext(os.path.basename(video_path))[0].replace(".", "_").replace(" ", "_")
        bgm_id = os.path.splitext(os.path.basename(bgm_name or audio_caption_path or "no_audio"))[0].replace(".", "_").replace(" ", "_")
        self._shot_plan_path = os.path.join(output_dir, f"shot_plan_{video_id}_{bgm_id}.json")
        self._shot_point_path = os.path.join(output_dir, f"shot_point_{video_id}_{bgm_id}.json")

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
        instruction = strategy_context if strategy_context else self.instruction

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

        print(f"[PlannerAgent] ✅ Shot plan generated in {elapsed:.1f}s")
        print(f"[PlannerAgent] 💾 Saved to: {self._shot_plan_path}")

        return {"shot_plan": shot_plan, "path": self._shot_plan_path, "elapsed": elapsed, "iteration": len(self._shot_plan_history)}

    def summarize_shot_plan(self) -> str:
        """
        锦书审查用：从 shot_plan JSON 提取可读摘要。
        """
        if not os.path.exists(self._shot_plan_path):
            return "⚠️ shot_plan 尚未生成"

        with open(self._shot_plan_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        lines = []
        lines.append("📋 分镜方案审查")
        lines.append("")

        # 整体主题
        theme = data.get("overall_theme", data.get("metadata", {}).get("overall_theme", "未指定"))
        lines.append(f"🎯 主题：{theme}")

        # 配乐段落
        meta = data.get("metadata", {})
        audio_start = meta.get("selected_audio_start", "?")
        audio_end = meta.get("selected_audio_end", "?")
        lines.append(f"🎵 配乐段落：{audio_start} → {audio_end}")

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
            lines.append("🎬 分镜列表：")
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
    #  Step 2: 生成 shot_point（精确剪辑点）
    # ═══════════════════════════════════════════════════════════════ #

    def generate_shot_point(self) -> Dict[str, Any]:
        """
        锦书调用：根据已有 shot_plan 生成精确剪辑点。
        如果 shot_plan 不存在，先生成默认 shot_plan。
        """
        from src.core import EditorCoreAgent, ParallelShotOrchestrator
        from src.video.preprocess import decode_video_to_frames as _decode

        os.makedirs(os.path.dirname(self._shot_point_path), exist_ok=True)

        # 确保 shot_plan 存在
        if not os.path.exists(self._shot_plan_path):
            print(f"[PlannerAgent] ⚠️  Shot plan not found, generating default first...")
            self.generate_shot_plan(strategy_context="制作一个15秒卡点短片")

        frames_dir = os.path.join(os.path.dirname(self.output_dir), "Video",
                                   os.path.splitext(os.path.basename(self.video_path))[0], "frames")
        caption_file = os.path.join(os.path.dirname(self.output_dir), "Video",
                                     os.path.splitext(os.path.basename(self.video_path))[0],
                                     "captions", "captions.json")
        use_parallel = (
            self.video_type == "film"
            and getattr(project_config, "PARALLEL_SHOT_ENABLED", True)
        )

        print(f"[PlannerAgent] ✂️  Running EditorCoreAgent...")
        t0 = time.time()

        if use_parallel:
            max_workers = getattr(project_config, "PARALLEL_SHOT_MAX_WORKERS", 4)
            max_reruns = getattr(project_config, "PARALLEL_SHOT_MAX_RERUNS", 2)
            orchestrator = ParallelShotOrchestrator(
                video_caption_path=caption_file,
                video_scene_path=self.scene_summaries_dir,
                audio_caption_path=self.audio_caption_path,
                output_path=self._shot_point_path,
                max_iterations=getattr(project_config, "EDITOR_MAX_ITERATIONS", 10),
                video_path=self.video_path,
                frame_folder_path=frames_dir,
                transcript_path=self.subtitle_path if (self.subtitle_path and os.path.exists(self.subtitle_path)) else None,
                max_workers=max_workers,
                max_reruns=max_reruns,
            )
            results = orchestrator.run_parallel(shot_plan_path=self._shot_plan_path)
        else:
            editor_agent = EditorCoreAgent(
                video_caption_path=caption_file,
                video_scene_path=self.scene_summaries_dir,
                audio_caption_path=self.audio_caption_path,
                output_path=self._shot_point_path,
                max_iterations=getattr(project_config, "EDITOR_MAX_ITERATIONS", 10),
                video_path=self.video_path,
                frame_folder_path=frames_dir,
                transcript_path=self.subtitle_path if (self.subtitle_path and os.path.exists(self.subtitle_path)) else None,
            )
            results = editor_agent.run(shot_plan_path=self._shot_plan_path)

        elapsed = round(time.time() - t0, 1)

        record = {
            "iteration": len(self._shot_point_history) + 1,
            "elapsed": elapsed,
            "path": self._shot_point_path,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "shots_count": len(results) if isinstance(results, list) else 0,
        }
        self._shot_point_history.append(record)

        print(f"[PlannerAgent] ✅ Shot point generated in {elapsed:.1f}s")
        print(f"[PlannerAgent] 💾 Saved to: {self._shot_point_path}")

        return {"shot_point": results, "path": self._shot_point_path, "elapsed": elapsed}

    def summarize_shot_point(self) -> str:
        """
        锦书审查用：从 shot_point JSON 提取可读的剪辑列表。
        """
        if not os.path.exists(self._shot_point_path):
            return "⚠️ shot_point 尚未生成"

        with open(self._shot_point_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        clips = data if isinstance(data, list) else data.get("clips", data.get("shots", []))

        lines = []
        lines.append("✂️  精确剪辑点审查")
        lines.append("")

        total_dur = 0.0
        for i, clip in enumerate(clips):
            status = clip.get("status", "success")
            shot_idx = clip.get("shot_idx", clip.get("id", i + 1))
            sub_clips = clip.get("clips", [clip])
            dur = clip.get("total_duration", clip.get("duration", 0))

            for sc in sub_clips:
                start = sc.get("start", "?")
                end = sc.get("end", "?")
                d = sc.get("duration", 0)
                total_dur += d
                lines.append(f"  #{shot_idx}  {start} → {end}  ({d:.1f}s)")

        lines.append("")
        lines.append(f"📊 统计：{len(clips)} 段剪辑 | 总时长 {total_dur:.1f}s")

        return "\n".join(lines)

    # ═══════════════════════════════════════════════════════════════ #
    #  Step 3: 生成给 Tree 的汇报
    # ═══════════════════════════════════════════════════════════════ #

    def generate_report(self) -> str:
        """
        锦书调用：生成结构化的汇报，发给 Tree 确认。

        输出格式包含：
        - 素材概览
        - 分镜方案表格
        - 配乐信息
        - 剪辑列表（精确时间）
        - 确认按钮
        """
        video_name = os.path.basename(self.video_path)

        # 获取视频时长
        if self.video_duration:
            dur_str = f"{self.video_duration:.0f}s"
        else:
            dur_str = "?"

        # 读取 shot_plan
        plan_summary = ""
        if os.path.exists(self._shot_plan_path):
            with open(self._shot_plan_path, "r", encoding="utf-8") as f:
                plan_data = json.load(f)
            theme = plan_data.get("overall_theme", plan_data.get("metadata", {}).get("overall_theme", "?"))
            meta = plan_data.get("metadata", {})
            audio_start = meta.get("selected_audio_start", "?")
            audio_end = meta.get("selected_audio_end", "?")
            plan_summary = f"- 分镜主题：{theme}\n- 配乐段落：{audio_start} → {audio_end}"

        # 读取 shot_point 生成剪辑表格
        clip_table = ""
        total_dur = 0.0
        if os.path.exists(self._shot_point_path):
            with open(self._shot_point_path, "r", encoding="utf-8") as f:
                point_data = json.load(f)
            clips = point_data if isinstance(point_data, list) else point_data.get("clips", point_data.get("shots", []))

            table_rows = []
            for i, clip in enumerate(clips):
                shot_idx = clip.get("shot_idx", clip.get("id", i + 1))
                sub_clips = clip.get("clips", [clip])
                for sc in sub_clips:
                    start = sc.get("start", "?")
                    end = sc.get("end", "?")
                    d = sc.get("duration", 0)
                    total_dur += d
                    table_rows.append(f"| {shot_idx} | {start} → {end} | {d:.1f}s |")

            if table_rows:
                clip_table = (
                    "🎥 剪辑列表：\n"
                    "| # | 时间范围 | 时长 |\n"
                    "|---|---------|------|\n"
                ) + "\n".join(table_rows)
                clip_table += f"\n\n📊 总时长：{total_dur:.1f}s"

        # 完整汇报
        report = f"""🎬 剪辑方案确认 — 请审阅

📋 素材概览
- 视频：{video_name}（{dur_str}）
- 配乐：{self.bgm_name}
- 类型：{self.video_type}

{plan_summary}

{clip_table}

🎵 配乐段落：{audio_start if 'audio_start' in dir() else '?'} → {audio_end if 'audio_end' in dir() else '?'}

✅ 请确认：
1. 分镜方案是否符合预期
2. 剪辑点是否正确
3. 配乐段是否合适

回复「确认执行」或提出修改意见 🎬
"""
        # 去掉多余的变量引用
        return report

    # ═══════════════════════════════════════════════════════════════ #
    #  便利方法：一键全流程（适合快速测试）
    # ═══════════════════════════════════════════════════════════════ #

    def run(
        self,
        strategy_context: str = "",
        skip_shot_plan: bool = False,
        skip_shot_point: bool = False,
    ) -> Dict[str, Any]:
        """一键生成 shot_plan + shot_point（用于快速测试，锦书建议用分步法）。"""
        result = {
            "video_path": self.video_path,
            "shot_plan_path": self._shot_plan_path,
            "shot_point_path": self._shot_point_path,
        }

        if not skip_shot_plan:
            r1 = self.generate_shot_plan(strategy_context=strategy_context)
            result["shot_plan_elapsed"] = r1["elapsed"]
        else:
            result["shot_plan_elapsed"] = 0

        if not skip_shot_point:
            r2 = self.generate_shot_point()
            result["shot_point_elapsed"] = r2["elapsed"]
        else:
            result["shot_point_elapsed"] = 0

        return result
