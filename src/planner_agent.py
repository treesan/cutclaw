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

        print(f"[PlannerAgent] [ok] Shot plan generated in {elapsed:.1f}s")
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
        output_dir=args.output_dir or None,
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
