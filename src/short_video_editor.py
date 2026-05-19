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
        video_path="resource/video/IMG_5747.MOV",
        shot_plan_path="Output/.../shot_plan_xxx.json",
        scene_summary_dir="Output/Video/.../scene_summaries_video",
        audio_caption_path="Output/Audio/.../captions.json",
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
import time
from typing import Any

from src import config


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
            print(f"❌ [短影] shot_plan 不存在: {self.shot_plan_path}")
            return None

        prompt = shot_point_context
        if iteration_note:
            prompt += f"\n\n【修改要求 - 第 ? 轮】\n{iteration_note}"

        use_direct = getattr(config, "DIRECT_SHOT_SELECTOR_ENABLED", True)

        print(f"\n{'='*60}")
        print(f"✂️  [短影] Generating shot point...")
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
            print(f"❌ [短影] Shot point generation failed ({elapsed:.1f}s)")
        else:
            n_shots = len(result.get("shots", result)) if isinstance(result, dict) else len(result) if isinstance(result, list) else 0
            print(f"✅ [短影] Shot point generated: {n_shots} shots in {elapsed:.1f}s")
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
        lines.append("📋 shot_point 预览 — 短影 ✂️")
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
            print(f"❌ [短影] shot_point 不存在: {sp_path}")
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
        print(f"🎥 [短影] Rendering video...")
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
            print(f"❌ [短影] Render failed ({elapsed:.1f}s)")
            if result.stderr:
                print(result.stderr[-1000:])
            return None

        print(f"✅ [短影] Render completed in {elapsed:.1f}s")
        print(f"💾 Output: {out}")
        return out


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
