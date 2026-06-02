"""
MultiSourceRenderer — 多源视频渲染器
====================================

从 shot_point v2.0 中读取每个 shot 的 clip_file_path，
从不同源视频提取片段、拼接、混音、验证。

渲染流程（借鉴 OpenMontage VideoStitch + VideoCompose）：
1. validate  — ffprobe 探测所有源片段的编码/分辨率/帧率，检查兼容性
2. extract   — ffmpeg -ss -to 提取每个片段（无损或重编码）
3. stitch    — concat demuxer 拼接所有片段
4. audio_mix — 叠加 BGM（volume expression + loudnorm）
5. review    — 渲染后验证（ffprobe + 黑帧 + 音量）
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import textwrap
import time
from dataclasses import dataclass, field
from typing import Optional

from src import config as app_config
from src.project.media_profiles import MediaProfile, ffmpeg_output_args


# ── 数据类 ──────────────────────────────────────────────────────

@dataclass
class RenderResult:
    """渲染报告"""
    status: str                         # "pass" / "fail"
    output_path: str
    duration_sec: float = 0.0
    file_size_mb: float = 0.0
    resolution: str = ""
    fps: float = 0.0
    codec: str = ""
    audio_codec: str = ""
    profile_name: str = ""
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


# ── 渲染器 ──────────────────────────────────────────────────────

class MultiSourceRenderer:
    """
    多源视频渲染器。

    从 shot_point v2.0 中每个 shot 的 clip_file_path 提取片段，
    拼接后叠加 BGM，输出最终视频。
    """

    def __init__(
        self,
        shot_point_path: str,
        profile: MediaProfile,
        bgm_path: str = "",
        bgm_start_sec: float = 0.0,
        bgm_duration_sec: float = 0.0,
        original_audio_volume: float = 0.0,
        extract_timeout: int = 600,
        ending_video_path: str = "",
        ending_duration: float = 0.0,
        ending_fade_duration: float = 0.5,
    ):
        """
        Args:
            shot_point_path: shot_point v2.0 JSON 路径
            profile: 输出平台配置
            bgm_path: BGM 音频文件路径
            bgm_start_sec: BGM 起始偏移（秒）
            bgm_duration_sec: BGM 使用时长（秒，0=自动）
            original_audio_volume: 原始音频音量（0=静音，1=原音量）
            extract_timeout: 单个片段提取超时（秒，默认 600，4K preset=slow 需要更长）
            ending_video_path: 结尾视频路径（如 resource/ending/ending.mp4）
            ending_duration: 截取结尾视频的时长（秒，0=完整）
            ending_fade_duration: 主体与结尾视频的交叉淡入淡出（秒）
        """
        self.shot_point_path = shot_point_path
        self.profile = profile
        self.bgm_path = bgm_path
        self.bgm_start_sec = bgm_start_sec
        self.bgm_duration_sec = bgm_duration_sec
        self.original_audio_volume = original_audio_volume
        self.extract_timeout = extract_timeout  # 修正：默认 120s → 600s (BUG-RT-05)
        self.ending_video_path = ending_video_path
        self.ending_duration = ending_duration
        self.ending_fade_duration = ending_fade_duration

        # 加载 shot_point
        with open(shot_point_path, "r", encoding="utf-8") as f:
            self._shot_point = json.load(f)

    # ══════════════════════════════════════════════════════════ #
    #  主流程：渲染                                               #
    # ══════════════════════════════════════════════════════════ #

    def render(self, output_path: str) -> RenderResult:
        """
        执行完整渲染流程。

        Returns:
            RenderResult 渲染报告
        """
        shots = self._shot_point.get("shots", [])
        if not shots:
            return RenderResult(status="fail", output_path=output_path, errors=["shot_point 中没有 shots"])

        print(f"\n{'='*60}")
        print(f"🎥 [MultiSourceRenderer] 开始渲染")
        print(f"   平台: {self.profile.name} ({self.profile.width}x{self.profile.height})")
        print(f"   片段数: {len(shots)}")
        print(f"{'='*60}\n")

        t0 = time.time()
        result = RenderResult(
            status="pass",
            output_path=output_path,
            profile_name=self.profile.name,
        )

        # 收集字幕数据
        subtitles = self._collect_subtitles(shots)

        with tempfile.TemporaryDirectory(prefix="cutclaw_render_") as tmpdir:
            # Step 1: 验证源片段
            print("📋 Step 1/5: 验证源片段...")
            compatible, reference = self._validate_segments(shots, result)
            if result.errors:
                result.status = "fail"
                return result

            # Step 2: 提取片段
            print(f"✂️  Step 2/5: 提取 {len(shots)} 个片段...")
            segment_paths = self._extract_segments(shots, tmpdir, compatible, result)
            if not segment_paths:
                result.status = "fail"
                if not result.errors:
                    result.errors.append("没有成功提取的片段")
                return result

            # 修正 (BUG-RT-10): 提取后再次检测兼容性，segment 实际格式可能已统一
            if not compatible:
                post_compatible = self._check_segments_compatible(segment_paths)
                if post_compatible:
                    print(f"  ✅ 提取后 segment 格式已统一，切换到 concat demuxer")
                    compatible = True

            # Step 3: 拼接
            print(f"🔗 Step 3/5: 拼接 {len(segment_paths)} 个片段...")
            concat_path = os.path.join(tmpdir, "concat_raw.mp4")
            self._concat_segments(segment_paths, concat_path, compatible)

            # Step 4: 混音
            if self.bgm_path and os.path.exists(self.bgm_path):
                print("🎵 Step 4/5: 叠加 BGM...")
                self._add_bgm(concat_path, output_path, result)
            else:
                print("⏭️  Step 4/5: 无 BGM，跳过混音")
                shutil.move(concat_path, output_path)

            # Step 5 (可选): 字幕叠加
            if subtitles:
                print(f"📝 Step 5/5: 叠加字幕（{len(subtitles)} 条）...")
                sub_output = output_path + ".sub.mp4"
                self._add_subtitles(output_path, sub_output, subtitles, tmpdir, result)
                if os.path.exists(sub_output):
                    shutil.move(sub_output, output_path)
            else:
                print("⏭️  Step 5/5: 无字幕")

            # Step 6 (可选): 拼接结尾视频
            if self.ending_video_path and os.path.exists(self.ending_video_path):
                print("🎬 Step 6/6: 拼接结尾视频...")
                self._append_ending(output_path, result)
            else:
                print("⏭️  Step 6/6: 无结尾视频")

        # 验证输出
        self._validate_render(output_path, result)

        elapsed = time.time() - t0
        result.duration_sec = self._get_duration(output_path)
        result.file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        result.resolution = f"{self.profile.width}x{self.profile.height}"
        result.fps = self.profile.fps
        result.codec = self.profile.codec
        result.audio_codec = self.profile.audio_codec

        status_icon = "✅" if result.status == "pass" else "❌"
        print(f"\n{status_icon} 渲染完成: {elapsed:.1f}s, {result.file_size_mb:.1f}MB, {result.duration_sec:.1f}s")
        if result.warnings:
            for w in result.warnings:
                print(f"  ⚠️ {w}")
        if result.errors:
            for e in result.errors:
                print(f"  ❌ {e}")

        return result

    # ══════════════════════════════════════════════════════════ #
    #  Step 1: 验证源片段                                         #
    # ══════════════════════════════════════════════════════════ #

    def _validate_segments(
        self,
        shots: list[dict],
        result: RenderResult,
    ) -> tuple[bool, dict]:
        """
        探测所有源片段的编码/分辨率/帧率，检查兼容性。

        Returns:
            (compatible, reference) — compatible=True 时可用 concat demuxer 无损拼接
        """
        reference = {}
        compatible = True
        has_hdr = False

        for i, shot in enumerate(shots):
            clip_path = shot.get("clip_file_path", "")
            if not clip_path or not os.path.exists(clip_path):
                result.errors.append(f"片段 {i}: 源文件不存在 ({clip_path})")
                continue

            try:
                probe = self._probe_video(clip_path)
            except Exception as e:
                result.errors.append(f"片段 {i}: ffprobe 失败 ({e})")
                continue

            streams = probe.get("streams", [])
            video = next((s for s in streams if s.get("codec_type") == "video"), {})
            audio = next((s for s in streams if s.get("codec_type") == "audio"), {})

            if not video:
                result.errors.append(f"片段 {i}: 无视频流 ({clip_path})")
                continue

            # 检测色彩空间
            cs_info = self._detect_color_space(probe)
            shot["_cs_info"] = cs_info
            if cs_info["is_hdr"]:
                has_hdr = True

            # 检查兼容性
            props = {
                "width": int(video.get("width", 0)),
                "height": int(video.get("height", 0)),
                "codec": video.get("codec_name", ""),
                "pix_fmt": video.get("pix_fmt", ""),
                "fps": self._parse_fps(video.get("r_frame_rate", "0/1")),
                "audio_codec": audio.get("codec_name", "") if audio else "",
                "sample_rate": int(audio.get("sample_rate", 0)) if audio else 0,
            }

            if not reference:
                reference = props
            else:
                # 检查关键属性是否一致
                for key in ["width", "height", "codec", "fps"]:
                    if props.get(key) != reference.get(key):
                        compatible = False
                        break

        # 色彩空间混合检测
        if has_hdr:
            target = getattr(app_config, "TARGET_COLOR_SPACE", "bt709").lower()
            if target == "bt709":
                result.warnings.append("检测到 HDR 素材，将自动转换为 BT.709 SDR")
                compatible = False  # HDR 转换需要重编码
                print("  🎨 检测到 HDR 素材，将进行色彩空间转换")

        if not compatible:
            result.warnings.append("源片段格式不一致，将重编码拼接")
            print("  ⚠️ 源片段格式不一致，需要重编码")

        return compatible, reference

    def _check_segments_compatible(self, segment_paths: list[str]) -> bool:
        """
        检测已提取的 segment 是否格式兼容（可用 concat demuxer 无损拼接）。

        修正 (BUG-RT-10): 源文件兼容性检查发生在提取前，但提取时所有 segment
        都被重编码到统一格式（profile 指定），所以即使源不兼容，segment 也可能兼容。
        """
        if len(segment_paths) < 2:
            return True
        try:
            ref = self._probe_video(segment_paths[0])
        except Exception:
            return False
        ref_video = next((s for s in ref.get("streams", []) if s.get("codec_type") == "video"), {})
        ref_audio = next((s for s in ref.get("streams", []) if s.get("codec_type") == "audio"), {})

        for path in segment_paths[1:]:
            try:
                info = self._probe_video(path)
            except Exception:
                return False
            video = next((s for s in info.get("streams", []) if s.get("codec_type") == "video"), {})
            audio = next((s for s in info.get("streams", []) if s.get("codec_type") == "audio"), {})
            # 关键属性比较：分辨率、codec、fps
            for key in ["width", "height", "codec_name"]:
                if video.get(key) != ref_video.get(key):
                    return False
            if int(video.get("r_frame_rate", "0/1").split("/")[0] or 0) != int(ref_video.get("r_frame_rate", "0/1").split("/")[0] or 0):
                return False
        return True

    # ══════════════════════════════════════════════════════════ #
    #  Step 2: 提取片段                                           #
    # ══════════════════════════════════════════════════════════ #

    def _extract_segments(
        self,
        shots: list[dict],
        tmpdir: str,
        compatible: bool,
        result: RenderResult,
    ) -> list[str]:
        """从各源视频提取每个 shot 的片段，自动处理色彩空间转换。"""
        segment_paths = []
        use_copy = compatible  # 格式一致时可用无损提取
        cs_converted_count = 0

        for i, shot in enumerate(shots):
            clip_path = shot.get("clip_file_path", "")
            start_sec = shot.get("start_sec", 0)
            end_sec = shot.get("end_sec", 0)
            speed = shot.get("speed", 1.0)

            if not clip_path or not os.path.exists(clip_path):
                continue

            duration = (end_sec - start_sec) / speed
            if duration <= 0:
                continue

            segment_path = os.path.join(tmpdir, f"segment_{i:04d}.mp4")

            # 检测色彩空间，决定是否需要转换
            cs_info = shot.get("_cs_info", {})
            cs_filter = self._color_space_filter(cs_info) if cs_info else ""
            needs_cs_conversion = bool(cs_filter)

            if use_copy and not needs_cs_conversion:
                # 无损提取（色彩空间一致，格式兼容）
                # 修正 (BUG-RT-16): 必须用 input seeking（-ss 在 -i 前），
                # output seeking + -c copy 会保留源文件的原始 PTS 偏移
                # （如 PTS 从 0.5s 开始），导致 segment 时长偏短，
                # concat 后总时长比预期少数秒（实测 59s → 54.5s）。
                cmd = ["ffmpeg", "-y",
                       "-ss", f"{start_sec:.3f}", "-t", f"{duration:.3f}",
                       "-i", clip_path,
                       "-c", "copy", "-avoid_negative_ts", "make_zero"]
            else:
                # 重编码路径：用 output seeking（-ss 在 -i 后）精确到帧
                cmd = ["ffmpeg", "-y",
                       "-i", clip_path,
                       "-ss", f"{start_sec:.3f}", "-t", f"{duration:.3f}"]

                if needs_cs_conversion:
                    # 色彩空间转换：必须重编码 + 分辨率标准化 + 帧率统一
                    # 修正 (BUG-RT-15): 必须在 -vf 中显式指定 fps 滤镜，
                    # 仅用 -r 输出选项会导致高帧率源（如 120fps）降帧不均匀，
                    # 产生多出帧（5s 产出 302 帧而非 300 帧），引起卡顿。
                    vf_parts = [cs_filter]
                    vf_parts.append(f"scale={self.profile.width}:{self.profile.height}:force_original_aspect_ratio=decrease")
                    vf_parts.append(f"pad={self.profile.width}:{self.profile.height}:(ow-iw)/2:(oh-ih)/2")
                    vf_parts.append(f"fps={self.profile.fps}")
                    vf = ",".join(vf_parts)
                    cmd += ["-vf", vf]
                    cmd += self._encode_args()
                    cs_converted_count += 1
                else:
                    # 重编码到目标格式（格式不兼容）— 需要 scale/pad + fps 确保分辨率和帧率一致
                    # 修正 (BUG-RT-15): 显式 fps 滤镜避免 NTSC→整数帧率转换时的多余帧
                    vf = (f"scale={self.profile.width}:{self.profile.height}:force_original_aspect_ratio=decrease,"
                          f"pad={self.profile.width}:{self.profile.height}:(ow-iw)/2:(oh-ih)/2,"
                          f"fps={self.profile.fps}")
                    cmd += ["-vf", vf]
                    cmd += self._encode_args()

            cmd.append(segment_path)

            try:
                self._run_ffmpeg(cmd, timeout=self.extract_timeout)
                if os.path.exists(segment_path) and os.path.getsize(segment_path) > 0:
                    # 修正 (BUG-RT-18): 检查提取结果是否真正有帧
                    # 当 -ss 超出源文件时长时，ffmpeg 会生成空容器（几百字节但 0 帧）
                    seg_dur = self._get_duration(segment_path)
                    if seg_dur < 0.05:
                        result.warnings.append(
                            f"片段 {i}: 提取后无有效帧（-ss={start_sec:.1f}s 可能超出源时长 {clip_path}）"
                        )
                    else:
                        segment_paths.append(segment_path)
                else:
                    result.warnings.append(f"片段 {i}: 提取后文件为空")
            except Exception as e:
                result.warnings.append(f"片段 {i}: 提取失败 ({e})")

        if cs_converted_count > 0:
            print(f"  ✅ 成功提取 {len(segment_paths)}/{len(shots)} 个片段 ({cs_converted_count} 个色彩空间转换)")
        else:
            print(f"  ✅ 成功提取 {len(segment_paths)}/{len(shots)} 个片段")
        return segment_paths

    # ══════════════════════════════════════════════════════════ #
    #  Step 3: 拼接                                               #
    # ══════════════════════════════════════════════════════════ #

    def _concat_segments(
        self,
        segment_paths: list[str],
        output_path: str,
        compatible: bool,
    ):
        """使用 concat demuxer 拼接所有片段。

        修正 (BUG-RT-12): 简化为始终用 concat demuxer（要求 segment 格式统一，
        通过 BUG-RT-10 修复后已统一）。重编码路径用 -c:v -c:a 重新编码成目标格式。
        之前的 filter_complex + anullsrc 方案因索引错位导致崩溃，已废弃。
        """
        # 只有一个片段时直接复制
        if len(segment_paths) == 1:
            shutil.copy2(segment_paths[0], output_path)
            print(f"  ✅ 单片段直接复制")
            return

        # 写入 concat 列表文件
        concat_list = output_path + ".txt"
        with open(concat_list, "w", encoding="utf-8") as f:
            for path in segment_paths:
                f.write(f"file '{path}'\n")

        # 始终用 concat demuxer 拼接（segment 格式已统一），-c copy 避免重编码
        # 但因 BGM/ending 等后续步骤可能需要重编码，这里仍走 -c copy
        cmd = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_list,
            "-c", "copy",
            output_path,
        ]
        try:
            self._run_ffmpeg(cmd, timeout=self.extract_timeout)
            print(f"  ✅ 拼接完成（concat demuxer, -c copy）")
        finally:
            if os.path.exists(concat_list):
                os.remove(concat_list)

    # ══════════════════════════════════════════════════════════ #
    #  Step 4: BGM 混合                                           #
    # ══════════════════════════════════════════════════════════ #

    def _add_bgm(
        self,
        video_path: str,
        output_path: str,
        result: RenderResult,
    ):
        """叠加 BGM 到已拼接的视频上。"""
        # 检查视频是否有音频流
        has_audio = False
        try:
            probe = self._probe_video(video_path)
            streams = probe.get("streams", [])
            has_audio = any(s.get("codec_type") == "audio" for s in streams)
        except Exception:
            pass

        video_duration = self._get_duration(video_path)
        bgm_duration = self.bgm_duration_sec if self.bgm_duration_sec > 0 else video_duration

        # 构建 volume 表达式（确保 fade 区间不重叠）
        fade_in = min(1.0, bgm_duration * 0.2)
        fade_out = min(2.0, bgm_duration * 0.3)
        if fade_in + fade_out > bgm_duration:
            fade_in = bgm_duration * 0.4
            fade_out = bgm_duration * 0.4
        start = self.bgm_start_sec
        end = start + bgm_duration

        vol_expr = (
            f"if(lt(t,{start}),0,"
            f"if(lt(t,{start + fade_in}),1.0*(t-{start})/{fade_in},"
            f"if(lt(t,{end - fade_out}),1.0,"
            f"if(lt(t,{end}),1.0*({end}-t)/{fade_out},"
            f"0))))"
        )

        if has_audio and self.original_audio_volume > 0:
            # 有原始音频且需要混合
            # amix duration=first 使混音时长 = 视频原始音频时长
            cmd = ["ffmpeg", "-y",
                   "-i", video_path,
                   "-stream_loop", "-1", "-i", self.bgm_path,
                   "-filter_complex",
                   f"[1:a]atrim=start={start},"
                   f"volume='{vol_expr}':eval=frame[bgm];"
                   f"[0:a]volume={self.original_audio_volume}[orig];"
                   f"[orig][bgm]amix=inputs=2:duration=first:normalize=0,"
                   f"loudnorm=I=-16:LRA=11:TP=-1.5[outa]",
                   "-map", "0:v", "-map", "[outa]",
                   "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                   "-t", f"{video_duration:.3f}",
                   output_path]
        else:
            # 无原始音频或静音 — 只用 BGM
            # 修正 (BUG-RT-17): 用 -t 替代 -shortest。
            # 当 BGM 时长 < 视频时长时，-shortest 会截断视频到 BGM 长度。
            # stream_loop -1 循环 BGM，volume 表达式处理 fade，-t 确保输出时长 = 视频时长。
            cmd = ["ffmpeg", "-y",
                   "-i", video_path,
                   "-stream_loop", "-1", "-i", self.bgm_path,
                   "-filter_complex",
                   f"[1:a]atrim=start={start},"
                   f"volume='{vol_expr}':eval=frame[bgm];"
                   f"[bgm]loudnorm=I=-16:LRA=11:TP=-1.5[outa]",
                   "-map", "0:v", "-map", "[outa]",
                   "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                   "-t", f"{video_duration:.3f}",
                   output_path]

        try:
            self._run_ffmpeg(cmd, timeout=300)
            print(f"  ✅ BGM 混合完成")
        except Exception as e:
            result.warnings.append(f"BGM 混合失败: {e}")
            # 降级：直接复制无 BGM 的视频
            shutil.copy2(video_path, output_path)

    # ══════════════════════════════════════════════════════════ #
    #  Step 4.5: 拼接结尾视频                                       #
    # ══════════════════════════════════════════════════════════ #

    def _append_ending(self, main_video_path: str, result: RenderResult):
        """在主体视频末尾拼接结尾视频（带交叉淡入淡出）。"""
        ending_path = self.ending_video_path
        ending_duration = self.ending_duration
        fade = self.ending_fade_duration

        # 截取 ending_duration 秒的结尾片段
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False, prefix="ending_") as tmp:
            trimmed_ending = tmp.name
        try:
            trim_cmd = ["ffmpeg", "-y", "-i", ending_path]
            if ending_duration > 0:
                trim_cmd += ["-t", f"{ending_duration:.3f}"]
            # 转码到与主体视频匹配的格式
            trim_cmd += ["-vf", f"scale={self.profile.width}:{self.profile.height}:force_original_aspect_ratio=decrease,pad={self.profile.width}:{self.profile.height}:(ow-iw)/2:(oh-ih)/2,fps={self.profile.fps}"]
            trim_cmd += ["-c:v", self.profile.codec, "-c:a", self.profile.audio_codec, "-ar", "48000", "-ac", "2", "-pix_fmt", self.profile.pixel_format]
            trim_cmd.append(trimmed_ending)
            try:
                self._run_ffmpeg(trim_cmd, timeout=self.extract_timeout)
            except Exception as e:
                result.warnings.append(f"结尾视频处理失败: {e}")
                if os.path.exists(trimmed_ending):
                    os.remove(trimmed_ending)
                return

            # 用 concat demuxer + xfade 滤镜拼接
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False, prefix="merged_") as tmp:
                merged_path = tmp.name
            try:
                if fade > 0:
                    main_dur = self._get_duration(main_video_path)
                    offset = max(0, main_dur - fade)
                    cmd = [
                        "ffmpeg", "-y",
                        "-i", main_video_path,
                        "-i", trimmed_ending,
                        "-filter_complex",
                        f"[0:v][1:v]xfade=transition=fade:duration={fade}:offset={offset}[v];"
                        f"[0:a][1:a]acrossfade=d={fade}[a]",
                        "-map", "[v]", "-map", "[a]",
                        "-c:v", "libx264", "-preset", "ultrafast",  # 拼接步骤用 ultrafast 加速
                        "-c:a", "aac",
                        merged_path,
                    ]
                else:
                    # 无淡入淡出，直接 concat demuxer（要求格式相同）
                    concat_list = merged_path + ".txt"
                    with open(concat_list, "w", encoding="utf-8") as f:
                        f.write(f"file '{main_video_path}'\n")
                        f.write(f"file '{trimmed_ending}'\n")
                    cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_list, "-c", "copy", merged_path]
                    os.remove(concat_list)
                self._run_ffmpeg(cmd, timeout=self.extract_timeout)
                # 覆盖原输出
                shutil.move(merged_path, main_video_path)
                print(f"  ✅ 结尾视频拼接完成（{fade}s 交叉淡入）")
            except Exception as e:
                result.warnings.append(f"结尾视频拼接失败: {e}")
                if os.path.exists(merged_path):
                    os.remove(merged_path)
        finally:
            if os.path.exists(trimmed_ending):
                os.remove(trimmed_ending)

    # ══════════════════════════════════════════════════════════ #
    #  Step 4.5: 字幕叠加                                         #
    # ══════════════════════════════════════════════════════════ #

    def _collect_subtitles(self, shots: list[dict]) -> list[dict]:
        """
        从 shot_point 的各 shot 中收集字幕数据，计算全局时间戳。

        每个 shot 可包含:
        - subtitle_lines: [{text, start, end}]（相对 shot 起始的秒数）
        - overlay_text: str（整段 shot 期间显示的文字）

        Returns:
            [{text, global_start, global_end}] 全局时间戳的字幕条目
        """
        entries = []
        cumulative = 0.0
        for shot in shots:
            duration = shot.get("end_sec", 0) - shot.get("start_sec", 0)
            speed = shot.get("speed", 1.0)
            effective_dur = duration / speed if speed > 0 else duration

            # 带时间的逐行字幕
            sub_lines = shot.get("subtitle_lines", [])
            if sub_lines:
                for sl in sub_lines:
                    entries.append({
                        "text": sl.get("text", ""),
                        "global_start": cumulative + sl.get("start", 0),
                        "global_end": cumulative + sl.get("end", 0),
                    })
            else:
                # 整段叠加文字
                overlay = shot.get("overlay_text", "")
                if overlay:
                    entries.append({
                        "text": overlay,
                        "global_start": cumulative,
                        "global_end": cumulative + effective_dur,
                    })

            cumulative += effective_dur
        return entries

    def _find_font(self, text: str = "") -> str:
        """
        查找可用字体。CJK 文本优先使用系统中文字体。

        与 render_video.py _resolve_drawtext_font 逻辑一致。
        """
        CJK_FONTS = [
            "/System/Library/Fonts/PingFang.ttc",
            "/System/Library/Fonts/STHeiti Medium.ttc",
            "/System/Library/Fonts/Hiragino Sans GB.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        ]
        DEJAVU = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

        # 检测 CJK 字符
        has_cjk = any("一" <= ch <= "鿿" or "㐀" <= ch <= "䶿" for ch in text)

        if has_cjk:
            for p in CJK_FONTS:
                if os.path.exists(p):
                    return p

        # 项目内置字体
        project_font = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                    "resource", "font", "Pulp Fiction M54.ttf")
        if os.path.exists(project_font):
            return project_font

        if os.path.exists(DEJAVU):
            return DEJAVU
        if os.path.exists("/System/Library/Fonts/Helvetica.ttc"):
            return "/System/Library/Fonts/Helvetica.ttc"
        return DEJAVU

    def _add_subtitles(
        self,
        video_path: str,
        output_path: str,
        subtitles: list[dict],
        tmpdir: str,
        result: RenderResult,
    ):
        """
        在已拼接（可选含 BGM）的视频上叠加字幕。

        使用 ffmpeg drawtext 滤镜逐条渲染，支持逐行定时和整段叠加。

        Args:
            video_path: 输入视频路径
            output_path: 输出视频路径
            subtitles: [{text, global_start, global_end}] 字幕条目
            tmpdir: 临时目录（写入字幕文本文件）
            result: 渲染结果（追加 warnings）
        """
        if not subtitles:
            shutil.copy2(video_path, output_path)
            return

        # 字体与样式
        all_text = " ".join(s["text"] for s in subtitles)
        font = self._find_font(all_text)
        font_size = getattr(app_config, "DIALOGUE_FONT_SIZE", 48)
        font_color = getattr(app_config, "DIALOGUE_FONT_COLOR", "white")
        box_color = getattr(app_config, "DIALOGUE_BOX_COLOR", "black@0.3")
        y_expr = getattr(app_config, "DIALOGUE_Y_POSITION", "h-text_h-40")

        # 自动换行宽度
        out_w = self.profile.width
        max_chars = max(10, int(out_w / (font_size * 0.6)))

        # 构建 drawtext 滤镜链
        vf_parts = []
        for i, sub in enumerate(subtitles):
            text_path = os.path.join(tmpdir, f"subtitle_{i:04d}.txt")
            wrapped = "\n".join(textwrap.wrap(sub["text"], width=max_chars))
            with open(text_path, "w", encoding="utf-8") as f:
                f.write(wrapped)

            # 转义 drawtext 路径中的特殊字符
            safe_text = text_path.replace("'", "'\\''").replace(":", "\\:")
            safe_font = font.replace("'", "'\\''").replace(":", "\\:")

            dt = (
                f"drawtext=textfile='{safe_text}':"
                f"fontfile='{safe_font}':"
                f"fontsize={font_size}:"
                f"fontcolor={font_color}:"
                f"x=(w-text_w)/2:y={y_expr}:"
                f"box=1:boxcolor={box_color}:boxborderw=12"
            )
            # 逐行字幕带时间窗口
            if sub["global_start"] > 0 or sub["global_end"] > 0:
                dt += f":enable='between(t,{sub['global_start']:.3f},{sub['global_end']:.3f})'"
            vf_parts.append(dt)

        vf = ",".join(vf_parts)

        # 重编码视频叠加字幕，音频直接复制
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-vf", vf,
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-c:a", "copy",
            output_path,
        ]

        try:
            self._run_ffmpeg(cmd, timeout=self.extract_timeout)
            print(f"  ✅ 字幕叠加完成（{len(subtitles)} 条）")
        except Exception as e:
            result.warnings.append(f"字幕叠加失败: {e}")
            shutil.copy2(video_path, output_path)

    # ══════════════════════════════════════════════════════════ #
    #  Step 5: 渲染后验证                                         #
    # ══════════════════════════════════════════════════════════ #

    def _validate_render(self, output_path: str, result: RenderResult):
        """渲染后自动验证：ffprobe 技术探针 + 黑帧检测 + 音量检测。"""
        if not os.path.exists(output_path):
            result.errors.append("输出文件不存在")
            return

        # 1. ffprobe 技术探针
        try:
            probe = self._probe_video(output_path)
            streams = probe.get("streams", [])
            video = next((s for s in streams if s.get("codec_type") == "video"), {})
            audio = next((s for s in streams if s.get("codec_type") == "audio"), {})

            if not video:
                result.errors.append("输出文件无视频流")
            if not audio:
                result.warnings.append("输出文件无音频流")

            # 分辨率检查
            out_w = int(video.get("width", 0))
            out_h = int(video.get("height", 0))
            if out_w != self.profile.width or out_h != self.profile.height:
                result.warnings.append(f"输出分辨率 {out_w}x{out_h} 与目标 {self.profile.width}x{self.profile.height} 不一致")

        except Exception as e:
            result.warnings.append(f"ffprobe 验证失败: {e}")

        # 2. 黑帧检测（采样 4 帧）
        try:
            self._check_black_frames(output_path, result)
        except Exception:
            pass  # 非关键检查

        # 3. 音量检测
        try:
            self._check_audio_levels(output_path, result)
        except Exception:
            pass  # 非关键检查

    def _check_black_frames(self, video_path: str, result: RenderResult):
        """采样 4 帧检测纯黑帧。"""
        try:
            import numpy as np
        except ImportError:
            return  # numpy 不可用时跳过黑帧检测

        duration = self._get_duration(video_path)
        if duration <= 0:
            return

        timestamps = [duration * p for p in [0.1, 0.35, 0.65, 0.9]]
        for ts in timestamps:
            cmd = [
                "ffmpeg", "-v", "quiet", "-ss", f"{ts:.2f}",
                "-i", video_path, "-frames:v", "1",
                "-vf", "scale=16:16", "-f", "rawvideo", "-pix_fmt", "gray",
                "pipe:1"
            ]
            try:
                proc = subprocess.run(cmd, capture_output=True, timeout=10)
                if proc.returncode == 0 and len(proc.stdout) == 16 * 16:
                    frame = np.frombuffer(proc.stdout, dtype=np.uint8)
                    mean_brightness = float(np.mean(frame))
                    if mean_brightness < 3:
                        result.warnings.append(f"黑帧检测: {ts:.1f}s 处疑似纯黑帧 (亮度={mean_brightness:.1f})")
            except Exception:
                pass

    def _check_audio_levels(self, video_path: str, result: RenderResult):
        """检测音量：静音 (< -60dB) 或削波 (> -0.5dB)。"""
        import re
        cmd = ["ffmpeg", "-i", video_path, "-af", "volumedetect", "-f", "null", "-"]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            stderr = proc.stderr
            # ffmpeg volumedetect 输出格式: "mean_volume: -23.0 dB"
            mean_match = re.search(r"mean_volume:\s*(-?[\d.]+)\s*dB", stderr)
            max_match = re.search(r"max_volume:\s*(-?[\d.]+)\s*dB", stderr)
            if mean_match:
                vol = float(mean_match.group(1))
                if vol < -60:
                    result.warnings.append(f"音量过低: mean={vol:.1f}dB")
            if max_match:
                vol = float(max_match.group(1))
                if vol > -0.5:
                    result.warnings.append(f"音量削波风险: max={vol:.1f}dB")
        except Exception:
            pass

    # ══════════════════════════════════════════════════════════ #
    #  工具方法                                                   #
    # ══════════════════════════════════════════════════════════ #

    def _encode_args(self) -> list[str]:
        """根据 MediaProfile 生成 ffmpeg 编码参数。修正 (优化 #9): 支持 config_my.py 覆盖。"""
        # 检查 config 中的覆盖值
        preset_override = getattr(app_config, "MEDIA_PROFILE_PRESET_OVERRIDE", None)
        crf_override = getattr(app_config, "MEDIA_PROFILE_CRF_OVERRIDE", None)
        effective_preset = preset_override if preset_override else self.profile.preset
        effective_crf = crf_override if crf_override is not None else self.profile.crf

        return [
            "-c:v", self.profile.codec,
            "-c:a", self.profile.audio_codec,
            "-crf", str(effective_crf),
            "-preset", effective_preset,
            "-pix_fmt", self.profile.pixel_format,
            "-r", str(self.profile.fps),
            "-ar", "48000", "-ac", "2",
        ]

    @staticmethod
    def _probe_video(path: str) -> dict:
        """ffprobe 获取视频元数据。"""
        cmd = [
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_format", "-show_streams",
            path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            raise RuntimeError(f"ffprobe 失败: {result.stderr}")
        return json.loads(result.stdout)

    @staticmethod
    def _run_ffmpeg(cmd: list[str], timeout: int = 300):
        """运行 ffmpeg 命令并检查返回码。"""
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg 失败: {result.stderr[-500:]}")

    @staticmethod
    def _get_duration(path: str) -> float:
        """获取视频时长（秒）。"""
        try:
            cmd = [
                "ffprobe", "-v", "quiet",
                "-show_entries", "format=duration",
                "-of", "json", path,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            data = json.loads(result.stdout)
            return float(data.get("format", {}).get("duration", 0))
        except Exception:
            return 0.0

    @staticmethod
    def _parse_fps(r_frame_rate: str) -> float:
        """解析 r_frame_rate（如 '60000/1001'）为浮点数。"""
        if "/" in r_frame_rate:
            parts = r_frame_rate.split("/")
            if float(parts[1]) > 0:
                return round(float(parts[0]) / float(parts[1]), 3)
        try:
            return float(r_frame_rate)
        except (ValueError, TypeError):
            return 0.0

    # ── 色彩空间检测与转换 ──────────────────────────────────────────

    @staticmethod
    def _detect_color_space(probe: dict) -> dict:
        """
        从 ffprobe 结果中提取色彩空间信息。

        Returns:
            {
                "color_space": str,    # 如 "bt2020nc", "bt709"
                "color_transfer": str, # 如 "arib-std-b67" (HLG), "smpte2084" (PQ), "bt709"
                "color_primaries": str,# 如 "bt2020", "bt709"
                "pix_fmt": str,        # 如 "yuv420p10le", "yuv420p"
                "is_hdr": bool,        # 是否 HDR 色彩空间
                "hdr_type": str,       # "hlg", "pq", "dlog_m", "none"
            }
        """
        streams = probe.get("streams", [])
        video = next((s for s in streams if s.get("codec_type") == "video"), {})

        color_space = (video.get("color_space", "") or "").lower()
        color_transfer = (video.get("color_transfer", "") or "").lower()
        color_primaries = (video.get("color_primaries", "") or "").lower()
        pix_fmt = (video.get("pix_fmt", "") or "").lower()

        # 判断是否 HDR
        is_hdr = False
        hdr_type = "none"

        # BT.2020 色彩空间 + HLG 传输特性 → HLG HDR
        if "bt2020" in color_primaries or "bt2020" in color_space:
            is_hdr = True
            if "arib-std-b67" in color_transfer or "hlg" in color_transfer:
                hdr_type = "hlg"
            elif "smpte2084" in color_transfer or "pq" in color_transfer:
                hdr_type = "pq"
            elif "10le" in pix_fmt:
                # BT.2020 + 10-bit 但无明确传输特性 → 可能是 D-Log M
                hdr_type = "bt2020_unknown"

        return {
            "color_space": color_space,
            "color_transfer": color_transfer,
            "color_primaries": color_primaries,
            "pix_fmt": pix_fmt,
            "is_hdr": is_hdr,
            "hdr_type": hdr_type,
        }

    @staticmethod
    def _select_tone_map_npl(cs_info: dict) -> int:
        """
        根据源色彩空间信息选择合适的 tone mapping 参考白电平 (npl)。

        修正 (优化 #11): 不同 HDR 标准的峰值亮度差异巨大：
        - HLG: 相对亮度（参考白 ~100 nits），但实际峰值可达 1000-4000 nits
        - PQ (HDR10): 静态元数据 maxCLL 通常 1000-4000 nits
        - DJI 无人机 HLG: 实测峰值约 1000 nits（参考 https://www.dji.com/...）
        - 通用 100 nits 对 HLG 太暗，通用 1000 nits 对 SDR-style HLG 太亮

        智能选择策略:
        1. 检查 color_transfer 字符串（HLG/PQ）
        2. 读取 master-display 元数据（CLLL/FALL）
        3. 默认值: HLG=1000, PQ=1000（更保守的高光保留）

        Returns:
            npl 值（nits）
        """
        transfer = cs_info.get("color_transfer", "").lower()
        primaries = cs_info.get("color_primaries", "").lower()

        # PQ: 典型峰值 1000-4000 nits，用 1000 作为安全中间值
        if "smpte2084" in transfer or "pq" in transfer:
            return 1000

        # HLG: 相对亮度系统，但实际峰值可达 1000+ nits
        # DJI/iPhone HLG 实测峰值约 1000 nits（保守值）
        if "arib-std-b67" in transfer or "hlg" in transfer:
            return 1000

        # BT.2020 未知 → 保守
        if "bt2020" in primaries:
            return 1000

        # SDR 不应进入这里（is_hdr 应为 False），但防御性返回
        return 203  # BT.709 标称白点

    def _color_space_filter(self, cs_info: dict) -> str:
        """
        根据源色彩空间和目标色彩空间，生成 ffmpeg 滤镜字符串。

        转换策略:
        - BT.709 → BT.709: 无操作（返回空字符串）
        - BT.2020 + HLG → BT.709: zscale tone mapping
        - BT.2020 + PQ → BT.709: zscale tone mapping
        - BT.709 → BT.2020 + HLG: zscale 上转换

        Returns:
            ffmpeg -vf 滤镜字符串，无需转换时返回空字符串
        """
        target = getattr(app_config, "TARGET_COLOR_SPACE", "bt709").lower()
        conversion = getattr(app_config, "COLOR_SPACE_CONVERSION", "auto").lower()

        if conversion == "skip":
            return ""

        if not cs_info["is_hdr"] and target == "bt709":
            # 源是 SDR + 目标是 SDR → 无需转换
            return ""

        if cs_info["is_hdr"] and target == "bt709":
            # HDR → SDR 转换（tone mapping）
            # 使用 zscale 进行高质量 tone mapping
            # desat=0 保留饱和度，避免色彩偏移
            # 修正 (优化 #11): 根据 color_transfer 智能选择 npl
            npl = self._select_tone_map_npl(cs_info)
            if cs_info["hdr_type"] == "hlg":
                # HLG → BT.709 (HLG 参考白电平约 100 nits)
                return (
                    f"zscale=t=linear:npl={npl},"
                    "tonemap=tonemap=hable:desat=0,"
                    "zscale=t=bt709:m=bt709:p=bt709,"
                    "format=yuv420p"
                )
            elif cs_info["hdr_type"] == "pq":
                # PQ (Dolby Vision / HDR10) → BT.709
                return (
                    f"zscale=t=linear:npl={npl},"
                    "tonemap=tonemap=hable:desat=0,"
                    "zscale=t=bt709:m=bt709:p=bt709,"
                    "format=yuv420p"
                )
            else:
                # BT.2020 未知传输特性 → BT.709
                return (
                    "zscale=t=bt709:m=bt709:p=bt709,"
                    "format=yuv420p"
                )

        if not cs_info["is_hdr"] and target == "bt2020_hlg":
            # SDR → HDR 上转换（安全上转换，不丢失信息）
            return (
                "zscale=t=bt2020:m=bt2020:p=bt2020,"
                "zscale=t=arib-std-b67,"
                "format=yuv420p10le"
            )

        return ""


# ── CLI 入口 ─────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="多源视频渲染器")
    parser.add_argument("--shot-point", required=True, help="shot_point v2.0 JSON 路径")
    parser.add_argument("--profile", default="bilibili_4k", help="输出平台配置名称")
    parser.add_argument("--bgm", default="", help="BGM 音频路径")
    parser.add_argument("--bgm-start", type=float, default=0.0, help="BGM 起始偏移（秒）")
    parser.add_argument("--bgm-duration", type=float, default=0.0, help="BGM 使用时长（秒）")
    parser.add_argument("--original-audio-volume", type=float, default=0.0, help="原始音频音量")
    parser.add_argument("--output", required=True, help="输出文件路径")
    args = parser.parse_args()

    from src.project.media_profiles import get_profile
    profile = get_profile(args.profile)

    renderer = MultiSourceRenderer(
        shot_point_path=args.shot_point,
        profile=profile,
        bgm_path=args.bgm,
        bgm_start_sec=args.bgm_start,
        bgm_duration_sec=args.bgm_duration,
        original_audio_volume=args.original_audio_volume,
    )
    result = renderer.render(args.output)
    print(f"\n渲染结果: {result.status}")
    if result.errors:
        for e in result.errors:
            print(f"  ❌ {e}")
