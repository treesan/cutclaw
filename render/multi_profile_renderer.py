"""
MultiProfileRenderer — 多平台输出渲染器
========================================

策略：
1. 先渲染长视频（bilibili_4k）— 完整叙事
2. 从长视频切片短视频（douyin/xiaohongshu）— 精华摘取
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from render.multi_source_renderer import MultiSourceRenderer, RenderResult

if TYPE_CHECKING:
    from src.project.project import Project
    from src.project.media_profiles import MediaProfile


class MultiProfileRenderer:
    """多平台输出渲染器。"""

    def render_all(
        self,
        project: "Project",
        shot_point_path: str,
        profiles: dict[str, "MediaProfile"],
        bgm_path: str = "",
        bgm_start_sec: float = 0.0,
        bgm_duration_sec: float = 0.0,
        original_audio_volume: float = 0.0,
    ) -> dict[str, RenderResult]:
        """
        按优先级渲染所有 Profile。

        Args:
            project: 批量项目
            shot_point_path: shot_point v2.0 路径
            profiles: {profile_name: MediaProfile} 字典
            bgm_path: BGM 路径
            bgm_start_sec: BGM 起始偏移
            bgm_duration_sec: BGM 使用时长

        Returns:
            {profile_name: RenderResult} 字典
        """
        output_dir = os.path.join(project.output_dir, "output")
        os.makedirs(output_dir, exist_ok=True)

        results = {}

        # 按 profile 排序：长视频优先（bilibili_4k > bilibili_1080p > douyin > xiaohongshu）
        priority_order = ["bilibili_4k", "bilibili_1080p", "douyin", "xiaohongshu"]
        sorted_profiles = sorted(
            profiles.items(),
            key=lambda x: priority_order.index(x[0]) if x[0] in priority_order else 99,
        )

        for profile_name, profile in sorted_profiles:
            output_path = os.path.join(output_dir, f"{profile_name}.mp4")
            print(f"\n{'='*60}")
            print(f"🎬 渲染 {profile_name} ({profile.width}x{profile.height})")
            print(f"{'='*60}")

            renderer = MultiSourceRenderer(
                shot_point_path=shot_point_path,
                profile=profile,
                bgm_path=bgm_path,
                bgm_start_sec=bgm_start_sec,
                bgm_duration_sec=bgm_duration_sec,
                original_audio_volume=original_audio_volume,
            )
            result = renderer.render(output_path)
            results[profile_name] = result

        # 汇总报告
        print(f"\n{'='*60}")
        print(f"📊 多平台渲染完成:")
        for name, r in results.items():
            icon = "✅" if r.status == "pass" else "❌"
            print(f"  {icon} {name}: {r.duration_sec:.1f}s, {r.file_size_mb:.1f}MB")
        print(f"{'='*60}")

        return results
