"""
共享路径工具 — 统一 shot_point / shot_plan 等产物路径构建与自动发现。

优化 #12：消除 local_run.py 与 app.py 之间的路径构建重复逻辑。
"""

import hashlib
import os
import re


def sanitize_id(name: str) -> str:
    """将文件名转为安全 ID（去扩展名、点→下划线、空格→下划线）。"""
    base = os.path.splitext(os.path.basename(name))[0]
    return base.replace(".", "_").replace(" ", "_")


def derive_instruction_id(instruction: str) -> str:
    """从编辑指令生成唯一 ID：可读前缀 + MD5 哈希。"""
    h = hashlib.md5(instruction.encode("utf-8")).hexdigest()[:8]
    safe = re.sub(r"[^\w\s-]", "", instruction)[:50].strip().replace(" ", "_")
    return f"{safe}_{h}" if safe else f"instruction_{h}"


def derive_artifact_path(
    video_path: str,
    audio_path: str,
    instruction: str,
    prefix: str,
    base_dir: str = "",
) -> str:
    """
    构建单视频模式的产物路径。

    格式: {base_dir}/Output/{video_id}_{audio_id}/{prefix}_{instruction_id}.json

    Args:
        video_path: 源视频路径
        audio_path: BGM 路径（可为空）
        instruction: 编辑指令
        prefix: 文件前缀，如 "shot_point"、"shot_plan"
        base_dir: 基础目录（默认 VIDEO_DATABASE_FOLDER）
    """
    from src import config

    if not base_dir:
        base_dir = config.VIDEO_DATABASE_FOLDER
    video_id = sanitize_id(video_path)
    audio_id = sanitize_id(audio_path) if audio_path else "no_audio"
    instruction_id = derive_instruction_id(instruction)
    return os.path.join(
        base_dir, "Output", f"{video_id}_{audio_id}",
        f"{prefix}_{instruction_id}.json",
    )


def discover_shot_point(output_dir: str, profile_name: str = "") -> str:
    """
    自动发现 shot_point 文件。

    优先匹配 profile 名称，退化为目录下最新 shot_point。

    Args:
        output_dir: shot_points 目录路径
        profile_name: 优先匹配的 profile 名称（如 "bilibili_1080p"）

    Returns:
        匹配的 shot_point 路径，未找到返回空字符串
    """
    if not os.path.isdir(output_dir):
        return ""

    # 优先：精确匹配 profile 名称
    if profile_name:
        candidate = os.path.join(output_dir, f"shot_point_{profile_name}.json")
        if os.path.exists(candidate):
            return candidate

    # 退化：目录下最新 shot_point
    try:
        candidates = sorted(
            [os.path.join(output_dir, f) for f in os.listdir(output_dir)
             if f.startswith("shot_point_") and f.endswith(".json")],
            key=os.path.getmtime,
            reverse=True,
        )
        return candidates[0] if candidates else ""
    except OSError:
        return ""
