#!/usr/bin/env python3
"""
DirectShotSelector — 替代 EditorCoreAgent 的 Agent 循环，改用一次 LLM 直推
一次性生成完整的 shot_point.json（所有 shot 的精确时间戳）。
"""

import json
import os
import re
import time
import random
from typing import Any

from src import config
from src.utils.media_utils import hhmmss_to_seconds, load_scene_summaries


# litellm 延迟导入（在 _call_litellm 中 import）


# ──────────────────────────────────────────────────────────────
# 轻量工具
# ──────────────────────────────────────────────────────────────

def _call_litellm(messages: list, max_tokens: int = None, temperature: float = 0.3):
    """调用 LiteLLM，返回 content 或 None。"""
    import litellm
    kwargs = dict(
        model= config.AGENT_LITELLM_MODEL,
        messages=messages,
        max_tokens=max_tokens or getattr(config, "DIRECT_SHOT_SELECTOR_MAX_TOKENS", 8192),
        api_key=config.AGENT_LITELLM_API_KEY,
        temperature=temperature,
        timeout=120,
    )
    if config.AGENT_LITELLM_URL:
        kwargs["api_base"] = config.AGENT_LITELLM_URL
    try:
        resp = litellm.completion(**kwargs)
        content = resp.choices[0].message.content
        if content is None:
            return None
        if isinstance(content, str):
            return content
        return str(content)
    except Exception as e:
        print(f"❌ [DirectShotSelector] LLM call failed: {e}")
        return None


def _extract_json(text: str):
    """从 LLM 输出中提取第一个 JSON 块。"""
    if not text:
        return None
    text = text.strip()
    # 尝试 ```json ... ``` 格式
    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL | re.IGNORECASE)
    if m:
        text = m.group(1).strip()
    # 尝试直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # 尝试找第一个 { ... }（可能跨行）
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return None


def _sec_to_hhmmss(sec: float) -> str:
    """将秒转换为 HH:MM:SS.sss 格式。"""
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def _load_scene_cuts(path: str) -> list[float]:
    """加载 shot_scenes.txt 中的场景切分时间点。"""
    if not path or not os.path.exists(path):
        return []
    cuts = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # 支持一行多个值（空格分隔）
            for token in line.split():
                try:
                    cuts.append(float(token))
                except ValueError:
                    pass
    return cuts


def _load_scene_summaries_raw(scene_folder: str) -> list[dict]:
    """加载 scene_summaries_video 目录下所有 JSON，返回结构化列表（非拼接字符串）。"""
    results = []
    if not os.path.isdir(scene_folder):
        return results

    scene_files = [f for f in os.listdir(scene_folder)
                   if f.startswith("scene_") and f.endswith(".json")]

    def _sn(fn: str) -> int:
        try:
            return int(fn.replace("scene_", "").replace(".json", ""))
        except ValueError:
            return 99999

    scene_files.sort(key=_sn)

    for fn in scene_files:
        fp = os.path.join(scene_folder, fn)
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        va = data.get("video_analysis", {})
        sc = va.get("scene_caption", {})
        scls = sc.get("scene_classification", {})

        # 跳过不可用场景
        if not scls.get("is_usable", True):
            continue
        # vlog 模式下降低重要性阈值，避免过滤掉所有场景
        min_imp = 2 if config.VIDEO_TYPE == "vlog" else 3
        if scls.get("importance_score", 5) < min_imp:
            continue

        summary = sc.get("scene_summary", {})
        time_range = data.get("time_range", {})

        def _to_seconds(v):
            if isinstance(v, (int, float)):
                return float(v)
            try:
                return float(v)
            except (ValueError, TypeError):
                return hhmmss_to_seconds(str(v), fps=config.VIDEO_FPS)

        results.append({
            "scene_index": int(data.get("scene_id", _sn(fn))),
            "start_sec": _to_seconds(time_range.get("start_seconds", 0)),
            "end_sec": _to_seconds(time_range.get("end_seconds", 0)),
            "content": summary.get("narrative", ""),
            "location": summary.get("location", ""),
            "event": summary.get("key_event", ""),
            "scene_type": scls.get("scene_type", ""),
            "importance": scls.get("importance_score", 3),
        })
    print(f"  [DirectShotSelector] Loaded {len(results)} usable scenes from {scene_folder}")
    return results


def _compress_scene_summaries(scenes: list[dict], max_chars: int = 6000) -> str:
    """压缩 scene_summaries 为紧凑格式以节省 Prompt Token。"""
    parts = []
    for s in scenes:
        content = s.get("content", "")
        # 截断长内容
        if len(content) > 80:
            content = content[:77] + "..."
        parts.append(
            f"  S{s['scene_index']:3d}: {s['start_sec']:8.2f}-{s['end_sec']:8.2f}s  "
            f"loc={s['location'][:15] if s['location'] else '-':15s}  "
            f"type={s['scene_type'][:12]:12s}  "
            f"imp={s['importance']}  {content}"
        )
    text = "\n".join(parts)
    if len(text) > max_chars:
        # 保留高重要性场景
        ranked = sorted(scenes, key=lambda x: x["importance"], reverse=True)
        keep = []
        for s in ranked:
            line = (
                f"  S{s['scene_index']:3d}: {s['start_sec']:8.2f}-{s['end_sec']:8.2f}s  "
                f"type={s['scene_type'][:12]:12s}  imp={s['importance']}  "
                f"{s['content'][:60]}"
            )
            if sum(len(l) for l in keep) + len(line) > max_chars:
                break
            keep.append(line)
        text = "\n".join(keep)
        print(f"  [DirectShotSelector] Scene summaries compressed: {len(parts)} → {len(keep)} (by importance)")
    return text


def _compress_shot_plan(shot_plan: dict) -> str:
    """压缩 shot_plan 为紧凑文本。"""
    parts = []
    for si, sec in enumerate(shot_plan.get("video_structure", [])):
        sec_name = sec.get("section_name", f"Section {si}")
        parts.append(f"[Section {si}: {sec_name}]")
        shots = (sec.get("shot_plan") or {}).get("shots", [])
        for sj, shot in enumerate(shots):
            related = shot.get("related_scene", [])
            rs = ",".join(str(r) for r in related) if isinstance(related, list) else str(related)
            content = shot.get("content", "")
            if len(content) > 60:
                content = content[:57] + "..."
            parts.append(
                f"  Shot{sj:2d}: dur={shot.get('time_duration', 0):5.1f}s  "
                f"scene=[{rs}]  emotion={shot.get('emotion', '-')[:12]:12s}  {content}"
            )
    return "\n".join(parts)


def _compress_audio_data(audio_db: dict, shot_plan: dict) -> str:
    """压缩音频数据：整体 summary + 按 section 的 caption 列表。"""
    parts = []
    overall = audio_db.get("overall_analysis", {}).get("summary", "")
    if overall:
        parts.append(f"Audio overall: {overall}")

    sections = audio_db.get("sections", [])
    # 按 section 索引输出 caption
    for si, sec in enumerate(sections):
        name = sec.get("name", f"Section {si}")
        start = sec.get("Start_Time", 0)
        end = sec.get("End_Time", 0)
        desc = sec.get("description", "")
        parts.append(f"  Audio[{si}] {name} ({start}-{end}s): {desc}")
        detailed = sec.get("detailed_analysis", {})
        if isinstance(detailed, dict):
            sun = detailed.get("summary", "")
            if sun:
                parts.append(f"    Summary: {sun}")
            sec_list = detailed.get("sections", [])
            if isinstance(sec_list, list):
                for sj, cap in enumerate(sec_list):
                    if len(cap) > 100:
                        cap = cap[:97] + "..."
                    parts.append(f"    Shot{sj}: {cap}")
    return "\n".join(parts)


# ──────────────────────────────────────────────────────────────
# Prompt 模板
# ──────────────────────────────────────────────────────────────

DIRECT_SHOT_SYSTEM_PROMPT = """You are a professional video editor. Your job is to take a shot plan (creative brief) and exact scene data, and select the BEST continuous time ranges from the source video.

Rules:
1. Each shot must be a single continuous clip from the source video — no stitching multiple clips.
2. Different shots must NOT overlap in time.
3. Each shot's actual duration should be ±{tolerance}s of the target duration.
4. Respect scene boundaries — keep each shot within its related_scene(s) when possible.
5. If a scene has detected characters, prefer segments where the protagonist is visible.
6. For scenic/landscape content, aesthetic quality matters more than character presence.
7. Output ONLY valid JSON — no markdown, no explanation."""

DIRECT_SHOT_USER_PROMPT = """Here is the complete editing brief and available footage.

== SHOT PLAN ==
{shot_plan_text}

== AUDIO ANALYSIS ==
{audio_text}

== SCENE SUMMARIES ==
{scene_text}

== SCENE CUTS (frame-level boundaries, avoid crossing these) ==
{scene_cuts_text}

== REQUIREMENTS ==
- Total output: {total_shots} shots
- Instructions from user: {instruction}
- Main characters: {main_character}
- Video type: {video_type}
- Face quality check mode: {face_check_mode}

== 锦书剪辑要求 ==
{shot_point_context}

Output exactly ONE JSON object with this structure:
{{
  "shots": [
    {{
      "section_idx": 0,
      "shot_idx": 0,
      "start_sec": 13.4,
      "end_sec": 16.8,
      "scene_index": 5,
      "reasoning": "Brief justification"
    }}
  ],
  "hook_dialogue_used": null
}}

No markdown, no code fences, no extra text. Just the JSON."""


# ──────────────────────────────────────────────────────────────
# DirectShotSelector 主类
# ──────────────────────────────────────────────────────────────

class DirectShotSelector:
    """一次性生成所有 shot 的精确时间戳，替代 EditorCoreAgent 的迭代循环。"""

    def __init__(
        self,
        video_path: str,
        shot_plan_path: str,
        scene_summary_dir: str,
        audio_caption_path: str,
        scene_cuts_path = None,
        instruction: str = "",
        main_character: str = "",
        output_path: str = "",
        subtitle_path = None,
        face_check_mode = None,
        shot_point_context: str = "",  # 锦书自定义提示词：注入到 LLM 的额外指令
    ):
        self.video_path = video_path
        self.shot_plan_path = shot_plan_path
        self.scene_summary_dir = scene_summary_dir
        self.audio_caption_path = audio_caption_path
        self.scene_cuts_path = scene_cuts_path
        self.instruction = instruction
        self.main_character = main_character
        self.output_path = output_path
        self.subtitle_path = subtitle_path
        self.face_check_mode = face_check_mode or getattr(config, "FACE_QUALITY_CHECK_MODE", "auto")
        self.shot_point_context = shot_point_context

        # 加载数据
        print(f"[DirectShotSelector] Loading shot plan: {shot_plan_path}")
        with open(shot_plan_path, "r", encoding="utf-8") as f:
            self.shot_plan = json.load(f)

        print(f"[DirectShotSelector] Loading audio captions: {audio_caption_path}")
        with open(audio_caption_path, "r", encoding="utf-8") as f:
            self.audio_db = json.load(f)

        self.raw_scenes = _load_scene_summaries_raw(scene_summary_dir)
        self.scene_cuts = _load_scene_cuts(scene_cuts_path) if scene_cuts_path else []

        self._result = None

    # ── 输入组装 ──

    def _count_total_shots(self) -> int:
        """统计 shot_plan 中所有 shot 的个数。"""
        total = 0
        for sec in self.shot_plan.get("video_structure", []):
            shots = (sec.get("shot_plan") or {}).get("shots", [])
            total += len(shots)
        return total

    def _build_input(self) -> dict:
        """组装所有输入数据为压缩文本。"""
        shot_plan_text = _compress_shot_plan(self.shot_plan)
        audio_text = _compress_audio_data(self.audio_db, self.shot_plan)
        scene_text = _compress_scene_summaries(self.raw_scenes)
        scene_cuts_text = (
            ", ".join(f"{c:.2f}s" for c in self.scene_cuts[:30])
            if self.scene_cuts
            else "None available"
        )
        if len(self.scene_cuts) > 30:
            scene_cuts_text += f" ... (total {len(self.scene_cuts)} cuts)"

        total_shots = self._count_total_shots()
        tolerance = getattr(config, "ALLOW_DURATION_TOLERANCE", 1.0)

        return {
            "shot_plan_text": shot_plan_text,
            "audio_text": audio_text,
            "scene_text": scene_text,
            "scene_cuts_text": scene_cuts_text,
            "total_shots": total_shots,
            "instruction": self.instruction or self.shot_plan.get("instruction", ""),
            "main_character": self.main_character or config.MAIN_CHARACTER_NAME,
            "video_type": config.VIDEO_TYPE,
            "face_check_mode": self.face_check_mode,
            "tolerance": tolerance,
            "shot_point_context": self.shot_point_context,
        }

    def _build_prompt(self, inp: dict) -> list[dict]:
        """构建 LLM 输入 messages。"""
        system = DIRECT_SHOT_SYSTEM_PROMPT.format(tolerance=inp["tolerance"])
        user = DIRECT_SHOT_USER_PROMPT.format(**inp)
        print(f"\n── Prompt Stats ──")
        print(f"  System: {len(system)} chars")
        print(f"  User:   {len(user)} chars")
        print(f"  Total:  {len(system) + len(user)} chars")
        print(f"─────────────────\n")
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    # ── LLM 调用 ──

    def _call_llm(self, messages: list):
        """调用 LLM 并解析 JSON 输出。"""
        max_retries = 2
        last_error = None
        for attempt in range(max_retries):
            raw = _call_litellm(messages)
            if not raw:
                last_error = "Empty response"
                continue

            parsed = _extract_json(raw)
            if parsed is None:
                last_error = f"JSON parse failed (response length: {len(raw)})"
                print(f"  ⚠️  Attempt {attempt + 1}: {last_error}")
                if attempt == 0:
                    # 第二次尝试时加上修复提示
                    messages.append({
                        "role": "user",
                        "content": (
                            "Your previous output was not valid JSON. "
                            "Output ONLY the JSON object with no markdown, no code fences, no extra text."
                        ),
                    })
                continue
            return parsed

        print(f"❌ [DirectShotSelector] Failed after {max_retries} attempts: {last_error}")
        return None

    # ── 后处理验证 ──

    def validate(self, shot_point: dict) -> list[str]:
        """验证 shot_point 的合法性和一致性。返回错误列表，空列表=完全通过。"""
        errors = []
        shots = shot_point.get("shots", [])
        if not shots:
            errors.append("No shots in output")
            return errors

        # 收集 shot_plan 中的目标时长
        target_durations = {}
        for sec in self.shot_plan.get("video_structure", []):
            for sj, shot in enumerate((sec.get("shot_plan") or {}).get("shots", [])):
                target_durations[(sj,)] = shot.get("time_duration", 0)

        tolerance = getattr(config, "ALLOW_DURATION_TOLERANCE", 1.0)

        used_ranges: list[tuple[float, float]] = []
        for i, s in enumerate(shots):
            start = s.get("start_sec", 0)
            end = s.get("end_sec", 0)
            duration = end - start

            # 时长检查
            idx = s.get("shot_idx", i)
            target = target_durations.get((idx,)) or 0
            if target > 0 and abs(duration - target) > tolerance:
                errors.append(
                    f"Shot {idx}: duration {duration:.2f}s deviates from target {target:.2f}s (±{tolerance}s)"
                )

            # 重叠检查
            for j, (ps, pe) in enumerate(used_ranges):
                if start < pe and end > ps:
                    errors.append(f"Shot {idx} overlaps with shot at index position {j}: "
                                   f"[{ps:.2f}, {pe:.2f}] vs [{start:.2f}, {end:.2f}]")
            used_ranges.append((start, end))

            # 时长下限
            min_dur = getattr(config, "MIN_ACCEPTABLE_SHOT_DURATION", 2.0)
            if duration < min_dur:
                errors.append(f"Shot {idx}: duration {duration:.2f}s below minimum {min_dur:.2f}s")

        return errors

    # ── 轻量修复 ──

    def _repair(self, errors: list[str], shot_point: dict) -> dict:
        """对轻量错误做自动修复，严重错误则返回原数据。"""
        if not errors:
            return shot_point

        # 仅修复 "轻微偏差" 类错误：简单地截断重叠区
        shots = shot_point.get("shots", [])
        for i in range(1, len(shots)):
            prev = shots[i - 1]
            curr = shots[i]
            if prev["end_sec"] > curr["start_sec"]:
                # 将当前 shot 的开始时间 push 到上一个之后
                new_start = prev["end_sec"] + 0.01
                curr["start_sec"] = new_start
                curr["end_sec"] = max(new_start + 0.1, curr["end_sec"])
                curr["reasoning"] = (curr.get("reasoning", "") +
                                     " [repaired: adjusted to avoid overlap]")
                print(f"  🔧 [Repair] Shot {curr.get('shot_idx', i)} moved to {new_start:.2f}s")

        return shot_point

    # ── 主入口 ──

    def run(self):
        """执行直接推理，返回 shot_point dict。"""
        print("\n" + "=" * 60)
        print("🚀 [DirectShotSelector] Starting direct shot selection")
        print("=" * 60)

        inp = self._build_input()
        messages = self._build_prompt(inp)

        t0 = time.time()
        result = self._call_llm(messages)
        elapsed = time.time() - t0

        if result is None:
            print("❌ [DirectShotSelector] LLM call failed, no result.")
            return None

        print(f"✅ [DirectShotSelector] LLM returned in {elapsed:.1f}s")

        # 验证
        errors = self.validate(result)
        if errors:
            print(f"\n⚠️  [DirectShotSelector] Validation found {len(errors)} issues:")
            for e in errors:
                print(f"  - {e}")
            result = self._repair(errors, result)
            # 再次验证
            remaining = self.validate(result)
            if remaining:
                print(f"  ⚠️  {len(remaining)} issues after repair, returning anyway")
                for e in remaining:
                    print(f"    - {e}")
            else:
                print("  ✅ All issues resolved after repair")
        else:
            print("✅ [DirectShotSelector] Validation passed")

        # 写入输出
        if self.output_path:
            os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
            with open(self.output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"💾 [DirectShotSelector] Output saved to: {self.output_path}")

        self._result = result
        return result


# ──────────────────────────────────────────────────────────────
# 入口：可作为独立脚本测试
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DirectShotSelector — single-pass shot point generation")
    parser.add_argument("--shot-plan", required=True, help="Path to shot_plan.json")
    parser.add_argument("--scene-summaries", required=True, help="Path to scene_summaries_video directory")
    parser.add_argument("--audio-captions", required=True, help="Path to audio/captions.json")
    parser.add_argument("--video", default="", help="Path to source video")
    parser.add_argument("--scene-cuts", default="", help="Path to shot_scenes.txt")
    parser.add_argument("--instruction", default="", help="Editing instruction")
    parser.add_argument("--main-character", default="", help="Main character name")
    parser.add_argument("--output", default="", help="Output path for shot_point.json")
    args = parser.parse_args()

    selector = DirectShotSelector(
        video_path=args.video,
        shot_plan_path=args.shot_plan,
        scene_summary_dir=args.scene_summaries,
        audio_caption_path=args.audio_captions,
        scene_cuts_path=args.scene_cuts or None,
        instruction=args.instruction,
        main_character=args.main_character,
        output_path=args.output,
    )
    result = selector.run()
    if result:
        print(f"\n🎉 Generated {len(result.get('shots', []))} shots")
    else:
        print("\n❌ Failed to generate shot point")
