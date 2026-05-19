"""
MiMo (Xiaomi) Audio Understanding API 封装
============================================

支持 mimo-v2.5 / mimo-v2-omni 全模态模型的音频理解能力。
完全兼容 OpenAI Chat Completion 格式。

特性：
- 支持 WAV / MP3 音频输入（base64 data URI）
- 支持文本 + 音频同时输入（传递 prompt）
- 带指数退避重试（应对限流）
- 支持批量并发处理
- 可选标准 endpoint / Token Plan endpoint

配置：
    环境变量 MIMO_API_KEY 或 config.py 中的 MIMO_AUDIO_API_KEY

用法：
    from src.audio.mimo_audio_api import mimo_audio, MimoAudioAPI

    # 函数式
    text = mimo_audio("/path/to/audio.wav", "描述这段音频")

    # 对象式（可复用）
    api = MimoAudioAPI(model="mimo-v2-omni")
    text = api.transcribe("/path/to/audio.wav")
"""

import asyncio
import base64
import os
import random
import subprocess
import tempfile
import time
from pathlib import Path
from typing import List, Optional, Union

from openai import OpenAI, AsyncOpenAI

try:
    from .. import config as project_config
except ImportError:
    project_config = None


# --------------------------------------------------------------------------- #
# 配置读取
# --------------------------------------------------------------------------- #


def _get_setting(config_key: str, env_key: str, default=None):
    """优先从 config.py 读取，其次环境变量，最后默认值。"""
    if project_config is not None and hasattr(project_config, config_key):
        value = getattr(project_config, config_key)
        if value is not None and value != "":
            return value
    env_value = os.getenv(env_key)
    if env_value is not None and env_value != "":
        return env_value
    return default


# ---------- 标准 endpoint（sk- 密钥） ----------
MIMO_MODEL = _get_setting(
    config_key="MIMO_AUDIO_MODEL",
    env_key="MIMO_AUDIO_MODEL",
    default="mimo-v2.5",
)

MIMO_API_KEY = _get_setting(
    config_key="MIMO_AUDIO_API_KEY",
    env_key="MIMO_API_KEY",
    default="",
)

MIMO_BASE_URL = _get_setting(
    config_key="MIMO_AUDIO_BASE_URL",
    env_key="MIMO_AUDIO_BASE_URL",
    default="https://api.xiaomimimo.com/v1",
)

# ---------- Token Plan endpoint（tp- 密钥） ----------
MIMO_TP_ENABLED = _get_setting(
    config_key="MIMO_AUDIO_TP_ENABLED",
    env_key="MIMO_AUDIO_TP_ENABLED",
    default=False,
)
MIMO_TP_API_KEY = _get_setting(
    config_key="MIMO_AUDIO_TP_API_KEY",
    env_key="MIMO_AUDIO_TP_API_KEY",
    default="",
)
MIMO_TP_BASE_URL = _get_setting(
    config_key="MIMO_AUDIO_TP_BASE_URL",
    env_key="MIMO_AUDIO_TP_BASE_URL",
    default="https://token-plan-cn.xiaomimimo.com/v1",
)

MIMO_TIMEOUT = 300  # 单次请求超时（秒）
MIMO_MAX_RETRIES = 5  # 限流重试次数


# --------------------------------------------------------------------------- #
# 音频转换工具
# --------------------------------------------------------------------------- #

SUPPORTED_FORMATS = {"wav", "mp3", "m4a", "ogg", "flac", "aac"}


def _get_audio_format(path: str) -> str:
    """根据文件扩展名判断音频格式。"""
    ext = Path(path).suffix.lower().lstrip(".")
    if ext in SUPPORTED_FORMATS:
        return ext
    # 未知格式，ffprobe
    return "wav"


def _audio_to_base64(audio_path: str, target_format: str = "wav") -> str:
    """
    将音频转为指定格式的 base64 字符串。
    优先跳过格式转换（如果已经是 target_format），
    否则用 ffmpeg 转码。
    """
    ext = Path(audio_path).suffix.lower().lstrip(".")
    if ext == target_format:
        with open(audio_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    # 需要转码
    with tempfile.NamedTemporaryFile(suffix=f".{target_format}", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        # 16kHz 单声道降低传输大小
        subprocess.run(
            ["ffmpeg", "-y", "-i", audio_path,
             "-ac", "1", "-ar", "16000",
             tmp_path],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        with open(tmp_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


# --------------------------------------------------------------------------- #
# 消息构建
# --------------------------------------------------------------------------- #


def _build_messages(
    prompt: str,
    audio_b64: str,
    audio_format: str = "wav",
) -> list:
    """构建 MiMo 多模态消息。"""
    mime_type = f"audio/{audio_format}"
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": f"data:{mime_type};base64,{audio_b64}",
                        "format": audio_format,
                    },
                },
            ],
        }
    ]


# --------------------------------------------------------------------------- #
# API 调用核心（带重试）
# --------------------------------------------------------------------------- #


def _do_retryable_call(callable_fn, max_retries: int = None) -> str:
    """
    带指数退避重试的同步调用。

    遇到 429 / 5xx 等可重试错误时自动重试。
    """
    max_retries = max_retries or MIMO_MAX_RETRIES
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            return callable_fn()
        except Exception as e:
            last_error = e
            err_str = str(e)
            is_retryable = any(
                kw in err_str.lower()
                for kw in ["429", "500", "502", "503", "rate", "throttl", "timeout",
                           "too many", "retry", "quota", "overload"]
            )
            if is_retryable and attempt < max_retries:
                delay = min(2 ** attempt + random.uniform(0, 2), 30)
                print(
                    f"[mimo_audio_api] Retry {attempt}/{max_retries} "
                    f"in {delay:.1f}s — {type(e).__name__}: {e!s:.120}"
                )
                time.sleep(delay)
            else:
                raise
    raise last_error  # 防御性


# --------------------------------------------------------------------------- #
# MiMo Audio API 类
# --------------------------------------------------------------------------- #


class MimoAudioAPI:
    """
    MiMo 音频理解 API 封装。

    参数：
        model:      模型名（默认 mimo-v2.5）
        api_key:    API 密钥（默认从环境/config 读取）
        base_url:   API 端点（默认 https://api.xiaomimimo.com/v1）
        use_tp:     是否使用 Token Plan 端点（需要 tp- 密钥）
        timeout:    请求超时（秒）
        max_retries: 限流重试次数
    """

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        use_tp: bool = False,
        timeout: int = MIMO_TIMEOUT,
        max_retries: int = MIMO_MAX_RETRIES,
    ):
        self.model = model or MIMO_MODEL
        self.max_retries = max_retries

        if use_tp:
            self.api_key = api_key or MIMO_TP_API_KEY
            self.base_url = base_url or MIMO_TP_BASE_URL
        else:
            self.api_key = api_key or MIMO_API_KEY
            self.base_url = base_url or MIMO_BASE_URL

        if not self.api_key:
            raise ValueError(
                "MiMo API key not configured. Set MIMO_API_KEY env var "
                "or pass api_key directly."
            )

        self.timeout = timeout
        self._sync_client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
        )
        self._async_client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
        )

    # ------------------------------------------------------------------ #
    # 同步调用
    # ------------------------------------------------------------------ #

    def call(
        self,
        audio_path: str,
        prompt: str = "",
        temperature: float = 0.0,
        max_tokens: int = 4096,
        top_p: Optional[float] = None,
    ) -> str:
        """
        同步调用 MiMo 音频理解。

        参数：
            audio_path:  本地音频文件路径
            prompt:      分析提示词（如 "描述这段音频"）
            temperature: 采样温度（ASR 建议 0.0）
            max_tokens:  最大输出 token 数
            top_p:       Top-p 采样

        返回：
            模型回复文本
        """
        fmt = _get_audio_format(audio_path)
        audio_b64 = _audio_to_base64(audio_path, fmt)

        kwargs = dict(
            model=self.model,
            messages=_build_messages(prompt, audio_b64, fmt),
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if top_p is not None:
            kwargs["top_p"] = top_p

        def _call():
            return self._sync_client.chat.completions.create(**kwargs)

        response = _do_retryable_call(_call, self.max_retries)
        return response.choices[0].message.content

    def transcribe(
        self,
        audio_path: str,
        language: str = "zh",
        temperature: float = 0.0,
    ) -> str:
        """
        便捷的语音转写：直接返回音频的文字内容。

        参数：
            audio_path: 本地音频文件路径
            language:   语言（zh/en）
            temperature: 采样温度
        """
        lang_hint = {"zh": "请用中文", "en": "Please"}.get(language, "")
        prompt = f"{lang_hint}逐字转写这段音频中的语音内容，返回纯文本，不要添加任何额外说明。"
        return self.call(audio_path, prompt=prompt, temperature=temperature)

    def describe(self, audio_path: str, detail: str = "general") -> str:
        """
        描述音频内容（音乐/环境/语音等）。

        参数：
            audio_path: 本地音频文件路径
            detail:     general（通用）/ music（音乐分析）/ soundscape（环境音）
        """
        prompts = {
            "general": "请用中文详细描述这段音频的内容，包括语音、背景音、音乐等所有声音元素。",
            "music": "你是一位专业音乐分析师。请分析这段音乐的节奏、乐器、风格、情绪等特征。",
            "soundscape": "请用中文描述这段音频中的环境声音和场景氛围。",
        }
        prompt = prompts.get(detail, prompts["general"])
        return self.call(audio_path, prompt=prompt, temperature=0.3)

    # ------------------------------------------------------------------ #
    # 异步调用
    # ------------------------------------------------------------------ #

    async def acall(
        self,
        audio_path: str,
        prompt: str = "",
        temperature: float = 0.0,
        max_tokens: int = 4096,
        top_p: Optional[float] = None,
    ) -> str:
        """异步版本，用法同 call()。"""
        loop = asyncio.get_running_loop()

        # 音频转 base64 在 executor 中执行（避免阻塞事件循环）
        fmt = _get_audio_format(audio_path)
        audio_b64 = await loop.run_in_executor(None, _audio_to_base64, audio_path, fmt)

        kwargs = dict(
            model=self.model,
            messages=_build_messages(prompt, audio_b64, fmt),
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if top_p is not None:
            kwargs["top_p"] = top_p

        async def _call():
            return await self._async_client.chat.completions.create(**kwargs)

        # 异步重试
        for attempt in range(1, self.max_retries + 1):
            try:
                response = await _call()
                return response.choices[0].message.content
            except Exception as e:
                err_str = str(e).lower()
                is_retryable = any(
                    kw in err_str
                    for kw in ["429", "500", "502", "503", "rate", "throttl", "timeout",
                               "too many", "retry", "quota", "overload"]
                )
                if is_retryable and attempt < self.max_retries:
                    delay = min(2 ** attempt + random.uniform(0, 2), 30)
                    print(
                        f"[mimo_audio_api] Async retry {attempt}/{self.max_retries} "
                        f"in {delay:.1f}s — {type(e).__name__}: {e!s:.120}"
                    )
                    await asyncio.sleep(delay)
                else:
                    raise

    async def atranscribe(
        self,
        audio_path: str,
        language: str = "zh",
        temperature: float = 0.0,
    ) -> str:
        """异步转写。"""
        lang_hint = {"zh": "请用中文", "en": "Please"}.get(language, "")
        prompt = f"{lang_hint}逐字转写这段音频中的语音内容，返回纯文本，不要添加任何额外说明。"
        return await self.acall(audio_path, prompt=prompt, temperature=temperature)

    # ------------------------------------------------------------------ #
    # 批量调用
    # ------------------------------------------------------------------ #

    def call_batch(
        self,
        audio_paths: List[str],
        prompt: str = "",
        temperature: float = 0.0,
        max_tokens: int = 4096,
        max_workers: int = 5,
    ) -> List[str]:
        """同步批量调用（内部使用 asyncio 并行）。"""
        return asyncio.run(
            self.acall_batch(
                audio_paths, prompt, temperature, max_tokens, max_workers,
                progress_interval=10,
            )
        )

    async def acall_batch(
        self,
        audio_paths: List[str],
        prompt: str = "",
        temperature: float = 0.0,
        max_tokens: int = 4096,
        max_concurrent: int = 3,
        progress_interval: int = 10,
    ) -> List[str]:
        """
        异步批量调用，带并发限制和进度日志。

        progress_interval: 每完成 N 个打印一次进度 (0=不打印)
        """
        if not audio_paths:
            return []

        total = len(audio_paths)
        completed = 0
        lock = asyncio.Lock()
        progress_logged_at = 0
        sem = asyncio.Semaphore(max_concurrent)
        results = [None] * total  # 预分配保持顺序

        async def _process(idx: int, path: str) -> None:
            nonlocal completed, progress_logged_at
            async with sem:
                try:
                    result = await self.acall(path, prompt, temperature, max_tokens)
                except Exception as e:
                    print(f"[mimo_audio_api] Batch failed [{idx}/{total}]: {path}: {e}")
                    result = ""

            async with lock:
                results[idx] = result
                completed += 1
                if progress_interval > 0 and (completed - progress_logged_at >= progress_interval or completed == total):
                    pct = completed * 100 // total
                    print(f"[mimo_audio_api] Batch progress: {completed}/{total} ({pct}%)")
                    progress_logged_at = completed

        tasks = [_process(i, p) for i, p in enumerate(audio_paths)]
        await asyncio.gather(*tasks, return_exceptions=False)

        return list(results)


# --------------------------------------------------------------------------- #
# 默认实例 & 快捷函数
# --------------------------------------------------------------------------- #

_default_api = None


def _get_default_api() -> MimoAudioAPI:
    global _default_api
    if _default_api is None:
        _default_api = MimoAudioAPI()
    return _default_api


def mimo_audio(
    audio_path: str,
    prompt: str = "",
    temperature: float = 0.0,
    max_tokens: int = 4096,
    **kwargs,
) -> str:
    """快捷函数：调用 MiMo 音频理解。"""
    return _get_default_api().call(audio_path, prompt, temperature, max_tokens, **kwargs)


def mimo_transcribe(audio_path: str, language: str = "zh") -> str:
    """快捷函数：转写音频。"""
    return _get_default_api().transcribe(audio_path, language)


def mimo_describe(audio_path: str, detail: str = "general") -> str:
    """快捷函数：描述音频。"""
    return _get_default_api().describe(audio_path, detail)


# --------------------------------------------------------------------------- #
# 入口：测试
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("用法: python mimo_audio_api.py <音频文件路径> [prompt]")
        sys.exit(1)

    path = sys.argv[1]
    prompt = sys.argv[2] if len(sys.argv) > 2 else "请用中文详细描述这段音频的内容"

    print(f"📁 音频: {path}")
    print(f"💬 Prompt: {prompt}")
    print(f"🤖 模型: {_get_default_api().model}")
    print(f"🔗 端点: {_get_default_api().base_url}")
    print()

    result = mimo_audio(path, prompt)
    print(f"📝 回复:\n{result}")
