"""
Qwen-Omni 多模态模型 API 封装
==============================

支持 Qwen3.5-Omni-Plus / Qwen3.5-Omni-Flash / Qwen3-Omni-Flash 等模型。

与 Qwen3-Omni-Captioner 不同，Qwen3.5-Omni 系列支持：
1. base64 data URI 直接传音频（无需文件上传）
2. text + audio 同一条消息共存（可以传 prompt）
3. `input_audio` 使用 ``data`` + ``format`` 两个字段

用法：
    from src.audio.qwen_audio_api import call_qwen_audio
    text = call_qwen_audio("path/to/audio.wav", "描述这段音频")
"""

import os
import asyncio
import base64
import time
import random
import tempfile
import subprocess
from pathlib import Path
from typing import List, Optional

from openai import OpenAI, AsyncOpenAI

try:
    from .. import config as project_config
except Exception:
    project_config = None


# --------------------------------------------------------------------------- #
# 配置
# --------------------------------------------------------------------------- #

def _get_setting(config_key: str, env_key: str, default=None):
    if project_config is not None and hasattr(project_config, config_key):
        value = getattr(project_config, config_key)
        if value is not None and value != "":
            return value
    env_value = os.getenv(env_key)
    if env_value is not None and env_value != "":
        return env_value
    return default


QWEN_MODEL = _get_setting(
    config_key="AUDIO_LITELLM_MODEL",
    env_key="QWEN_AUDIO_MODEL",
    default="qwen3.5-omni-plus",
)
if QWEN_MODEL.startswith("openai/"):
    QWEN_MODEL = QWEN_MODEL[len("openai/"):]

QWEN_API_KEY = _get_setting(
    config_key="AUDIO_LITELLM_API_KEY",
    env_key="QWEN_AUDIO_API_KEY",
    default="",
)

QWEN_BASE_URL = _get_setting(
    config_key="AUDIO_LITELLM_BASE_URL",
    env_key="QWEN_AUDIO_BASE_URL",
    default="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

QWEN_TIMEOUT = 300
QWEN_MAX_RETRIES = 3  # rate limit 重试次数


# --------------------------------------------------------------------------- #
# 音频 → base64 MP3（16kHz 低码率）
# --------------------------------------------------------------------------- #

def _audio_to_base64(audio_path: str) -> str:
    """将音频转为 16kHz 低码率 MP3，返回 base64 字符串。"""
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", audio_path,
             "-ac", "1", "-ar", "16000",
             "-ab", "16k",
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
# 消息构建（Qwen-Omni 系列支持 text + input_audio 共存）
# --------------------------------------------------------------------------- #

def _build_messages(prompt: str, audio_b64: str) -> list:
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": f"data:;base64,{audio_b64}",
                        "format": "mp3",
                    },
                },
            ],
        }
    ]


# --------------------------------------------------------------------------- #
# 重试装饰器（应对 Throttling.RateQuota）
# --------------------------------------------------------------------------- #

def _retry_on_throttle(func):
    """简易重试：遇到 429 或 5xx 时最多重试 QWEN_MAX_RETRIES 次。"""
    for attempt in range(1, QWEN_MAX_RETRIES + 1):
        try:
            return func()
        except Exception as e:
            err_str = str(e)
            is_retryable = any(
                keyword in err_str
                for keyword in ["Throttling", "RateQuota", "429", "500", "502", "503"]
            )
            if is_retryable and attempt < QWEN_MAX_RETRIES:
                delay = 2 ** attempt + random.uniform(0, 1)
                print(f"[qwen_audio_api] Rate limited, retry {attempt}/{QWEN_MAX_RETRIES} in {delay:.1f}s")
                time.sleep(delay)
            else:
                raise


# --------------------------------------------------------------------------- #
# 单条音频调用
# --------------------------------------------------------------------------- #

def call_qwen_audio(
    audio_path: str,
    prompt: str = "",
    temperature: float = 0.0,
    max_tokens: int = 4096,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> str:
    """
    调用 Qwen-Omni 模型分析一段音频。

    Qwen3.5-Omni-Plus 支持 text + audio 同时输入，
    ``prompt`` 参数不会被忽略（与 Qwen3-Omni-Captioner 不同）。

    参数：
        audio_path: 本地音频文件路径
        prompt:     分析提示词（Qwen3.5-Omni 支持！）
        temperature: 采样温度（ASR 建议 0.0）
        max_tokens:  最大输出 token 数
    """
    return asyncio.run(
        acall_qwen_audio(audio_path, prompt, temperature, max_tokens, model, api_key, base_url)
    )


async def acall_qwen_audio(
    audio_path: str,
    prompt: str = "",
    temperature: float = 0.0,
    max_tokens: int = 4096,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> str:
    model = model or QWEN_MODEL
    api_key = api_key or QWEN_API_KEY
    base_url = base_url or QWEN_BASE_URL

    # 转换音频
    loop = asyncio.get_running_loop()
    audio_b64 = await loop.run_in_executor(None, _audio_to_base64, audio_path)

    # 使用同步 OpenAI 客户端（避免 async 协程的 await 问题）
    def _do_call():
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url=base_url, timeout=QWEN_TIMEOUT)
        return client.chat.completions.create(
            model=model,
            messages=_build_messages(prompt, audio_b64),
            temperature=temperature,
            max_tokens=max_tokens,
        )

    response = await loop.run_in_executor(None, _do_call_with_retry, _do_call)
    return response.choices[0].message.content


def _do_call_with_retry(callable_fn):
    """同步版，带 rate limit 重试。"""
    for attempt in range(1, QWEN_MAX_RETRIES + 1):
        try:
            return callable_fn()
        except Exception as e:
            err_str = str(e)
            is_retryable = any(
                kw in err_str
                for kw in ["Throttling", "RateQuota", "429", "500", "502", "503"]
            )
            if is_retryable and attempt < QWEN_MAX_RETRIES:
                delay = 2 ** attempt + random.uniform(0, 1)
                print(f"[qwen_audio_api] Rate limited, retry {attempt}/{QWEN_MAX_RETRIES} in {delay:.1f}s")
                time.sleep(delay)
            else:
                raise


# --------------------------------------------------------------------------- #
# 批量音频调用
# --------------------------------------------------------------------------- #

async def acall_qwen_audio_batch(
    audio_paths: List[str],
    prompt: str = "",
    temperature: float = 0.0,
    max_tokens: int = 4096,
    max_concurrent: int = 5,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> List[str]:
    if not audio_paths:
        return []

    sem = asyncio.Semaphore(max_concurrent)

    async def _limited(path: str) -> str:
        async with sem:
            try:
                return await acall_qwen_audio(
                    path, prompt, temperature, max_tokens, model, api_key, base_url,
                )
            except Exception as e:
                print(f"[qwen_audio_api] Failed: {path}: {e}")
                return ""

    results = await asyncio.gather(*[_limited(p) for p in audio_paths], return_exceptions=False)
    return list(results)


def call_qwen_audio_batch(
    audio_paths: List[str],
    prompt: str = "",
    temperature: float = 0.0,
    max_tokens: int = 4096,
    max_workers: int = 5,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> List[str]:
    return asyncio.run(
        acall_qwen_audio_batch(
            audio_paths, prompt, temperature, max_tokens, max_workers, model, api_key, base_url,
        )
    )
