"""
Lightweight litellm wrapper for audio analysis via cloud API.

路由规则（按优先级）：
1. 模型名含 ``mimo`` → 路由到 ``mimo_audio_api``（MiMo 原生 input_audio 协议）
2. 模型名为 Qwen-Omni 系列 → 路由到 ``qwen_audio_api``（Qwen 原生 input_audio 协议）
3. 其他 → 使用 litellm（回退兼容行为）

Reads configuration from src/audio/.env:
  AUDIO_MODEL    - LiteLLM model string (e.g. openai/Qwen3-Omni-30B-A3B-Instruct)
  AUDIO_API_KEY  - API key (use EMPTY for no auth)
  AUDIO_BASE_URL - Base URL for OpenAI-compatible endpoints
"""

import os
import asyncio
import base64
import subprocess
import tempfile
from pathlib import Path
from typing import List

import litellm
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

try:
    from .. import config as project_config
except Exception:
    project_config = None

# -- 各个后端的 API 封装 --
from .qwen_audio_api import (
    call_qwen_audio,
    call_qwen_audio_batch,
    acall_qwen_audio,
    acall_qwen_audio_batch,
    QWEN_MODEL as _RAW_QWEN_MODEL,
)
from .mimo_audio_api import (
    mimo_audio,
    mimo_transcribe,
    MimoAudioAPI,
)

# Load .env from the same directory as this file
load_dotenv(Path(__file__).parent / ".env")


def _get_setting(config_key: str, env_key: str, default=None):
    """Read setting from src/config.py first, then fallback to environment/.env."""
    if project_config is not None and hasattr(project_config, config_key):
        value = getattr(project_config, config_key)
        if value is not None and value != "":
            return value
    env_value = os.getenv(env_key)
    if env_value is not None and env_value != "":
        return env_value
    return default


AUDIO_MODEL = _get_setting(
    config_key="AUDIO_LITELLM_MODEL",
    env_key="AUDIO_MODEL",
    default="openai/Qwen3-Omni-30B-A3B-Instruct",
)
AUDIO_API_KEY = _get_setting(
    config_key="AUDIO_LITELLM_API_KEY",
    env_key="AUDIO_API_KEY",
    default="EMPTY",
) or "EMPTY"
AUDIO_BASE_URL = _get_setting(
    config_key="AUDIO_LITELLM_BASE_URL",
    env_key="AUDIO_BASE_URL",
    default=None,
)


# --------------------------------------------------------------------------- #
# 路由检测
# --------------------------------------------------------------------------- #

# ---- MiMo 检测 ----

def _is_mimo() -> bool:
    """检测当前音频模型是否为 MiMo 系列。"""
    name = AUDIO_MODEL.lower()
    return "mimo" in name


def _mimo_kwargs() -> dict:
    """提取 mimo_audio_api 需要的 kwargs。"""
    model = AUDIO_MODEL
    if model.startswith("openai/"):
        model = model[len("openai/"):]
    # 如果配置了 base_url 且包含 "token-plan"，自动使用 Token Plan 密钥
    base_url = AUDIO_BASE_URL
    use_tp = base_url and "token-plan" in base_url.lower()
    return dict(
        model=model,
        api_key=AUDIO_API_KEY,
        base_url=base_url,
        use_tp=use_tp,
    )


# ---- Qwen-Omni 检测 ----

def _is_qwen_omni() -> bool:
    """检测当前音频模型是否为 Qwen-Omni 系列（含 3.5）。"""
    name = AUDIO_MODEL.lower()
    return ("qwen-omni" in name or "qwen_omni" in name
            or "qwen3-omni" in name or "qwen3_omni" in name
            or "qwen3.5-omni" in name)


def _qwen_kwargs() -> dict:
    """提取 qwen_audio_api 需要的 kwargs（去掉 litellm ``openai/`` 前缀）。"""
    model = AUDIO_MODEL
    if model.startswith("openai/"):
        model = model[len("openai/"):]
    return dict(model=model, api_key=AUDIO_API_KEY, base_url=AUDIO_BASE_URL)


# --------------------------------------------------------------------------- #
# 音频转 base64（回退用）
# --------------------------------------------------------------------------- #

def _audio_to_base64_mp3(audio_path: str) -> str:
    """Convert audio file to base64-encoded MP3 for cloud API submission."""
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", audio_path, "-ac", "1", "-ar", "16000", "-ab", "32k", tmp_path],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        with open(tmp_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


# --------------------------------------------------------------------------- #
# 单条音频调用
# --------------------------------------------------------------------------- #

@retry(
    reraise=True,
    stop=stop_after_attempt(4),  # 1 次正常 + 最多 3 次重试
    wait=wait_exponential(multiplier=2, min=1, max=30),
    retry=retry_if_exception_type(Exception),
    before_sleep=lambda rs: print(
        f"[litellm_client] Retry {rs.attempt_number}/3 "
        f"after {rs.outcome.exception()} — sleeping {rs.next_action.sleep:.1f}s"
    ),
)
async def acall_audio_api(
    audio_path: str,
    prompt: str,
    temperature: float = 0.7,
    top_p: float = 0.95,
    max_tokens: int = 4096,
) -> str:
    """
    Async: Call the cloud audio API with a prompt about the given audio file.

    Automatically routes to:
    - MiMo (mimo-v2.5 / mimo-v2-omni)
    - Qwen-Omni (qwen3-omni / qwen3.5-omni)
    - Plain litellm (fallback)

    Args:
        audio_path: Path to audio file
        prompt: Text prompt for analysis
        temperature: Sampling temperature
        top_p: Top-p sampling
        max_tokens: Max tokens to generate

    Returns:
        Response text from the model
    """
    # 优先级 1: MiMo
    if _is_mimo():
        api = MimoAudioAPI(**_mimo_kwargs())
        return await api.acall(
            audio_path, prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    # 优先级 2: Qwen-Omni
    if _is_qwen_omni():
        return await acall_qwen_audio(
            audio_path, prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            **_qwen_kwargs(),
        )

    # 优先级 3: 旧模型 → litellm + image_url
    loop = asyncio.get_running_loop()
    audio_b64 = await loop.run_in_executor(None, _audio_to_base64_mp3, audio_path)

    response = await litellm.acompletion(
        model=AUDIO_MODEL,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:audio/mp3;base64,{audio_b64}"}},
            ],
        }],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        timeout=300,
        api_key=AUDIO_API_KEY,
        **({"api_base": AUDIO_BASE_URL} if AUDIO_BASE_URL else {}),
    )

    return response.choices[0].message.content


def call_audio_api(
    audio_path: str,
    prompt: str,
    temperature: float = 0.7,
    top_p: float = 0.95,
    max_tokens: int = 4096,
) -> str:
    """
    Sync wrapper around acall_audio_api.

    Args:
        audio_path: Path to audio file
        prompt: Text prompt for analysis
        temperature: Sampling temperature
        top_p: Top-p sampling
        max_tokens: Max tokens to generate

    Returns:
        Response text from the model
    """
    return asyncio.run(acall_audio_api(audio_path, prompt, temperature, top_p, max_tokens))


# --------------------------------------------------------------------------- #
# 批量调用（频率已调低：max_concurrent=3）
# --------------------------------------------------------------------------- #

async def acall_audio_api_batch(
    audio_paths: List[str],
    prompt: str,
    temperature: float = 0.7,
    top_p: float = 0.95,
    max_tokens: int = 4096,
    max_concurrent: int = 3,  # ⬅ 从 5 降为 3，降低并发压力
) -> List[str]:
    """
    Async: Call the cloud audio API concurrently for multiple audio files.

    Uses asyncio.gather with a semaphore to limit concurrent requests.
    Each request is retried automatically via the @retry decorator on acall_audio_api.

    Args:
        audio_paths: List of paths to audio files
        prompt: Text prompt for analysis
        temperature: Sampling temperature
        top_p: Top-p sampling
        max_tokens: Max tokens to generate
        max_concurrent: Max simultaneous API requests (默认 3，已降低并发压力)

    Returns:
        List of response texts (same order as audio_paths, "" on error)
    """
    if not audio_paths:
        return []

    # MiMo 批量
    if _is_mimo():
        api = MimoAudioAPI(**_mimo_kwargs())
        return await api.acall_batch(
            audio_paths, prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            max_concurrent=max_concurrent,
        )

    # Qwen-Omni 批量
    if _is_qwen_omni():
        return await acall_qwen_audio_batch(
            audio_paths, prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            max_concurrent=max_concurrent,
            **_qwen_kwargs(),
        )

    # 旧模型批量
    sem = asyncio.Semaphore(max_concurrent)

    async def _limited(path: str) -> str:
        async with sem:
            return await acall_audio_api(path, prompt, temperature, top_p, max_tokens)

    results = await asyncio.gather(
        *[_limited(p) for p in audio_paths],
        return_exceptions=True,
    )

    processed = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"[litellm_client] Failed after retries: {audio_paths[i]}: {result}")
            processed.append("")
        else:
            processed.append(result)
    return processed


def call_audio_api_batch(
    audio_paths: List[str],
    prompt: str,
    temperature: float = 0.7,
    top_p: float = 0.95,
    max_tokens: int = 4096,
    max_workers: int = 3,  # ⬅ 从 5 降为 3
) -> List[str]:
    """
    Sync wrapper around acall_audio_api_batch.

    Args:
        audio_paths: List of paths to audio files
        prompt: Text prompt for analysis
        temperature: Sampling temperature
        top_p: Top-p sampling
        max_tokens: Max tokens to generate
        max_workers: Max simultaneous API requests（默认 3）

    Returns:
        List of response texts (same order as audio_paths, "" on error)
    """
    return asyncio.run(
        acall_audio_api_batch(audio_paths, prompt, temperature, top_p, max_tokens, max_workers)
    )
