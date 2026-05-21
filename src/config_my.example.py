"""
config_my.example.py — 本地配置模板

使用方式：
  cp src/config_my.example.py src/config_my.py
  然后填入你的 API Key

config_my.py 已在 .gitignore 中，不会被提交到 git。
"""

import os

# ==============================================================================
# LLM / Agent 配置
# ==============================================================================

AGENT_LITELLM_API_KEY   = os.environ.get("AGENT_LITELLM_API_KEY", "")  # 或直接填 Key
AGENT_LITELLM_MODEL     = "openai/MiniMax-M2.7"
AGENT_LITELLM_URL       = "https://api.minimaxi.com/v1"

# ==============================================================================
# 视频分析模型（火山引擎豆包）
# ==============================================================================

VIDEO_ANALYSIS_API_KEY  = os.environ.get("VIDEO_ANALYSIS_API_KEY", "")
VIDEO_ANALYSIS_MODEL    = "openai/doubao-seed-2.0-lite"
VIDEO_ANALYSIS_ENDPOINT = "https://ark.cn-beijing.volces.com/api/plan/v3"

# ==============================================================================
# 音频分析模型（Mimo）
# ==============================================================================

AUDIO_LITELLM_API_KEY   = os.environ.get("AUDIO_LITELLM_API_KEY", "")
AUDIO_LITELLM_MODEL     = "openai/mimo-v2.5"
AUDIO_LITELLM_BASE_URL  = "https://api.xiaomimimo.com/v1"

# ==============================================================================
# 个性化配置
# ==============================================================================

MAIN_CHARACTER_NAME     = ""  # 修改为你的名字/角色名
