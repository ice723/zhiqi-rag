"""大模型配置。

- 业务对话/信息抽取：默认硅基流动 Qwen/Qwen3-8B（OpenAI 兼容接口）
- 图片理解（VL）：复用 DashScope qwen3-vl-plus
- 两套配置分开，Key 也分开，方便分别替换
"""

from dataclasses import dataclass, field
from typing import Any, Dict

from config.env_loader import env_float, env_json, env_str

DEFAULT_LLM_BASE_URL = "https://api.siliconflow.cn/v1"
DEFAULT_LLM_MODEL = "Qwen/Qwen3-8B"
DEFAULT_VL_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_VL_MODEL = "qwen3-vl-plus"


@dataclass
class LLMConfig:
    base_url: str
    api_key: str
    llm_model: str
    item_model: str
    llm_temperature: float
    extra_body: Dict[str, Any] = field(default_factory=dict)

    vl_base_url: str = DEFAULT_VL_BASE_URL
    vl_api_key: str = ""
    vl_model: str = DEFAULT_VL_MODEL


def _build_extra_body() -> Dict[str, Any]:
    """Qwen3 系列默认关闭思考模式，避免 JSON 抽取被思维链干扰。"""
    explicit = env_json("LLM_EXTRA_BODY")
    if explicit is not None:
        return explicit

    enable_thinking = env_str("LLM_ENABLE_THINKING", default="0").lower()
    if enable_thinking in {"1", "true", "yes", "on"}:
        return {"enable_thinking": True}
    return {"enable_thinking": False}


lm_config = LLMConfig(
    base_url=env_str("LLM_API_BASE", "OPENAI_API_BASE", default=DEFAULT_LLM_BASE_URL),
    api_key=env_str("LLM_API_KEY", "SILICONFLOW_API_KEY", "OPENAI_API_KEY"),
    llm_model=env_str("LLM_MODEL", "LLM_DEFAULT_MODEL", default=DEFAULT_LLM_MODEL),
    item_model=env_str("ITEM_MODEL", "LLM_MODEL", "LLM_DEFAULT_MODEL", default=DEFAULT_LLM_MODEL),
    llm_temperature=env_float("LLM_DEFAULT_TEMPERATURE", default=0.1),
    extra_body=_build_extra_body(),
    vl_base_url=env_str("VL_API_BASE", "DASHSCOPE_API_BASE", "OPENAI_API_BASE", default=DEFAULT_VL_BASE_URL),
    vl_api_key=env_str("VL_API_KEY", "DASHSCOPE_API_KEY", "OPENAI_API_KEY"),
    vl_model=env_str("VL_MODEL", default=DEFAULT_VL_MODEL),
)
