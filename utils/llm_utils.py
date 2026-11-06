"""大模型客户端工具。

- get_llm_client()      业务对话/抽取，默认硅基流动 Qwen/Qwen3-8B
- get_vl_client()       图片理解，DashScope qwen3-vl-plus
- invoke_json()         JSON 输出的稳健调用（自动去 ``` 围栏、兜底重试）

所有 Key 都从环境变量读取，代码中不出现明文。
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from langchain_openai import ChatOpenAI

from config.lm_config import lm_config

logger = logging.getLogger(__name__)

_llm_client_cache: Dict[tuple, ChatOpenAI] = {}


def _extra_body() -> Optional[Dict[str, Any]]:
    return lm_config.extra_body or None


def get_llm_client(model: str | None = None, json_model: bool = False) -> ChatOpenAI:
    model_name = model or lm_config.llm_model
    key = (model_name, json_model)

    if key in _llm_client_cache:
        return _llm_client_cache[key]

    model_kwargs: Dict[str, Any] = {}
    if json_model:
        model_kwargs["response_format"] = {"type": "json_object"}

    client = ChatOpenAI(
        model=model_name,
        temperature=lm_config.llm_temperature,
        base_url=lm_config.base_url,
        api_key=lm_config.api_key,
        model_kwargs=model_kwargs,
        extra_body=_extra_body(),
        timeout=120,
        max_retries=3,
    )
    _llm_client_cache[key] = client
    return client


def get_item_client(json_model: bool = True) -> ChatOpenAI:
    """商品名识别专用客户端（可用 ITEM_MODEL 单独指定小模型）。"""
    return get_llm_client(lm_config.item_model, json_model=json_model)


def get_vl_client() -> ChatOpenAI:
    """视觉模型客户端，走 DashScope 的 OpenAI 兼容接口。"""
    key = ("__vl__", lm_config.vl_model)
    if key in _llm_client_cache:
        return _llm_client_cache[key]

    client = ChatOpenAI(
        model=lm_config.vl_model,
        temperature=0.1,
        base_url=lm_config.vl_base_url,
        api_key=lm_config.vl_api_key,
        extra_body=_extra_body(),
        timeout=120,
        max_retries=3,
    )
    _llm_client_cache[key] = client
    return client


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """从模型输出里尽量抠出一个 JSON 对象。"""
    if not text:
        return None

    cleaned = text.strip()
    if cleaned.startswith("```"):
        # 去掉 ```json ... ``` 围栏
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()

    try:
        value = json.loads(cleaned)
        return value if isinstance(value, dict) else None
    except ValueError:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            value = json.loads(cleaned[start:end + 1])
            return value if isinstance(value, dict) else None
        except ValueError:
            return None
    return None


def invoke_json(
    messages: Any,
    model: Optional[str] = None,
    fallback: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """调用模型并返回 JSON 对象。

    先尝试 JSON 模式；部分网关/模型不支持 response_format，会退化成普通调用，
    再对返回文本做容错解析。
    """
    try:
        client = get_llm_client(model, json_model=True)
        result = client.invoke(messages)
        parsed = _extract_json(getattr(result, "content", "") or "")
        if parsed is not None:
            return parsed
    except Exception as exc:  # noqa: BLE001 - 需要兜底到普通模式
        logger.warning("JSON 模式调用失败，改用普通模式重试：%s", exc)

    try:
        client = get_llm_client(model, json_model=False)
        result = client.invoke(messages)
        parsed = _extract_json(getattr(result, "content", "") or "")
        if parsed is not None:
            return parsed
    except Exception as exc:  # noqa: BLE001
        logger.error("模型 JSON 调用失败：%s", exc)

    return fallback if fallback is not None else {}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    client = get_llm_client()
    print(client.invoke("你好，你是什么模型？"))
