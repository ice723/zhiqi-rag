"""环境变量加载与类型转换工具。

统一从项目根目录的 .env 加载，避免因为启动目录不同而读不到配置。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_env() -> None:
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=False)
    else:
        # 容器部署时可能通过 env_file 注入，这里再兜底一份
        load_dotenv(override=False)


_load_env()


def env_str(*names: str, default: str = "") -> str:
    """按顺序取第一个非空环境变量。"""
    for name in names:
        value = os.getenv(name)
        if value is not None and str(value).strip() != "":
            return str(value).strip()
    return default


def env_int(*names: str, default: int = 0) -> int:
    raw = env_str(*names, default=str(default))
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def env_float(*names: str, default: float = 0.0) -> float:
    raw = env_str(*names, default=str(default))
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


_TRUE_VALUES = {"1", "true", "yes", "on", "y"}


def env_bool(*names: str, default: bool = False) -> bool:
    raw = env_str(*names)
    if raw == "":
        return default
    return raw.strip().lower() in _TRUE_VALUES


def env_json(*names: str, default: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """解析 JSON 对象型环境变量，解析失败时回退默认值。"""
    raw = env_str(*names)
    if not raw:
        return default
    try:
        value = json.loads(raw)
    except (TypeError, ValueError):
        return default
    return value if isinstance(value, dict) else default


def first_env(names: Iterable[str]) -> str:
    return env_str(*names)
