"""资料库清单接口。

前端左侧「资料库」原来只靠 localStorage 记着本机传过什么，换浏览器、
换设备或服务重启后就和真实知识库对不上了。这里以 MinIO 里的原始文件
为事实来源，再合并内存里的导入状态，给前端一份可信的清单。

数据来源：
1. MinIO `pdf_files/YYYYMMDD/<文件名>` —— 上传即落库，持久可信；
2. utils/doc_registry —— 本次进程里「解析中 / 失败」的文件，补足状态。

注意：MinIO 客户端是同步的，接口必须写成普通 def，交给 FastAPI 线程池执行。
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import PurePosixPath
from typing import Any, Dict, List

from fastapi import APIRouter

from config.minio_config import minio_config
from utils import doc_registry
from utils.minio_utils import get_minio_probe_client

logger = logging.getLogger(__name__)

router = APIRouter(tags=["documents"])

PREFIX = "pdf_files/"
DEFAULT_LIMIT = 60


def _local_time(value: Any) -> datetime:
    """把 MinIO 的 last_modified 统一成本地时区的 datetime。"""
    if not isinstance(value, datetime):
        return datetime.now()
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone()


def _minio_items() -> List[Dict[str, Any]]:
    """遍历 MinIO 里的原始资料文件。"""
    client = get_minio_probe_client()
    items: List[Dict[str, Any]] = []
    for obj in client.list_objects(minio_config.bucket_name, prefix=PREFIX, recursive=True):
        if getattr(obj, "is_dir", False):
            continue
        local = _local_time(obj.last_modified)
        items.append(
            {
                "id": obj.object_name,
                "name": PurePosixPath(obj.object_name).name,
                "size": int(obj.size or 0),
                "time": local.strftime("%H:%M"),
                "date": local.strftime("%Y-%m-%d"),
                "status": "ready",
                "source": "minio",
                "_ts": local.timestamp(),
            }
        )
    return items


def _merge_pending(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """把本次进程里还在解析 / 已失败的文件合并进来。"""
    by_name = {item["name"]: item for item in items}

    for record in doc_registry.snapshot():
        name = record.get("name") or ""
        if not name:
            continue

        status = record.get("status") or "parsing"
        existing = by_name.get(name)

        if status == "ready":
            # 已经落进 MinIO 了，以 MinIO 的记录为准
            continue

        if existing is not None:
            # 同名文件又传了一次：沿用 MinIO 的元信息，状态以本次导入为准
            existing["status"] = status
            existing["id"] = record.get("id") or existing["id"]
            continue

        local = datetime.fromtimestamp(record.get("time") or 0)
        item = {
            "id": record.get("id") or name,
            "name": name,
            "size": int(record.get("size") or 0),
            "time": local.strftime("%H:%M"),
            "date": local.strftime("%Y-%m-%d"),
            "status": status,
            "source": "upload",
            "_ts": local.timestamp(),
        }
        items.append(item)
        by_name[name] = item

    return items


@router.get("/documents", summary="资料库清单")
def list_documents(limit: int = DEFAULT_LIMIT) -> dict:
    """返回知识库里的资料文件（新的在前），供前端侧栏展示。"""
    items: List[Dict[str, Any]] = []
    error = ""

    try:
        items = _minio_items()
    except Exception as exc:  # noqa: BLE001
        # MinIO 不可达时仍然返回本次上传记录，前端至少能看到自己刚传的文件
        logger.warning("读取 MinIO 资料清单失败：%s", exc)
        error = str(exc)

    items = _merge_pending(items)
    items.sort(key=lambda item: item.get("_ts") or 0, reverse=True)

    for item in items:
        item.pop("_ts", None)

    return {
        "items": items[: max(1, min(int(limit or DEFAULT_LIMIT), 200))],
        "error": error,
    }
