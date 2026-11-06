"""上传中的文件登记表（单进程内存）。

作用：`GET /documents` 的数据主要来自 MinIO 对象清单（持久事实），
但「刚上传、还在解析」「解析失败」这两种状态 MinIO 上看不出来，
这里补一份轻量登记，和 utils/task_utils.py 一样只活在当前进程里。

重启后登记表清空，MinIO 里的文件仍然会被列出来，只是状态统一按 ready 处理。
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, List

_lock = threading.Lock()

# task_id -> {id, name, size, time(epoch 秒), status}
_docs: Dict[str, Dict[str, Any]] = {}


def register(task_id: str, name: str, size: int = 0) -> None:
    """文件落盘后登记，状态默认 parsing。"""
    with _lock:
        _docs[task_id] = {
            "id": task_id,
            "name": name,
            "size": int(size or 0),
            "time": time.time(),
            "status": "parsing",
        }


def mark(task_id: str, status: str) -> None:
    """更新登记状态：parsing / ready / failed。"""
    with _lock:
        record = _docs.get(task_id)
        if record is not None:
            record["status"] = status


def snapshot() -> List[Dict[str, Any]]:
    """返回登记表的副本，调用方可以随意改。"""
    with _lock:
        return [dict(item) for item in _docs.values()]
