"""上传中的文件登记表（单进程内存）。

作用：`GET /documents` 的数据主要来自 MinIO 对象清单（持久事实），
但「刚上传、还在解析」「解析失败」这两种状态 MinIO 上看不出来，
这里补一份轻量登记，和 utils/task_utils.py 一样只活在当前进程里。

登记只是短期状态：解析完成的记录由 mark(..., "ready") 直接删掉，超时未完成的
记录在 snapshot() 时回收，避免一条失败记录永久粘在清单上。重启后登记表清空，
MinIO 里的文件仍然会被列出来，只是状态统一按 ready 处理。
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, List

_lock = threading.Lock()

# 解析中超过这个时长仍未结束，视为残留（任务卡死 / 线程丢失），不再展示该状态。
# 和前端的轮询上限（30 分钟）保持一致。
PARSING_TTL = 30 * 60
# 失败记录保留一天：用户还能看到失败，但不会永久占着「失败」。
FAILED_TTL = 24 * 3600
# 兜底上限，避免极端情况下登记表无限增长。
MAX_RECORDS = 200

# task_id -> {id, name, size, time(epoch 秒), status}
_docs: Dict[str, Dict[str, Any]] = {}


def _prune_locked(now: float) -> None:
    """按状态回收过期记录，并把总量压在上限内；调用方需持有 _lock。"""
    for task_id in list(_docs):
        record = _docs[task_id]
        ttl = FAILED_TTL if record.get("status") == "failed" else PARSING_TTL
        if now - float(record.get("time") or 0) > ttl:
            _docs.pop(task_id, None)

    if len(_docs) > MAX_RECORDS:
        ordered = sorted(_docs, key=lambda task_id: _docs[task_id].get("time") or 0)
        for task_id in ordered[: len(_docs) - MAX_RECORDS]:
            _docs.pop(task_id, None)


def register(task_id: str, name: str, size: int = 0) -> None:
    """文件落盘后登记，状态默认 parsing。"""
    now = time.time()
    with _lock:
        _prune_locked(now)
        # 清单按文件名合并，同一个文件只保留最新一次导入的状态，
        # 否则旧的「失败」记录会覆盖这次的成功结果
        for old_id in [key for key, record in _docs.items() if record.get("name") == name]:
            _docs.pop(old_id, None)
        _docs[task_id] = {
            "id": task_id,
            "name": name,
            "size": int(size or 0),
            "time": now,
            "status": "parsing",
        }


def mark(task_id: str, status: str) -> None:
    """更新登记状态：parsing / ready / failed。

    ready 表示文件已经在 MinIO 里，清单以 MinIO 为准，登记表不再需要这条记录。
    """
    with _lock:
        if status == "ready":
            record = _docs.get(task_id)
            name = record.get("name") if record else None
            # 连同该文件名下的旧记录一起清掉，避免历史失败状态继续覆盖
            for old_id in [key for key, item in _docs.items() if key == task_id or (name and item.get("name") == name)]:
                _docs.pop(old_id, None)
            return
        record = _docs.get(task_id)
        if record is not None:
            record["status"] = status


def snapshot() -> List[Dict[str, Any]]:
    """返回登记表的副本，调用方可以随意改。"""
    now = time.time()
    with _lock:
        _prune_locked(now)
        return [dict(item) for item in _docs.values()]
