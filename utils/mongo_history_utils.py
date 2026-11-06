"""MongoDB 历史对话读写工具。

与原版相比：
1. 改为懒加载单例，服务未就绪时不会在 import 阶段直接崩掉；
2. 修正 get_recent_messages 的取数逻辑（应先取最近 N 条再按时间正序返回）；
3. 统一从 config.env_loader 读取环境变量。
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional

from bson import ObjectId
from pymongo import ASCENDING, DESCENDING, MongoClient

from config.env_loader import env_str

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_history_mongo_tool: Optional["HistoryMongoTool"] = None


class HistoryMongoTool:
    """MongoDB 历史对话记录读写工具类。"""

    def __init__(self) -> None:
        self.mongo_url = env_str("MONGO_URL", default="mongodb://127.0.0.1:27017")
        self.db_name = env_str("MONGO_DB_NAME", default="kb001")

        self.client = MongoClient(self.mongo_url, serverSelectionTimeoutMS=5000)
        # 提前触发一次握手，配置错误能尽早暴露
        self.client.admin.command("ping")
        self.db = self.client[self.db_name]
        self.chat_message = self.db["chat_message"]
        self.chat_message.create_index([("session_id", 1), ("ts", -1)])

        logger.info("MongoDB 连接成功：%s/%s", self.mongo_url, self.db_name)


def get_history_mongo_tool() -> HistoryMongoTool:
    """获取懒加载单例。"""
    global _history_mongo_tool
    if _history_mongo_tool is not None:
        return _history_mongo_tool

    with _lock:
        if _history_mongo_tool is None:
            _history_mongo_tool = HistoryMongoTool()
        return _history_mongo_tool


def clear_history(session_id: str) -> int:
    """清空指定会话的所有历史记录，返回删除条数。"""
    mongo_tool = get_history_mongo_tool()
    try:
        result = mongo_tool.chat_message.delete_many({"session_id": session_id})
        logger.info("已删除会话 %s 的 %d 条历史", session_id, result.deleted_count)
        return result.deleted_count
    except Exception as exc:  # noqa: BLE001
        logger.error("清空会话 %s 历史失败：%s", session_id, exc)
        return 0


def save_chat_message(
    session_id: str,
    role: str,
    text: str,
    rewritten_query: str = "",
    item_names: List[str] = None,
    image_urls: List[str] = None,
    message_id: str = None,
) -> str:
    """写入/更新单条会话记录。有 message_id 时为更新，否则新增。"""
    ts = datetime.now().timestamp()

    document = {
        "session_id": session_id,
        "role": role,
        "text": text,
        "rewritten_query": rewritten_query or "",
        "item_names": item_names,
        "image_urls": image_urls,
        "ts": ts,
    }

    mongo_tool = get_history_mongo_tool()
    if message_id:
        mongo_tool.chat_message.update_one(
            {"_id": ObjectId(message_id)},
            {"$set": document},
        )
        return message_id

    result = mongo_tool.chat_message.insert_one(document)
    return str(result.inserted_id)


def update_message_item_names(ids: List[str], item_names: List[str]) -> int:
    """批量更新历史记录的关联商品名称。"""
    if not ids:
        return 0

    mongo_tool = get_history_mongo_tool()
    try:
        object_ids = [ObjectId(i) for i in ids]
        result = mongo_tool.chat_message.update_many(
            {"_id": {"$in": object_ids}},
            {"$set": {"item_names": item_names}},
        )
        logger.info("已更新 %d 条历史的 item_names：%s", result.modified_count, item_names)
        return result.modified_count
    except Exception as exc:  # noqa: BLE001
        logger.error("更新历史 item_names 失败：%s", exc)
        return 0


def get_recent_messages(session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
    """查询指定会话最近的 N 条记录，按时间正序返回。"""
    mongo_tool = get_history_mongo_tool()
    try:
        cursor = (
            mongo_tool.chat_message.find({"session_id": session_id})
            .sort("ts", DESCENDING)
            .limit(limit)
        )
        messages = list(cursor)
        messages.reverse()
        return messages
    except Exception as exc:  # noqa: BLE001
        logger.error("查询历史消息失败：%s", exc)
        return []


def get_history_messages(session_id: str, limit: int = 50) -> List[Dict[str, Any]]:
    """给前端历史面板使用的同义接口。"""
    messages = get_recent_messages(session_id, limit=limit)
    result: List[Dict[str, Any]] = []
    for msg in messages:
        result.append(
            {
                "id": str(msg.get("_id", "")),
                "role": msg.get("role", ""),
                "text": msg.get("text", ""),
                "item_names": msg.get("item_names") or [],
                "image_urls": msg.get("image_urls") or [],
                "ts": msg.get("ts", 0),
            }
        )
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sid = "test_session_cpu"
    save_chat_message(sid, "user", "你好，有烫金机吗？")
    save_chat_message(sid, "assistant", "你好！请问你想询问哪个型号？")
    print(get_history_messages(sid))
