"""查询相关接口（挂在统一服务上）。"""

import logging
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Request
from pydantic import BaseModel
from starlette.responses import StreamingResponse

from processor.query_processor.main_graph import KBQueryWorkflow
from utils.mongo_history_utils import clear_history, get_history_messages
from utils.sse_utils import SSEEvent, create_sse_queue, push_to_session, sse_generator
from utils.task_utils import (
    TASK_STATUS_PROCESSING,
    clear_task,
    get_done_task_list,
    get_task_result,
    update_task_status,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["query"])


class QueryRequest(BaseModel):
    query: str
    session_id: str
    is_stream: bool = True


@router.post("/query", summary="知识库问答", description="流式返回会通过 /stream/{session_id} 推送 SSE 事件")
async def query(background: BackgroundTasks, query: QueryRequest):
    user_query = query.query
    session_id = query.session_id
    is_stream = query.is_stream

    if not user_query.strip():
        return {"message": "问题不能为空", "session_id": session_id, "answer": ""}

    if is_stream:
        create_sse_queue(session_id)

    # 同一个 session 会被连续追问：先清掉上一轮遗留的节点进度和答案，
    # 否则新一轮的第一个 progress 事件仍带着旧问题的「已完成」节点，
    # 前端会把上一轮的步骤错记到这一轮上。
    clear_task(session_id)

    update_task_status(session_id, TASK_STATUS_PROCESSING, is_stream)

    if is_stream:
        background.add_task(run_query_graph, session_id, user_query, is_stream)
        return {"message": "任务已经开始，请耐心等待", "session_id": session_id}

    run_query_graph(session_id, user_query, is_stream)
    return {
        "message": "处理完成",
        "session_id": session_id,
        "answer": get_task_result(session_id, "answer", ""),
        "error": get_task_result(session_id, "error", ""),
        "image_urls": get_task_result(session_id, "image_urls", []),
        "done_list": get_done_task_list(session_id),
    }


def run_query_graph(session_id: str, user_query: str, is_stream: bool):
    logger.info("调用查询工作流：session=%s, stream=%s", session_id, is_stream)
    init_state = {
        "original_query": user_query,
        "session_id": session_id,
        "is_stream": is_stream,
    }
    workflow = KBQueryWorkflow()
    try:
        for _ in workflow.run(init_state, stream=is_stream):
            # stream 是惰性对象，必须迭代才会真正执行
            pass
    except Exception as exc:  # noqa: BLE001
        logger.exception("查询流程失败：%s", exc)
        from utils.task_utils import TASK_STATUS_FAILED, set_task_result

        set_task_result(session_id, "error", str(exc))
        update_task_status(session_id, TASK_STATUS_FAILED, is_stream)
    finally:
        if is_stream:
            # 流程结束后主动关闭 SSE：否则生成器一直阻塞在队列上，
            # 连接要等浏览器 es.close() 才断开，服务端队列也会一直挂着。
            push_to_session(session_id, SSEEvent.CLOSE, {})


@router.get("/stream/{session_id}", summary="SSE 流式输出")
async def stream(session_id: str, request: Request):
    return StreamingResponse(
        sse_generator(session_id, request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/history/{session_id}", summary="获取会话历史")
def history(session_id: str, limit: int = 50):
    # Mongo 是同步客户端，用普通 def 交给线程池，别堵事件循环
    try:
        items = get_history_messages(session_id, limit=limit)
    except Exception as exc:  # noqa: BLE001
        # 历史服务不可用时页面仍要能打开：返回空列表 + 原因，前端不弹错
        logger.warning("读取会话历史失败：%s", exc)
        return {"session_id": session_id, "items": [], "error": str(exc)}
    return {"session_id": session_id, "items": items}


@router.delete("/history/{session_id}", summary="清空会话历史")
def delete_history(session_id: str) -> dict:
    try:
        deleted = clear_history(session_id)
    except Exception as exc:  # noqa: BLE001
        logger.warning("清空会话历史失败：%s", exc)
        return {"session_id": session_id, "deleted": 0, "error": str(exc)}
    return {"session_id": session_id, "deleted": deleted}


if __name__ == "__main__":
    raise SystemExit("请通过 `python -m web.api.app` 或 `uvicorn web.api.app:app` 启动统一服务")
