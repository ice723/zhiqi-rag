"""统一 FastAPI 服务：文档导入 + 知识库问答 + 单页前端。

启动：
    python -m web.api.app
或：
    uvicorn web.api.app:app --host 0.0.0.0 --port 8000 --workers 1

注意：SSE 队列和任务进度都保存在进程内存里，必须使用单进程（--workers 1）。
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, RedirectResponse
from starlette.middleware.cors import CORSMiddleware

from config.env_loader import env_str
from web.api.document_service import router as document_router
from web.api.import_service import router as import_router
from web.api.query_service import router as query_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s",
)

PAGE_DIR = Path(__file__).resolve().parent.parent / "page"

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    # 确保本地目录存在（Linux 上首次启动时常见问题）
    for key, default in (("DATA_BASED_ROOT_DIR", "./output"), ("MD_ROOT_DIR", "./temp-files")):
        directory = Path(env_str(key, default=default)).expanduser()
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.warning("创建目录失败 %s: %s", directory, exc)
    logger.info("知识库服务启动完成")
    yield
    logger.info("知识库服务已停止")


app = FastAPI(
    title="知器 · 知识库问答",
    description="文档导入 + 知识库问答统一服务",
    version="0.2.0",
    lifespan=lifespan,
)

# 前端与接口同源部署时其实不需要跨域；保留开关方便本地分端口调试
_origins = [item.strip() for item in env_str("CORS_ALLOW_ORIGINS", default="*").split(",") if item.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins or ["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(import_router)
app.include_router(query_router)
app.include_router(document_router)


# 单页应用每次都要回源校验：没有 Cache-Control 时浏览器会按启发式缓存 HTML，
# 发版后手机端可能长期停留在旧页面。带 ETag 的 no-cache 仍然走 304，不浪费带宽。
PAGE_HEADERS = {"Cache-Control": "no-cache, must-revalidate"}


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(PAGE_DIR / "index.html", headers=PAGE_HEADERS)


@app.get("/index.html", include_in_schema=False)
async def index_alias() -> FileResponse:
    return FileResponse(PAGE_DIR / "index.html", headers=PAGE_HEADERS)


@app.get("/import/html", include_in_schema=False)
@app.get("/import.html", include_in_schema=False)
async def legacy_import_page() -> RedirectResponse:
    # 旧入口保留，统一跳到合并后的单页
    return RedirectResponse(url="/", status_code=307)


@app.get("/chat/html", include_in_schema=False)
@app.get("/chat.html", include_in_schema=False)
async def legacy_chat_page() -> RedirectResponse:
    return RedirectResponse(url="/", status_code=307)


@app.get("/health", summary="健康检查")
async def health() -> dict:
    return {
        "status": "ok",
        "service": "kb-rag",
        "inference_backend": env_str("INFERENCE_BACKEND", default="siliconflow"),
        "chunks_collection": env_str("CHUNKS_COLLECTION", default="kb_chunks_v2"),
    }


@app.get("/health/deep", summary="依赖健康检查")
def health_deep() -> dict:
    """逐项检查 Milvus / MongoDB / MinIO，任何一项失败不影响接口本身返回。

    这里调用的是同步客户端，必须写成普通 def，由 FastAPI 丢到线程池执行。
    写成 async def 会在事件循环里发起阻塞连接，中间件不可达时会整站卡死。
    """
    from config.milvus_config import milvus_config
    from utils.milvus_utils import get_milvus_client
    from utils.minio_utils import get_minio_probe_client
    from utils.mongo_history_utils import get_history_mongo_tool

    result = {}

    try:
        client = get_milvus_client()
        result["milvus"] = "ok" if client.has_collection(milvus_config.chunks_collection) else "collection_missing"
    except Exception as exc:  # noqa: BLE001
        result["milvus"] = f"error: {exc}"

    try:
        get_history_mongo_tool().client.admin.command("ping")
        result["mongodb"] = "ok"
    except Exception as exc:  # noqa: BLE001
        result["mongodb"] = f"error: {exc}"

    try:
        from config.minio_config import minio_config

        client = get_minio_probe_client()
        result["minio"] = "ok" if client.bucket_exists(minio_config.bucket_name) else "bucket_missing"
    except Exception as exc:  # noqa: BLE001
        result["minio"] = f"error: {exc}"

    return result


if __name__ == "__main__":
    # 单进程运行：SSE 队列和任务进度都在内存中，不能用多 worker
    uvicorn.run(
        app,
        host=env_str("APP_HOST", default="0.0.0.0"),
        port=int(env_str("APP_PORT", default="8000")),
    )
