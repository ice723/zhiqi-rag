"""推理客户端工厂。

通过 INFERENCE_BACKEND 切换后端：
- siliconflow（默认）：CPU 云端部署，嵌入/重排都走硅基流动 API
- local_gpu：切回 GPU 服务器上的 BGE-M3 / reranker 自定义 HTTP 服务

上层业务只调用 get_embedding_client() / get_rerank_client()。
"""

from __future__ import annotations

import os
import threading
from typing import Optional

from utils.inference.base import EmbeddingClient, InferenceError, RerankClient

_lock = threading.Lock()
_embedding_client: Optional[EmbeddingClient] = None
_rerank_client: Optional[RerankClient] = None


def _backend() -> str:
    return (os.getenv("INFERENCE_BACKEND") or "siliconflow").strip().lower()


def get_embedding_client() -> EmbeddingClient:
    global _embedding_client
    if _embedding_client is not None:
        return _embedding_client

    with _lock:
        if _embedding_client is not None:
            return _embedding_client

        backend = _backend()
        if backend in {"local", "local_gpu", "gpu"}:
            from utils.inference.local_gpu import BgeM3HttpEmbeddingClient

            url = os.getenv("BGE_M3_URL", "").strip()
            if not url:
                raise InferenceError(
                    "INFERENCE_BACKEND=local_gpu 时必须配置 BGE_M3_URL，"
                    "例如 http://127.0.0.1:6006/v1/embeddings"
                )
            _embedding_client = BgeM3HttpEmbeddingClient(
                url=url,
                timeout=int(os.getenv("BGE_M3_TIMEOUT", "120")),
            )
        else:
            from utils.inference.siliconflow import SiliconFlowEmbeddingClient

            _embedding_client = SiliconFlowEmbeddingClient()

        return _embedding_client


def get_rerank_client() -> RerankClient:
    global _rerank_client
    if _rerank_client is not None:
        return _rerank_client

    with _lock:
        if _rerank_client is not None:
            return _rerank_client

        backend = _backend()
        if backend in {"local", "local_gpu", "gpu"}:
            from utils.inference.local_gpu import BgeRerankHttpClient

            url = os.getenv("BGE_RERANKER_URL", "").strip()
            if not url:
                raise InferenceError(
                    "INFERENCE_BACKEND=local_gpu 时必须配置 BGE_RERANKER_URL，"
                    "例如 http://127.0.0.1:6008/v1/rerank"
                )
            _rerank_client = BgeRerankHttpClient(
                url=url,
                timeout=int(os.getenv("BGE_RERANKER_TIMEOUT", "120")),
            )
        else:
            from utils.inference.siliconflow import SiliconFlowRerankClient

            _rerank_client = SiliconFlowRerankClient()

        return _rerank_client


def reset_clients() -> None:
    """测试或运行期切换配置时清空单例。"""
    global _embedding_client, _rerank_client
    with _lock:
        _embedding_client = None
        _rerank_client = None
