"""推理客户端抽象层。

上层 RAG 业务只依赖本包暴露的接口，不关心底层是：
- 硅基流动云端 API（CPU 部署默认）
- GPU 服务器上的 BGE-M3 / reranker 自定义 HTTP 服务
- 后续的本地 vLLM / 其他 OpenAI 兼容服务
"""

from utils.inference.base import (
    EmbeddingClient,
    EmbeddingResult,
    InferenceError,
    RerankClient,
)
from utils.inference.factory import get_embedding_client, get_rerank_client, reset_clients

__all__ = [
    "EmbeddingClient",
    "EmbeddingResult",
    "InferenceError",
    "RerankClient",
    "get_embedding_client",
    "get_rerank_client",
    "reset_clients",
]
