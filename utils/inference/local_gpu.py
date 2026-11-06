"""本地 / GPU 推理客户端。

用于以后切回 GPU 部署：直接对接《GPU模型部署.md》里发布的两个自定义
HTTP 服务，不需要改上层 RAG 业务代码。

环境变量：
- BGE_M3_URL       例如 http://127.0.0.1:6006/v1/embeddings
- BGE_RERANKER_URL 例如 http://127.0.0.1:6008/v1/rerank
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import requests

from utils.inference.base import (
    EmbeddingClient,
    EmbeddingResult,
    InferenceError,
    RerankClient,
)

logger = logging.getLogger(__name__)


def _post(url: str, payload: Dict[str, Any], timeout: int) -> Dict[str, Any]:
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
    except requests.RequestException as exc:
        raise InferenceError(f"本地推理服务网络异常：{url}，{exc}") from exc

    if resp.status_code != 200:
        raise InferenceError(
            f"本地推理服务调用失败：{url}，HTTP {resp.status_code}，{resp.text[:300]}"
        )
    try:
        return resp.json()
    except ValueError as exc:
        raise InferenceError(f"本地推理服务返回非 JSON：{resp.text[:300]}") from exc


class BgeM3HttpEmbeddingClient(EmbeddingClient):
    """对接 GPU 服务器上的 BGE-M3 自定义服务，返回 dense + sparse。"""

    def __init__(self, url: str, timeout: int = 120) -> None:
        self.url = url
        self.timeout = timeout

    def embed(self, texts: List[str]) -> EmbeddingResult:
        if not texts:
            return EmbeddingResult(dense=[], sparse=None)

        data = _post(self.url, {"embedding_documents": texts}, self.timeout)
        dense = data.get("dense")
        sparse = data.get("sparse")
        if not isinstance(dense, list) or len(dense) != len(texts):
            raise InferenceError("BGE-M3 服务返回的 dense 数量与输入不一致")

        processed_sparse: Optional[List[Dict[int, float]]] = None
        if sparse is not None:
            if not isinstance(sparse, list) or len(sparse) != len(texts):
                raise InferenceError("BGE-M3 服务返回的 sparse 数量与输入不一致")
            processed_sparse = []
            for item in sparse:
                # JSON 的 key 一定是字符串，这里统一转回 int，Milvus 稀疏向量要求 int key
                processed_sparse.append({int(k): float(v) for k, v in item.items()})

        return EmbeddingResult(dense=dense, sparse=processed_sparse)


class BgeRerankHttpClient(RerankClient):
    """对接 GPU 服务器上的 reranker 自定义服务。

    兼容两种请求体：
    - {"query": ..., "documents": [...]}
    - {"sentence_pairs": [[query, doc], ...]}
    """

    def __init__(self, url: str, timeout: int = 120) -> None:
        self.url = url
        self.timeout = timeout

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_n: Optional[int] = None,
    ) -> List[float]:
        if not documents:
            return []

        payload = {"query": query, "documents": documents}
        try:
            data = _post(self.url, payload, self.timeout)
        except InferenceError:
            # 部分服务只实现了 sentence_pairs 入参，做一次兼容重试
            pairs_payload = {"sentence_pairs": [[query, doc] for doc in documents]}
            data = _post(self.url, pairs_payload, self.timeout)

        scores = data.get("scores")
        if not isinstance(scores, list) or len(scores) != len(documents):
            raise InferenceError("reranker 服务返回的 scores 数量与 documents 不一致")
        return [float(x) for x in scores]
