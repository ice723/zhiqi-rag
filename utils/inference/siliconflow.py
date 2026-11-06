"""硅基流动（SiliconFlow）推理客户端。

对外接口：
- Embedding: POST {base_url}/embeddings
- Rerank:    POST {base_url}/rerank

注意：
硅基流动的 BAAI/bge-m3 embeddings 只返回稠密向量，不返回 sparse。
因此本客户端的 EmbeddingResult.sparse 恒为 None，混合检索的稀疏一路
由 Milvus 2.5 的 BM25 Function 在服务端生成。
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import requests

from config.embedding_config import embedding_config
from config.reranker_config import reranker_config
from utils.inference.base import (
    EmbeddingClient,
    EmbeddingResult,
    InferenceError,
    RerankClient,
)

logger = logging.getLogger(__name__)


def _post_json(
    url: str,
    payload: Dict[str, Any],
    api_key: str,
    timeout: int,
    max_retries: int,
) -> Dict[str, Any]:
    """发送 JSON POST 请求，带指数退避重试。

    只对网络异常、429、5xx 重试；4xx（除 429）直接抛错，避免无意义重试。
    """
    if not api_key:
        raise InferenceError(f"未配置 API Key，无法调用 {url}")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    last_error: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
        except requests.RequestException as exc:
            last_error = exc
            if attempt >= max_retries:
                break
            wait = min(2 ** attempt, 10)
            logger.warning("推理请求网络异常，%ss 后重试：%s", wait, exc)
            time.sleep(wait)
            continue

        if resp.status_code == 200:
            try:
                return resp.json()
            except ValueError as exc:
                raise InferenceError(f"推理服务返回非 JSON 内容：{resp.text[:300]}") from exc

        # 429 / 5xx 可重试
        if resp.status_code == 429 or resp.status_code >= 500:
            last_error = InferenceError(
                f"推理服务暂时不可用：HTTP {resp.status_code}，{resp.text[:300]}"
            )
            if attempt >= max_retries:
                break
            retry_after = resp.headers.get("Retry-After")
            wait = int(retry_after) if retry_after and retry_after.isdigit() else min(2 ** attempt, 10)
            logger.warning("推理服务返回 %s，%ss 后重试", resp.status_code, wait)
            time.sleep(wait)
            continue

        # 其余 4xx 直接失败，带上服务端信息，方便排查模型名/Key/额度问题
        raise InferenceError(
            f"推理服务调用失败：HTTP {resp.status_code}，{resp.text[:500]}"
        )

    raise InferenceError(f"推理服务调用失败（已重试 {max_retries} 次）：{last_error}")


class SiliconFlowEmbeddingClient(EmbeddingClient):
    """硅基流动 /v1/embeddings 客户端。"""

    def __init__(self) -> None:
        self.api_base = embedding_config.api_base.rstrip("/")
        self.api_key = embedding_config.api_key
        self.model = embedding_config.model
        self.dim = embedding_config.dim
        self.batch_size = max(1, embedding_config.batch_size)
        self.timeout = embedding_config.timeout
        self.max_retries = embedding_config.max_retries
        self.min_interval = embedding_config.min_interval

    @property
    def embeddings_url(self) -> str:
        return f"{self.api_base}/embeddings"

    def embed(self, texts: List[str]) -> EmbeddingResult:
        if not texts:
            return EmbeddingResult(dense=[], sparse=None)

        all_dense: List[List[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]
            all_dense.extend(self._embed_batch(batch))
            if self.min_interval > 0 and start + self.batch_size < len(texts):
                time.sleep(self.min_interval)

        return EmbeddingResult(dense=all_dense, sparse=None)

    def _embed_batch(self, batch: List[str]) -> List[List[float]]:
        payload = {
            "model": self.model,
            "input": batch,
            "encoding_format": "float",
        }
        logger.info("硅基流动 embeddings：model=%s, batch=%d", self.model, len(batch))
        data = _post_json(
            self.embeddings_url,
            payload,
            self.api_key,
            self.timeout,
            self.max_retries,
        )

        rows = data.get("data")
        if not isinstance(rows, list) or len(rows) != len(batch):
            raise InferenceError(
                f"embeddings 返回数量异常：期望 {len(batch)} 条，实际 {len(rows) if isinstance(rows, list) else '非列表'}"
            )

        # 按 index 回填，避免服务端乱序
        ordered: List[Optional[List[float]]] = [None] * len(batch)
        for row in rows:
            index = row.get("index")
            embedding = row.get("embedding")
            if not isinstance(index, int) or not isinstance(embedding, list):
                raise InferenceError(f"embeddings 返回格式异常：{str(row)[:300]}")
            if index < 0 or index >= len(batch):
                raise InferenceError(f"embeddings 返回 index 越界：{index}")
            ordered[index] = [float(x) for x in embedding]

        if any(item is None for item in ordered):
            raise InferenceError("embeddings 返回缺少部分 index 的结果")

        result = [item for item in ordered if item is not None]
        if self.dim and result and len(result[0]) != self.dim:
            raise InferenceError(
                f"向量维度不匹配：配置 EMBEDDING_DIM={self.dim}，"
                f"模型实际返回 {len(result[0])}。请确认 .env 与 Milvus 集合维度一致。"
            )
        return result


class SiliconFlowRerankClient(RerankClient):
    """硅基流动 /v1/rerank 客户端。"""

    def __init__(self) -> None:
        self.api_base = reranker_config.api_base.rstrip("/")
        self.api_key = reranker_config.api_key
        self.model = reranker_config.model
        self.timeout = reranker_config.timeout
        self.max_retries = reranker_config.max_retries
        self.default_top_n = reranker_config.top_n

    @property
    def rerank_url(self) -> str:
        return f"{self.api_base}/rerank"

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_n: Optional[int] = None,
    ) -> List[float]:
        if not documents:
            return []

        n = top_n or self.default_top_n or len(documents)
        n = min(n, len(documents))

        payload = {
            "model": self.model,
            "query": query,
            "documents": documents,
            "top_n": n,
            "return_documents": False,
        }
        logger.info("硅基流动 rerank：model=%s, documents=%d, top_n=%d", self.model, len(documents), n)
        data = _post_json(
            self.rerank_url,
            payload,
            self.api_key,
            self.timeout,
            self.max_retries,
        )

        results = data.get("results")
        if not isinstance(results, list) or not results:
            raise InferenceError(f"rerank 返回结果为空或格式异常：{str(data)[:300]}")

        # 服务端可能只返回 top_n 条，未返回的按 0 分处理，保证顺序与 documents 一致
        scores = [0.0] * len(documents)
        for item in results:
            index = item.get("index")
            score = item.get("relevance_score", item.get("score", 0.0))
            if not isinstance(index, int) or index < 0 or index >= len(documents):
                logger.warning("rerank 返回非法 index，已忽略：%s", item)
                continue
            try:
                scores[index] = float(score)
            except (TypeError, ValueError):
                scores[index] = 0.0
        return scores
