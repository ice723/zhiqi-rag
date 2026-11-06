"""重排序工具。

上层只调用 rerank_documents(query, documents)，具体后端由
utils.inference.factory 决定：
- siliconflow：硅基流动 /v1/rerank，模型 BAAI/bge-reranker-v2-m3
- local_gpu：GPU 服务器上的 reranker 自定义服务
"""

from typing import List

from utils.inference.factory import get_rerank_client


def rerank_documents(query: str, documents: List[str]) -> List[float]:
    """返回与 documents 顺序一致的 relevance score 列表。"""
    if not documents:
        return []

    scores = get_rerank_client().rerank(query, documents)
    print(f"rerank_scores:{scores}")
    return scores
