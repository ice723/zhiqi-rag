"""向量化工具。

上层只调用 generate_embeddings(texts)，具体走硅基流动 API 还是 GPU 上的
BGE-M3 服务由 utils.inference.factory 决定。

返回结构与旧版保持一致：
    {"dense": [[...], ...], "sparse": [{token_id: weight}, ...] | None}

CPU 部署（硅基流动）下 sparse 为 None，稀疏检索由 Milvus 2.5 的 BM25
Function 在写入/查询时自动生成。
"""

from typing import Dict, List, Optional

from utils.inference.factory import get_embedding_client


def generate_embeddings(texts: List[str]) -> Dict[str, Optional[list]]:
    """为一批文本生成向量，顺序与输入一致。"""
    if not texts:
        return {"dense": [], "sparse": None}

    result = get_embedding_client().embed(texts)
    return {"dense": result.dense, "sparse": result.sparse}
