from typing import Dict, List, Tuple

from processor.query_processor.base import NodeBase
from processor.query_processor.state import QueryGraphState
from tool.logger import logger
from utils.milvus_utils import flatten_hits


class NodeRrf(NodeBase):
    """
    节点功能：Reciprocal Rank Fusion
    将多路召回的结果（向量、HyDE、Web）进行加权融合排序。
    """

    name: str = "node_rrf"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        logger.info(f"【{self.name}】节点逻辑")

        #1.参数处理：Milvus 命中先拍平成普通字典
        embedding_chunks = flatten_hits(state.get("embedding_chunks"))
        hyde_embedding_chunks = flatten_hits(state.get("hyde_embedding_chunks"))

        #2.封装待处理的参数
        rrf_inputs = [
            (embedding_chunks, 1),
            (hyde_embedding_chunks, 1),
        ]

        #3.RRF 融合
        rrf_merge_results = self._rrf_merge(rrf_inputs)

        #4.结果只回写自己负责的字段
        return {"rrf_chunks": [doc for doc, _ in rrf_merge_results]}

    def _rrf_merge(self, rrf_inputs: List[Tuple], k: int = 60, max_results: int = 5):
        print("向量搜索和假设性搜索的融合函数")
        chunk_scores: Dict[str, float] = {}
        chunk_data: Dict[str, Dict] = {}

        for rrf_input, weight in rrf_inputs:
            for rank, doc in enumerate(rrf_input or []):
                chunk_id = doc.get("chunk_id")
                if chunk_id is None:
                    continue
                chunk_scores[chunk_id] = chunk_scores.get(chunk_id, 0) + weight / (k + rank)
                chunk_data.setdefault(chunk_id, doc)

        unsorted_results = [(chunk_data[cid], score) for cid, score in chunk_scores.items()]
        sorted_results = sorted(unsorted_results, key=lambda x: x[1], reverse=True)
        return sorted_results[:max_results] if max_results else sorted_results


if __name__ == '__main__':
    mock_state = {
        "embedding_chunks": [[
            {"id": 1, "distance": 0.9, "entity": {"chunk_id": 1, "content": "向量搜索结果#1"}},
            {"id": 2, "distance": 0.8, "entity": {"chunk_id": 2, "content": "向量搜索结果#2"}},
        ]],
        "hyde_embedding_chunks": [[
            {"id": 1, "distance": 0.7, "entity": {"chunk_id": 1, "content": "HyDE搜索结果#1"}},
            {"id": 3, "distance": 0.6, "entity": {"chunk_id": 3, "content": "HyDE搜索结果#2"}},
        ]],
    }
    print(NodeRrf()(mock_state))
