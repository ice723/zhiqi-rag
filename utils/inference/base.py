"""推理客户端接口定义。

这里是整个项目“可切换推理后端”的契约：
- 业务代码只调用 embed() / rerank()；
- 具体走硅基流动、GPU 自定义服务还是别的实现，由 factory.py 决定；
- 以后切回本地 GPU，只需要新增/替换客户端实现，不改上层 RAG 逻辑。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional


class InferenceError(RuntimeError):
    """推理服务调用失败（网络、鉴权、限流、返回格式异常等）。"""


@dataclass
class EmbeddingResult:
    """一次向量化调用的结果。

    dense:
        稠密向量列表，顺序与输入文本一致。
    sparse:
        稀疏向量列表，每个元素是 {token_id: weight}。
        硅基流动的 BAAI/bge-m3 只返回稠密向量，此时为 None；
        GPU 上的 BGE-M3 自定义服务会同时返回 dense 和 sparse。
    """

    dense: List[List[float]]
    sparse: Optional[List[Dict[int, float]]] = None


class EmbeddingClient(ABC):
    """向量化客户端接口。"""

    @abstractmethod
    def embed(self, texts: List[str]) -> EmbeddingResult:
        """把一组文本转换为向量，顺序必须与输入一致。"""
        raise NotImplementedError


class RerankClient(ABC):
    """重排序客户端接口。"""

    @abstractmethod
    def rerank(
        self,
        query: str,
        documents: List[str],
        top_n: Optional[int] = None,
    ) -> List[float]:
        """返回与 documents 顺序一致的 relevance score 列表。"""
        raise NotImplementedError
