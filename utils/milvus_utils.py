"""Milvus 工具（Milvus 2.5+，稠密向量 + 服务端 BM25 稀疏检索）。

为什么用 BM25：
硅基流动的 BAAI/bge-m3 只返回稠密向量，不返回稀疏向量。原项目依赖
BGE-M3 的 dense + sparse 做混合检索。这里改用 Milvus 2.5 原生的 BM25
Function：写入时 Milvus 用中文分词器从 content 字段自动生成 sparse_vector，
查询时把原始查询文本直接作为稀疏检索的输入，仍然保持“稠密 + 稀疏”双路召回。

集合结构：
- 切片表：content 为可分析字段，sparse_vector 由 BM25 Function 生成
- 商品名表：item_name 为可分析字段，sparse_vector 由 BM25 Function 生成

若后续把 INFERENCE_BACKEND 切回 local_gpu，稠密向量仍由 BGE-M3 生成，
而稀疏一路继续用服务端 BM25，无需重建集合结构。
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, List, Optional

from pymilvus import (
    AnnSearchRequest,
    DataType,
    Function,
    FunctionType,
    MilvusClient,
    WeightedRanker,
)

from config.milvus_config import milvus_config

logger = logging.getLogger(__name__)

SPARSE_FIELD = "sparse_vector"
DENSE_FIELD = "dense_vector"
BM25_FUNCTION_NAME = "content_bm25"
CHINESE_ANALYZER = {"type": "chinese"}

_client_lock = threading.Lock()
_milvus_client: Optional[MilvusClient] = None


def get_milvus_client() -> MilvusClient:
    """获取全局单例的 MilvusClient。"""
    global _milvus_client
    if _milvus_client is not None:
        return _milvus_client

    with _client_lock:
        if _milvus_client is None:
            _milvus_client = MilvusClient(uri=milvus_config.milvus_url)
            logger.info("Milvus 客户端已创建：%s", milvus_config.milvus_url)
        return _milvus_client


def escape_milvus_string(value: str) -> str:
    """转义 Milvus 过滤表达式中的字符串。"""
    value = (value or "").replace("\\", "\\\\").replace('"', '\\"').replace("'", "\\'")
    return value


def _add_dense_index(index_params, field_name: str = DENSE_FIELD) -> None:
    index_params.add_index(
        field_name=field_name,
        index_name=f"{field_name}_index",
        index_type="AUTOINDEX",
        metric_type=milvus_config.dense_metric_type,
    )


def _add_bm25_index(index_params, field_name: str = SPARSE_FIELD) -> None:
    index_params.add_index(
        field_name=field_name,
        index_name=f"{field_name}_bm25_index",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="BM25",
        params={"inverted_index_algo": "DAAT_MAXSCORE"},
    )


def ensure_chunks_collection(client: MilvusClient, dim: Optional[int] = None) -> str:
    """创建（若不存在）切片集合，返回集合名。"""
    collection_name = milvus_config.chunks_collection
    if client.has_collection(collection_name):
        return collection_name

    dim = dim or milvus_config.dense_dim

    schema = client.create_schema(auto_id=True, enable_dynamic_field=True)
    schema.add_field("chunk_id", DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field("content", DataType.VARCHAR, max_length=65535,
                     enable_analyzer=True, analyzer_params=CHINESE_ANALYZER)
    schema.add_field("title", DataType.VARCHAR, max_length=512)
    schema.add_field("parent_title", DataType.VARCHAR, max_length=512)
    schema.add_field("part", DataType.INT8)
    schema.add_field("file_title", DataType.VARCHAR, max_length=512)
    schema.add_field("item_name", DataType.VARCHAR, max_length=512)
    schema.add_field(SPARSE_FIELD, DataType.SPARSE_FLOAT_VECTOR)
    schema.add_field(DENSE_FIELD, DataType.FLOAT_VECTOR, dim=dim)

    # 由 Milvus 在服务端根据 content 生成稀疏向量
    schema.add_function(
        Function(
            name=BM25_FUNCTION_NAME,
            function_type=FunctionType.BM25,
            input_field_names=["content"],
            output_field_names=[SPARSE_FIELD],
        )
    )

    index_params = client.prepare_index_params()
    _add_dense_index(index_params)
    _add_bm25_index(index_params)

    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        index_params=index_params,
    )
    logger.info("已创建切片集合：%s（dim=%s, BM25）", collection_name, dim)
    return collection_name


def ensure_item_name_collection(client: MilvusClient, dim: Optional[int] = None) -> str:
    """创建（若不存在）商品名集合，返回集合名。"""
    collection_name = milvus_config.item_name_collection
    if client.has_collection(collection_name):
        return collection_name

    dim = dim or milvus_config.dense_dim

    schema = client.create_schema(auto_id=True, enable_dynamic_field=True)
    schema.add_field("pk", DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field("file_title", DataType.VARCHAR, max_length=512)
    schema.add_field("item_name", DataType.VARCHAR, max_length=512,
                     enable_analyzer=True, analyzer_params=CHINESE_ANALYZER)
    schema.add_field(SPARSE_FIELD, DataType.SPARSE_FLOAT_VECTOR)
    schema.add_field(DENSE_FIELD, DataType.FLOAT_VECTOR, dim=dim)

    schema.add_function(
        Function(
            name="item_name_bm25",
            function_type=FunctionType.BM25,
            input_field_names=["item_name"],
            output_field_names=[SPARSE_FIELD],
        )
    )

    index_params = client.prepare_index_params()
    _add_dense_index(index_params)
    _add_bm25_index(index_params)

    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        index_params=index_params,
    )
    logger.info("已创建商品名集合：%s（dim=%s, BM25）", collection_name, dim)
    return collection_name


def ensure_loaded(client: MilvusClient, collection_name: str) -> None:
    """确保集合已加载到查询节点（幂等）。"""
    try:
        state = client.get_load_state(collection_name=collection_name)
        raw = state.get("state") if isinstance(state, dict) else getattr(state, "state", None)
        # Loaded 的枚举值/字符串在不同版本下表现不同，统一按字符串判断
        if raw is not None and "Loaded" in str(raw):
            return
    except Exception:  # noqa: BLE001 - 查询状态失败时直接尝试 load
        pass

    try:
        client.load_collection(collection_name)
    except Exception as exc:  # noqa: BLE001 - 已加载时会报错，忽略
        logger.debug("load_collection(%s) 忽略异常：%s", collection_name, exc)


def build_item_name_expr(item_names: Optional[List[str]]) -> Optional[str]:
    """构造 item_name 过滤表达式；为空时返回 None（不过滤）。"""
    if not item_names:
        return None
    quoted = ", ".join(f'"{escape_milvus_string(str(name))}"' for name in item_names)
    return f"item_name in [{quoted}]"


def flatten_hits(response) -> List[Dict[str, Any]]:
    """把 MilvusClient 的检索结果拍平成普通字典列表。

    MilvusClient 返回结构：[[{"id":..., "distance":..., "entity": {...}}, ...]]
    这里统一输出 {"chunk_id", "distance", **entity}，方便 RRF / 重排阶段使用。
    """
    if not response:
        return []

    hits = response[0] if isinstance(response[0], list) else response
    flattened: List[Dict[str, Any]] = []
    for hit in hits:
        if not isinstance(hit, dict):
            continue
        entity = hit.get("entity") or {}
        data: Dict[str, Any] = dict(entity)
        if "chunk_id" not in data:
            data["chunk_id"] = hit.get("id")
        data["distance"] = hit.get("distance", 0.0)
        flattened.append(data)
    return flattened


def create_hybrid_search_requests(
    dense_vector: List[float],
    query_text: Optional[str] = None,
    sparse_vector: Optional[Dict[int, float]] = None,
    dense_params: Optional[Dict[str, Any]] = None,
    sparse_params: Optional[Dict[str, Any]] = None,
    expr: Optional[str] = None,
    limit: int = 5,
) -> List[AnnSearchRequest]:
    """构建稠密 + 稀疏双路检索请求。

    优先使用 query_text 走服务端 BM25（新集合）；
    传入 sparse_vector（dict）时走 IP 稀疏检索（兼容旧集合 / GPU 模式）。
    """
    if dense_params is None:
        dense_params = {"metric_type": milvus_config.dense_metric_type}

    if query_text is None and sparse_vector is None:
        raise ValueError("create_hybrid_search_requests 需要 query_text 或 sparse_vector")

    dense_req = AnnSearchRequest(
        data=[dense_vector],
        anns_field=DENSE_FIELD,
        param=dense_params,
        expr=expr,
        limit=limit,
    )

    if query_text is not None:
        sparse_req = AnnSearchRequest(
            data=[query_text],
            anns_field=SPARSE_FIELD,
            param=sparse_params or {"metric_type": "BM25"},
            expr=expr,
            limit=limit,
        )
    else:
        sparse_req = AnnSearchRequest(
            data=[sparse_vector],
            anns_field=SPARSE_FIELD,
            param=sparse_params or {"metric_type": "IP"},
            expr=expr,
            limit=limit,
        )

    return [dense_req, sparse_req]


def hybrid_search(
    client: MilvusClient,
    collection_name: str,
    reqs: List[AnnSearchRequest],
    ranker_weights: tuple = (0.5, 0.5),
    norm_score: bool = True,
    limit: int = 5,
    output_fields: Optional[List[str]] = None,
    dense_vector: Optional[List[float]] = None,
    expr: Optional[str] = None,
) -> List[List[dict]]:
    """执行稠密 + BM25 混合检索。

    BM25 分数没有上界，必须 norm_score=True 后再加权，否则会压制稠密分数。
    混合检索失败时，若提供了 dense_vector，会退化为纯稠密检索，保证问答可用。
    """
    if output_fields is None:
        output_fields = ["item_name"]

    try:
        ranker = WeightedRanker(ranker_weights[0], ranker_weights[1], norm_score=norm_score)
        res = client.hybrid_search(
            collection_name=collection_name,
            reqs=reqs,
            ranker=ranker,
            limit=limit,
            output_fields=output_fields,
        )
        logger.info("Milvus 混合检索完成：集合[%s]，%d 条结果", collection_name, len(res[0]) if res else 0)
        return res
    except Exception as exc:  # noqa: BLE001
        logger.error("Milvus 混合检索失败：集合[%s]，%s", collection_name, exc, exc_info=True)
        if dense_vector is None:
            return []

    # 退化路径：纯稠密检索
    try:
        res = client.search(
            collection_name=collection_name,
            data=[dense_vector],
            anns_field=DENSE_FIELD,
            search_params={"metric_type": milvus_config.dense_metric_type},
            limit=limit,
            output_fields=output_fields,
            filter=expr,
        )
        logger.warning("已退化为纯稠密检索：集合[%s]，%d 条结果", collection_name, len(res[0]) if res else 0)
        return res
    except Exception as exc:  # noqa: BLE001
        logger.error("Milvus 纯稠密检索同样失败：集合[%s]，%s", collection_name, exc)
        return []
