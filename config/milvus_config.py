"""Milvus 配置。"""

from dataclasses import dataclass

from config.env_loader import env_int, env_str


@dataclass
class MilvusConfig:
    milvus_url: str
    chunks_collection: str
    item_name_collection: str
    dense_metric_type: str
    dense_dim: int


milvus_config = MilvusConfig(
    milvus_url=env_str("MILVUS_URL", default="http://127.0.0.1:19530"),
    chunks_collection=env_str("CHUNKS_COLLECTION", default="kb_chunks_v2"),
    item_name_collection=env_str("ITEM_NAME_COLLECTION", default="kb_item_names_v2"),
    dense_metric_type=env_str("MILVUS_METRIC_TYPE", default="COSINE"),
    dense_dim=env_int("EMBEDDING_DIM", default=1024),
)
