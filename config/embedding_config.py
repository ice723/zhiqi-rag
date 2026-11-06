"""Embedding 配置。

CPU 部署默认使用硅基流动的 BAAI/bge-m3（1024 维，仅稠密向量）。
切回 GPU 时把 INFERENCE_BACKEND 改为 local_gpu，配置 BGE_M3_URL 即可，
上层代码无需修改。
"""

from dataclasses import dataclass

from config.env_loader import env_float, env_int, env_str

DEFAULT_BASE_URL = "https://api.siliconflow.cn/v1"
DEFAULT_MODEL = "BAAI/bge-m3"
DEFAULT_DIM = 1024


@dataclass
class EmbeddingConfig:
    api_base: str
    api_key: str
    model: str
    dim: int
    batch_size: int
    timeout: int
    max_retries: int
    min_interval: float


embedding_config = EmbeddingConfig(
    api_base=env_str("EMBEDDING_API_BASE", "SILICONFLOW_BASE_URL", default=DEFAULT_BASE_URL),
    api_key=env_str("EMBEDDING_API_KEY", "SILICONFLOW_API_KEY", "OPENAI_API_KEY"),
    model=env_str("EMBEDDING_MODEL", default=DEFAULT_MODEL),
    dim=env_int("EMBEDDING_DIM", default=DEFAULT_DIM),
    batch_size=env_int("EMBEDDING_BATCH_SIZE", default=16),
    timeout=env_int("EMBEDDING_TIMEOUT", default=60),
    max_retries=env_int("EMBEDDING_MAX_RETRIES", default=3),
    min_interval=env_float("EMBEDDING_MIN_INTERVAL", default=0.0),
)
