"""Rerank 配置。

CPU 部署默认使用硅基流动的 BAAI/bge-reranker-v2-m3。
切回 GPU 时把 INFERENCE_BACKEND 改为 local_gpu，配置 BGE_RERANKER_URL。
"""

from dataclasses import dataclass

from config.env_loader import env_int, env_str

DEFAULT_BASE_URL = "https://api.siliconflow.cn/v1"
DEFAULT_MODEL = "BAAI/bge-reranker-v2-m3"


@dataclass
class RerankerConfig:
    api_base: str
    api_key: str
    model: str
    timeout: int
    max_retries: int
    top_n: int  # 0 表示全部返回


reranker_config = RerankerConfig(
    api_base=env_str("RERANK_API_BASE", "SILICONFLOW_BASE_URL", default=DEFAULT_BASE_URL),
    api_key=env_str("RERANK_API_KEY", "SILICONFLOW_API_KEY", "OPENAI_API_KEY"),
    model=env_str("RERANK_MODEL", default=DEFAULT_MODEL),
    timeout=env_int("RERANK_TIMEOUT", default=60),
    max_retries=env_int("RERANK_MAX_RETRIES", default=3),
    top_n=env_int("RERANK_TOP_N", default=0),
)
