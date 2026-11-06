"""MinIO 工具。

与原版相比：
1. 改为懒加载：导入模块时不再连接 MinIO，服务未就绪也不会把客户端永久置空；
2. 增加 build_public_url / rewrite_minio_url，保证写入 Markdown 的图片地址
   在云服务器上对浏览器可访问（可用 MINIO_PUBLIC_BASE_URL 指定公网/反代地址）。
"""

from __future__ import annotations

import json
import logging
import threading
from typing import Optional

from minio import Minio

from config.minio_config import minio_config

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_minio_client: Optional[Minio] = None

_PUBLIC_READ_POLICY = {
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {"AWS": ["*"]},
            "Action": ["s3:GetObject"],
            "Resource": [f"arn:aws:s3:::{minio_config.bucket_name}/*"],
        }
    ],
}


def get_minio_client() -> Minio:
    """获取 MinIO 客户端（懒加载 + 首次连接时确保桶存在且可公共读）。"""
    global _minio_client
    if _minio_client is not None:
        return _minio_client

    with _lock:
        if _minio_client is not None:
            return _minio_client

        if not minio_config.access_key or not minio_config.secret_key:
            raise RuntimeError("未配置 MINIO_ACCESS_KEY / MINIO_SECRET_KEY")

        client = Minio(
            endpoint=minio_config.endpoint,
            access_key=minio_config.access_key,
            secret_key=minio_config.secret_key,
            secure=minio_config.secure,
        )

        if not client.bucket_exists(minio_config.bucket_name):
            client.make_bucket(minio_config.bucket_name)
            logger.info("已创建 MinIO 桶：%s", minio_config.bucket_name)

        client.set_bucket_policy(
            minio_config.bucket_name,
            json.dumps(_PUBLIC_READ_POLICY),
        )

        _minio_client = client
        logger.info("MinIO 连接成功：%s/%s", minio_config.endpoint, minio_config.bucket_name)
        return _minio_client


def build_public_url(object_name: str) -> str:
    """生成浏览器可访问的对象 URL。"""
    return minio_config.build_public_url(object_name)


def rewrite_minio_url(url: str) -> str:
    """把 MinIO 内部地址改写成对外地址。

    兼容：
    - http://192.168.88.130:9000/bucket/a/b.png
    - https://old-host/bucket/a/b.png
    统一按 "<...>/<bucket>/<object>" 截取，再用 MINIO_PUBLIC_BASE_URL 重组。
    """
    if not url:
        return url

    marker = f"/{minio_config.bucket_name}/"
    index = url.find(marker)
    if index == -1:
        return url

    object_name = url[index + len(marker):]
    if not object_name:
        return url
    return build_public_url(object_name)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(get_minio_client().bucket_exists(minio_config.bucket_name))
    print(build_public_url("kb01/demo.png"))
