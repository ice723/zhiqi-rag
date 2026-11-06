"""MinIO 配置。

endpoint 用于服务端 SDK 访问；
public_base_url 用于拼接浏览器可访问的图片 URL（云服务器需要公网地址或反向代理地址）。
"""

from dataclasses import dataclass

from config.env_loader import env_bool, env_str


@dataclass
class MinIOConfig:
    endpoint: str
    access_key: str
    secret_key: str
    bucket_name: str
    img_dir: str
    secure: bool
    public_base_url: str

    @property
    def scheme(self) -> str:
        return "https" if self.secure else "http"

    def build_public_url(self, object_name: str) -> str:
        """生成浏览器可访问的对象 URL。"""
        base = self.public_base_url.rstrip("/") if self.public_base_url else f"{self.scheme}://{self.endpoint}"
        return f"{base}/{self.bucket_name}/{object_name.lstrip('/')}"


minio_config = MinIOConfig(
    endpoint=env_str("MINIO_ENDPOINT", default="127.0.0.1:9000"),
    access_key=env_str("MINIO_ACCESS_KEY"),
    secret_key=env_str("MINIO_SECRET_KEY"),
    bucket_name=env_str("MINIO_BUCKET_NAME", default="knowledge-base"),
    img_dir=env_str("MINIO_IMG_DIR", default="kb01"),
    secure=env_bool("MINIO_SECURE", default=False),
    public_base_url=env_str("MINIO_PUBLIC_BASE_URL"),
)
