# CPU 版知识库统一服务镜像
# 说明：不安装 torch / transformers / FlagEmbedding，模型推理全部走 SiliconFlow API
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_ROOT_USER_ACTION=ignore \
    PYTHONPATH=/app \
    PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple \
    PIP_DEFAULT_TIMEOUT=300 \
    PIP_RETRIES=10 \
    TZ=Asia/Shanghai

WORKDIR /app

# 系统依赖：libgomp1 供 numpy/pymilvus 使用；curl 用于容器健康检查
RUN sed -i \
        's|http://deb.debian.org/debian|https://mirrors.aliyun.com/debian|g; \
         s|http://deb.debian.org/debian-security|https://mirrors.aliyun.com/debian-security|g' \
        /etc/apt/sources.list.d/debian.sources && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        libgomp1 \
        tzdata && \
    rm -rf /var/lib/apt/lists/*

# 先复制依赖清单，利用 Docker 层缓存
COPY deploy/requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir --default-timeout=300 --retries=10 -r /app/requirements.txt

# 复制项目代码（.env 已在 .dockerignore 中排除，配置由 compose 的 env_file 注入）
COPY . /app

RUN mkdir -p /app/temp-files /app/output /app/logs

EXPOSE 8000

# 单进程：SSE 队列与任务进度都在进程内存里，禁止 --workers > 1
CMD ["python", "-m", "uvicorn", "web.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
