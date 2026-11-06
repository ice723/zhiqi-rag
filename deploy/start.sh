#!/usr/bin/env bash
# 启动顺序：中间件（etcd/minio/milvus/mongodb）→ 业务服务
# 用法：在 deploy 目录下执行 bash start.sh
set -euo pipefail

cd "$(dirname "$0")"

if [ ! -f .env.prod ]; then
  echo "缺少 deploy/.env.prod，请先复制 .env.prod.example 并填写" >&2
  exit 1
fi

echo "==> 启动基础中间件"
docker compose --env-file .env.prod up -d etcd minio milvus mongodb

echo "==> 等待 Milvus 健康检查通过"
for i in $(seq 1 60); do
  status=$(docker inspect -f '{{.State.Health.Status}}' kb-milvus 2>/dev/null || echo "starting")
  if [ "$status" = "healthy" ]; then
    echo "Milvus 已就绪"
    break
  fi
  sleep 5
done

echo "==> 构建并启动业务服务"
docker compose --env-file .env.prod up -d --build kb-app

echo "==> 当前状态"
docker compose --env-file .env.prod ps
echo "访问：http://<服务器地址>:8000/    健康检查：/health 与 /health/deep"
