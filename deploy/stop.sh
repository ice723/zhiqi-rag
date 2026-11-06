#!/usr/bin/env bash
# 停止服务。加参数 --down 时同时删除容器（数据卷目录 ./volumes 保留）
set -euo pipefail

cd "$(dirname "$0")"

if [ "${1:-}" = "--down" ]; then
  docker compose --env-file .env.prod down
else
  docker compose --env-file .env.prod stop
fi
