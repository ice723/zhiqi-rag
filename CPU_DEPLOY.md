# CPU 版知识库部署手册（SiliconFlow + Docker Compose）

本文参考你提供的两份原始文档：
`应用服务部署.md`（应用侧 Docker 部署）与 `GPU模型部署.md`（GPU 模型服务），
按其思路改写为 **纯 CPU 服务器 + 硅基流动 API** 的方案。

> 重要约定
> 1. 硅基流动的 API Key、模型名、接口地址等一律由你在 `.env.prod` 中填写，本文不提供默认值，代码里也没有硬编码。
> 2. 服务器上的每条命令都只做示例输出，由你在 **FinalShell** 中手动执行，本文不会代你操作服务器。
> 3. 文中标注「参考值」的内容来自原始文档（Ubuntu 22.04、端口、镜像版本等），**执行前请与你服务器实际情况核对**。

---

## 0. 与原始两份文档的差异

| 项目 | 原始文档 | 本方案 | 原因 |
|---|---|---|---|
| 模型推理 | 本地 GPU 服务（vLLM + BGE 自定义服务） | 硅基流动 API | 服务器只有 CPU |
| 业务服务 | `kb-import`(8000) + `kb-query`(8001) 两个进程 | **一个** `kb-app`(8000) | 页面与接口已合并，SSE 队列在进程内存，必须单进程 |
| 稀疏向量 | BGE-M3 返回 dense+sparse | Milvus 2.5 **服务端 BM25**（中文分词） | 硅基流动 embedding 只返回稠密向量 |
| Milvus | v2.4.0 | **v2.5.6**（2.5 起步） | BM25 Function / 分词器是 2.5 新增能力 |
| Neo4j | 参与编排 | 不部署 | 当前代码已无任何 Neo4j 调用 |
| 依赖 | 含 torch/transformers | 无 torch/transformers/FlagEmbedding | 不需要本地推理 |
| 前端 | 两个页面 | 单页 `web/page/index.html` | 上传 + 问答合并 |
| 图片地址 | `MINIO_PUBLIC_ENDPOINT` | `MINIO_PUBLIC_BASE_URL` | 命名沿用当前代码 |

---

## 1. 部署架构

```text
浏览器
  │  http://<公网IP>:8000  （或 Nginx 80/443）
  ▼
┌──────────────────────────── 云 CPU 服务器 ─────────────────────────────┐
│  Docker Compose                                                        │
│  ┌────────────┐   ┌──────────┐   ┌────────┐   ┌─────────┐   ┌────────┐│
│  │ kb-app     │──▶│ milvus   │   │ etcd   │   │ minio   │   │ mongo  ││
│  │ FastAPI    │   │ 2.5.6    │   │ 3.5    │   │ 9000    │   │ 7.0    ││
│  │ :8000      │   └──────────┘   └────────┘   └─────────┘   └────────┘│
│  └─────┬──────┘                                                        │
└────────┼───────────────────────────────────────────────────────────────┘
         │ HTTPS
         ▼
  硅基流动（Embedding / Rerank / LLM）
  阿里云百炼（VL 图片理解 / MCP 联网搜索）
  MinerU（PDF 解析）
```

RAG 链路保持不变：
上传 → MinerU 解析 → 图片上传 MinIO → 切片 → 商品名识别（LLM）→ 向量化（Embedding API）
→ 写入 Milvus（dense + BM25 稀疏）→ 提问 → 混合检索 → RRF → Rerank → LLM 作答 → SSE 流式返回。

---

## 2. 部署前需要你确认/提供的信息

| # | 信息 | 说明 | 状态 |
|---|---|---|---|
| 1 | **硅基流动 API Key** | 写入 `deploy/.env.prod` 的 `SILICONFLOW_API_KEY`；也可分别填 EMBEDDING/RERANK/LLM 三个 Key | 待你提供 |
| 2 | 服务器公网 IP、SSH 端口、账号、密码或密钥 | FinalShell 登录用 | 待你提供 |
| 3 | 操作系统与版本 | 参考文档是 Ubuntu 22.04；本文命令以 Debian/Ubuntu 为准 | 待确认 |
| 4 | 服务器是否已装 Docker / Docker Compose | 未装则执行第 4.2 节 | 待确认 |
| 5 | 业务端口 | 参考文档开放 8000；若用 Nginx 则只开放 80/443 | 待确认 |
| 6 | MinIO 公网访问方式 | 直接暴露 9000（`http://IP:9000`）还是走 Nginx（`https://域名/minio`） | 待确认 |
| 7 | 数据目录 | 本文用 `/data/kb`（参考文档用 `/data/atguigu_knowledge_base_02`） | 待确认 |
| 8 | MongoDB / MinIO 密码 | 生产环境务必改强密码 | 待你设定 |
| 9 | 域名与证书（可选） | 需要 HTTPS 时提供 | 待定 |

> 第 1 项未就绪时，服务可以正常启动、页面能打开，但**上传解析到“向量生成”这一步会失败，问答也会失败**。这是正常现象。

---

## 3. 本地准备

1. 确认 `D:\python_lib\knowlege_base311_cpu` 为本方案代码目录（原项目 `knowlege_base311` 未做任何修改）。
2. 打包上传（两种方式任选）：
   - **FinalShell 文件管理器**：把整个项目目录拖到服务器 `/data/kb/app`。
   - 命令行（在本地 Git Bash 执行，端口/账号按实际替换）：

```bash
# 本地执行：打包（排除无关产物）
cd /d/python_lib/knowlege_base311_cpu
tar --exclude=__pycache__ --exclude=.venv --exclude=output --exclude=temp-files \
    --exclude=logs --exclude=.git -czf kb_cpu.tar.gz .
```

```bash
# 本地执行：上传（SSH 端口按实际修改）
scp -P <SSH端口> kb_cpu.tar.gz <账号>@<公网IP>:/data/kb/
```

> 若你更习惯 Git 流程，也可以按原始文档把代码推到 Gitee 再 `git clone`，效果一致。

---

## 4. 服务器初始化（FinalShell 中手动执行）

以下命令均为**参考命令**，请逐条复制到 FinalShell 执行并观察输出。

### 4.1 创建部署目录

```bash
sudo mkdir -p /data/kb
sudo chown -R $USER:$USER /data/kb
cd /data/kb
```

解压（若用 tar 方式上传）：

```bash
tar -xzf /data/kb/kb_cpu.tar.gz -C /data/kb/app --strip-components=0
```

如果没有 `app` 目录，先 `mkdir -p /data/kb/app` 再把压缩包解压进去。

### 4.2 安装 Docker 与 Compose 插件（已安装可跳过）

```bash
sudo apt update
sudo apt install -y ca-certificates curl gnupg lsb-release

sudo mkdir -p /etc/apt/keyrings
curl -fsSL http://mirrors.aliyun.com/docker-ce/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] http://mirrors.aliyun.com/docker-ce/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

sudo usermod -aG docker $USER
newgrp docker
docker --version && docker compose version
```

> 注意：`docker-compose-plugin` 提供的是 `docker compose`（空格）子命令，不是老的 `docker-compose`。

### 4.3 配置镜像加速（可选，拉取慢时使用）

```bash
sudo tee /etc/docker/daemon.json > /dev/null <<'EOF'
{
  "registry-mirrors": [
    "http://hub-mirror.c.163.com",
    "https://mirror.ccs.tencentyun.com",
    "https://docker.m.daocloud.io"
  ]
}
EOF
sudo systemctl daemon-reload
sudo systemctl restart docker
```

### 4.4 准备数据目录

```bash
cd /data/kb/app/deploy
mkdir -p volumes/etcd volumes/minio volumes/milvus volumes/mongo volumes/output temp-files logs
```

### 4.5 写生产环境配置

```bash
cd /data/kb/app/deploy
cp .env.prod.example .env.prod
chmod 600 .env.prod
vi .env.prod
```

必须逐项确认的字段：

- `SILICONFLOW_API_KEY=`：**由你填写**（硅基流动控制台获取）
- `DASHSCOPE_API_KEY=`：沿用你现有的百炼 Key
- `MINERU_API_TOKEN=`：沿用现有 MinerU Token
- `MONGO_ROOT_PASSWORD` / `MONGO_URL`：两处密码保持一致
- `MINIO_SECRET_KEY`：改成强密码
- `MINIO_PUBLIC_BASE_URL`：浏览器能访问到的 MinIO 地址，例如 `http://<公网IP>:9000`
- `EMBEDDING_DIM=1024`：与 `BAAI/bge-m3` 一致，不要随意改

> 安全提示：`.env.prod` 含密钥，只放在服务器上，`chmod 600`，不要提交到代码仓库。

### 4.6 安全组 / 防火墙

参考原始文档开放：

| 端口 | 用途 | 建议 |
|---|---|---|
| 22 | SSH | 仅限你的 IP |
| 8000 | 业务服务 | 对外开放（或只开 80/443 走 Nginx） |
| 9000 | MinIO 图片访问 | 对外开放（否则回答里的图片加载不出来） |
| 80 / 443 | Nginx | 使用反向代理时开放 |
| 19530 / 27017 / 2379 / 9091 | Milvus / Mongo / etcd | **不要对公网开放**（compose 已只绑定 127.0.0.1） |

---

## 5. 启动

```bash
cd /data/kb/app/deploy
bash start.sh
```

脚本等价于：

```bash
docker compose --env-file .env.prod up -d etcd minio milvus mongodb   # 先起中间件
docker compose --env-file .env.prod up -d --build kb-app              # 再起业务
```

首次构建镜像需要下载 Python 依赖，耗时几分钟到十几分钟，属正常。

查看状态与日志：

```bash
docker compose --env-file .env.prod ps
docker compose --env-file .env.prod logs -f kb-app
docker compose --env-file .env.prod logs -f milvus
```

---

## 6. 验证

### 6.1 接口健康检查

```bash
curl http://127.0.0.1:8000/health
# {"status":"ok", ...}

curl http://127.0.0.1:8000/health/deep
# milvus / mongodb / minio 三项都应为 ok 或 collection_missing
```

`collection_missing` 是正常的：集合会在第一次导入时自动创建（BM25 Function 要求 2.5+，若出现建表报错请先看第 9 节）。

### 6.2 页面

浏览器打开 `http://<公网IP>:8000/`，右上角应显示「API: 已连接」。
`/import/html`、`/chat/html` 会 307 跳转到新单页，老书签不会失效。

### 6.3 推理链路冒烟（填完 Key 后）

```bash
cd /data/kb/app/deploy
docker compose --env-file .env.prod exec kb-app python scripts/smoke_inference.py --all
```

预期输出：Embedding 维度 1024、Rerank 三档分数里前两条高于无关条、LLM 有回答、VL 有回答。
任一项失败会打印具体原因，先修好再继续。

> 也可在本地 venv 直接跑：`python scripts/smoke_inference.py --all`。

### 6.4 端到端

1. 页面点「导入资料」，选一个小 PDF/Markdown 上传；
2. 卡片进度会依次显示：上传 → 解析 → 图片处理 → 切分 → 商品名识别 → 向量生成 → 入库；
3. 完成后在输入框提问文档里的内容，应看到流式回答，涉及图片的会显示在答案下方。

### 6.5 原有知识库数据需要重新导入

旧数据不能直接复用，原因有三点：

- 旧集合用的是另一套 embedding 模型（维度不同），向量不能混用；
- 新集合需要 Milvus 2.5 的 BM25 Function，与旧集合结构不兼容；
- 集合名已换成 `kb_chunks_v2` / `kb_item_names_v2`，避免和旧数据冲突。

因此部署完成后，请把原来的 PDF/Markdown 通过页面重新上传一遍。
如果文档量较大，可以先用 `MinerU` 之外的批量脚本（如 `eval/import_manuals.py` 的思路）串行导入，注意单进程与接口限流。

---

## 7. Nginx 反向代理（可选）

```bash
sudo apt install -y nginx
sudo cp /data/kb/app/deploy/nginx/kb.conf /etc/nginx/conf.d/kb.conf
sudo vi /etc/nginx/conf.d/kb.conf    # 改 server_name
sudo nginx -t && sudo systemctl reload nginx
```

配置里已经为 SSE 关闭了 `proxy_buffering` 并放大超时；**不要**改成默认缓冲，否则流式回答会“憋一大段才出来”。

使用 Nginx 后：

- 业务端口可改为只监听本机：把 compose 里 `"8000:8000"` 改成 `"127.0.0.1:8000:8000"`；
- 若同时代理 MinIO（配置里被注释的 `/minio/` 段），把 `MINIO_PUBLIC_BASE_URL` 改为 `https://<域名>/minio`。

---

## 8. 日常运维

```bash
cd /data/kb/app/deploy

docker compose --env-file .env.prod ps                    # 状态
docker compose --env-file .env.prod logs -f kb-app        # 业务日志
docker compose --env-file .env.prod restart kb-app        # 只重启业务
docker compose --env-file .env.prod up -d --build kb-app  # 代码更新后重建
bash stop.sh                                              # 停止容器（保留数据）
bash stop.sh --down                                       # 删除容器（volumes 目录保留）
```

更新代码：

```bash
cd /data/kb/app
# FinalShell 上传新代码或 git pull
cd deploy && docker compose --env-file .env.prod up -d --build kb-app
```

备份（至少包含 Milvus/MinIO/Mongo 的数据目录与配置）：

```bash
tar -czf /data/kb/backup-$(date +%F).tar.gz \
  /data/kb/app/deploy/volumes /data/kb/app/deploy/.env.prod
```

> Milvus 建议停服后备份，或使用其备份工具；MinIO/Mongo 的数据在 `volumes/` 下。

---

## 9. 故障排查

| 现象 | 常见原因 | 处理 |
|---|---|---|
| 页面显示「API: 未连接」 | 服务未起 / 端口不对 / 反向代理 502 | `docker compose ps`、`curl 127.0.0.1:8000/health` |
| 启动报 `Function`/`analyzer` 相关错误 | Milvus 版本低于 2.5 | 确认镜像是 `v2.5.6`，`docker compose down` 后重起 |
| 导入卡在「向量生成」 | Key 未填 / 余额不足 / 网络不通 | 跑 `smoke_inference.py`，看具体报错 |
| 导入报维度不匹配 | `EMBEDDING_DIM` 与模型不一致 | bge-m3 固定 1024，改回来并删掉旧集合后重建 |
| 回答不流式、一次性吐出 | Nginx 缓冲 / 非流式开关 | 关闭 `proxy_buffering`，页面勾选「流式输出」 |
| 回答里的图片裂开 | `MINIO_PUBLIC_BASE_URL` 不对 / 9000 未放行 | 浏览器直接打开图片 URL 验证 |
| `mcp` 调用 500 | mcp SDK 2.x 与百炼网关不兼容 | 依赖固定 `mcp>=1.30,<2`（已写入 requirements） |
| 容器反复重启 | 内存不足 / 端口被占用 | `docker compose logs`，必要时加 swap 或换端口 |
| 拉不到 `quay.io/coreos/etcd` | 网络问题 | 配置镜像加速，或换可访问的 etcd 镜像源 |
| `milvusdb/milvus:v2.5.6` 拉取失败 | 该 tag 不存在或网络问题 | 在 Docker Hub 上换用同系列最新 `2.5.x` tag，改 compose 里的 `image` |
| pymilvus 与服务端版本告警 | 客户端 3.x / 服务端 2.5 | 一般不影响使用；若真报协议错误，把 requirements 固定为 `pymilvus==2.5.16` 后重建镜像 |
| 中文出现乱码 | 终端编码 | 服务器为 UTF-8 一般无碍，Windows 本地看日志可用 `chcp 65001` |

查看详细错误：

```bash
docker compose --env-file .env.prod logs --tail=200 kb-app
```

---

## 10. 切回 GPU 本地推理

代码里推理层是抽象出来的，切换只改配置、不改业务代码：

```dotenv
INFERENCE_BACKEND=local_gpu
BGE_M3_URL=http://<GPU服务器A>:8101/v1/embeddings
BGE_RERANKER_URL=http://<GPU服务器A>:8102/v1/rerank
LLM_API_BASE=http://<GPU服务器B>:8100/v1
LLM_API_KEY=sk-any
```

对应原始文档 `GPU模型部署.md` 的两个模型服务：

- BGE-M3：`/v1/embeddings`，请求 `{"embedding_documents":[...]}`，返回 `{"dense":[[...]],"sparse":[{...}]}`
- Reranker：`/v1/rerank`，请求 `{"query":...,"documents":[...]}`，返回 `{"scores":[...]}` 或 `results[].relevance_score`

说明：

- Milvus 集合结构保持 BM25（服务端稀疏），切回 GPU 后**不需要**重建集合，也不需要稀疏向量；
- 如果以后要改回“用 BGE-M3 的 sparse 向量”，需要同步改 `utils/milvus_utils.py` 与导入/查询节点，属于结构性改动。

---

## 11. 安全清单

- [ ] `.env.prod` 权限 600，不入库、不进镜像（`.dockerignore` 已排除）
- [ ] MongoDB / MinIO 使用强密码，且只绑定本机端口
- [ ] 安全组只放行 22 / 8000（或 80/443）/ 9000
- [ ] SSH 建议改非默认端口 + 仅密钥登录
- [ ] 定期备份 `deploy/volumes` 与 `.env.prod`
- [ ] API Key 泄露时立即在硅基流动控制台吊销并更换

---

## 12. 文件清单

| 文件 | 作用 |
|---|---|
| `deploy/Dockerfile` | CPU 镜像（python:3.11-slim，无 torch） |
| `deploy/docker-compose.yml` | etcd + MinIO + Milvus 2.5 + Mongo + 业务服务 |
| `deploy/.env.prod.example` | 生产环境变量模板（无任何真实 Key） |
| `deploy/nginx/kb.conf` | 反向代理（SSE 优化） |
| `deploy/start.sh` / `stop.sh` | 分步启动 / 停止 |
| `scripts/smoke_inference.py` | Embedding / Rerank / LLM / VL 冒烟测试 |
| `.dockerignore` | 避免把 `.env`、数据目录打进镜像 |
| `web/page/index.html` | 上传 + 问答合并单页 |
| `web/api/app.py` | 统一 FastAPI 入口（`python -m web.api.app`） |
