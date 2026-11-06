# 知器 · 知识库问答（CPU / SiliconFlow 版）

知器是一个面向私有资料的知识库问答系统：上传 PDF / Markdown，即可通过混合检索 + RAG 在线提问。

本仓库基于原 `knowlege_base311` 改造的 CPU 部署版本：**不加载本地大模型**，Embedding / Rerank / LLM 全部通过硅基流动 API 调用，适合部署在普通云 CPU 服务器上。

原项目保持不动，所有改动只发生在本目录。

## 功能

- 单页完成「上传资料 → 解析入库 → 在线问答」
- 文档解析：MinerU API（PDF → Markdown），图片上传 MinIO
- 向量库：Milvus 2.5（稠密向量 + 服务端 BM25 中文稀疏检索，混合召回）
- 检索链路：混合检索 → RRF 融合 → Rerank 重排 → LLM 作答（SSE 流式）
- 会话历史：MongoDB
- 联网搜索（可选）：百炼 MCP WebSearch

## 架构

```text
web/page/index.html  ←  单页前端（上传抽屉 + 对话）
        │
web/api/app.py       ←  统一 FastAPI（import + query 两个 router，单进程）
        │
LangGraph 导入图 / 查询图（processor/）
        │
utils/inference/     ←  推理抽象层：siliconflow | local_gpu
        │
硅基流动 API / GPU 自定义 HTTP 服务
```

推理层通过 `INFERENCE_BACKEND` 切换，业务代码不感知具体实现：

| 后端 | Embedding | Rerank | LLM |
|---|---|---|---|
| `siliconflow`（默认） | `BAAI/bge-m3` | `BAAI/bge-reranker-v2-m3` | `Qwen/Qwen3-8B` |
| `local_gpu` | `BGE_M3_URL` | `BGE_RERANKER_URL` | `LLM_API_BASE` |

## 本地运行

```bash
# 1. 准备配置（Key 只放环境文件，不写进代码）
cp .env.example .env
vi .env

# 2. 安装依赖（Python 3.11）
pip install -r deploy/requirements.txt

# 3. 推理链路冒烟
python scripts/smoke_inference.py --all

# 4. 启动（单进程，SSE 依赖进程内存）
python -m web.api.app
```

打开 <http://127.0.0.1:8000/>。

## 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/upload` | 多文件上传，返回 `task_ids` |
| GET | `/status/{task_id}` | 导入进度（done_list / running_list / status / error） |
| POST | `/query` | 问答（`is_stream` 决定流式或一次性） |
| GET | `/stream/{session_id}` | SSE（progress / delta / final / error） |
| GET/DELETE | `/history/{session_id}` | 会话历史 / 清空 |
| GET | `/health`、`/health/deep` | 服务与依赖健康检查 |

## 配置

所有参数见 `.env.example`，关键项：

- `SILICONFLOW_API_KEY`：必须由你填写，不要提交到仓库
- `EMBEDDING_DIM=1024`：与 bge-m3 一致，改动会导致集合维度不匹配
- `MILVUS_URL`、`MONGO_URL`、`MINIO_*`：中间件地址
- `MINIO_PUBLIC_BASE_URL`：浏览器访问图片用的地址（部署后必填）
- `CHUNKS_COLLECTION=kb_chunks_v2`：BM25 集合与旧版不兼容，使用新名字

## 部署

完整的服务器部署步骤（Docker Compose、Nginx、安全组、排障）见 **[CPU_DEPLOY.md](CPU_DEPLOY.md)**。

## 目录

```text
config/      配置（env_loader 统一读取项目根目录 .env）
processor/   LangGraph 导入图与查询图
utils/       推理客户端、Milvus、MinIO、Mongo、SSE、任务进度
web/         api/（FastAPI）+ page/（单页前端）
deploy/      Dockerfile、docker-compose、env 模板、Nginx、启停脚本
scripts/     冒烟测试等运维脚本
test/ eval/  示例与评测脚本
```
