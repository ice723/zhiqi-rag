# 知器 Zhiqi · 私有资料 RAG 知识库问答

上传 PDF / Markdown，用自然语言提问，系统在你的资料里做混合检索，再带着依据作答。

线上演示：**<https://rag.gcmg.cn>**

知器是从 `knowlege_base311` 改造出的 **CPU 部署版**：不加载本地大模型，Embedding / Rerank / LLM 全部走硅基流动 API，一台 2 核 4G 级别的普通云服务器就能跑完整链路（Milvus 2.5 + MinIO + MongoDB + 单进程 FastAPI）。

## 能力

- **一条链路跑完导入**：PDF 经 MinerU 解析成 Markdown → 图片进 MinIO → 语义切片 → 识别资料主体（商品名）→ bge-m3 向量化 → 写入 Milvus
- **混合检索**：稠密向量 + Milvus 2.5 服务端 BM25（中文分词）双路召回，不依赖本地 GPU
- **三路召回 + RRF + 精排**：向量检索、HyDE 假设文档检索、联网搜索并行，RRF 融合后由 Cross-Encoder 重排，再交给 LLM 作答
- **主体对齐（防答错手册）**：提问先对齐"问的是哪台设备"，双阈值（≥0.8 确认 / 0.6~0.8 反问 / <0.6 拒答），同一份资料带出配图
- **SSE 流式**：节点级进度（识别提问主体 → 知识库检索 → 联网搜索 → HyDE → RRF → 重排 → 生成回答）+ 逐字输出
- **可降级的界面**：中间件不可达时页面照样打开，资料清单、历史记录返回空并说明原因；生成失败会收敛成一句人话，不会一直转圈
- **资料清单一律以 MinIO 为事实来源**：换浏览器、换设备、服务重启后，左侧列表依旧和知识库对得上

## 架构

```text
web/page/index.html        单页前端（导入抽屉 + 对话 + 资料库，零依赖 vanilla JS）
        │  /upload /status /query /stream /documents /history
web/api/app.py             统一 FastAPI（单进程，SSE 队列与任务进度在进程内存）
        │
processor/                 LangGraph 两张图
  ├─ import_processor/     入口 → PDF转MD → 图片处理 → 切片 → 主体识别 → 向量化 → 入库
  └─ query_processor/      主体确认 → 向量 / HyDE / 联网 三路召回 → RRF → 重排 → 生成
        │
utils/inference/           推理抽象层：siliconflow（默认） | local_gpu
        │
硅基流动 API / 自建 GPU HTTP 服务
```

推理层由 `INFERENCE_BACKEND` 切换，业务代码不感知实现：

| 后端 | Embedding | Rerank | LLM |
|---|---|---|---|
| `siliconflow`（默认，CPU 服务器） | `BAAI/bge-m3` | `BAAI/bge-reranker-v2-m3` | `Qwen/Qwen3-8B` |
| `local_gpu` | `BGE_M3_URL` | `BGE_RERANKER_URL` | `LLM_API_BASE` |

## 检索链路评测

详见 **[eval/README.md](eval/README.md)**（12 本设备手册 / 276 chunks / 16 题，含消融与实体对齐）：

| 配置 | 线上链路（带主体过滤） | 全库压力（无过滤） |
|---|---|---|
| C1 稠密+稀疏混合检索 | 100% | 100% |
| C2 +HyDE / RRF | 100% | 94% |
| C3 +Cross-Encoder 重排（全链路） | **100%** | **81%** |

- 主体过滤是最关键的一环：去掉后全链路命中率 100% → 81%（4 本手册章节高度雷同），**元数据过滤比精排更本质**
- HyDE 在本库是负收益（压力模式 -6%），保留它是为了三路并行对冲单路波动，同时是主要延迟来源（约 +11s）
- 基线检索约 284ms、重排约 148ms（`local_gpu` 后端实测）

## 快速开始

```bash
# 1. 配置（Key 只放环境文件，不提交）
cp .env.example .env
vi .env

# 2. 装依赖（Python 3.11）
uv pip install -e .                 # 推荐；或 pip install -r deploy/requirements.txt

# 3. 推理链路冒烟
python scripts/smoke_inference.py --all

# 4. 启动（必须单进程，SSE 依赖进程内存）
python -m web.api.app
```

打开 <http://127.0.0.1:8000/>。Milvus / MongoDB / MinIO 的地址写在 `.env`，可以指向本机、内网虚拟机或远程实例；只想起中间件的话用 `deploy/docker-compose.yml` 里的四个服务即可。

## 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/upload` | 多文件上传，返回 `task_ids`，后台跑导入图 |
| GET | `/status/{task_id}` | 导入进度（`done_list` / `running_list` / `status` / `error`） |
| GET | `/documents` | 资料库清单（以 MinIO 为准，合并本次进程的解析中/失败状态） |
| POST | `/query` | 提问（`is_stream=true` 时事件投递到 `/stream/{session_id}`） |
| GET | `/stream/{session_id}` | SSE：`ready` / `progress` / `delta` / `final` / `error` |
| GET / DELETE | `/history/{session_id}` | 会话历史 / 清空 |
| GET | `/health`、`/health/deep` | 服务 / 依赖（Milvus、Mongo、MinIO）健康检查 |

## 配置要点

完整键名见 `.env.example`，容易踩的几项：

- `SILICONFLOW_API_KEY`：必填，不要提交
- `EMBEDDING_DIM=1024`：与 bge-m3 一致，改动会导致已有集合维度不匹配
- `CHUNKS_COLLECTION=kb_chunks_v2`、`ITEM_NAME_COLLECTION=kb_item_names_v2`：BM25 集合结构与旧版不兼容，换名字避免误用旧数据；集合在首次导入时自动创建
- `MINIO_PUBLIC_BASE_URL`：回答里图片链接的对外前缀（反代到 `/minio/` 时填 `https://你的域名/minio`）
- `MINERU_API_TOKEN` / `MINERU_BASE_URL`：PDF 解析走 MinerU 云端 API（Markdown 上传不经过它）
- `WEB_SEARCH_ENABLED`：联网搜索开关，关闭后查询图只走两路召回

## 部署

Docker Compose 全栈、Nginx 反代（SSE 必须关缓冲）、安全组、镜像加速与排障，见 **[CPU_DEPLOY.md](CPU_DEPLOY.md)**。

另外两条经验：

- 业务服务只能单进程（`--workers 1`），多 worker 会让 SSE 队列和任务进度对不上
- 首页响应要带 `Cache-Control: no-cache`，否则手机浏览器会按启发式缓存把旧页面留住

## 目录

```text
config/      配置（env_loader 统一读项目根目录 .env）
processor/   import_processor（导入图）+ query_processor（查询图）
utils/       推理客户端、Milvus、MinIO、Mongo、SSE、任务进度、资料登记
web/         api/（FastAPI 路由）+ page/（单页前端）
deploy/      Dockerfile、docker-compose、env 模板、Nginx、启停脚本
scripts/     推理冒烟等运维脚本
eval/        检索链路评测（题库、脚本、结果、结论）
test/        零散联调脚本
```

## 已知限制（设计取舍，不是待修的 bug）

- **提问需要能对上资料主体**：查询图会先确认"问的是哪个产品/设备"并据此过滤切片。问题是"今天天气怎么样"这类没有主体的问法会直接拒答；同一设备的不同资料如果被识别成不同主体名，跨资料检索会漏
- 演示库很小时，短文档会被切成单块，检索粒度偏粗
- HyDE 是延迟大头；优化方向是小模型生成、限制假设文档长度，或只对改写类问句启用

## 关于

个人作品集项目，基于原 `knowlege_base311`（GPU + 本地 BGE 方案）改造而来，原项目保持不动，所有改动只发生在当前目录。
