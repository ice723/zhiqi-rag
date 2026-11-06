# Zhiqi · RAG knowledge base over your own documents

Upload PDFs / Markdown, ask questions in natural language, and get answers grounded in your own material through hybrid retrieval.

Live demo: **<https://rag.gcmg.cn>**

Zhiqi is the **CPU-only edition** of `knowlege_base311`. No local LLM is loaded: embedding, reranking and generation all go through the SiliconFlow API, so the full stack (Milvus 2.5 + MinIO + MongoDB + single-process FastAPI) runs on a modest 2 vCPU / 4 GB cloud server.

## What it does

- **One pipeline for ingestion**: PDF → Markdown (MinerU) → images to MinIO → semantic chunking → entity (product name) recognition → bge-m3 embedding → Milvus
- **Hybrid retrieval**: dense vectors plus server-side BM25 with a Chinese analyzer in Milvus 2.5 — no local GPU required
- **Three-way recall + RRF + rerank**: vector search, HyDE, and web search run in parallel, fused by RRF, reranked by a Cross-Encoder, then answered by the LLM
- **Entity alignment**: before searching, the query graph decides *which product* is being asked about, with dual thresholds (≥0.8 confirmed / 0.6–0.8 ask back / <0.6 refuse). Answers can carry the matching figures from the source document
- **SSE streaming**: node-level progress (entity → KB search → web search → HyDE → RRF → rerank → answer) plus token-by-token output
- **Degrades instead of blanking**: with middleware down the page still loads, document list and history return empty with a reason, and a failed generation ends in one readable sentence rather than a spinner
- **MinIO is the source of truth for the document list**: switching browsers or restarting the service keeps the sidebar in sync with the actual knowledge base

## Architecture

```text
web/page/index.html        single-page UI (import drawer + chat + library, dependency-free vanilla JS)
        │  /upload /status /query /stream /documents /history
web/api/app.py             unified FastAPI (single process; SSE queues and task progress live in memory)
        │
processor/                 two LangGraph workflows
  ├─ import_processor/     entry → PDF→MD → images → split → entity → embed → import
  └─ query_processor/      entity confirm → vector / HyDE / web recall → RRF → rerank → answer
        │
utils/inference/           inference abstraction: siliconflow (default) | local_gpu
        │
SiliconFlow API / self-hosted GPU HTTP service
```

The backend is selected by `INFERENCE_BACKEND`; business code is unaware of the implementation:

| Backend | Embedding | Rerank | LLM |
|---|---|---|---|
| `siliconflow` (default, CPU server) | `BAAI/bge-m3` | `BAAI/bge-reranker-v2-m3` | `Qwen/Qwen3-8B` |
| `local_gpu` | `BGE_M3_URL` | `BGE_RERANKER_URL` | `LLM_API_BASE` |

## Retrieval evaluation

Details in **[eval/README.md](eval/README.md)** (12 device manuals / 276 chunks / 16 questions, ablations and entity alignment):

| Configuration | Online path (entity filter) | Full-corpus stress (no filter) |
|---|---|---|
| C1 dense + sparse hybrid | 100% | 100% |
| C2 + HyDE / RRF | 100% | 94% |
| C3 + Cross-Encoder rerank (full chain) | **100%** | **81%** |

- The entity filter matters most: removing it drops the full chain from 100% to 81% on four manuals with near-identical chapters — **metadata filtering beats reranking here**
- HyDE is a negative trade in this corpus (-6% in stress mode). It is kept for robustness against single-path variance and is the main latency cost (about +11 s)
- Baseline retrieval ≈ 284 ms, rerank ≈ 148 ms (measured on the `local_gpu` backend)

## Quick start

```bash
# 1. Configuration (keys live in the env file only)
cp .env.example .env
vi .env

# 2. Dependencies (Python 3.11)
uv pip install -e .                 # recommended; or: pip install -r deploy/requirements.txt

# 3. Smoke-test the inference path
python scripts/smoke_inference.py --all

# 4. Run (single process — SSE relies on in-process state)
python -m web.api.app
```

Open <http://127.0.0.1:8000/>. Milvus / MongoDB / MinIO addresses come from `.env`; use the four services in `deploy/docker-compose.yml` if you only need the middleware.

## API

| Method | Path | Description |
|---|---|---|
| POST | `/upload` | Upload one or more files, returns `task_ids`; ingestion runs in the background |
| GET | `/status/{task_id}` | Import progress (`done_list` / `running_list` / `status` / `error`) |
| GET | `/documents` | Library listing (MinIO as source of truth, merged with in-flight states) |
| POST | `/query` | Ask a question (with `is_stream=true` events go to `/stream/{session_id}`) |
| GET | `/stream/{session_id}` | SSE: `ready` / `progress` / `delta` / `final` / `error` |
| GET / DELETE | `/history/{session_id}` | Chat history / clear |
| GET | `/health`, `/health/deep` | Service and dependency (Milvus, Mongo, MinIO) checks |

## Configuration notes

See `.env.example` for the full list. The usual traps:

- `SILICONFLOW_API_KEY`: required, never commit it
- `EMBEDDING_DIM=1024`: must match bge-m3; changing it breaks existing collections
- `CHUNKS_COLLECTION=kb_chunks_v2`, `ITEM_NAME_COLLECTION=kb_item_names_v2`: the BM25 schema is incompatible with older collections, hence the new names; collections are created on first import
- `MINIO_PUBLIC_BASE_URL`: public prefix for figures in answers (e.g. `https://your-domain/minio` when reverse-proxied)
- `MINERU_API_TOKEN` / `MINERU_BASE_URL`: MinerU cloud API for PDF parsing (Markdown uploads skip it)
- `WEB_SEARCH_ENABLED`: toggles the web-search branch of the query graph

## Deployment

Full-stack Docker Compose, Nginx (SSE must disable buffering), security groups, image mirrors and troubleshooting: see **[CPU_DEPLOY.md](CPU_DEPLOY.md)**.

Two hard-won notes:

- The app must run single-process (`--workers 1`); multiple workers break the in-memory SSE queues and task progress
- The index response must carry `Cache-Control: no-cache`, otherwise mobile browsers keep serving a stale page

## Layout

```text
config/      configuration (env_loader reads the project-root .env)
processor/   import_processor (ingest graph) + query_processor (query graph)
utils/       inference clients, Milvus, MinIO, Mongo, SSE, task progress, document registry
web/         api/ (FastAPI routers) + page/ (single-page UI)
deploy/      Dockerfile, docker-compose, env template, Nginx, start/stop scripts
scripts/     inference smoke test and ops helpers
eval/        retrieval evaluation (questions, script, results, findings)
test/        ad-hoc integration scripts
```

## Known limitations (design trade-offs)

- **The question must map to a known entity**: the query graph confirms which product/device is being asked about and filters chunks by it. Questions with no such entity ("what's the weather") are refused, and documents recognized under different entity names will not be retrieved together
- With a tiny demo corpus, short documents become a single chunk, so retrieval granularity is coarse
- HyDE dominates latency; the roadmap is a smaller generator, a length cap on the hypothetical document, or enabling it only for paraphrased questions

## About

A personal portfolio project, built on top of `knowlege_base311` (the GPU + local BGE edition). The original project is left untouched; all changes live in this directory.
