"""推理链路冒烟测试。

所有请求参数全部来自 .env / 环境变量，脚本里不写死任何模型名、地址或 Key。

用法：
    python scripts/smoke_inference.py          # Embedding + Rerank
    python scripts/smoke_inference.py --llm    # 再测 LLM
    python scripts/smoke_inference.py --vl     # 再测视觉模型
    python scripts/smoke_inference.py --all    # 全部

退出码：0 全部通过；1 有调用失败；2 缺少必要配置（例如 API Key 为空）。
"""

from __future__ import annotations

import argparse
import base64
import os
import struct
import sys
import time
import zlib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Windows 控制台默认 GBK，统一成 UTF-8 避免中文乱码（服务器上本来就超集）
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

from config.env_loader import env_str  # noqa: E402
from config.embedding_config import embedding_config  # noqa: E402
from config.lm_config import lm_config  # noqa: E402
from config.reranker_config import reranker_config  # noqa: E402
from utils.inference.factory import get_embedding_client, get_rerank_client  # noqa: E402

def _make_png(width: int = 32, height: int = 32, rgb: tuple = (255, 0, 0)) -> bytes:
    """生成一张纯色 PNG，用于验证视觉接口连通性（百炼要求宽高 > 10px）。"""
    raw = b"".join(b"\x00" + bytes(rgb) * width for _ in range(height))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


TEST_IMAGE_B64 = base64.b64encode(_make_png()).decode()

QUESTION = "如何用万用表测量交流电压？"
DOCUMENTS = [
    "将万用表档位调到交流电压档，选择合适量程，红黑表笔并联到被测电路两端。",
    "使用前请检查表笔绝缘层是否破损，测量高压时务必佩戴绝缘手套。",
    "今天天气不错，适合户外活动。",
]


def _mask(value: str) -> str:
    if not value:
        return "(空)"
    if len(value) <= 8:
        return "*" * len(value)
    return f"{value[:4]}...{value[-4:]}(len={len(value)})"


def _print_config() -> list[str]:
    """打印当前配置概览，返回缺失项的说明列表。"""
    print("=" * 60)
    print("当前配置")
    print("=" * 60)
    print(f"INFERENCE_BACKEND = {env_str('INFERENCE_BACKEND', default='siliconflow')}")
    print(f"Embedding: {embedding_config.model} @ {embedding_config.api_base}")
    print(f"  key={_mask(embedding_config.api_key)}  dim={embedding_config.dim}  batch={embedding_config.batch_size}")
    print(f"Rerank   : {reranker_config.model} @ {reranker_config.api_base}")
    print(f"  key={_mask(reranker_config.api_key)}")
    print(f"LLM      : {lm_config.llm_model} @ {lm_config.base_url}")
    print(f"  key={_mask(lm_config.api_key)}")
    print(f"VL       : {lm_config.vl_model} @ {lm_config.vl_base_url}")
    print(f"  key={_mask(lm_config.vl_api_key)}")
    print()

    missing = []
    if not embedding_config.api_key:
        missing.append("EMBEDDING_API_KEY / SILICONFLOW_API_KEY")
    if not reranker_config.api_key:
        missing.append("RERANK_API_KEY / SILICONFLOW_API_KEY")
    if not lm_config.api_key:
        missing.append("LLM_API_KEY / SILICONFLOW_API_KEY")
    return missing


def check_embedding() -> bool:
    print("[1/4] Embedding ...")
    client = get_embedding_client()
    started = time.time()
    result = client.embed([QUESTION])
    cost = time.time() - started
    if not result.dense or not result.dense[0]:
        print("  ✗ 返回为空")
        return False
    dim = len(result.dense[0])
    print(f"  ✓ 向量维度={dim}，期望={embedding_config.dim}，耗时={cost:.2f}s，sparse={'有' if result.sparse else '无'}")
    if embedding_config.dim and dim != embedding_config.dim:
        print("  ! 维度与 EMBEDDING_DIM 不一致，Milvus 集合会建不起来，请核对配置")
        return False
    return True


def check_rerank() -> bool:
    print("[2/4] Rerank ...")
    client = get_rerank_client()
    started = time.time()
    scores = client.rerank(QUESTION, DOCUMENTS)
    cost = time.time() - started
    if len(scores) != len(DOCUMENTS):
        print(f"  ✗ 分数数量={len(scores)}，期望={len(DOCUMENTS)}")
        return False
    print(f"  ✓ 分数={[round(s, 4) for s in scores]}，耗时={cost:.2f}s")
    if not (scores[0] > scores[2] and scores[1] > scores[2]):
        print("  ! 相关性排序看起来不对，请检查 RERANK_MODEL 是否为真正的重排模型")
        return False
    return True


def check_llm() -> bool:
    print("[3/4] LLM ...")
    from utils.llm_utils import get_llm_client

    llm = get_llm_client()
    started = time.time()
    answer = llm.invoke("用一句话回答：万用表测量电压前要注意什么？")
    cost = time.time() - started
    text = getattr(answer, "content", "") or ""
    print(f"  ✓ 耗时={cost:.2f}s，回答长度={len(text)}")
    print(f"    回答：{text[:120].replace(chr(10), ' ')}")
    return bool(text.strip())


def check_vl() -> bool:
    print("[4/4] 视觉模型 ...")
    from utils.llm_utils import get_vl_client

    vl = get_vl_client()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "这张图片是什么颜色？只回答颜色。"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{TEST_IMAGE_B64}"}},
            ],
        }
    ]
    started = time.time()
    answer = vl.invoke(messages)
    cost = time.time() - started
    text = getattr(answer, "content", "") or ""
    print(f"  ✓ 耗时={cost:.2f}s，回答：{text[:80]}")
    return bool(text.strip())


def main() -> int:
    parser = argparse.ArgumentParser(description="推理接口冒烟测试")
    parser.add_argument("--llm", action="store_true", help="同时测试 LLM")
    parser.add_argument("--vl", action="store_true", help="同时测试视觉模型")
    parser.add_argument("--all", action="store_true", help="测试全部")
    args = parser.parse_args()

    missing = _print_config()
    if missing:
        print("缺少必要配置，请先在 .env（或 deploy/.env.prod）中填写：")
        for item in missing:
            print(f"  - {item}")
        print("\n提示：API Key 只写在环境配置文件中，不要写进代码。")
        return 2

    results = [("Embedding", check_embedding()), ("Rerank", check_rerank())]
    if args.all or args.llm:
        results.append(("LLM", check_llm()))
    if args.all or args.vl:
        results.append(("VL", check_vl()))

    print("=" * 60)
    for name, ok in results:
        print(f"{'✓' if ok else '✗'} {name}")
    print("=" * 60)
    return 0 if all(ok for _, ok in results) else 1


if __name__ == "__main__":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    raise SystemExit(main())
