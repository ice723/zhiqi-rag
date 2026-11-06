# -*- coding: utf-8 -*-
"""
批量导入手册驱动脚本：把 D:/BaiduNetdiskDownload 中的新设备手册逐本灌入知识库。
每本导入的结果（成功/失败、chunk 数、耗时）实时写入 eval/import_results.json，
中途某本失败（如 MinerU 配额耗尽）不影响后续手册继续导入。

运行前提：VM 虚拟机内 Milvus / MinIO 已启动，.env 中 MinerU 与 DashScope 配置有效
运行方式（项目根目录下）：
    D:/python_lib/kb311/Scripts/python.exe eval/import_manuals.py
"""
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from processor.import_processor.main_graph import KBImportWorkflow
from processor.import_processor.base import setup_logging

PDF_DIR = Path(r"D:/BaiduNetdiskDownload")

# 已按 MD5 去重、并排除知识库中已有的 hak180；H3C取命名规范的一份
PDFS = [
    "Aolynk CB304n Cable网桥 用户手册-5W100-整本手册.pdf",
    "H3C LA2608室内无线网关 用户手册-6W100-整本手册.pdf",
    "HUAWEI MateBook B7-420 用户指南-(MRGFZ,Windows11_02,zh-cn).pdf",
    "HUAWEI MateStation S 12代酷睿版 用户指南-(PUC,Windows11_02,zh-cn).pdf",
    "华为擎云 M272Q 用户指南-(XSN-27QBZ,02,zh-cn).pdf",
    "华为擎云 W525 用户指南-(PGUW-WBX0,KOS&UOS_02,zh-cn).pdf",
    "华为擎云B530 用户指南-(PUCZ,Windows11_03,zh-cn).pdf",
    "华为擎云W585 用户指南-(PGUV,KOS&UOS_01,zh-cn).pdf",
    "华为显示器 B3-211H 用户指南-(NSN-21BZ,02,zh-cn).pdf",
]

OUT_FILE = Path(__file__).parent / "import_results.json"


def main():
    setup_logging()
    results = []
    for name in PDFS:
        pdf = PDF_DIR / name
        rec = {"file": name, "start": time.strftime("%H:%M:%S")}
        t0 = time.time()
        try:
            workflow = KBImportWorkflow()
            final_state = workflow.run({"import_file_path": str(pdf)}, stream=False)
            chunks = final_state.get("chunks") or []
            rec.update({
                "ok": True,
                "chunks": len(chunks),
                "item_name": final_state.get("item_name"),
                "seconds": round(time.time() - t0, 1),
            })
            print(f"[OK] {name} -> {len(chunks)} chunks, "
                  f"item_name={rec['item_name']}, 耗时 {rec['seconds']}s", flush=True)
        except Exception as e:
            rec.update({
                "ok": False,
                "error": f"{type(e).__name__}: {e}",
                "seconds": round(time.time() - t0, 1),
            })
            print(f"[FAIL] {name} -> {rec['error']}", flush=True)
            traceback.print_exc()
        results.append(rec)
        OUT_FILE.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    ok_n = sum(1 for r in results if r["ok"])
    print(f"\n===== 导入完成：成功 {ok_n}/{len(PDFS)}，"
          f"总耗时 {sum(r['seconds'] for r in results)}s =====", flush=True)


if __name__ == "__main__":
    main()
