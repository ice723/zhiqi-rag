# -*- coding: utf-8 -*-
"""
检索链路评测脚本（简历量化数据来源）。

评测设计（两部分）：
一、检索命中率@5（16 题：8 单实体直问 + 8 口语化改写）
    C1 基线：BGE-M3 稠密+稀疏混合检索（node_search_embedding）
    C2 +HyDE：基线 + HyDE 假设文档检索，两路结果 RRF 融合（node_rrf）
    C3 +重排：C2 结果经 Cross-Encoder 精排 + 断崖截断（node_rerank）
    命中判定：前 5 条结果中存在包含全部期望关键词的 chunk

二、实体对齐评测（对应节点 A 的双阈值三档策略）
    库内 12 个标准主体名 → 期望 top1 归一化得分 >= 0.8（自动确认）
    库外 8 个不存在的设备名 → 期望 top1 得分 < 0.6（正确拒答）
    检索 kb_item_names 集合，权重 (0.8, 0.2)，norm_score=True，与线上节点 A 一致

说明：检索评测直接驱动 B/C/E/F 节点，绕过节点 A（依赖 MongoDB 历史对话）
与节点 D（联网搜索）——联网结果无 chunk_id 不参与本地命中率，属合理简化。

运行方式（项目根目录下）：
    D:/python_lib/kb311/Scripts/python.exe eval/run_eval.py
"""
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from processor.query_processor.nodes.b_node_search_embedding import NodeSearchEmbedding
from processor.query_processor.nodes.c_node_search_embedding_hyde import NodeSearchEmbeddingHyde
from processor.query_processor.nodes.e_node_rrf import NodeRrf
from processor.query_processor.nodes.f_node_rerank import NodeRerank

from config.milvus_config import milvus_config
from utils.embedding_utils import generate_embeddings
from utils.milvus_utils import get_milvus_client, create_hybrid_search_requests, hybrid_search

EVAL_FILE = Path(__file__).parent / "eval_questions.json"
OUT_FILE = Path(__file__).parent / "eval_results.json"


# ---------- 第一部分：检索命中率 ----------

def contents_from_hits(hits):
    out = []
    for h in hits or []:
        try:
            out.append(h.entity.get("content"))
        except AttributeError:
            entity = h.get("entity") if isinstance(h, dict) else {}
            out.append((entity or {}).get("content"))
    return out


def is_hit(content, keywords):
    return bool(content) and all(k in content for k in keywords)


def run_question(node_b, node_c, node_rrf, node_rerank, q):
    state = {"item_names": [q["item_name"]], "rewritten_query": q["query"]}
    record = dict(q)

    # ---- C1 基线：混合检索 ----
    t0 = time.perf_counter()
    r1 = node_b.process(dict(state))
    t1 = time.perf_counter()
    base_docs = contents_from_hits(r1.get("embedding_chunks"))

    # ---- C2 +HyDE 两路召回 -> RRF 融合 ----
    r2 = node_c.process(dict(state))
    t2 = time.perf_counter()
    rrf_state = {
        "embedding_chunks": r1.get("embedding_chunks") or [],
        "hyde_embedding_chunks": r2.get("hyde_embedding_chunks") or [],
    }
    rrf_out = node_rrf.process(dict(rrf_state))
    t3 = time.perf_counter()
    rrf_chunks = rrf_out.get("rrf_chunks") or []
    hyde_docs = [d.get("content") for d in rrf_chunks]

    # ---- C3 +重排（含断崖截断） ----
    if rrf_chunks:
        rer_state = {"rewritten_query": q["query"], "rrf_chunks": rrf_chunks, "web_search_docs": []}
        rer_out = node_rerank.process(dict(rer_state))
        t4 = time.perf_counter()
        reranked = rer_out.get("reranked_docs") or []
    else:
        t4 = t3
        reranked = []
    rer_docs = [d.get("content") for d in reranked]
    rer_scores = [d.get("score") for d in reranked]

    keywords = q["expected_keywords"]
    record.update({
        "base_hit": any(is_hit(c, keywords) for c in base_docs[:5]),
        "hyde_hit": any(is_hit(c, keywords) for c in hyde_docs[:5]),
        "rerank_hit": any(is_hit(c, keywords) for c in rer_docs[:5]),
        "rerank_top1_score": round(rer_scores[0], 4) if rer_scores else None,
        "t_base_ms": round((t1 - t0) * 1000),
        "t_hyde_ms": round((t2 - t1) * 1000),
        "t_rrf_ms": round((t3 - t2) * 1000),
        "t_rerank_ms": round((t4 - t3) * 1000),
    })
    return record


# ---------- 压力模式：全库无过滤检索（不启用实体过滤，276 chunks 跨 12 本手册竞争） ----------

def hits_to_pairs(hits):
    """hit 对象列表 -> (content, item_name) 元组列表"""
    pairs = []
    for h in hits or []:
        try:
            pairs.append((h.entity.get("content"), h.entity.get("item_name")))
        except AttributeError:
            e = h.get("entity") if isinstance(h, dict) else {}
            pairs.append(((e or {}).get("content"), (e or {}).get("item_name")))
    return pairs


def run_question_global(node_rrf, node_rerank, client, llm, hyde_prompt_tpl, q):
    """全库检索版：直接用 milvus_utils 复刻节点 B/C 的检索（不带 expr），下游 RRF/重排用原节点"""
    query = q["query"]
    record = {"id": q["id"], "type": q["type"], "query": query}

    def search_no_filter(text):
        emb = generate_embeddings([text])
        reqs = create_hybrid_search_requests(
            dense_vector=emb["dense"][0], sparse_vector=emb["sparse"][0], limit=5)
        res = hybrid_search(client=client, collection_name=milvus_config.chunks_collection,
                            reqs=reqs, ranker_weights=(0.8, 0.2),
                            output_fields=["chunk_id", "content", "item_name"])
        return res[0] if res else []

    t0 = time.perf_counter()
    base_pairs = hits_to_pairs(search_no_filter(query))
    t1 = time.perf_counter()

    hyde_doc = llm.invoke(hyde_prompt_tpl.format(rewritten_query=query)).content
    hyde_pairs = hits_to_pairs(search_no_filter(query + "" + hyde_doc))
    t2 = time.perf_counter()

    rrf_out = node_rrf.process({
        "embedding_chunks": [{"entity": {"chunk_id": i, "content": c, "item_name": n}}
                             for i, (c, n) in enumerate(base_pairs)],
        "hyde_embedding_chunks": [{"entity": {"chunk_id": 1000 + i, "content": c, "item_name": n}}
                                  for i, (c, n) in enumerate(hyde_pairs)],
    })
    t3 = time.perf_counter()
    rrf_chunks = rrf_out.get("rrf_chunks") or []

    if rrf_chunks:
        rer_out = node_rerank.process({"rewritten_query": query, "rrf_chunks": rrf_chunks,
                                       "web_search_docs": []})
        t4 = time.perf_counter()
        reranked = rer_out.get("reranked_docs") or []
    else:
        t4, reranked = t3, []
    content_to_item = {d.get("content"): d.get("item_name") for d in rrf_chunks}
    rer_pairs = [(d.get("content"), content_to_item.get(d.get("content"))) for d in reranked]

    def hit(pairs):
        return any(is_hit(c, q["expected_keywords"]) and n == q["item_name"]
                   for c, n in pairs[:5])

    record.update({
        "base_hit": hit(base_pairs),
        "hyde_hit": hit([(d.get("content"), d.get("item_name")) for d in rrf_chunks]),
        "rerank_hit": hit(rer_pairs),
        "rerank_top1_score": round(reranked[0].get("score"), 4) if reranked else None,
        "t_base_ms": round((t1 - t0) * 1000),
        "t_hyde_ms": round((t2 - t1) * 1000),
        "t_rrf_ms": round((t3 - t2) * 1000),
        "t_rerank_ms": round((t4 - t3) * 1000),
    })
    return record


# ---------- 第二部分：实体对齐（节点A同款检索逻辑） ----------

def align_entity(client, name):
    """对单个名称在 kb_item_names 上做混合检索，返回 top1 得分与命中名"""
    emb = generate_embeddings([name])
    reqs = create_hybrid_search_requests(
        dense_vector=emb["dense"][0],
        sparse_vector=emb["sparse"][0],
        limit=5,
    )
    res = hybrid_search(
        client=client,
        collection_name=milvus_config.item_name_collection,
        reqs=reqs,
        ranker_weights=(0.8, 0.2),
        norm_score=True,
        output_fields=["item_name"],
    )
    hits = res[0] if res else []
    if not hits:
        return 0.0, None
    top = hits[0]
    score = top.get("distance")
    return float(score), top.entity.get("item_name")


def run_entity_eval(oov_entities):
    client = get_milvus_client()
    client.load_collection(milvus_config.item_name_collection)
    stored = [r["item_name"] for r in client.query(
        collection_name=milvus_config.item_name_collection,
        filter="pk >= 0", output_fields=["item_name"], limit=100)]

    results = {"in_kb": [], "oov": []}
    print("\n----- 实体对齐评测（阈值: >=0.8 确认 / 0.6~0.8 澄清 / <0.6 拒答）-----", flush=True)
    for name in stored:
        score, hit = align_entity(client, name)
        ok = score >= 0.8
        results["in_kb"].append({"name": name, "top1": round(score, 4), "hit": hit, "auto_confirm": ok})
        print(f"  库内 {name:<24} top1={score:.3f} {'√确认' if ok else '×未达确认'}", flush=True)
    for name in oov_entities:
        score, hit = align_entity(client, name)
        ok = score < 0.6
        results["oov"].append({"name": name, "top1": round(score, 4), "hit": hit, "reject": ok})
        print(f"  库外 {name:<24} top1={score:.3f} {'√拒答' if ok else '×误放行'}", flush=True)

    in_ok = sum(1 for r in results["in_kb"] if r["auto_confirm"])
    oov_ok = sum(1 for r in results["oov"] if r["reject"])
    summary = {
        "库内主体数": len(results["in_kb"]),
        "库内自动确认率": round(100 * in_ok / len(results["in_kb"]), 1),
        "库外实体数": len(results["oov"]),
        "库外正确拒答率": round(100 * oov_ok / len(results["oov"]), 1),
        "库外最高分": round(max(r["top1"] for r in results["oov"]), 3),
        "库内最低分": round(min(r["top1"] for r in results["in_kb"]), 3),
    }
    return summary, results


# ---------- 主流程 ----------

def main():
    data = json.loads(EVAL_FILE.read_text(encoding="utf-8"))
    questions = data["questions"]
    oov_entities = data.get("oov_entities", [])
    print(f"加载评测题 {len(questions)} 道（直问+改写），库外实体 {len(oov_entities)} 个", flush=True)

    node_b, node_c = NodeSearchEmbedding(), NodeSearchEmbeddingHyde()
    node_rrf, node_rerank = NodeRrf(), NodeRerank()
    from utils.llm_utils import get_llm_client
    from processor.query_processor.prompt.search_embedding_hyde import HYDE_PROMPT
    llm = get_llm_client()
    client = get_milvus_client()
    client.load_collection(milvus_config.chunks_collection)

    # ============ 模式一：线上链路（实体过滤，单手册范围内检索） ============
    print("=" * 60)
    print("模式一：线上链路评测（item_name 过滤，单手册范围）")
    print("=" * 60, flush=True)
    results_a = []
    for i, q in enumerate(questions, 1):
        rec = run_question(node_b, node_c, node_rrf, node_rerank, q)
        results_a.append(rec)
        print(f"[{i:>2}/{len(questions)}] ({q['type']}) {q['query'][:18]:<20}"
              f" 基线={'√' if rec['base_hit'] else '×'}"
              f" HyDE={'√' if rec['hyde_hit'] else '×'}"
              f" 重排={'√' if rec['rerank_hit'] else '×'}", flush=True)

    # ============ 模式二：压力模式（全库无过滤，12 手册 276 chunks 竞争） ============
    print("=" * 60)
    print("模式二：全库压力评测（无实体过滤，命中须来自正确手册）")
    print("=" * 60, flush=True)
    results_b = []
    for i, q in enumerate(questions, 1):
        rec = run_question_global(node_rrf, node_rerank, client, llm, HYDE_PROMPT, q)
        results_b.append(rec)
        print(f"[{i:>2}/{len(questions)}] ({q['type']}) {q['query'][:18]:<20}"
              f" 基线={'√' if rec['base_hit'] else '×'}"
              f" HyDE={'√' if rec['hyde_hit'] else '×'}"
              f" 重排={'√' if rec['rerank_hit'] else '×'}", flush=True)

    def summarize_mode(results):
        n = len(results) or 1
        out = {
            "命中率@5": {
                "C1_基线混合检索": round(100 * sum(1 for r in results if r["base_hit"]) / n),
                "C2_加HyDE_RRF": round(100 * sum(1 for r in results if r["hyde_hit"]) / n),
                "C3_加CrossEncoder重排": round(100 * sum(1 for r in results if r["rerank_hit"]) / n),
            },
            "延迟中位数ms": {
                "基线检索(embedding+混合检索)": int(statistics.median([r["t_base_ms"] for r in results])),
                "HyDE增路(LLM生成+检索+RRF)": int(statistics.median([r["t_hyde_ms"] + r["t_rrf_ms"] for r in results])),
                "CrossEncoder重排": int(statistics.median([r["t_rerank_ms"] for r in results])),
            },
        }
        by_type = {}
        for t in ("直问", "改写"):
            sub = [r for r in results if r["type"] == t]
            if sub:
                by_type[t] = {
                    "基线": round(100 * sum(1 for r in sub if r["base_hit"]) / len(sub)),
                    "重排": round(100 * sum(1 for r in sub if r["rerank_hit"]) / len(sub)),
                }
        out["分题型(基线vs重排)"] = by_type
        return out

    ent_summary, ent_results = run_entity_eval(oov_entities)

    summary = {
        "线上链路(实体过滤)": summarize_mode(results_a),
        "全库压力(无过滤)": summarize_mode(results_b),
        "实体对齐评测": ent_summary,
    }
    OUT_FILE.write_text(
        json.dumps({"summary": summary,
                    "mode_a_detail": results_a, "mode_b_detail": results_b,
                    "entity_detail": ent_results},
                   ensure_ascii=False, indent=2),
        encoding="utf-8")
    print("\n" + "=" * 60)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n明细已写入 {OUT_FILE}")


if __name__ == "__main__":
    main()
