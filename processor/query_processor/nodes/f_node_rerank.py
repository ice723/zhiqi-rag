from typing import List, Dict, Any

from processor.query_processor.base import NodeBase
from processor.query_processor.state import QueryGraphState
from tool.logger import logger
from utils.reranker_http_utils import rerank_documents


class NodeRerank(NodeBase):
    """
    节点功能：使用 Cross-Encoder 模型对 RRF 后的结果进行精确打分重排。
    """

    # 覆盖基类的 name 属性，标识节点名称
    name: str = "node_rerank"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        """
        节点逻辑
        """
        logger.info(f"【{self.name}】节点逻辑")
        #1.参数合并(web_search_docs和rrf_chunks)
        merged_multi_docs: List[Dict[str, Any]] = self._step_1_merged_multi_sroce_docs(state)
        # print(merged_multi_docs)
        #2.调用模型重排序
        reranked_docs: List[Dict[str,Any]] = self._step_2_rerank_merged_docs(state,merged_multi_docs)

        #3.断崖测试(去掉过低的返回结果)
        cutoff_docs = self._step_3_cliff_cutoff(reranked_docs)

        #4.只返回本节点负责的字段
        print("步骤4，返回结果封装")
        return {"reranked_docs": cutoff_docs}

    def _step_1_merged_multi_sroce_docs(self, state):
        print("步骤1：将web_doc和rrf_chunks节点的数据合并")
        final_docs = []
        web_docs = state.get("web_search_docs") or []
        rrf_chunks = state.get("rrf_chunks") or []

        #统一两个的格式添加到final_docs里；空内容无法参与重排，直接跳过
        for rrf_doc in rrf_chunks:
            content = (rrf_doc.get("content") or "").strip()
            if not content:
                continue
            final_docs.append({
                "content": content,
                "title": rrf_doc.get("title"),
                "chunk_id": rrf_doc.get("chunk_id"),
                "url": None,
                "source": "local"
            })

        for web_doc in web_docs:
            content = (web_doc.get("snippet") or "").strip()
            if not content:
                continue
            final_docs.append({
                "content": content,
                "title": web_doc.get("title"),
                "chunk_id": None,
                "url": web_doc.get("url"),
                "source": "web"
            })
        return final_docs

    def _step_2_rerank_merged_docs(self, state, merged_multi_docs):
        print("步骤2：调用排序模型对合并后结果进行排序")
        if not merged_multi_docs:
            return []

        #原始问题(经过大模型加工的)
        rewritten_query = state.get("rewritten_query") or state.get("original_query") or ""
        #结合的问题
        contents = [doc.get("content") for doc in merged_multi_docs]
        #调用排序模型拿到相关文本的分数
        rerank_scores = rerank_documents(rewritten_query, contents)
        #把分数添加到原本数据列表中
        scored_docs = [{**doc, "score": score} for doc, score in zip(merged_multi_docs, rerank_scores)]
        #等价于
        # scored_docs = []
        # for doc, score in zip(merged_multi_docs, rerank_scores):
        #     scored_docs.append({
        #         "content": doc.get("content"),
        #         "title": doc.get("title"),
        #         "chunk_id": doc.get("chunk_id"),
        #         "url": doc.get("url"),
        #         "source": doc.get("source"),
        #         "score": float(score),
        #     })

        #结果排序
        sorted_scores = sorted(scored_docs, key=lambda x: x["score"], reverse=True)
        return sorted_scores

    def _step_3_cliff_cutoff(self, reranked_docs):
        print("步骤3：断崖测试(分数差距，分数比率)去掉过低的匹配结果")
        if not reranked_docs:
            return []
        #设置最多要几条数据
        upper_bound = min(10, len(reranked_docs))
        #设置最少要几条数据
        lower_bound = min(3, len(reranked_docs))
        #设置断崖位置（默认全部数据都保存）
        cutoff_pos = upper_bound
        for index in range(lower_bound-1, upper_bound-1):
            current_doc = reranked_docs[index].get("score")
            next_doc = reranked_docs[index+1].get("score")
            #设置断崖位置前移的条件
            #条件一：分差
            abc_gap = current_doc - next_doc
            #条件二：比率 （前一个人跟后一个人的分差除以前一个人的分数）
            rel_gap = abc_gap / (current_doc+ 1e-6) #1e-6 表示十的负六次方
            if abc_gap >= 0.5 or rel_gap >= 0.25:
                cutoff_pos = index +1 #确定在此位置断崖了
                break
            #返回断崖前的数据
        return reranked_docs[:cutoff_pos]



if __name__ == "__main__":
    mock_state = {
        "rewritten_query": "怎么测这块主板的短路问题？",
        "rrf_chunks": [
            {
                "chunk_id": "local_1",
                "title": "主板维修手册",
                "content": "主板短路通常表现为通电后风扇转一下就停，可以使用万用表的蜂鸣档测量。"
            },
            {
                "chunk_id": "local_2",
                "title": "闲聊",
                "content": "今天中午去吃猪脚饭吧，这块主板外观很漂亮。"
            },
        ],
        "web_search_docs": [
            {
                "url": "https://example.com/repair",
                "title": "短路查修指南",
                "snippet": "主板通电前先打各主供电电感对地阻值，阻值偏低就是短路。"
            },
            {
                "url": "https://example.com/news",
                "title": "科技新闻",
                "snippet": "苹果发布新款手机，A系列芯片性能提升20%。"
            },
        ],
    }

    node_rerank = NodeRerank()
    result = node_rerank(mock_state)
    logger.info(result)


            
