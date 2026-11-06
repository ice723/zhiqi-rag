import json
from typing import Dict, List

from langchain_core.messages import SystemMessage, HumanMessage

from config.milvus_config import milvus_config
from processor.query_processor.base import NodeBase
from processor.query_processor.prompt.item_name_confirm import ITEM_NAME_EXTRACT_TEMPLATE, \
    ITEM_NAME_EXTRACT_SYSTEM_PROMPT
from processor.query_processor.state import QueryGraphState
from tool.logger import logger
from utils.embedding_utils import generate_embeddings
from utils.llm_utils import invoke_json
from utils.milvus_utils import (
    create_hybrid_search_requests,
    flatten_hits,
    get_milvus_client,
    hybrid_search,
)
from utils.mongo_history_utils import get_recent_messages, save_chat_message, update_message_item_names


class NodeItemNameConfirm(NodeBase):
    """
    节点功能：确认用户问题中的核心商品名称。
    """
    name: str = "node_item_name_confirm"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        logger.info(f"【{self.name}】节点逻辑")
        # 1.参数校验 (session_id:主键)
        session_id, original_query = self._step_1_validate_param(state)

        # 2.获取历史对话记录（在保存本轮用户消息之前取，避免重复）
        history = get_recent_messages(session_id)
        state["history"] = history

        # 3.保存用户对话信息
        message_id = save_chat_message(session_id, "user", original_query)

        # 4.大模型主体意图识别
        extract_res = self._step_4_extract_info(original_query, history)
        item_names = extract_res["item_names"]
        rewritten_query = extract_res["rewritten_query"]
        state["rewritten_query"] = rewritten_query
        state["item_names"] = item_names

        # 5/6.向量检索 + 结果对齐
        align_result: Dict[str, List[str]] = {}
        if item_names:
            query_result = self._step_5_vectorize_and_query(item_names)
            align_result = self._step_6_align_item_names(query_result)
        else:
            logger.info("Node: 未提取到商品名，跳过向量检索")

        # 7.信息写入 state
        state = self._step_7_check_confirmation(state, align_result, history)

        # 8.保存对话信息
        self._step_8_write_history(state, session_id, rewritten_query, message_id)

        return state

    def _step_1_validate_param(self, state):
        print("step_1: 参数校验")
        session_id = state.get("session_id")
        if not session_id:
            raise ValueError("核心参数session_id缺失")
        original_query = state.get("original_query")
        if not original_query:
            raise ValueError("核心original_query缺失")
        return session_id, original_query

    def _step_4_extract_info(self, original_query, history):
        print("step_4: 模型提取意图主体")
        # 拼接历史：Mongo 中正文字段是 text（旧代码误用了 content）
        history_text = ""
        for msg in history:
            role = msg.get("role", "")
            content = msg.get("text", "")
            if content:
                history_text += f"{role}: {content}\n"

        user_prompt = ITEM_NAME_EXTRACT_TEMPLATE.format(
            history_text=history_text,
            query=original_query,
        )
        messages = [
            SystemMessage(content=ITEM_NAME_EXTRACT_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt)
        ]

        # JSON 模式调用 + 容错解析（模型偶尔会带 ``` 围栏）
        result = invoke_json(messages, fallback={"item_names": [], "rewritten_query": original_query})

        if not isinstance(result.get("item_names"), list):
            result["item_names"] = []
        if not result.get("rewritten_query"):
            result["rewritten_query"] = original_query

        cleaned_names = []
        for name in result["item_names"]:
            if not isinstance(name, str):
                continue
            name = name.replace(" ", "").replace("\n", "").replace("\t", "").replace("\r", "").strip()
            if name:
                cleaned_names.append(name)
        result["item_names"] = cleaned_names
        return result

    def _step_5_vectorize_and_query(self, item_names):
        print("step_5: 向量化并检索商品名称")
        results: List[Dict] = []

        milvus_client = get_milvus_client()
        collection_name = milvus_config.item_name_collection
        embeddings = generate_embeddings(item_names)

        for i, name in enumerate(item_names):
            dense_vector = embeddings["dense"][i]
            # 稀疏一路由 Milvus 的 BM25 生成，直接传原始文本
            reqs = create_hybrid_search_requests(
                dense_vector=dense_vector,
                query_text=name,
                limit=5,
            )
            search_res = hybrid_search(
                client=milvus_client,
                collection_name=collection_name,
                reqs=reqs,
                ranker_weights=(0.8, 0.2),
                norm_score=True,
                output_fields=["item_name"],
                dense_vector=dense_vector,
            )

            matches = [
                {
                    "item_name": hit.get("item_name"),
                    "score": hit.get("distance", 0.0),
                }
                for hit in flatten_hits(search_res)
                if hit.get("item_name")
            ]

            results.append({
                "extracted_name": name,
                "matches": matches,
            })

        return results

    def _step_6_align_item_names(self, query_result: List[Dict]) -> Dict:
        print("step_6: 对齐结果,高分结果和低分结果整合")
        confirmed_item_names: List[str] = []
        options: List[str] = []

        for res in query_result:
            extracted_name = (res.get("extracted_name") or "").strip()
            matches = res.get("matches") or []
            if not matches:
                continue

            high_results = [m for m in matches if m.get("score", 0) >= 0.8]
            mid_results = [m for m in matches if 0.8 > m.get("score", 0) >= 0.6]

            if len(high_results) == 1:
                confirmed_item_names.append(high_results[0].get("item_name"))
                continue

            if len(high_results) > 1:
                picked = None
                if extracted_name:
                    for hr in high_results:
                        if hr.get("item_name") == extracted_name:
                            picked = hr
                            break
                if not picked:
                    picked = high_results[0]
                confirmed_item_names.append(picked.get("item_name"))
                continue

            if mid_results:
                for mr in mid_results[:5]:
                    options.append(mr.get("item_name"))

        return {
            "confirmed_item_names": list(dict.fromkeys(confirmed_item_names)),
            "options": list(dict.fromkeys(options)),
        }

    def _step_7_check_confirmation(self, state, align_result, history):
        print("step_7: 状态state信息,根据第六步高分低分对齐结果整理")
        confirmed = align_result.get("confirmed_item_names") or []
        options = align_result.get("options") or []

        if confirmed:
            # 历史里 item_names 为空的用户消息，补上本次确认的商品名
            ids_to_update = []
            for msg in history:
                if not msg.get("item_names"):
                    msgid = msg.get("_id")
                    if msgid:
                        ids_to_update.append(str(msgid))
            if ids_to_update:
                update_message_item_names(ids_to_update, confirmed)

            state["item_names"] = confirmed
            state["answer"] = ""
            return state

        if options:
            state["item_names"] = []
            options_str = "、".join(options)
            state["answer"] = f"先生，你是想想问{options_str}这里面产品的哪一个产品，请明确一下产品型号重新输入喔"
            return state

        state["item_names"] = []
        state["answer"] = "抱歉，未找到相关产品，请提供准确型号以便我为您查询。"
        return state

    def _step_8_write_history(self, state, session_id, rewritten_query, message_id):
        print("step_8: 写入历史会话，更新")
        # 先落用户消息、再落澄清回复：save_chat_message 每次都重取 ts，
        # 顺序反了会让历史里「回复」排在「提问」前面。
        save_chat_message(
            session_id=session_id,
            role="user",
            text=state.get("original_query"),
            rewritten_query=rewritten_query,
            item_names=state.get("item_names", []),
            message_id=message_id,
        )

        # 只有本节点直接给出答案（澄清/未找到）时才写 assistant 记录；
        # 正常问答的答案由 g_node_answer_output 统一写入，避免重复
        if state.get("answer"):
            save_chat_message(
                session_id=session_id,
                role="assistant",
                text=state["answer"],
                rewritten_query="",
                item_names=state.get("item_names", []),
            )


if __name__ == '__main__':
    init_state = {
        "original_query": "hak180这个玩意咋鼓捣呀？",
        "session_id": "123"
    }
    node = NodeItemNameConfirm()
    node(init_state)
