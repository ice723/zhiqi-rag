

from config.milvus_config import milvus_config
from processor.query_processor.base import NodeBase
from processor.query_processor.prompt.search_embedding_hyde import HYDE_PROMPT
from processor.query_processor.state import QueryGraphState
from tool.logger import logger
from utils.embedding_utils import generate_embeddings
from utils.llm_utils import get_llm_client
from utils.milvus_utils import (
    build_item_name_expr,
    create_hybrid_search_requests,
    get_milvus_client,
    hybrid_search,
)


class NodeSearchEmbeddingHyde(NodeBase):
    """
    节点功能：HyDE (Hypothetical Document Embedding)
    先让 LLM 生成假设性答案，再对答案进行向量检索，提高召回率。
    """

    # 覆盖基类的 name 属性，标识节点名称
    name: str = "node_search_embedding_hyde"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        """
        节点逻辑
        :param state: 工作流状态对象
        :return: 更新后的状态对象
        """

        # TODO
        logger.info(f"【{self.name}】节点逻辑")
        #1.参数处理
        item_names = state.get("item_names")  # 命中的设备名
        rewritten_query = state.get("rewritten_query")  # 重写的问题

        if not rewritten_query:
            return {"hyde_embedding_chunks": [], "hyde_doc": ""}

        #2.假设性文档生成
        hyde_doc = self._step_1_creat_embedding_hyde_doc(rewritten_query)

        #3.用“重写的问题+假设性回答”搜索向量文本块
        response = self._setp_2_search_embedding_hyde(rewritten_query, hyde_doc, item_names)

        #4.结果解析（并行节点只返回自己负责的字段）
        return {
            "hyde_embedding_chunks": response[0] if response else [],
            "hyde_doc": hyde_doc,
        }

    def _step_1_creat_embedding_hyde_doc(self,rewritten_query):
        print("步骤1：基于大模型生成假设性文档")
        #1.连接模型
        ai_client = get_llm_client()

        #2.构建提示词
        hyde_prompt = HYDE_PROMPT.format(rewritten_query=rewritten_query) #.format用来填写提示词的空缺

        #3.假设性回答
        hyde_doc = ai_client.invoke(hyde_prompt).content

        return hyde_doc

    def _setp_2_search_embedding_hyde(self,rewritten_query,hyde_doc,item_names):
        print("步骤2：根据假设性文档向量搜索")
        #1.拼接问题+假设性回答文本
        embedding_context = f"{rewritten_query}\n{hyde_doc}".strip()
        expr = build_item_name_expr(item_names)

        #2.文本向量化（仅稠密；BM25 由 Milvus 服务端生成）
        embeddings = generate_embeddings([embedding_context])
        dense_vector = embeddings["dense"][0]

        #3.向量搜索
        milvus_client = get_milvus_client()
        collection_name = milvus_config.chunks_collection
        reqs = create_hybrid_search_requests(
            dense_vector=dense_vector,
            query_text=embedding_context,
            expr=expr,
        )
        response = hybrid_search(
            client=milvus_client,
            collection_name=collection_name,
            reqs=reqs,
            limit=5,
            output_fields=["chunk_id", "content", "item_name"],
            dense_vector=dense_vector,
            expr=expr,
        )
        return response

if __name__ == "__main__":
    node_search_embedding_hyde = NodeSearchEmbeddingHyde()
    init_state = {
        "item_names": ["兄弟HAK180烫金机", "百度一下"],
        "rewritten_query": "兄弟请帮我查一下HAK180烫金机是什么?"
    }
    process = node_search_embedding_hyde.process(init_state)
    # print(json.dumps(process, ensure_ascii=False, indent=4))