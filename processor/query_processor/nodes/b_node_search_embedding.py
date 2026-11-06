import json


from config.milvus_config import milvus_config
from processor.query_processor.base import NodeBase
from processor.query_processor.state import QueryGraphState
from tool.logger import logger
from utils.embedding_utils import generate_embeddings
from utils.milvus_utils import (
    build_item_name_expr,
    create_hybrid_search_requests,
    get_milvus_client,
    hybrid_search,
)


class NodeSearchEmbedding(NodeBase):
    """
   节点功能：基于已确认主体名+改写后的用户问题，执行Milvus向量数据库混合检索
   """

    # 覆盖基类的 name 属性，标识节点名称
    name: str = "node_search_embedding"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        """
        节点逻辑
        """
        #1.参数处理
        logger.info(f"【{self.name}节点执行逻辑】")
        item_names = state.get("item_names") #milvus元数据
        query = state.get("rewritten_query") #语义搜索条件
        if not query:
            return {"embedding_chunks": []}

        query_embedding = generate_embeddings([query]) #语义搜索向量化
        dense_vector = query_embedding["dense"][0]

        #2.连接milvus客户端
        milvus_client = get_milvus_client()  #连接单例
        chunks_collection = milvus_config.chunks_collection #获取表名

        #3.构建请求体：稠密向量 + BM25（传原始查询文本）
        reqs = create_hybrid_search_requests(
            dense_vector=dense_vector,
            query_text=query,
            expr=build_item_name_expr(item_names),  #expr: 搜索过滤表达式，用于精准筛选数据
        )

        #4.发送请求
        response = hybrid_search(
            client=milvus_client,
            collection_name=chunks_collection,
            reqs=reqs,
            ranker_weights=(0.8,0.2),
            output_fields=["chunk_id", "content", "item_name"],
            dense_vector=dense_vector,
            expr=build_item_name_expr(item_names),
        )

        # TODO 并行节点只返回「自己负责的那个字段」，不能整份 state 返回
        return {"embedding_chunks": response[0] if response else []}


if __name__ == '__main__':
    node = NodeSearchEmbedding()
    init_state = {
        "item_names": ["兄弟HAK180烫金机", "百度一下"],
        "rewritten_query": "兄弟请帮我查一下HAK180烫金机是什么?"
    }
    process = node.process(init_state)
    print(json.dumps(process[0],ensure_ascii=False , indent=4))