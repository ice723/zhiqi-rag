import asyncio
import json

from agents.mcp import MCPServerStreamableHttp

from config.bailian_mcp_config import mcp_config
from processor.query_processor.base import NodeBase
from processor.query_processor.state import QueryGraphState
from tool.logger import logger


class NodeWebSearchMcp(NodeBase):
    """
    节点功能，调用外部搜索引擎补充信息
    """
    # 覆盖基类的 name 属性，标识节点名称
    name: str = "node_web_search_mcp"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        """
        节点逻辑
        """
        logger.info(f"【{self.name}】节点逻辑")
        #1.参数处理
        query = state.get("rewritten_query")
        docs = []

        if not mcp_config.enabled:
            logger.info("联网搜索已关闭（WEB_SEARCH_ENABLED=0），跳过")
            return {"web_search_docs": []}
        if not query:
            return {"web_search_docs": []}

        #2.调用外部搜索引擎：失败不影响本地知识库问答
        try:
            result = asyncio.run(self._mcp_call(query))
            json_data = result.content[0].text
            result_json_obj = json.loads(json_data)
            pages = result_json_obj.get("pages") or []
            for page in pages:
                docs.append({
                    "snippet": page.get("snippet"),
                    "url": page.get("url"),
                    "title": page.get("title"),
                })
        except Exception as exc:  # noqa: BLE001
            logger.error(f"联网搜索失败，已跳过：{exc}")

        #3.并行节点只返回「自己负责的那个字段」
        return {"web_search_docs": docs}

    async def _mcp_call(self, query):
        print("利用python库来发送mcp请求")
        #1.构建阿里平台mcp的请求工具
        search_mcp_tool = MCPServerStreamableHttp(
            name="search_mcp",
            params={
                "url": mcp_config.mcp_base_url,
                "headers": {"Authorization": f"Bearer {mcp_config.api_key}"}, #联网的mcp工具要求的调用方式
                "timeout": 10 #最大http请求超时时间
            },
            cache_tools_list=True,  #缓存设置是否开启，开启缓存速度会快一点
            max_retry_attempts=3  #设置最大尝试次数
        )
        #2.建立mcp服务连接，拉取工具列表
        await search_mcp_tool.connect()
        try:
            #执行搜索请求
            result = await search_mcp_tool.call_tool(
                tool_name="bailian_web_search", #这是百炼平台的具体mcp工具
                arguments={"query": query, "count": 3} #count是设置返回的条数
            )
            return result
        finally:
            #释放连接，清理资源
            await search_mcp_tool.cleanup()

if __name__ == '__main__':
    init_state = {
        "rewritten_query": "关于brother HAK180烫金机，如何调节转印温度？",
        "session_id": "123"
    }
    # 执行节点的业务调用
    node_web_search_mcp = NodeWebSearchMcp()
    result = node_web_search_mcp(init_state)
        