"""百炼 MCP（联网搜索）配置。

MCP 走 DashScope 的 Key，与对话模型 Key 解耦。
"""

from dataclasses import dataclass

from config.env_loader import env_bool, env_str


@dataclass
class McpConfig:
    mcp_base_url: str
    api_key: str
    enabled: bool


mcp_config = McpConfig(
    mcp_base_url=env_str(
        "MCP_DASHSCOPE_BASE_URL",
        default="https://dashscope.aliyuncs.com/api/v1/mcps/WebSearch/mcp",
    ),
    api_key=env_str("DASHSCOPE_API_KEY", "OPENAI_API_KEY"),
    enabled=env_bool("WEB_SEARCH_ENABLED", default=True),
)
