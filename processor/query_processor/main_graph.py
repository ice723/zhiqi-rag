from dotenv import load_dotenv
from langgraph.constants import END
from langgraph.graph import StateGraph

from processor.query_processor.nodes.a_node_item_name_confirm import NodeItemNameConfirm
from processor.query_processor.nodes.b_node_search_embedding import NodeSearchEmbedding
from processor.query_processor.nodes.c_node_search_embedding_hyde import NodeSearchEmbeddingHyde
from processor.query_processor.nodes.d_node_web_search_mcp import NodeWebSearchMcp
from processor.query_processor.nodes.e_node_rrf import NodeRrf
from processor.query_processor.nodes.f_node_rerank import NodeRerank
from processor.query_processor.nodes.g_node_answer_output import NodeAnswerOutput
from processor.query_processor.state import QueryGraphState

load_dotenv()
class KBQueryWorkflow:
    #构造
    def __init__(self):
        print("初始化查询工作流...")
        #1.设置工作流状态
        self.workflow = StateGraph(QueryGraphState)
        #2.实例化节点
        self._init_nodes()
        #3.注册节点
        self._register_nodes()
        #4.设置路由（设置边）
        self._setup_routes()
        #5.编译图(用来编译后续存放的实例，因为未编译所以为空)
        self._compiled_app = None

    #调用
    def run(self, initial_state: QueryGraphState, stream: bool = False) -> QueryGraphState:
        print("调用图...")
        """
        统一执行入口，支持切换invoke/stream
        """
        if not self._compiled_app:
            self.compile()
        if stream:
            return self._compiled_app.stream(initial_state)
        else:
            return self._compiled_app.invoke(initial_state)


    #实例化节点
    def _init_nodes(self):
        print("实例化所有节点...")
        self.a_node_item_name_confirm = NodeItemNameConfirm()
        self.b_node_search_embedding = NodeSearchEmbedding()
        self.c_node_search_embedding_hyde = NodeSearchEmbeddingHyde()
        self.d_node_web_search_mcp = NodeWebSearchMcp()
        self.e_node_rrf = NodeRrf()
        self.f_node_rerank = NodeRerank()
        self.g_node_answer_output = NodeAnswerOutput()

    #注册节点
    def _register_nodes(self):
        print("注册所有节点...")
        self.workflow.add_node("a_node_item_name_confirm", self.a_node_item_name_confirm)
        self.workflow.add_node("b_node_search_embedding", self.b_node_search_embedding)
        self.workflow.add_node("c_node_search_embedding_hyde", self.c_node_search_embedding_hyde)
        self.workflow.add_node("d_node_web_search_mcp", self.d_node_web_search_mcp)
        self.workflow.add_node("e_node_rrf", self.e_node_rrf)
        self.workflow.add_node("f_node_rerank", self.f_node_rerank)
        self.workflow.add_node("g_node_answer_output", self.g_node_answer_output)

    # 设置路由
    def _setup_routes(self):
        print("设置路由规则...")
        #设置入口(两种写法取其一)
        self.workflow.set_entry_point("a_node_item_name_confirm")
        #self.workflow.add_node(START, "a_node_item_name_confirm")

        #设置条件边
        self.workflow.add_conditional_edges(
            "a_node_item_name_confirm",
            self._route_after_item_name_confirm,
            {
                "b_node_search_embedding": "b_node_search_embedding",
                "c_node_search_embedding_hyde": "c_node_search_embedding_hyde",
                "d_node_web_search_mcp": "d_node_web_search_mcp",
                "g_node_answer_output": "g_node_answer_output"
            }
        )

        #注册条件边
        self.workflow.add_edge("b_node_search_embedding", "e_node_rrf")
        self.workflow.add_edge("c_node_search_embedding_hyde", "e_node_rrf")
        self.workflow.add_edge("d_node_web_search_mcp", "e_node_rrf")

        self.workflow.add_edge("e_node_rrf", "f_node_rerank")
        self.workflow.add_edge("f_node_rerank", "g_node_answer_output")
        self.workflow.add_edge("g_node_answer_output", END)

    #定义哪条边的条件
    def _route_after_item_name_confirm(self, state: QueryGraphState) ->str:
        print("路由-条件...")
        if state.get("answer"):
            return "g_node_answer_output"
        return ["b_node_search_embedding", "c_node_search_embedding_hyde", "d_node_web_search_mcp"]

    #编译图
    def compile(self):
        print("编译图...")
        if not self._compiled_app:
            self._compiled_app = self.workflow.compile()
        return self._compiled_app

if __name__ == '__main__':
    #stream是惰性方法，不调用就不执行 用for循环调用
    workflow = KBQueryWorkflow()
    response = workflow.run({"original_query": "哥们儿，奔图P3500黑白激光打印机这个东东咋鼓捣上啊？", "session_id": "123"},stream=True)
    for res in response:
        print(res)








