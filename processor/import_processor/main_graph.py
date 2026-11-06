import json
import logging

from langgraph.constants import END
from langgraph.graph import StateGraph

from processor.import_processor.base import setup_logging
from processor.import_processor.nodes.a_node_entry import NodeEntry
from processor.import_processor.nodes.b_node_pdf_to_md import NodePDFToMD
from processor.import_processor.nodes.c_node_md_img import NodeMDImg
from processor.import_processor.nodes.d_node_document_split import NodeDocumentSplit
from processor.import_processor.nodes.e_node_item_name_recognition import NodeItemNameRecognition
from processor.import_processor.nodes.f_node_bge_embedding import NodeBGEEmbedding
from processor.import_processor.nodes.g_node_import_milvus import NodeImportMilvus
from processor.import_processor.state import ImportGraphState


class KBImportWorkflow:

    def __init__(self,config=None):
        self._compiled_graph =None
    @property # propert是将一个方法当作一个属性来用
    def graph(self):
        """
        返回图实例
        """
        logging.info("获取图实例") #logging是用来开发级别的专业日志
        if self._compiled_graph is None:
            self._compiled_graph = self.build_graph() #创建一个图
        return self._compiled_graph

    @staticmethod  #把这个函数标记成**类里面的静态方法
    def route_after_entry(state:ImportGraphState): #ImportGraphState是自定义的名字，专门用来定义这张状态图在执行过程里，**全程要携带哪些数据字段**
        if state.get("is_pdf_read_enabled"):
            return "b_node_pdf_to_md"
        elif state.get("is_md_read_enabled"):
            return "c_node_md_img"
        else:
            return END
    def build_graph(self):
        """
        创建主图
        """
        graph = StateGraph(ImportGraphState) #新建一张 LangGraph 主图，并且规定这张图在运行全过程中，使用 `ImportGraphState` 作为全局状态容器
        #1.注册节点
         #定义节点
        graph.add_node("a_node_entry",NodeEntry())
        graph.add_node("b_node_pdf_to_md",NodePDFToMD())
        graph.add_node("c_node_md_img",NodeMDImg())
        graph.add_node("d_node_document_split",NodeDocumentSplit())
        graph.add_node("e_node_item_name_recognition",NodeItemNameRecognition())
        graph.add_node("f_node_bge_embedding",NodeBGEEmbedding())
        graph.add_node("g_node_import_milvus",NodeImportMilvus())
         #定义节点入口
        graph.set_entry_point("a_node_entry")
        #2.节点边
         #条件边
        graph.add_conditional_edges(
            "a_node_entry",
        self.route_after_entry,  # 自定义一个工具方法用做判断
        {
            "c_node_md_img": "c_node_md_img",
            "b_node_pdf_to_md": "b_node_pdf_to_md",
            END: END  # 是langgraph的包
        }
        )
        # ============ 这里新增顺序边 ============
        graph.add_edge("b_node_pdf_to_md", "c_node_md_img")
        graph.add_edge("c_node_md_img", "d_node_document_split")
        graph.add_edge("d_node_document_split", "e_node_item_name_recognition")
        graph.add_edge("e_node_item_name_recognition", "f_node_bge_embedding")
        graph.add_edge("f_node_bge_embedding", "g_node_import_milvus")
        graph.add_edge("g_node_import_milvus", END)

        #3.编译图
        graph_compile = graph.compile()
        return graph_compile
    def run(self,state:ImportGraphState,stream:bool=False):
        if stream:
            # updates 模式返回 {节点名: 该节点更新的字段}，前端才能显示节点级进度；
            # values 模式返回的是整个 state，key 是字段名，进度会错乱
            return self.graph.stream(state,stream_mode="updates")
        else:
            return self.graph.invoke(state)

if __name__ == '__main__':
    #启用日志
    setup_logging()
    workflow = KBImportWorkflow()
    inint_state = {"import_file_path":r"D:\BaiduNetdiskDownload\hak180产品安全手册.pdf"}
    #方式一，流式输出
    for envent in workflow.run(inint_state,stream=True):
        print(f"state:{envent}")
    #方式二，非流式输出
    # final_state = workflow.run(inint_state,stream=False)
    # print(json.dumps(final_state,ensure_ascii=False,indent=4))
