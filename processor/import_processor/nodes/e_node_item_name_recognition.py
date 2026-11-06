# processor/import_processor/nodes/node_item_name_recognition.py
import json
from pathlib import Path
from typing import List, Dict

from langchain_core.messages import SystemMessage, HumanMessage


from config.lm_config import lm_config
from config.milvus_config import milvus_config
from processor.import_processor.base import BaseNode
from processor.import_processor.exceptions import StateFieldError
from processor.import_processor.state import ImportGraphState
from processor.import_processor.import_config import get_config
from utils.embedding_utils import generate_embeddings
from utils.llm_utils import get_llm_client
from utils.milvus_utils import (
    ensure_item_name_collection,
    escape_milvus_string,
    get_milvus_client,
)


class NodeItemNameRecognition(BaseNode):
    """
    主体识别节点：主体识别与标签提取
    """

    name = "node_item_name_recognition"

    def __init__(self):
        super().__init__()
        self.config = get_config()

    def process(self, state: ImportGraphState):
        #1.参数处理
        file_title, chunks = self._step_1_get_inputs(state)

        #2.上下文拼接
        context = self._step_2_build_context(file_title, chunks)

        #3.模型识别（总结）
        item_name = self._step_3_call_llm(file_title, context)
        print(f"item_name: {item_name}")

        #4.回填数据（item_name -> chunks）
        self.step_4_update_chunks(state, chunks, item_name)
        path = f'{Path(state.get("md_path")).parent}/{state.get("file_title")}_new_new_chunks.json'
        with open(path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=4) #chunks是方法里面已经有的列表，添加chunks列表中

        #5.主体名称向量化（仅稠密向量；稀疏一路由 Milvus BM25 生成）
        dense_vector = self._step_5_generate_vectors(item_name)

        #6.存入milvus向量数据库
        state = self._step_6_save_to_milvus(state, file_title, item_name, dense_vector)
        return state

    def _step_1_get_inputs(self, state) ->(str, List[Dict]):
        print("node_item_name_recognition: 步骤1：参数处理")
        file_title = state.get("file_title")
        if not file_title:
            raise StateFieldError(field_name="file_title", message="文件标题不能为空", expected_type=str)

        chunks = state.get("chunks")
        if not chunks:
            raise StateFieldError(field_name="chunks", message="chunks不能为空", expected_type=list)

        return file_title, chunks

    def _step_2_build_context(self, file_title, chunks: List[Dict]) ->str:
        print("node_item_name_recognition: 步骤2：上下文拼接")
        #上下文限制的片数
        k = self.config.item_name_chunk_k
        #上下文限制的切片长度
        chunk_size = self.config.item_name_chunk_size

        parts: List[str] = []
        total_chars = 0
        for index, chunk in enumerate(chunks[:k], start=1): # enumerate方法，同时拿到下标跟元素
            chunk_title = chunk.get("title", "").strip()
            chunk_content = chunk.get("content", "").strip()

            #格式化
            piece = f"【切片{index}】\n标题：{chunk_title}\n内容：{chunk_content}"
            parts.append(piece)
            #计算长度
            total_chars += len(piece)
            #检测长度
            if total_chars >= chunk_size:
                break

        #截断处理
        content = "\n\n".join(parts).strip()
        final_content = content[:chunk_size]
        return final_content

    def _step_3_call_llm(self,file_title, context):
        print("node_item_name_recognition: 步骤3：模型识别")
        if not context:
            return file_title
        #调用模型（ITEM_MODEL 可单独指定更便宜的抽取模型）
        llm_ai = get_llm_client(lm_config.item_model)
        #提示词
        prompt = f"""
                请从以下信息中识别出商品名称与型号：
                文件名：{file_title}
                
                正文切片（用于辅助识别）：
                {context}
                
                要求：
                1. 返回内容为字符串形式，最好是带品牌、型号和名称的完整商品名称。比如：苏伯尓5000W大功率电磁炉；
                2. 返回结果应该只包含商品名称，不要添加任何解释或其他内容；
                3. 如果无法识别商品名称,请返回空字符串。
        """
        message = [
            SystemMessage("你是一个专业的商品名称识别模型，请根据提供的信息，识别商品名称。名称最好不要超过20个字"),
            HumanMessage(content=prompt)
        ]
        #调用
        response = llm_ai.invoke(message)
        #解析，数据清洗
        item_name = getattr(response, "content", "").strip()
        item_name = item_name.replace(" ", "").replace("\n", "").replace("\t", "").replace("\r", "")
        #兜底
        if not item_name:
            item_name = file_title
        #返回
        return item_name

    def step_4_update_chunks(self,state, chunks, item_name):
        print("node_item_name_recognition: 步骤4：回填数据")
        state["item_name"] = item_name
        for chunk in chunks:
            chunk["item_name"] = item_name
        return state

    def _step_5_generate_vectors(self, item_name):
        print("node_item_name_recognition: 步骤5：主体名称向量化（稠密）")
        embeddings = generate_embeddings([item_name])
        return embeddings["dense"][0]

    def _step_6_save_to_milvus(self, state, file_title, item_name, dense_vector):
        print("node_item_name_recognition: 步骤6：保存到milvus")
        if not milvus_config.milvus_url or not milvus_config.item_name_collection:
            raise Exception("Milvus配置错误")

        client = get_milvus_client()
        ensure_item_name_collection(client, dim=len(dense_vector))

        # 幂等性清理同名（item_name）数据
        safe_item_name = escape_milvus_string(item_name)
        client.delete(
            collection_name=milvus_config.item_name_collection,
            filter=f'item_name=="{safe_item_name}"',
        )

        # 稀疏向量由 BM25 Function 生成，不能手动传入
        data = {
            "file_title": file_title,
            "item_name": item_name,
            "dense_vector": dense_vector,
        }
        client.insert(milvus_config.item_name_collection, [data])

        state["item_name"] = item_name
        return state


if __name__ == '__main__':
    node=NodeItemNameRecognition()
    path = r"D:\output\PantumP3500用户手册zh_CNV1.2_1644316283788\PantumP3500用户手册zh_new_chunk.json"
    with open(path, "r", encoding="utf-8")as f:
        chunks_json_data = f.read()

    init_state = {
        "file_title": "PantumP3500用户手册",
        "chunks": json.loads(chunks_json_data),
        "md_path": r"D:\output\PantumP3500用户手册zh_CNV1.2_1644316283788\PantumP3500用户手册zh_CNV1.2_1644316283788_new.md"
    }
    process = node.process(init_state)
    for chunk in process["chunks"]:
        print(f'{chunk.get("title")}章节：{chunk.get("item_name")}')

    print(process)
