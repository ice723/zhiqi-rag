# processor/import_processor/nodes/node_bge_embedding.py
import json
from pathlib import Path

from config.embedding_config import embedding_config
from processor.import_processor.base import BaseNode
from processor.import_processor.exceptions import StateFieldError
from processor.import_processor.state import ImportGraphState
from utils.embedding_utils import generate_embeddings


class NodeBGEEmbedding(BaseNode):
    """
    混合向量化节点：使用 BGE-M3 模型将文本转换为向量
    """

    name = "node_bge_embedding"

    def process(self, state: ImportGraphState):
        #1.参数校验
        chunks = self._step_1_validate_paths(state)

        #2.数据向量化
        output_data = self._step_2_generate_embeddings(chunks)
        # for item in output_data:
        #     item_name = item.get("item_name")
        #     content = item.get("content")
        #     print(f"{item_name}")
        #     sparse_vector = item.get("sparse_vector")
        #     print(sparse_vector)
        path = f"{Path(state.get('md_path')).parent}/{state.get('file_title')}_new_new_new_chunks.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                output_data,
                f,
                ensure_ascii=False,
                indent=2
            )

        #3.返回结果
        state["chunks"] = output_data

        return state

    def _step_1_validate_paths(self, state):
        print("node_bge_embedding: 步骤1：参数校验")
        # 校验参数
        chunks = state.get("chunks")
        if not chunks:
            raise ValueError("参数错误：chunks")
        if not isinstance(chunks, list): #判断chunks是不是列表类型
            raise StateFieldError(field_name="chunks", message="数据不是列表类型", expected_type=list)
        return chunks

    def _step_2_generate_embeddings(self, chunks):
        """把 item_name + content 转成稠密向量。

        稀疏一路由 Milvus 的 BM25 Function 在写入时自动生成，这里不再产出
        sparse_vector；即使切回本地 GPU（BGE-M3 返回 sparse）也统一忽略。
        """
        print("node_bge_embedding: 步骤2：数据向量化（稠密）")
        output_data = []
        batch_size = embedding_config.batch_size or 16

        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i: i + batch_size]
            batch_texts = []
            for doc in batch_chunks:
                item_name = doc.get("item_name") or ""
                content = doc.get("content") or ""
                batch_texts.append(f"{item_name}\n{content}" if item_name else content)

            embeddings = generate_embeddings(batch_texts)
            for j, doc in enumerate(batch_chunks):
                item = doc.copy()
                item["dense_vector"] = embeddings["dense"][j]
                item.pop("sparse_vector", None)
                output_data.append(item)

            print(f"node_bge_embedding: 已向量化 {min(i + batch_size, len(chunks))}/{len(chunks)}")
        return output_data

if __name__ == '__main__':
    node = NodeBGEEmbedding()
    path=r"D:\output\PantumP3500用户手册zh_CNV1.2_1644316283788\PantumP3500用户手册_new_new_chunks.json"
    with open(file=path,mode="r",encoding="utf-8") as f:
        chunks_content=f.read()
    json_loads = json.loads(chunks_content)
    init_state = {
        "chunks": json_loads,
        "file_title": "PantumP3500用户手册",
        "md_path": r"D:\output\PantumP3500用户手册zh_CNV1.2_1644316283788\PantumP3500用户手册zh_CNV1.2_1644316283788_new.md"
    }
    process = node(init_state)
    # print(process)



