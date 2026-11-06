# processor/import_processor/nodes/g_node_import_milvus.py
"""导入向量库节点：把切片写入 Milvus。

CPU 部署下：
- 稠密向量由硅基流动 BAAI/bge-m3 生成，随 chunk 一起写入；
- 稀疏向量不再由本地 BGE-M3 生成，而是由 Milvus 2.5 的 BM25 Function
  根据 content 字段在服务端自动生成，因此插入数据里不能带 sparse_vector。
"""

import json
from typing import Any, Dict, List

from config.milvus_config import milvus_config
from processor.import_processor.base import BaseNode
from processor.import_processor.exceptions import MilvusError, StateFieldError
from processor.import_processor.state import ImportGraphState
from utils.milvus_utils import (
    ensure_chunks_collection,
    ensure_item_name_collection,
    escape_milvus_string,
    get_milvus_client,
)

# 只写入 schema 中声明的字段，避免 enable_dynamic_field 把中间字段一起存进去
INSERT_FIELDS = ("content", "title", "parent_title", "part", "file_title", "item_name", "dense_vector")


class NodeImportMilvus(BaseNode):
    """导入向量库节点：数据持久化。"""

    name = "node_import_milvus"

    def process(self, state: ImportGraphState):
        # 1.参数校验
        chunks_json_data, vector_dimension = self._step_1_check_inputs(state)

        # 2.构建连接和数据结构
        milvus_client = self._step_2_prepare_collection(vector_dimension)

        # 3.清理可能冗余数据（幂等性处理）
        self._step_3_clean_old_data(milvus_client, chunks_json_data)

        # 4.数据入库
        update_chunks = self._step_4_insert_data(milvus_client, chunks_json_data)
        for chunk in update_chunks[:3]:
            print(f"chunk_id:{chunk['chunk_id']}")
        print(f"node_import_milvus: 本次插入 {len(update_chunks)} 条切片")

        # 5.更新状态
        state["chunks"] = update_chunks
        return state

    # 步骤1 数据校验
    def _step_1_check_inputs(self, state: Dict[str, Any]) -> tuple[List[Dict[str, Any]], int]:
        print("node_import_milvus: 步骤1：数据校验")
        chunks_json_data = state.get("chunks")
        if not chunks_json_data:
            raise StateFieldError(field_name="chunks", message="chunks参数为空", expected_type=list)

        vector_dimension = len(chunks_json_data[0].get("dense_vector") or [])
        if vector_dimension <= 0:
            raise StateFieldError(
                field_name="dense_vector",
                message="切片缺少 dense_vector，请先执行向量化节点",
                expected_type=list,
            )
        print(f"向量维度:{vector_dimension}")
        return chunks_json_data, vector_dimension

    # 步骤2 客户端和集合准备
    def _step_2_prepare_collection(self, vector_dimension: int):
        print("node_import_milvus: 步骤2：结构准备")
        milvus_client = get_milvus_client()
        try:
            ensure_chunks_collection(milvus_client, dim=vector_dimension)
            # 商品名集合也在这里确保存在，避免首次问答时才发现缺失
            ensure_item_name_collection(milvus_client, dim=vector_dimension)
        except Exception as exc:  # noqa: BLE001
            raise MilvusError(f"Milvus 集合初始化失败: {exc}") from exc
        return milvus_client

    # 步骤3 清理旧数据
    def _step_3_clean_old_data(self, client, chunks_json_data):
        print("node_import_milvus: 步骤3：清理旧数据")
        file_title = chunks_json_data[0].get("file_title")
        if not file_title:
            raise StateFieldError(field_name="file_title", message="file_title 为空", expected_type=str)
        self.clear_chunks_by_file_title(client, file_title)

    # 步骤3方法1
    def clear_chunks_by_file_title(self, client, file_title):
        try:
            safe_title = escape_milvus_string(file_title)
            client.delete(
                collection_name=milvus_config.chunks_collection,
                filter=f'file_title=="{safe_title}"',
            )
        except Exception as e:  # noqa: BLE001
            self.logger.error(f"Milvus 数据删除失败: {str(e)}")
            raise MilvusError(f"Milvus 数据删除失败: {str(e)}") from e

    # 4.数据入库
    def _step_4_insert_data(self, milvus_client, chunks_json_data: List[Dict[str, Any]]):
        print("node_import_milvus: 步骤4：插入数据")
        data_to_insert: List[Dict[str, Any]] = []
        skipped = 0

        for item in chunks_json_data:
            content = (item.get("content") or "").strip()
            if not content:
                skipped += 1
                continue

            row = {field: item.get(field) for field in INSERT_FIELDS if field in item}
            row["content"] = content
            row.setdefault("part", 0)
            row.setdefault("title", "")
            row.setdefault("parent_title", "")
            row.setdefault("item_name", "")
            row["dense_vector"] = item["dense_vector"]
            data_to_insert.append(row)

        if skipped:
            self.logger.warning("跳过 %d 条空内容切片", skipped)
        if not data_to_insert:
            raise MilvusError("没有可插入的切片（content 全为空）")

        # 分批插入，避免单次请求体过大
        batch_size = 200
        inserted_ids: List[int] = []
        try:
            for start in range(0, len(data_to_insert), batch_size):
                batch = data_to_insert[start:start + batch_size]
                result = milvus_client.insert(
                    collection_name=milvus_config.chunks_collection,
                    data=batch,
                )
                inserted_ids.extend(result.get("ids") or [])
        except Exception as exc:  # noqa: BLE001
            raise MilvusError(f"Milvus 数据写入失败: {exc}") from exc

        print(f"node_import_milvus: 插入数据成功，插入条数：{len(inserted_ids)}")

        # 回写主键（跳过的空切片不对应主键）
        valid_chunks = [c for c in chunks_json_data if (c.get("content") or "").strip()]
        for index, item in enumerate(valid_chunks):
            if index < len(inserted_ids):
                item["chunk_id"] = inserted_ids[index]
        return valid_chunks


if __name__ == "__main__":
    node = NodeImportMilvus()
    path = r"D:\output\PantumP3500用户手册zh_CNV1.2_1644316283788\PantumP3500用户手册_new_new_new_chunks.json"
    with open(path, "r", encoding="utf-8") as f:
        json_data = json.load(f)
    node.process({"chunks": json_data})
