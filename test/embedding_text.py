"""Embedding 调用示例。

CPU 部署走 SiliconFlow API，只返回稠密向量（sparse 为 None）；
切回 local_gpu 时 sparse 可能为 BGE-M3 的词权重字典。
"""

from utils.embedding_utils import generate_embeddings

embeddings = generate_embeddings(["你好", "hello world"])

print("dense 维度:", len(embeddings["dense"][0]))
print("dense 前 5 个值:", embeddings["dense"][0][:5])
print("sparse:", embeddings["sparse"])
