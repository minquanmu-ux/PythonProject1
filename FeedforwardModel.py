from unittest import result

import numpy as np

# 假设我们已经学习到简化的二维向量词
embeddings = {
    "king": np.array([0.9, 0.8]),
    "queen": np.array([0.9, 0.2]),
    "man": np.array([0.9, 0.2]),
    "woman": np.array([0.7, 0.3]),
}


def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    return dot_product / norm_product
    # king - man + woman


result_vec = embeddings["king"] - embeddings["man"] + embeddings["woman"]
# 计算结果向量与queen的相似度
sim = cosine_similarity(result_vec, embeddings["queen"])
print(f"king - man + woman的结果向量：{result_vec}")
print(f"该结果：'queen' 的相似度：{sim:4f}")
# 神经网络语言模型通过词嵌入，成功解决了 N-gram
# 模型的泛化能力差的问题。然而，
# 它仍然有一个类似 N-gram 的限制：上下文窗口是固定的。
# 它只能考虑固定数量的前文，这为能处理任意长序列的循环神经网络埋下了伏笔。