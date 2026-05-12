import torch
import torch.nn.functional as F
import math


def scaled_dot_product_attention(Q, K, V, mask=None):
    # 1. 计算 Q 和 K 的相似度
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    # 2. 应用 mask（如果需要）
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    # 3. softmax 转为权重
    weights = F.softmax(scores, dim=-1)

    # 4. 加权求和 V
    output = torch.matmul(weights, V)

    return output, weights