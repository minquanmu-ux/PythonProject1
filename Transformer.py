#---占位符模块，将在后续小节中实现---
import torch
import  torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """
    位置编码模块
    """
    def forward(self, x):
        pass
    class MultiHeadAttention(nn.Module):
        """
        多头注意力机制模块
        """
        def forward(self, query, key, value, mask):
            pass
        class PositionWiseFeedForward(nn.Module):