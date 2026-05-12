import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ========== 1. 位置编码 ==========
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ========== 2. 多头注意力 ==========
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def split_heads(self, x):
        batch, seq_len, _ = x.size()
        x = x.view(batch, seq_len, self.num_heads, self.d_k)
        return x.transpose(1, 2)
    
    def combine_heads(self, x):
        batch, _, seq_len, _ = x.size()
        x = x.transpose(1, 2).contiguous()
        return x.view(batch, seq_len, self.d_model)
    
    def forward(self, Q, K, V, mask=None):
        Q = self.W_q(Q)
        K = self.W_k(K)
        V = self.W_v(V)
        
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)
        
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        attn_output = torch.matmul(weights, V)
        
        output = self.combine_heads(attn_output)
        output = self.W_o(output)
        return output


# ========== 3. 前馈网络 ==========
class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x


# ========== 4. 编码器层 ==========
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # 子层1：多头注意力 + 残差 + 归一化
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 子层2：前馈网络 + 残差 + 归一化
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


# ========== 5. 完整编码器 ==========
class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, max_len, dropout=0.1):
        super().__init__()
        # 词嵌入
        self.embedding = nn.Embedding(vocab_size, d_model)
        # 位置编码
        self.positional_encoding = PositionalEncoding(d_model, max_len, dropout)
        # N 个编码器层
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model
    
    def forward(self, x, mask=None):
        # x: [batch, seq_len] token IDs
        # 1. 词嵌入 + 缩放
        x = self.embedding(x) * math.sqrt(self.d_model)
        # 2. 加位置编码
        x = self.positional_encoding(x)
        # 3. 通过 N 个编码器层
        for layer in self.layers:
            x = layer(x, mask)
        return x


# ========== 测试 ==========
if __name__ == "__main__":
    # 参数设置
    vocab_size = 10000  # 词汇表大小
    d_model = 512       # 模型维度
    num_heads = 8       # 注意力头数
    d_ff = 2048         # 前馈网络维度
    num_layers = 6      # 编码器层数
    max_len = 100       # 最大序列长度
    dropout = 0.1
    
    # 创建编码器
    encoder = Encoder(vocab_size, d_model, num_heads, d_ff, num_layers, max_len, dropout)
    
    # 模拟输入
    batch_size = 2
    seq_len = 10
    x = torch.randint(1, vocab_size, (batch_size, seq_len))  # 随机 token IDs
    
    print(f"输入形状: {x.shape}  (batch_size, seq_len)")
    
    # 前向传播
    output = encoder(x)
    
    print(f"输出形状: {output.shape}  (batch_size, seq_len, d_model)")
    print(f"输入输出形状不同（输出多了 d_model 维）")
    
    # 统计参数量
    total_params = sum(p.numel() for p in encoder.parameters())
    print(f"\n编码器参数量: {total_params:,}")