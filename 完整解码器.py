import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ========== 1. 位置编码（同编码器） ==========
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


# ========== 2. 多头注意力（同编码器） ==========
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


# ========== 3. 前馈网络（同编码器） ==========
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


# ========== 4. 解码器层（关键！） ==========
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        # 3 个子层（编码器只有2个）
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)      # 掩码自注意力
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)     # 交叉注意力（新！）
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        
        # 3 个层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        # 子层1：掩码自注意力（对自己）
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 子层2：交叉注意力（对编码器输出）⭐ 解码器独有
        cross_output = self.cross_attn(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout(cross_output))
        
        # 子层3：前馈网络
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x


# ========== 5. 完整解码器 ==========
class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, max_len, dropout=0.1):
        super().__init__()
        # 词嵌入
        self.embedding = nn.Embedding(vocab_size, d_model)
        # 位置编码
        self.positional_encoding = PositionalEncoding(d_model, max_len, dropout)
        # N 个解码器层
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model
    
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        # x: [batch, tgt_len] token IDs
        # 1. 词嵌入 + 缩放
        x = self.embedding(x) * math.sqrt(self.d_model)
        # 2. 加位置编码
        x = self.positional_encoding(x)
        # 3. 通过 N 个解码器层
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return x


# ========== 6. 创建掩码辅助函数 ==========
def create_padding_mask(seq, pad_idx=0):
    """创建填充掩码"""
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2)

def create_causal_mask(seq_len):
    """创建因果掩码（下三角）"""
    mask = torch.tril(torch.ones(seq_len, seq_len)).bool()
    return mask.unsqueeze(0).unsqueeze(0)


# ========== 测试 ==========
if __name__ == "__main__":
    # 参数设置
    vocab_size = 10000
    d_model = 512
    num_heads = 8
    d_ff = 2048
    num_layers = 6
    max_len = 100
    dropout = 0.1
    
    # 创建编码器和解码器
    encoder = Encoder(vocab_size, d_model, num_heads, d_ff, num_layers, max_len, dropout)
    decoder = Decoder(vocab_size, d_model, num_heads, d_ff, num_layers, max_len, dropout)
    
    # 模拟输入
    batch_size = 2
    src_len = 10
    tgt_len = 8
    
    src = torch.randint(1, vocab_size, (batch_size, src_len))  # 源语言
    tgt = torch.randint(1, vocab_size, (batch_size, tgt_len))  # 目标语言
    
    # 创建掩码
    src_mask = create_padding_mask(src)  # 编码器掩码
    tgt_mask = create_padding_mask(tgt) & create_causal_mask(tgt_len)  # 解码器掩码
    
    print(f"源语言输入: {src.shape}")
    print(f"目标语言输入: {tgt.shape}")
    
    # 编码器前向
    encoder_output = encoder(src, src_mask)
    print(f"\n编码器输出: {encoder_output.shape}")
    
    # 解码器前向
    decoder_output = decoder(tgt, encoder_output, src_mask, tgt_mask)
    print(f"解码器输出: {decoder_output.shape}")
    
    # 统计参数量
    total_params = sum(p.numel() for p in decoder.parameters())
    print(f"\n解码器参数量: {total_params:,}")