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
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


# ========== 5. 编码器 ==========
class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, max_len, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model
    
    def forward(self, x, mask=None):
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)
        for layer in self.layers:
            x = layer(x, mask)
        return x


# ========== 6. 解码器层 ==========
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        cross_output = self.cross_attn(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout(cross_output))
        
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x


# ========== 7. 解码器 ==========
class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, max_len, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model
    
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return x


# ========== 8. 完整 Transformer ==========
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8,
                 d_ff=2048, num_layers=6, max_len=5000, dropout=0.1, pad_idx=0):
        super().__init__()
        self.pad_idx = pad_idx
        self.d_model = d_model
        
        # 编码器和解码器
        self.encoder = Encoder(src_vocab_size, d_model, num_heads, d_ff, num_layers, max_len, dropout)
        self.decoder = Decoder(tgt_vocab_size, d_model, num_heads, d_ff, num_layers, max_len, dropout)
        
        # 输出投影层
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)
    
    def create_masks(self, src, tgt):
        """创建所有需要的掩码"""
        # 编码器掩码：忽略源语言填充
        src_mask = (src != self.pad_idx).unsqueeze(1).unsqueeze(2)
        
        # 解码器掩码：忽略目标语言填充 + 因果掩码
        tgt_pad_mask = (tgt != self.pad_idx).unsqueeze(1).unsqueeze(2)
        tgt_len = tgt.size(1)
        tgt_causal_mask = torch.tril(torch.ones(tgt_len, tgt_len)).bool().to(tgt.device)
        tgt_causal_mask = tgt_causal_mask.unsqueeze(0).unsqueeze(0)
        tgt_mask = tgt_pad_mask & tgt_causal_mask
        
        return src_mask, tgt_mask
    
    def forward(self, src, tgt):
        """
        src: [batch, src_len] 源语言 token IDs
        tgt: [batch, tgt_len] 目标语言 token IDs（训练时）
        """
        # 创建掩码
        src_mask, tgt_mask = self.create_masks(src, tgt)
        
        # 编码器
        encoder_output = self.encoder(src, src_mask)
        
        # 解码器
        decoder_output = self.decoder(tgt, encoder_output, src_mask, tgt_mask)
        
        # 输出投影（logits）
        logits = self.output_projection(decoder_output)
        
        return logits


# ========== 9. 训练示例 ==========
def train_example():
    # 参数设置
    src_vocab_size = 10000  # 源语言词汇表大小
    tgt_vocab_size = 10000  # 目标语言词汇表大小
    d_model = 512
    num_heads = 8
    d_ff = 2048
    num_layers = 6
    max_len = 100
    dropout = 0.1
    pad_idx = 0
    
    # 创建模型
    model = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads,
                        d_ff, num_layers, max_len, dropout, pad_idx)
    
    # 模拟一个 batch 的数据
    batch_size = 2
    src_len = 10
    tgt_len = 8
    
    src = torch.randint(1, src_vocab_size, (batch_size, src_len))
    tgt = torch.randint(1, tgt_vocab_size, (batch_size, tgt_len))
    
    # 前向传播
    logits = model(src, tgt)
    
    print("=" * 50)
    print("Transformer 模型信息")
    print("=" * 50)
    print(f"源语言输入形状: {src.shape}")
    print(f"目标语言输入形状: {tgt.shape}")
    print(f"输出 logits 形状: {logits.shape}")
    print(f"  (batch_size, tgt_len, tgt_vocab_size)")
    
    # 计算损失（示例）
    # 假设目标是 tgt 右移一位
    target = tgt[:, 1:]  # 去掉第一个 <BOS>
    logits_for_loss = logits[:, :-1, :]  # 去掉最后一个位置
    
    loss = F.cross_entropy(
        logits_for_loss.reshape(-1, tgt_vocab_size),
        target.reshape(-1)
    )
    print(f"\n示例损失: {loss.item():.4f}")
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n总参数量: {total_params:,}")
    
    return model


# ========== 10. 推理示例（生成） ==========
def inference_example(model, src, max_len=20, start_token=1, end_token=2):
    """使用训练好的模型进行翻译/生成"""
    model.eval()
    with torch.no_grad():
        # 编码器
        src_mask = (src != model.pad_idx).unsqueeze(1).unsqueeze(2)
        encoder_output = model.encoder(src, src_mask)
        
        # 初始化解码器输入（只有 <BOS>）
        batch_size = src.size(0)
        tgt = torch.full((batch_size, 1), start_token).to(src.device)
        
        # 逐词生成
        for _ in range(max_len - 1):
            # 创建目标掩码
            tgt_mask = model.create_masks(src, tgt)[1]
            
            # 解码器前向
            decoder_output = model.decoder(tgt, encoder_output, src_mask, tgt_mask)
            
            # 预测下一个词
            logits = model.output_projection(decoder_output[:, -1, :])
            next_token = logits.argmax(dim=-1, keepdim=True)
            
            # 拼接
            tgt = torch.cat([tgt, next_token], dim=1)
            
            # 如果生成了结束符，停止
            if (next_token == end_token).all():
                break
    
    return tgt


# ========== 主程序 ==========
if __name__ == "__main__":
    print("训练示例:")
    model = train_example()
    
    print("\n" + "=" * 50)
    print("推理示例:")
    print("=" * 50)
    
    # 模拟输入
    src = torch.randint(1, 10000, (1, 10))
    print(f"源语言输入: {src[0].tolist()}")
    
    # 生成
    generated = inference_example(model, src, max_len=15)
    print(f"生成的序列: {generated[0].tolist()}")