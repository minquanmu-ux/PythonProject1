# 交叉注意力的三个输入
cross_output = self.cross_attn(
    query=x,                    # Q 来自解码器（我要生成什么）
    key=encoder_output,         # K 来自编码器（源语言有什么）
    value=encoder_output,       # V 来自编码器（源语言的内容）
    mask=src_mask               # 忽略源语言的填充符
)