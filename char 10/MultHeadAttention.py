import math
import torch
from torch import nn
from d2l import torch as d2l
import matplotlib.pyplot as plt
class MultiHeadAttention(nn.Module):
    def __init__(self, key_size, query_size, value_size, num_hiddens, num_heads, dropout, bias = False,  **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = d2l.DotProductAttention(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(self, queries, keys, values, valid_lens):
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)

        if valid_lens is not None:
            valid_lens = torch.repeat_interleave(valid_lens, repeats = self.num_heads, dim = 0)
        output = self.attention(queries, keys, values, valid_lens)

        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)



def transpose_qkv(X, num_heads):
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)
    X = X.permute(0, 2, 1, 3)
    return X.reshape(-1, X.shape[2], X.shape[3])

def transpose_output(X, num_heads):
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)

num_hiddens, num_heads = 100, 5
attention = MultiHeadAttention(key_size=num_hiddens, query_size=num_hiddens
                               ,num_heads=num_heads, value_size=num_hiddens
                               , num_hiddens=num_hiddens, dropout=0.5)


def plot_multi_head_attention(attn_weights, num_heads, batch_idx=0):
    """
    可视化多个头的注意力热力图
    attn_weights: 形状为 (batch_size * num_heads, num_queries, num_keys)
    """
    # 计算 batch_size
    batch_size = attn_weights.shape[0] // num_heads
    num_queries = attn_weights.shape[1]
    num_keys = attn_weights.shape[2]
    
    # 变形回 (batch_size, num_heads, num_queries, num_keys)
    weights = attn_weights.reshape(batch_size, num_heads, num_queries, num_keys)
    
    # 提取指定样本的所有头的权重
    sample_weights = weights[batch_idx].detach().cpu().numpy()
    
    # 创建网格图 (1行, num_heads列)
    fig, axes = plt.subplots(1, num_heads, figsize=(4 * num_heads, 4))
    if num_heads == 1:
        axes = [axes]
        
    for i in range(num_heads):
        ax = axes[i]
        # 绘制热力图
        im = ax.imshow(sample_weights[i], cmap='viridis', aspect='auto')
        ax.set_title(f"Head {i+1}", fontsize=14)
        ax.set_xlabel("Keys (被查询的词)")
        ax.set_ylabel("Queries (查询者)")
        
        # 添加颜色条
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
    plt.suptitle(f"Attention Weights Visualization (Sample {batch_idx})", fontsize=18)
    plt.tight_layout()
    plt.show()


batch_size, num_queries = 2, 4
num_kvpairs, valid_lens = 6, torch.tensor([3,2])

X = torch.randn((batch_size, num_queries, num_hiddens))
Y = torch.randn((batch_size, num_kvpairs, num_hiddens))
# print(attention(X, Y, Y, valid_lens))

attention.eval()

with torch.no_grad():
        # 1. 前向传播，产生动态权重
        output = attention(X, Y, Y, valid_lens)
        
        # 2. 提取封装在内部的动态注意力权重
        attn_weights = attention.attention.attention_weights
        
        # 3. 调用绘图函数（看第 0 个样本的 5 个头）
        plot_multi_head_attention(attn_weights, num_heads, batch_idx=0)