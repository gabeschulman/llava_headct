import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.blocks.mlp import MLPBlock

class LoraLinear(nn.Module):
    """
    Implements a LoRA (Low-Rank Adaptation) linear layer.
    """
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 8
    ):
        super().__init__()
        self.lora_matrix_B = nn.Parameter(torch.zeros(out_features, r))
        self.lora_matrix_A = nn.Parameter(torch.randn(r, in_features))
        
    def forward(self, x):
        lora_weights = torch.matmul(self.lora_matrix_B, self.lora_matrix_A)
        return F.linear(x, lora_weights)

class SelfAttention(nn.Module):
    """
    Implements a self-attention mechanism with optional LoRA.
    """
    def __init__(
        self, 
        hidden_size: int, 
        num_heads: int = 12, 
        dropout: float = 0.0, 
        qkv_proj_bias: bool = False, 
        lora: bool = False
    ):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.lora = lora
        
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=qkv_proj_bias)
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.proj_drop = nn.Dropout(dropout)
        
        if self.lora:
            self.lora_q = LoraLinear(hidden_size, hidden_size, r=128)
            self.lora_v = LoraLinear(hidden_size, hidden_size, r=128)

        self.dropout = dropout

    def forward(self, x, attn_mask=None):
        B, N, C = x.shape
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.lora:
            q = q + self.lora_q(x).reshape(B, self.num_heads, N, C // self.num_heads)
            v = v + self.lora_v(x).reshape(B, self.num_heads, N, C // self.num_heads)
            
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0, is_causal=False)
        y = y.transpose(1, 2).contiguous().view(B, N, C)
        
        y = self.proj(y)
        y = self.proj_drop(y)
        return y

class AttentionBlock(nn.Module):
    """
    A transformer block with Attention.
    """
    def __init__(
        self, 
        hidden_size: int, 
        mlp_dim: int, 
        num_heads: int, 
        dropout_rate: float = 0.0, 
        qkv_bias: bool = False, 
        save_attn: bool = False, 
        lora: bool = False, 
        norm_layer: nn.Module = nn.LayerNorm
    ):
        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.dropout_rate = dropout_rate
        self.mlp = MLPBlock(hidden_size, mlp_dim, dropout_rate)
        self.att_norm = norm_layer(hidden_size)
        self.ffn_norm = norm_layer(hidden_size)
        self.attn = SelfAttention(hidden_size, num_heads, dropout=dropout_rate, qkv_proj_bias=qkv_bias, lora=lora)

    def forward(self, hidden_states, residual=None):
        hidden_states = hidden_states + self.attn(self.att_norm(hidden_states))
        hidden_states = hidden_states + self.mlp(self.ffn_norm(hidden_states))
        return hidden_states, residual
