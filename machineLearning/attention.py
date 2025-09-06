import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0, bias: bool = True):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.o_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x_q: torch.Tensor,
        x_kv: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        x_q: (B, Tq, C)
        x_kv: (B, Tk, C) or None (self-attention)
        attn_mask: (Tq, Tk) or (B, Tq, Tk) additive mask where masked = -inf
        key_padding_mask: (B, Tk) bool where True means pad (mask out)
        Returns: (B, Tq, C), (B, num_heads, Tq, Tk) if need_weights else None
        """
        if x_kv is None:
            x_kv = x_q
        B, Tq, C = x_q.shape
        Tk = x_kv.shape[1]

        q = self.q_proj(x_q)
        k = self.k_proj(x_kv)
        v = self.v_proj(x_kv)

        # (B, T, H, Dh)
        q = q.view(B, Tq, self.num_heads, self.head_dim)
        k = k.view(B, Tk, self.num_heads, self.head_dim)
        v = v.view(B, Tk, self.num_heads, self.head_dim)

        # Attention scores: (B, H, Tq, Tk)
        attn_scores = torch.einsum('bqhd,bkhd->bhqk', q, k) * self.scale

        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_scores = attn_scores + attn_mask.unsqueeze(0).unsqueeze(0)
            elif attn_mask.dim() == 3:
                attn_scores = attn_scores + attn_mask.unsqueeze(1)
            else:
                raise ValueError("attn_mask rank must be 2 or 3")

        if key_padding_mask is not None:
            # mask shape (B, 1, 1, Tk)
            mask = key_padding_mask[:, None, None, :].to(dtype=attn_scores.dtype)
            attn_scores = attn_scores.masked_fill(mask.bool(), float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Output: (B, Tq, H, Dh)
        out = torch.einsum('bhqk,bkhd->bqhd', attn_weights, v)
        out = out.reshape(B, Tq, C)
        out = self.o_proj(out)
        return out, (attn_weights if need_weights else None)
