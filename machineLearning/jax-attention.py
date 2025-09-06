from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn


class MultiHeadAttention(nn.Module):
    embed_dim: int
    num_heads: int
    dropout: float = 0.0
    use_bias: bool = True

    def setup(self):
        assert self.embed_dim % self.num_heads == 0, "embed_dim must be divisible by num_heads"
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = 1.0 / jnp.sqrt(jnp.array(self.head_dim, dtype=jnp.float32))

        self.q_proj = nn.Dense(self.embed_dim, use_bias=self.use_bias)
        self.k_proj = nn.Dense(self.embed_dim, use_bias=self.use_bias)
        self.v_proj = nn.Dense(self.embed_dim, use_bias=self.use_bias)
        self.o_proj = nn.Dense(self.embed_dim, use_bias=self.use_bias)
        self.dropout_layer = nn.Dropout(rate=self.dropout)

    @nn.compact
    def __call__(
        self,
        x_q: jnp.ndarray,
        x_kv: Optional[jnp.ndarray] = None,
        attn_mask: Optional[jnp.ndarray] = None,
        key_padding_mask: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
        return_weights: bool = False,
    ) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
        """
        x_q: (B, Tq, C)
        x_kv: (B, Tk, C) or None for self-attention
        attn_mask: (Tq, Tk) or (B, Tq, Tk) additive mask; masked positions should be -inf
        key_padding_mask: (B, Tk) bool; True means pad (mask out)
        Returns: (B, Tq, C), (B, H, Tq, Tk) if return_weights else None
        """
        # Use setup-defined layers (access via self directly in setup, but inside @compact we rebind)
        q_proj = self.q_proj
        k_proj = self.k_proj
        v_proj = self.v_proj
        o_proj = self.o_proj
        dropout_layer = self.dropout_layer

        if x_kv is None:
            x_kv = x_q
        B, Tq, C = x_q.shape
        Tk = x_kv.shape[1]

        q = q_proj(x_q)
        k = k_proj(x_kv)
        v = v_proj(x_kv)

        # (B, T, H, Dh)
        q = q.reshape(B, Tq, self.num_heads, self.head_dim)
        k = k.reshape(B, Tk, self.num_heads, self.head_dim)
        v = v.reshape(B, Tk, self.num_heads, self.head_dim)

        # (B, H, Tq, Tk)
        attn_scores = jnp.einsum('bqhd,bkhd->bhqk', q, k) * self.scale

        if attn_mask is not None:
            if attn_mask.ndim == 2:
                attn_scores = attn_scores + attn_mask[None, None, :, :]
            elif attn_mask.ndim == 3:
                attn_scores = attn_scores + attn_mask[:, None, :, :]
            else:
                raise ValueError("attn_mask rank must be 2 or 3")

        if key_padding_mask is not None:
            # (B, 1, 1, Tk)
            mask = key_padding_mask[:, None, None, :]
            attn_scores = jnp.where(mask, -jnp.inf, attn_scores)

        attn_weights = jax.nn.softmax(attn_scores, axis=-1)
        attn_weights = dropout_layer(attn_weights, deterministic=deterministic)

        # (B, Tq, H, Dh)
        out = jnp.einsum('bhqk,bkhd->bqhd', attn_weights, v)
        out = out.reshape(B, Tq, C)
        out = o_proj(out)
        return out, (attn_weights if return_weights else None)
