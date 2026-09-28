"""
Phase 1 — ExplicitTransformerBlock

Replaces nn.MultiheadAttention with explicit QKV projection +
F.scaled_dot_product_attention + output projection.

The operation order is IDENTICAL to the original TransformerBlock:
    QKV → Attention → OutProj → Residual+Norm1 → FFN → Residual+Norm2

The only difference is that QKV is exposed as a nn.Linear, making it
accessible for future kernel fusion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ExplicitTransformerBlock(nn.Module):
    """TransformerBlock with exposed QKV projection.

    Architecturally identical to TransformerBlock in attention.py.
    QKV weights can be copied from nn.MultiheadAttention's in_proj_weight/bias.
    """

    def __init__(self, d_model=256, n_heads=8):
        super().__init__()

        assert d_model % n_heads == 0, \
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Exposed QKV projection — packed [Q, K, V] like nn.MHA's in_proj
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)

        # LayerNorm after attention
        self.norm1 = nn.LayerNorm(d_model)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),   # 256 -> 1024
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)    # 1024 -> 256
        )

        # LayerNorm after FFN
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):

        B, S, D = x.shape

        # 1. QKV projection (exposed — this is what we'll fuse later)
        qkv = self.qkv_proj(x)                                    # (B, S, 3D)
        qkv = qkv.reshape(B, S, 3, self.n_heads, self.head_dim)   # (B, S, 3, H, hd)
        qkv = qkv.permute(2, 0, 3, 1, 4)                         # (3, B, H, S, hd)
        Q, K, V = qkv.unbind(dim=0)                               # each: (B, H, S, hd)

        # 2. Scaled dot-product attention
        attn_out = F.scaled_dot_product_attention(Q, K, V)         # (B, H, S, hd)

        # 3. Reshape back and output projection
        attn_out = attn_out.transpose(1, 2).reshape(B, S, D)      # (B, S, D)
        attn_out = self.out_proj(attn_out)                         # (B, S, D)

        # 4. Residual connection + LayerNorm
        x = self.norm1(x + attn_out)

        # 5. Feed-forward network
        ffn_out = self.ffn(x)

        # 6. Residual connection + LayerNorm
        x = self.norm2(x + ffn_out)

        return x

    @staticmethod
    def from_original(original_block):
        """Create an ExplicitTransformerBlock by copying weights from a
        standard TransformerBlock (which uses nn.MultiheadAttention).

        Args:
            original_block: a TransformerBlock instance from attention.py

        Returns:
            ExplicitTransformerBlock with identical weights.
        """
        mha = original_block.attention
        d_model = mha.embed_dim
        n_heads = mha.num_heads

        v2 = ExplicitTransformerBlock(d_model=d_model, n_heads=n_heads)

        # Copy QKV weights (packed identically: in_proj_weight is [3D, D])
        with torch.no_grad():
            v2.qkv_proj.weight.copy_(mha.in_proj_weight)
            v2.qkv_proj.bias.copy_(mha.in_proj_bias)

            # Copy output projection
            v2.out_proj.weight.copy_(mha.out_proj.weight)
            v2.out_proj.bias.copy_(mha.out_proj.bias)

            # Copy norm1
            v2.norm1.weight.copy_(original_block.norm1.weight)
            v2.norm1.bias.copy_(original_block.norm1.bias)

            # Copy FFN
            v2.ffn[0].weight.copy_(original_block.ffn[0].weight)
            v2.ffn[0].bias.copy_(original_block.ffn[0].bias)
            v2.ffn[2].weight.copy_(original_block.ffn[2].weight)
            v2.ffn[2].bias.copy_(original_block.ffn[2].bias)

            # Copy norm2
            v2.norm2.weight.copy_(original_block.norm2.weight)
            v2.norm2.bias.copy_(original_block.norm2.bias)

        return v2
