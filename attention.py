# import torch
# import torch.nn as nn


# class TransformerBlock(nn.Module):
#     def __init__(self, d_model=8, n_heads=2):
#         super().__init__()

#         self.attention = nn.MultiheadAttention(
#             embed_dim=d_model,
#             num_heads=n_heads,
#             batch_first=True
#         )

#         self.norm1 = nn.LayerNorm(d_model)

#         self.ffn = nn.Sequential(
#             nn.Linear(d_model, 16),
#             nn.ReLU(),
#             nn.Linear(16, d_model)
#         )

#         self.norm2 = nn.LayerNorm(d_model)

#     # def forward(self, x):

#     #     # Self-attention
#     #     attention_output, _ = self.attention(x, x, x)

#     #     # Residual + normalization
#     #     x = self.norm1(x + attention_output)

#     #     # FFN
#     #     ffn_output = self.ffn(x)

#     #     # Residual + normalization
#     #     x = self.norm2(x + ffn_output)

#     #     return x
#     def forward(self, x):

#         # Get the Q, K, V weights
#         q_weight, k_weight, v_weight = \
#             self.attention.in_proj_weight.chunk(3, dim=0)

#         q_bias, k_bias, v_bias = \
#             self.attention.in_proj_bias.chunk(3, dim=0)

#         # Manually calculate Q, K, V
#         Q = x @ q_weight.T + q_bias
#         K = x @ k_weight.T + k_bias
#         V = x @ v_weight.T + v_bias

#         print("Q:")
#         print(Q)

#         print("K:")
#         print(K)

#         print("V:")
#         print(V)

#         # Normal PyTorch attention
#         attention_output, attention_weights = self.attention(x, x, x, need_weights=True, average_attn_weights=False)
#         print("Attention weights:")
#         print(attention_weights)

#         x = self.norm1(x + attention_output)

#         ffn_output = self.ffn(x)

#         x = self.norm2(x + ffn_output)

#         return x


# # block = TransformerBlock()

# # x = torch.randn(1, 2, 8)

# # output = block(x)

# # print("input shape :", x.shape)
# # print("output shape:", output.shape)

# # block = TransformerBlock()

# # h = x

# # for i in range(3):
# #     h = block(h)

# # print("Number of parameters:", sum(p.numel() for p in block.parameters()))

import torch
import torch.nn as nn


class TransformerBlock(nn.Module):
    def __init__(self, d_model=8, n_heads=2):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            batch_first=True
        )

        self.norm1 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, 16),
            nn.ReLU(),
            nn.Linear(16, d_model)
        )

        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):

        # 1. Self-attention
        attention_output, _ = self.attention(x, x, x)

        # 2. Residual connection + LayerNorm
        x = self.norm1(x + attention_output)

        # 3. Feed-forward network
        ffn_output = self.ffn(x)

        # 4. Residual connection + LayerNorm
        x = self.norm2(x + ffn_output)

        return x