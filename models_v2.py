"""
Phase 2 — LoopTransformerV2 using ExplicitTransformerBlock.

Architecturally identical to the original LoopTransformer.
Uses ExplicitTransformerBlock so QKV is accessible for future kernel fusion.
"""

import torch
import torch.nn as nn

from attention_v2 import ExplicitTransformerBlock


class LoopTransformerV2(nn.Module):
    """Loop Transformer V2 — uses ExplicitTransformerBlock with exposed QKV."""

    def __init__(self, loops=8, d_model=256, n_heads=8):
        super().__init__()

        self.block = ExplicitTransformerBlock(d_model=d_model, n_heads=n_heads)
        self.loops = loops

    def forward(self, x):

        h = x

        for _ in range(self.loops):
            h = self.block(h)

        return h

    @staticmethod
    def from_v1(v1_model):
        """Create LoopTransformerV2 from an original LoopTransformer.

        Copies all weights so outputs match.
        """
        v2 = LoopTransformerV2(loops=v1_model.loops)
        v2.block = ExplicitTransformerBlock.from_original(v1_model.block)
        return v2
