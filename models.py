"""
Shared model definitions for the Loop Transformer project.

LoopTransformer: a single TransformerBlock reused N times (weight-sharing).
"""

import torch
import torch.nn as nn

from attention import TransformerBlock


class LoopTransformer(nn.Module):
    """Loop Transformer — one shared TransformerBlock applied `loops` times."""

    def __init__(self, loops=8):
        super().__init__()

        self.block = TransformerBlock()
        self.loops = loops

    def forward(self, x):

        h = x

        for _ in range(self.loops):
            h = self.block(h)

        return h
