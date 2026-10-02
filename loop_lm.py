"""
Phase 8 — Autoregressive loop LM (Huginn-style, toy scale).

    tokens → embed → Prelude → [ Core ] × loops → Coda → LM head
                                  ↑ shared weights

Mirrors the prelude / recurrent core / coda split of Huginn-0125, but small
enough to train from scratch on FineWeb. Only the Core is weight-shared,
so only the Core is the target for low-rank compression.

The `shared=False` variant gives every loop its own Core block (same compute,
`loops`× the core parameters) — the control for "does compression behave
differently when the compressed weight is reused?".
"""

import glob
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from attention_v2 import ExplicitTransformerBlock


class LoopLM(nn.Module):

    def __init__(self, vocab_size=50304, d_model=512, n_heads=8,
                 loops=4, max_seq_len=1024, shared=True):
        super().__init__()

        self.loops = loops
        self.shared = shared

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        def block():
            return ExplicitTransformerBlock(d_model, n_heads, causal=True)

        self.prelude = block()
        if shared:
            self.core = block()
        else:
            self.cores = nn.ModuleList(block() for _ in range(loops))
        self.coda = block()

        self.norm_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight   # weight tying

        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)

    def core_block(self, i):
        return self.core if self.shared else self.cores[i]

    def forward(self, idx, targets=None, return_states=False):

        B, S = idx.shape
        pos = torch.arange(S, device=idx.device)

        h = self.tok_emb(idx) + self.pos_emb(pos)
        h = self.prelude(h)

        states = []
        for i in range(self.loops):
            h = self.core_block(i)(h)
            if return_states:
                states.append(h)

        h = self.coda(h)
        logits = self.head(self.norm_f(h))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.float().view(-1, logits.size(-1)), targets.view(-1))

        if return_states:
            return logits, loss, states
        return logits, loss


# ============================================================
# FineWeb data (modded-nanogpt GPT-2 token shards)
# ============================================================

DEFAULT_DATA_DIR = os.path.expanduser(
    "~/Documents/Coding/modded-nanogpt/data/fineweb10B")

HEADER_INTS = 256
MAGIC = 20240520


def _load_shard(path):
    header = np.fromfile(path, dtype=np.int32, count=HEADER_INTS)
    assert header[0] == MAGIC, f"bad magic in {path}"
    n_tokens = int(header[2])
    return np.memmap(path, dtype=np.uint16, mode="r",
                     offset=HEADER_INTS * 4, shape=(n_tokens,))


class FineWeb:
    """Random-window sampler over GPT-2 tokenized FineWeb shards (read-only)."""

    def __init__(self, data_dir=DEFAULT_DATA_DIR, split="train", seed=0):
        paths = sorted(glob.glob(os.path.join(data_dir, f"fineweb_{split}_*.bin")))
        assert paths, f"no {split} shards found in {data_dir}"
        self.shards = [_load_shard(p) for p in paths]
        self.rng = np.random.default_rng(seed)

    def batch(self, batch_size, seq_len, device):
        xs, ys = [], []
        for _ in range(batch_size):
            shard = self.shards[self.rng.integers(len(self.shards))]
            i = self.rng.integers(0, len(shard) - seq_len - 1)
            chunk = torch.from_numpy(shard[i:i + seq_len + 1].astype(np.int64))
            xs.append(chunk[:-1])
            ys.append(chunk[1:])
        x = torch.stack(xs).to(device, non_blocking=True)
        y = torch.stack(ys).to(device, non_blocking=True)
        return x, y

    def fixed_batches(self, n, batch_size, seq_len, device):
        """Deterministic sequential batches (for validation)."""
        shard = self.shards[0]
        out = []
        for b in range(n):
            xs, ys = [], []
            for j in range(batch_size):
                i = (b * batch_size + j) * (seq_len + 1)
                chunk = torch.from_numpy(shard[i:i + seq_len + 1].astype(np.int64))
                xs.append(chunk[:-1])
                ys.append(chunk[1:])
            out.append((torch.stack(xs).to(device), torch.stack(ys).to(device)))
        return out
