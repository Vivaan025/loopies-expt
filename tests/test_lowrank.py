"""
Phase 8 — tests for low-rank shared weights (lowrank.py, loop_lm.py).

Run: python tests/test_lowrank.py
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from attention_v2 import ExplicitTransformerBlock
from loop_lm import LoopLM
from lowrank import LowRankLinear, compress_block, count_params


def test_full_rank_svd_is_exact():
    lin = torch.nn.Linear(64, 96).double()
    lr = LowRankLinear.from_linear(lin, rank=64)
    x = torch.randn(5, 64, dtype=torch.double)
    assert torch.allclose(lr(x), lin(x), atol=1e-10)
    print("  PASS  full-rank SVD factorization reproduces the layer")


def test_eckart_young_error():
    """Rank-r SVD error must equal sqrt(sum of discarded σ²)."""
    lin = torch.nn.Linear(64, 96).double()
    s = torch.linalg.svdvals(lin.weight.detach())
    for r in (1, 8, 32):
        lr = LowRankLinear.from_linear(lin, r)
        err = (lin.weight - lr.weight).norm()
        assert torch.allclose(err, s[r:].pow(2).sum().sqrt(), atol=1e-8)
    print("  PASS  rank-r error matches discarded singular values")


def test_param_count():
    lr = LowRankLinear(512, 1536, 32)
    assert count_params(lr) == 32 * (512 + 1536) + 1536
    print("  PASS  parameter count r(d_in + d_out) + bias")


def test_compress_block_keeps_original():
    torch.manual_seed(0)
    block = ExplicitTransformerBlock(64, 4, causal=True).double()
    x = torch.randn(2, 10, 64, dtype=torch.double)
    ref = block(x)

    exact = compress_block(block, rank=64)
    assert torch.allclose(exact(x), ref, atol=1e-10)

    small = compress_block(block, rank=8)
    assert isinstance(small.qkv_proj, LowRankLinear)
    assert isinstance(small.ffn[0], LowRankLinear)
    assert count_params(small) < count_params(block)
    assert torch.equal(block(x), ref), "original block was modified"
    print("  PASS  compress_block copies, compresses, leaves original intact")


def test_shared_core_reuses_factors():
    model = LoopLM(vocab_size=128, d_model=64, n_heads=4, loops=3, max_seq_len=16)
    model.core = compress_block(model.core, rank=8)
    idx = torch.randint(0, 128, (2, 16))
    _, loss, states = model(idx, idx, return_states=True)
    assert len(states) == 3
    loss.backward()
    g = model.core.qkv_proj.U.weight.grad
    assert g is not None and g.abs().sum() > 0
    print("  PASS  one set of U,V shared across all loops, receives gradients")


def test_causal():
    torch.manual_seed(0)
    model = LoopLM(vocab_size=128, d_model=64, n_heads=4, loops=2, max_seq_len=16).eval()
    a = torch.randint(0, 128, (1, 16))
    b = a.clone()
    b[0, 10:] = (b[0, 10:] + 1) % 128
    la, _ = model(a)
    lb, _ = model(b)
    assert torch.allclose(la[0, :10], lb[0, :10], atol=1e-5)
    print("  PASS  future tokens do not affect earlier positions")


if __name__ == "__main__":
    test_full_rank_svd_is_exact()
    test_eckart_young_error()
    test_param_count()
    test_compress_block_keeps_original()
    test_shared_core_reuses_factors()
    test_causal()
    print("\nAll low-rank tests passed.")
