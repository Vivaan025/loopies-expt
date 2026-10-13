"""
Phase 9 — Low-rank shared weights on a pretrained loop LM (Huginn-0125).

Same question as Phase 8 (benchmarks/lowrank_shared.py), now on a real
3.5B recurrent-depth model instead of a toy one:

    Huginn:  tokens → Prelude (2 blocks) → [ adapter + Core (4 blocks) ] × num_steps → Coda (2 blocks)
                                              ↑ shared weights, ~1.6B params

Each core projection (attn Wqkv, attn proj, mlp fc, mlp proj) is replaced by
its truncated-SVD factorization W ≈ UV, the same U,V reused on every step.
Because Huginn accepts any num_steps at inference, we can ask directly how
the compression error behaves as the same compressed core is applied
4, 8, 16, 32, 64 times.

Stages (--stage all runs spectrum, compress, error, latency):
  spectrum  singular-value energy of every core projection
  compress  post-training compression: val loss at each rank × num_steps
  error     per-iteration hidden-state error vs. the full model (up to max steps)
  recover   compression + fine-tune of the U,V factors only (all else frozen)
  latency   full forward time, full vs. low-rank core

Evaluation text = the FineWeb val shard already on the server (GPT-2 tokens),
decoded back to text and re-tokenized with Huginn's tokenizer.
"""

import argparse
import copy
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loop_lm import DEFAULT_DATA_DIR, _load_shard
from lowrank import LowRankLinear, count_params

MODEL_ID = "tomg-group-umd/huginn-0125"
BOS_ID = 65504          # <|begin_text|>
GPT2_EOT = 50256        # document separator in the GPT-2 FineWeb shards

# Projections inside each Huginn SandwichBlock that can be factorized.
TARGETS = {
    "qkv":      ("attn", "Wqkv"),
    "attn_out": ("attn", "proj"),
    "mlp_fc":   ("mlp", "fc"),
    "mlp_proj": ("mlp", "proj"),
}
ALL_TARGETS = tuple(TARGETS)


# ============================================================
# Model + data
# ============================================================

def load_huginn(device):
    """Load Huginn with its own modeling code (trust_remote_code).

    The repo's code targets transformers 4.44; transformers 5 expects
    `_tied_weights_keys` to be a {target: source} dict rather than a list,
    so the class is patched at runtime before loading. Nothing on disk changes.
    """
    from transformers import AutoTokenizer
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    cls = get_class_from_dynamic_module("raven_modeling_minimal.RavenForCausalLM", MODEL_ID)
    if isinstance(cls._tied_weights_keys, list):
        cls._tied_weights_keys = {"lm_head.weight": "transformer.wte.weight"}

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = cls.from_pretrained(MODEL_ID, dtype=torch.bfloat16)
    model.to(device).eval()
    assert model.lm_head.weight.data_ptr() == model.transformer.wte.weight.data_ptr(), \
        "embedding / lm_head tie was lost while loading"
    return model, tok


def build_token_set(tok, data_dir, split, n_seqs, seq_len, cache_dir, skip_docs=0):
    """Decode GPT-2 FineWeb docs to text, re-tokenize with Huginn's tokenizer,
    pack into (n_seqs, seq_len + 1). Cached on disk."""
    path = os.path.join(cache_dir, f"tokens_{split}_{n_seqs}x{seq_len}_skip{skip_docs}.pt")
    if os.path.exists(path):
        return torch.load(path)

    import glob
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    shard_path = sorted(glob.glob(os.path.join(data_dir, f"fineweb_{split}_*.bin")))[0]
    shard = _load_shard(shard_path)

    need = n_seqs * (seq_len + 1)
    stream = []
    doc_starts = np.flatnonzero(shard[: min(len(shard), 50_000_000)] == GPT2_EOT)
    for d, (a, b) in enumerate(zip(doc_starts[:-1], doc_starts[1:])):
        if d < skip_docs:
            continue
        text = enc.decode(shard[a + 1:b].tolist())
        stream.append(BOS_ID)
        stream.extend(tok(text, add_special_tokens=False)["input_ids"])
        if len(stream) >= need:
            break
    assert len(stream) >= need, "not enough text in shard"
    data = torch.tensor(stream[:need], dtype=torch.long).view(n_seqs, seq_len + 1)
    torch.save(data, path)
    return data


def batches(data, batch_size):
    for i in range(0, len(data), batch_size):
        chunk = data[i:i + batch_size]
        yield chunk[:, :-1], chunk[:, 1:]


# ============================================================
# Low-rank swapping (in place, reversible)
# ============================================================

def _proj(block, target):
    a, b = TARGETS[target]
    return getattr(getattr(block, a), b)


def _set_proj(block, target, module):
    a, b = TARGETS[target]
    setattr(getattr(block, a), b, module)


class CoreCompressor:
    """Computes each core projection's SVD once (top `max_rank` kept on CPU),
    then swaps rank-r factors in / originals back out without recomputing."""

    def __init__(self, model, targets, max_rank):
        self.model = model
        self.targets = targets
        self.blocks = list(model.transformer.core_block)
        self.orig = {}      # (i, target) -> nn.Linear
        self.svd = {}       # (i, target) -> (P_r, S, Qt_r) on CPU
        self.spectrum = {}  # (i, target) -> full singular values

        for i, block in enumerate(self.blocks):
            for t in targets:
                lin = _proj(block, t)
                self.orig[(i, t)] = lin
                W = lin.weight.detach().float()
                P, S, Qt = torch.linalg.svd(W, full_matrices=False)
                self.spectrum[(i, t)] = S.cpu()
                self.svd[(i, t)] = (P[:, :max_rank].bfloat16().cpu(),
                                    S[:max_rank].cpu(),
                                    Qt[:max_rank].bfloat16().cpu())
                del W, P, S, Qt
                torch.cuda.empty_cache()
                print(f"    SVD core[{i}].{t} {tuple(lin.weight.shape)}", flush=True)

    def apply(self, rank):
        """Replace every target projection with its rank-r factors."""
        for (i, t), (P, S, Qt) in self.svd.items():
            lin = self.orig[(i, t)]
            sqrt_s = S[:rank].sqrt()
            lr = LowRankLinear(lin.in_features, lin.out_features, rank, bias=False)
            with torch.no_grad():
                lr.U.weight.copy_(P[:, :rank].float() * sqrt_s)
                lr.V.weight.copy_(sqrt_s[:, None] * Qt[:rank].float())
            _set_proj(self.blocks[i], t, lr.to(lin.weight.device, torch.bfloat16))

    def restore(self):
        for (i, t), lin in self.orig.items():
            _set_proj(self.blocks[i], t, lin)

    def core_target_params(self, rank=None):
        total = 0
        for (i, t), lin in self.orig.items():
            m, n = lin.out_features, lin.in_features
            total += m * n if rank is None else rank * (m + n)
        return total


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def eval_loss(model, data, num_steps, batch_size, device, seed=1234):
    """Mean next-token loss. The recurrent state is initialized randomly, so
    each batch is seeded identically across models for a fair comparison."""
    total, n = 0.0, 0
    for b, (x, y) in enumerate(batches(data, batch_size)):
        torch.manual_seed(seed + b)
        out = model(input_ids=x.to(device), labels=y.to(device),
                    num_steps=num_steps)
        total += out.loss.item() * x.numel()
        n += x.numel()
    return total / n


class StepRecorder:
    """Forward hook on the last core block: fires once per recurrence step."""

    def __init__(self, block):
        self.states = []
        self.ref = None
        self.errs = None
        self.handle = block.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        if self.ref is None:
            self.states.append(h.detach())
        else:
            r = self.ref[self._t].float()
            self.errs[self._t] += ((h.float() - r).norm() / r.norm()).item()
            self._t += 1

    def remove(self):
        self.handle.remove()


@torch.no_grad()
def per_step_error(model, comp, rank, data, num_steps, batch_size, device, seed=1234):
    """||h_t^lowrank - h_t^full|| / ||h_t^full|| after each recurrence step t."""
    block = model.transformer.core_block[-1]
    errs = torch.zeros(num_steps)
    nb = 0
    for b, (x, _) in enumerate(batches(data, batch_size)):
        x = x.to(device)

        comp.restore()
        rec = StepRecorder(block)
        torch.manual_seed(seed + b)
        model(input_ids=x, num_steps=num_steps)
        rec.remove()

        comp.apply(rank)
        cmp = StepRecorder(block)
        cmp.ref, cmp.errs, cmp._t = rec.states, errs, 0
        torch.manual_seed(seed + b)
        model(input_ids=x, num_steps=num_steps)
        cmp.remove()

        del rec
        nb += 1
    comp.restore()
    return (errs / nb).tolist()


# ============================================================
# Stages
# ============================================================

def stage_spectrum(comp, ranks):
    print("\n" + "=" * 78)
    print("SPECTRUM — energy kept at rank r, mean over the 4 core blocks")
    print("=" * 78)
    out = {}
    for t in comp.targets:
        rows = []
        for i in range(len(comp.blocks)):
            s2 = comp.spectrum[(i, t)].double() ** 2
            cum = (s2.cumsum(0) / s2.sum()).tolist()
            rows.append({r: cum[min(r, len(cum)) - 1] for r in ranks})
        mean = {r: sum(row[r] for row in rows) / len(rows) for r in ranks}
        out[t] = dict(per_block=rows, mean=mean)
        print(f"  {t:<9} " + "  ".join(f"r{r}:{mean[r]:.3f}" for r in ranks))
    return out


def stage_compress(model, comp, data, args):
    print("\n" + "=" * 78)
    print(f"POST-TRAINING SVD COMPRESSION  (targets={','.join(comp.targets)}, "
          f"{len(data)}x{data.shape[1] - 1} tokens)")
    print("=" * 78)
    full_p = comp.core_target_params()
    header = "  " + f"{'rank':>6} {'params':>14} {'ratio':>6} | " + \
        " ".join(f"{'s=' + str(s):>8}" for s in args.steps)
    print(header)

    rows = []
    comp.restore()
    base = {s: eval_loss(model, data, s, args.batch_size, args.device) for s in args.steps}
    rows.append(dict(rank="full", params=full_p, loss=base))
    print(f"  {'full':>6} {full_p:>14,} {1.0:>5.1f}x | " +
          " ".join(f"{base[s]:>8.4f}" for s in args.steps), flush=True)

    for r in args.ranks:
        comp.apply(r)
        loss = {s: eval_loss(model, data, s, args.batch_size, args.device) for s in args.steps}
        p = comp.core_target_params(r)
        rows.append(dict(rank=r, params=p, loss=loss))
        print(f"  {r:>6} {p:>14,} {full_p / p:>5.1f}x | " +
              " ".join(f"{loss[s]:>8.4f}" for s in args.steps), flush=True)
    comp.restore()
    return rows


def stage_error(model, comp, data, args):
    S = max(args.steps)
    print("\n" + "=" * 78)
    print(f"PER-STEP HIDDEN-STATE ERROR vs. full model  ({S} steps, "
          f"{len(data)} seqs)")
    print("=" * 78)
    show = [s for s in (1, 2, 4, 8, 16, 32, 64, 128) if s <= S]
    print("  " + f"{'rank':>6} | " + " ".join(f"{'t=' + str(s):>7}" for s in show))
    out = {}
    for r in args.ranks:
        errs = per_step_error(model, comp, r, data, S, args.batch_size, args.device)
        out[r] = errs
        print(f"  {r:>6} | " + " ".join(f"{errs[s - 1]:>7.3f}" for s in show), flush=True)
    return out


@torch.no_grad()
def stage_eval_loops(model, comp, data, args):
    """Evaluate full, compressed, and recovered models at multiple loop counts."""
    steps = args.steps
    print("\n" + "=" * 78)
    print(f"LOOP-COUNT EVALUATION  (rank {args.ranks}, steps={steps}, "
          f"{len(data)}x{data.shape[1] - 1} tokens)")
    if args.resume_factors:
        print(f"  recovered checkpoint(s): {args.resume_factors}")
    print("=" * 78)

    header = f"  {'model':<22} | " + " ".join(f"{'s=' + str(s):>8}" for s in steps)
    print(header)
    print("  " + "-" * (len(header) - 2))

    rows = []

    # Check if reusing baselines from an earlier run
    cached_full = None
    cached_svd = {}
    if getattr(args, 'reuse_baselines', None) and os.path.exists(args.reuse_baselines):
        with open(args.reuse_baselines) as f:
            base_json = json.load(f)
        for row in base_json.get("eval_loops", []):
            m = row["model"]
            if m == "full":
                cached_full = {int(k): v for k, v in row["loss"].items()}
            elif m.startswith("svd_r"):
                r_val = int(m.split("_r")[1])
                cached_svd[r_val] = {int(k): v for k, v in row["loss"].items()}

    # 1. Full model (original)
    if cached_full is not None:
        full = cached_full
        rows.append(dict(model="full", loss=full))
        print(f"  {'full (original)':<22} | " +
              " ".join(f"{full[s]:>8.4f}" for s in steps), flush=True)
    else:
        comp.restore()
        full = {s: eval_loss(model, data, s, args.batch_size, args.device) for s in steps}
        rows.append(dict(model="full", loss=full))
        print(f"  {'full (original)':<22} | " +
              " ".join(f"{full[s]:>8.4f}" for s in steps), flush=True)

    for r in args.ranks:
        # 2. Compressed, untrained (raw SVD)
        if r in cached_svd:
            compressed = cached_svd[r]
            rows.append(dict(model=f"svd_r{r}", loss=compressed))
            print(f"  {f'svd r={r}':<22} | " +
                  " ".join(f"{compressed[s]:>8.4f}" for s in steps), flush=True)
            comp.apply(r)
        else:
            comp.apply(r)
            compressed = {s: eval_loss(model, data, s, args.batch_size, args.device) for s in steps}
            rows.append(dict(model=f"svd_r{r}", loss=compressed))
            print(f"  {f'svd r={r}':<22} | " +
                  " ".join(f"{compressed[s]:>8.4f}" for s in steps), flush=True)

        # 3. Recovered (from checkpoint(s))
        if args.resume_factors:
            ckpts = args.resume_factors if isinstance(args.resume_factors, list) else [args.resume_factors]
            for ckpt_path in ckpts:
                ckpt = torch.load(ckpt_path, map_location=args.device,
                                  weights_only=False)
                if isinstance(ckpt, dict) and "factors" in ckpt:
                    factor_states = ckpt["factors"]
                else:
                    factor_states = ckpt
                for (i, t) in comp.orig:
                    lr_mod = _proj(comp.blocks[i], t)
                    lr_mod.load_state_dict(
                        {k: v.to(lr_mod.U.weight.device) for k, v in factor_states[(i, t)].items()}
                    )
                recovered = {s: eval_loss(model, data, s, args.batch_size, args.device) for s in steps}
                ckpt_stem = Path(ckpt_path).stem
                label = f"rec {ckpt_stem}"
                rows.append(dict(model=f"recovered_r{r}_{ckpt_stem}", loss=recovered, checkpoint=ckpt_path))
                print(f"  {label:<22} | " +
                      " ".join(f"{recovered[s]:>8.4f}" for s in steps), flush=True)

                # Print recovery % at each step count
                rec_pcts = {
                    s: (100 * (compressed[s] - recovered[s]) / (compressed[s] - full[s]))
                    if compressed[s] != full[s] else 0.0
                    for s in steps
                }
                rows.append(dict(model=f"recovery_pct_r{r}_{ckpt_stem}", pct=rec_pcts))
                print(f"  {'  recovery %':<22} | " +
                      " ".join(f"{rec_pcts[s]:>7.1f}%" for s in steps), flush=True)

        comp.restore()
    return rows


def stage_recover(model, comp, train_data, val_data, args):
    step_offset = getattr(args, 'start_step', 0)
    total_steps = step_offset + args.recover_steps
    print("\n" + "=" * 78)
    print(f"COMPRESSION + RECOVERY  (U,V of the core only; {args.recover_steps} steps "
          f"[global {step_offset+1}..{total_steps}], "
          f"lr {args.recover_lr}, grad through last {args.grad_steps} of "
          f"{args.train_num_steps} recurrences)")
    resume_path = args.resume_factors
    if isinstance(resume_path, list):
        resume_path = resume_path[0] if len(resume_path) > 0 else None
    if resume_path:
        print(f"  resuming from: {resume_path}")
    print("=" * 78)
    out = []
    s_eval = args.recover_eval_steps
    for r in args.ranks:
        comp.apply(r)

        # --- Resume from checkpoint if provided ---
        if resume_path:
            ckpt = torch.load(resume_path, map_location=args.device,
                              weights_only=False)
            # Support both old format (bare factor dicts) and new format
            if isinstance(ckpt, dict) and "factors" in ckpt:
                factor_states = ckpt["factors"]
            else:
                factor_states = ckpt
            for (i, t) in comp.orig:
                lr_mod = _proj(comp.blocks[i], t)
                lr_mod.load_state_dict(
                    {k: v.to(lr_mod.U.weight.device) for k, v in factor_states[(i, t)].items()}
                )
            print(f"    Loaded factor weights from checkpoint")

        before = eval_loss(model, val_data, s_eval, args.batch_size, args.device)

        for p in model.parameters():
            p.requires_grad_(False)
        params = []
        for (i, t) in comp.orig:
            lr_mod = _proj(comp.blocks[i], t)
            for p in lr_mod.parameters():
                p.data = p.data.float()       # train factors in fp32
                p.requires_grad_(True)
                params.append(p)
        opt = torch.optim.AdamW(params, lr=args.recover_lr, betas=(0.9, 0.95),
                                weight_decay=0.0)

        # Restore optimizer state if available in checkpoint
        if resume_path and isinstance(ckpt, dict) and "optimizer" in ckpt:
            opt.load_state_dict(ckpt["optimizer"])
            print(f"    Restored optimizer state from checkpoint")
        elif resume_path:
            print(f"    NOTE: checkpoint has no optimizer state; Adam restarts fresh")

        model.train()
        model.gradient_checkpointing = True
        no_grad = args.train_num_steps - args.grad_steps
        warmup = max(1, args.recover_steps // 10)
        t0 = time.perf_counter()
        it = getattr(args, 'start_it', 0)
        best_val = before
        best_path = None

        def save_factors(step_num, val_loss):
            nonlocal best_path
            ckpt_dir = Path(args.out_dir) / "factors"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            best_path = ckpt_dir / f"factors_r{r}_step{step_num}.pt"
            torch.save(
                {
                    "factors": {k: _proj(comp.blocks[k[0]], k[1]).state_dict()
                                for k in comp.orig},
                    "optimizer": opt.state_dict(),
                    "step": step_num,
                    "it": it,
                    "val_loss": val_loss,
                },
                best_path,
            )

        for step in range(args.recover_steps):
            global_step = step_offset + step
            lr = args.recover_lr * min(1.0, (step + 1) / warmup)
            for g in opt.param_groups:
                g["lr"] = lr
            opt.zero_grad(set_to_none=True)
            train_loss_sum = 0.0
            for _ in range(args.accum):
                row = train_data[it % len(train_data)].unsqueeze(0).to(args.device)
                it += 1
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    result = model(
                        input_ids=row[:, :-1],
                        labels=row[:, 1:],
                        num_steps=(no_grad, args.grad_steps),
                        use_cache=False,
                    )
                train_loss_sum += result.loss.detach().item()
                (result.loss / args.accum).backward()
                del result
            if step == 0:
                missing = sum(p.grad is None for p in params)
                print(f"    Parameters without gradients: {missing}/{len(params)}")
                if missing:
                    raise RuntimeError("Some intended trainable parameters have no gradient.")
            grad_norm = torch.nn.utils.clip_grad_norm_(
                params, 1.0, error_if_nonfinite=True
            )
            opt.step()
            global_completed = global_step + 1
            if (step + 1) % 25 == 0 or (step + 1) == args.recover_steps:
                cuda_device = torch.device(args.device)
                cuda_index = (
                    cuda_device.index
                    if cuda_device.index is not None
                    else torch.cuda.current_device()
                )
                model.eval()
                model.gradient_checkpointing = False
                try:
                    with torch.random.fork_rng(devices=[cuda_index]):
                        torch.manual_seed(0)
                        with torch.no_grad(), torch.autocast(
                            "cuda", dtype=torch.bfloat16
                        ):
                            val = float(eval_loss(
                                model, val_data, s_eval,
                                args.batch_size, args.device,
                            ))
                finally:
                    model.train()
                    model.gradient_checkpointing = True
                if val < best_val:
                    best_val = val
                    save_factors(global_completed, val)
                print(
                    f"    [r={r}] update {global_completed}/{total_steps} "
                    f"train_avg={train_loss_sum / args.accum:.4f} "
                    f"val={val:.4f} best={best_val:.4f} "
                    f"grad_norm={float(grad_norm):.4f} "
                    f"lr={lr:.2e} "
                    f"({time.perf_counter() - t0:.0f}s)",
                    flush=True,
                )
        print(f"    Best factor checkpoint: {best_path}", flush=True)
        model.eval()
        model.gradient_checkpointing = False
        for p in params:
            p.data = p.data.bfloat16()
            p.requires_grad_(False)

        after = eval_loss(model, val_data, s_eval, args.batch_size, args.device)
        out.append(dict(rank=r, loss_before=before, loss_after=after, eval_steps=s_eval))
        print(f"  rank {r}: {before:.4f} → {after:.4f}  (num_steps={s_eval})", flush=True)
        comp.restore()
    return out


@torch.no_grad()
def stage_latency(model, comp, args, B=1, S=1024, iters=10, warmup=3):
    steps_list = args.steps
    print("\n" + "=" * 78)
    print(f"FORWARD LATENCY  (B={B}, S={S}, bf16, eager, num_steps={steps_list})")
    print("=" * 78)
    x = torch.randint(0, 65000, (B, S), device=args.device)

    def run(ns):
        for _ in range(warmup):
            model(input_ids=x, num_steps=ns)
        torch.cuda.synchronize()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(iters):
            model(input_ids=x, num_steps=ns)
        b.record()
        torch.cuda.synchronize()
        return a.elapsed_time(b) / iters

    header = f"  {'model':>12} {'params':>10} | " + " ".join(f"{'s=' + str(s):>8}" for s in steps_list)
    print(header)
    print("  " + "-" * (len(header) - 2))

    rows = []

    # Full model at all step counts
    comp.restore()
    full_ms = {}
    for ns in steps_list:
        full_ms[ns] = run(ns)
    rows.append(dict(rank="full", params=count_params(model), ms=full_ms))
    print(f"  {'full':>12} {count_params(model) / 1e9:>9.2f}B | " +
          " ".join(f"{full_ms[s]:>7.1f}ms" for s in steps_list), flush=True)

    # Compressed at all step counts
    for r in args.ranks:
        comp.apply(r)
        comp_ms = {}
        for ns in steps_list:
            comp_ms[ns] = run(ns)
        rows.append(dict(rank=r, params=count_params(model), ms=comp_ms))
        print(f"  {f'r={r}':>12} {count_params(model) / 1e9:>9.2f}B | " +
              " ".join(f"{comp_ms[s]:>7.1f}ms" for s in steps_list), flush=True)
        # Speedup row
        print(f"  {'speedup':>12} {'':>10} | " +
              " ".join(f"{full_ms[s] / comp_ms[s]:>7.2f}x" for s in steps_list), flush=True)
        comp.restore()

    return rows


# ============================================================
# Activation-Aware SVD (ASVD)
# ============================================================

def factorize_asvd(W, S_cov, rank, eps=1e-5):
    """
    Activation-weighted SVD for W (d_out, d_in) given input second-moment S_cov (d_in, d_in).
    Minimizes ||(W - UV) X^T||_F.
    Returns U (d_out, rank) and V (rank, d_in) in bfloat16.
    """
    device = W.device
    W_f = W.detach().float()
    S_f = S_cov.to(device=device, dtype=torch.float32)

    # Ensure exact symmetry
    S_f = 0.5 * (S_f + S_f.T)
    eigvals, eigvecs = torch.linalg.eigh(S_f)

    # Ridge regularization: clamp eigenvalues to at least eps * max_eigval
    max_ev = eigvals.max().item()
    min_ev = max(1e-12, eps * max_ev)
    eigvals_reg = torch.clamp(eigvals, min=min_ev)

    sqrt_ev = torch.sqrt(eigvals_reg)
    inv_sqrt_ev = 1.0 / sqrt_ev

    # Factored multiplication without allocating (d_in, d_in)
    # W_scaled = (W_f @ eigvecs * sqrt_ev) @ eigvecs.T
    W_scaled = torch.matmul(W_f, eigvecs)
    W_scaled = W_scaled * sqrt_ev.unsqueeze(0)
    W_scaled = torch.matmul(W_scaled, eigvecs.T)

    P, Sigma, Qt = torch.linalg.svd(W_scaled, full_matrices=False)

    P_r = P[:, :rank]
    Sigma_r = Sigma[:rank]
    Qt_r = Qt[:rank, :]

    sqrt_sig = torch.sqrt(Sigma_r)
    U = P_r * sqrt_sig.unsqueeze(0)

    V_temp = sqrt_sig.unsqueeze(1) * Qt_r
    V_temp = torch.matmul(V_temp, eigvecs)
    V_temp = V_temp * inv_sqrt_ev.unsqueeze(0)
    V = torch.matmul(V_temp, eigvecs.T)

    return U.bfloat16(), V.bfloat16()


class CalibrationCollector:
    """Collects projection-input second-moment statistics (X^T @ X) for each core projection.
    Variant B: loop 16 for all M calibration sequences.
    Variant C: equal split across loops (1, 4, 8, 16, 32), M/5 sequences per loop.
    Both variants accumulate the exact same total number of token vectors."""

    def __init__(self, comp, sample_loops, total_seqs, seq_len):
        self.comp = comp
        self.sample_loops = sorted(sample_loops) # e.g. [1, 4, 8, 16, 32]
        self.step_to_loop = {s - 1: s for s in self.sample_loops}
        self.total_seqs = total_seqs
        self.num_loops = len(self.sample_loops)
        self.per_loop_seqs = total_seqs // self.num_loops

        # Mapping for Variant C: which sequence index is assigned to which loop
        self.seq_to_c_loop = {}
        for idx, loop_val in enumerate(self.sample_loops):
            start = idx * self.per_loop_seqs
            end = (idx + 1) * self.per_loop_seqs if idx < self.num_loops - 1 else total_seqs
            for k in range(start, end):
                self.seq_to_c_loop[k] = loop_val

        # Second moments (on CPU to preserve GPU memory) and token counts
        self.cov_B = {}
        self.count_B = {}
        self.cov_C = {}
        self.count_C = {}
        self.cov_per_loop = {}
        self.count_per_loop = {}

        for (i, t), lin in comp.orig.items():
            d_in = lin.in_features
            self.cov_B[(i, t)] = torch.zeros(d_in, d_in, dtype=torch.float32)
            self.count_B[(i, t)] = 0
            self.cov_C[(i, t)] = torch.zeros(d_in, d_in, dtype=torch.float32)
            self.count_C[(i, t)] = 0
            for l_val in self.sample_loops:
                self.cov_per_loop[(i, t, l_val)] = torch.zeros(d_in, d_in, dtype=torch.float32)
                self.count_per_loop[(i, t, l_val)] = 0

        self.current_seq_offset = 0
        self.step_counter = {k: 0 for k in comp.orig}
        self.handles = []
        self._register_hooks()

    def _register_hooks(self):
        for (i, t), lin in self.comp.orig.items():
            h = lin.register_forward_hook(self._make_hook(i, t))
            self.handles.append(h)

    def _make_hook(self, i, t):
        def hook(module, inputs, output):
            step = self.step_counter[(i, t)]
            self.step_counter[(i, t)] += 1
            if step in self.step_to_loop:
                loop_val = self.step_to_loop[step]
                x = inputs[0].detach() # (B, S, d_in)
                B, S, d_in = x.shape
                for b in range(B):
                    seq_idx = self.current_seq_offset + b
                    if seq_idx >= self.total_seqs:
                        continue
                    x_b = x[b].float() # (S, d_in) on GPU
                    xx = (x_b.T @ x_b).cpu() # (d_in, d_in) on CPU

                    # Variant B: accumulate loop 16 for ALL sequences
                    if loop_val == 16:
                        self.cov_B[(i, t)] += xx
                        self.count_B[(i, t)] += S

                    # Variant C: accumulate if seq_idx is assigned to loop_val
                    if self.seq_to_c_loop.get(seq_idx) == loop_val:
                        self.cov_C[(i, t)] += xx
                        self.count_C[(i, t)] += S

                    # Per-loop stats
                    self.cov_per_loop[(i, t, loop_val)] += xx
                    self.count_per_loop[(i, t, loop_val)] += S
        return hook

    def reset_step_counters(self, seq_offset):
        self.current_seq_offset = seq_offset
        for k in self.step_counter:
            self.step_counter[k] = 0

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()


def stage_asvd(model, comp, tok, val_data, args):
    """Activation-aware compression experiment:
    1. Collect projection-input second moments at loops 1, 4, 8, 16, 32 from train calibration set.
    2. Build three rank-2048 variants:
       Variant A: Plain SVD (weight-only baseline)
       Variant B: Single-loop calibration (loop 16)
       Variant C: Multi-loop calibration (split equally among loops 1, 4, 8, 16, 32)
    3. Evaluate without recovery at 16 and 32 loops on the 256-sequence validation set."""
    print("\n" + "=" * 78)
    print("ACTIVATION-AWARE SVD (ASVD) COMPRESSION")
    print(f"  calibration sequences: {args.calib_seqs} (train split)")
    print(f"  sample loops: {args.asvd_loops}")
    print(f"  target ranks: {args.ranks}")
    print("=" * 78)

    comp.restore()
    calib_data = build_token_set(tok, args.data_dir, "train", args.calib_seqs,
                                 args.seq_len, args.out_dir, skip_docs=args.calib_skip_docs)
    M = len(calib_data)
    print(f"Loaded calibration set: {M} sequences x {args.seq_len} tokens")

    collector = CalibrationCollector(comp, args.asvd_loops, M, args.seq_len)
    batch_size = min(2, args.batch_size)
    model.eval()

    t_calib = time.perf_counter()
    print("Collecting input second-moment statistics across loops...")
    with torch.no_grad():
        for b_idx, i in enumerate(range(0, M, batch_size)):
            chunk = calib_data[i:i + batch_size]
            collector.reset_step_counters(i)
            x = chunk[:, :-1].to(args.device)
            torch.manual_seed(1234 + b_idx)
            model(input_ids=x, num_steps=max(args.asvd_loops))
            if (b_idx + 1) % 5 == 0 or (i + batch_size) >= M:
                print(f"  processed {min(i + batch_size, M)}/{M} sequences "
                      f"({time.perf_counter() - t_calib:.1f}s)", flush=True)

    collector.remove()
    print(f"Calibration collection complete in {time.perf_counter() - t_calib:.1f}s.")

    sample_key = (0, "qkv")
    print(f"  Tokens per projection: Variant B = {collector.count_B[sample_key]:,}, "
          f"Variant C = {collector.count_C[sample_key]:,}")
    for l_val in sorted(args.asvd_loops):
        print(f"    Loop {l_val:>2} in Variant C: {collector.count_per_loop[(0, 'qkv', l_val)]:,} tokens")

    ckpt_dir = Path(args.out_dir) / "asvd_checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    ckpt_paths = []
    for rank in args.ranks:
        print(f"\nFactorizing core projections at rank {rank}...")
        t_fact = time.perf_counter()

        factors_plain = {}
        factors_b = {}
        factors_c = {}

        for (i, t), lin in comp.orig.items():
            W = lin.weight

            # Variant A: Plain SVD
            P, S, Qt = comp.svd[(i, t)]
            sqrt_s = S[:rank].sqrt()
            u_plain = (P[:, :rank].float() * sqrt_s).bfloat16()
            v_plain = (sqrt_s[:, None] * Qt[:rank].float()).bfloat16()
            factors_plain[(i, t)] = {"U.weight": u_plain, "V.weight": v_plain}

            # Variant B: Single-loop ASVD (loop 16)
            S_B = collector.cov_B[(i, t)] / max(1, collector.count_B[(i, t)])
            u_b, v_b = factorize_asvd(W, S_B, rank)
            factors_b[(i, t)] = {"U.weight": u_b, "V.weight": v_b}

            # Variant C: Multi-loop ASVD (loops 1, 4, 8, 16, 32)
            S_C = collector.cov_C[(i, t)] / max(1, collector.count_C[(i, t)])
            u_c, v_c = factorize_asvd(W, S_C, rank)
            factors_c[(i, t)] = {"U.weight": u_c, "V.weight": v_c}
            print(f"  factorized block[{i}].{t} (ASVD B and C)", flush=True)

        p_plain = ckpt_dir / f"factors_r{rank}_plain.pt"
        p_b = ckpt_dir / f"factors_r{rank}_asvd_single16.pt"
        p_c = ckpt_dir / f"factors_r{rank}_asvd_multi.pt"

        torch.save({"factors": factors_plain, "variant": "A_plain", "rank": rank}, p_plain)
        torch.save({"factors": factors_b, "variant": "B_asvd_single16", "rank": rank}, p_b)
        torch.save({"factors": factors_c, "variant": "C_asvd_multi", "rank": rank}, p_c)

        print(f"Saved factor checkpoints ({time.perf_counter() - t_fact:.1f}s):")
        print(f"  Variant A: {p_plain}")
        print(f"  Variant B: {p_b}")
        print(f"  Variant C: {p_c}")

        ckpt_paths.extend([str(p_plain), str(p_b), str(p_c)])

    # Evaluate without recovery on the validation set at specified loop steps
    print("\n" + "=" * 78)
    print(f"EVALUATING ASVD VARIANTS WITHOUT RECOVERY (steps={args.steps})")
    print("=" * 78)

    eval_args = copy.copy(args)
    eval_args.resume_factors = ckpt_paths
    eval_rows = stage_eval_loops(model, comp, val_data, eval_args)

    return {
        "calib_seqs": args.calib_seqs,
        "sample_loops": args.asvd_loops,
        "token_count": collector.count_B[sample_key],
        "checkpoints": ckpt_paths,
        "eval_results": eval_rows,
    }


# ============================================================
# Main
# ============================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", default="all",
                   choices=["all", "spectrum", "compress", "error", "recover",
                            "eval-loops", "latency", "asvd"])
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--out-dir", default="results/huginn")
    p.add_argument("--ranks", type=int, nargs="+", default=[256, 512, 1024, 2048])
    p.add_argument("--targets", nargs="+", default=list(ALL_TARGETS), choices=list(ALL_TARGETS))
    p.add_argument("--steps", type=int, nargs="+", default=[4, 8, 16, 32, 64],
                   help="num_steps (recurrences) to evaluate")
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--val-seqs", type=int, default=32)
    p.add_argument("--error-seqs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--latency-steps", type=int, default=32)
    # recovery
    p.add_argument("--recover-steps", type=int, default=200)
    p.add_argument("--recover-lr", type=float, default=3e-5)
    p.add_argument("--accum", type=int, default=8)
    p.add_argument("--train-seqs", type=int, default=2000)
    p.add_argument("--train-num-steps", type=int, default=32)
    p.add_argument("--grad-steps", type=int, default=8)
    p.add_argument("--recover-eval-steps", type=int, default=32)
    # resume / continuation / eval
    p.add_argument("--resume-factors", nargs="*", default=None,
                   help="Path(s) to saved factor checkpoint(s) to resume from or evaluate")
    p.add_argument("--reuse-baselines", default=None,
                   help="Path to previous results_eval-loops.json to reuse full and svd losses")
    p.add_argument("--start-it", type=int, default=0,
                   help="Starting data iterator position for continuation")
    p.add_argument("--start-step", type=int, default=0,
                   help="Global step offset for continuation runs")
    # activation-aware SVD (ASVD) calibration
    p.add_argument("--calib-seqs", type=int, default=50,
                   help="Number of training sequences for calibration")
    p.add_argument("--calib-skip-docs", type=int, default=0,
                   help="Document skip offset for calibration split")
    p.add_argument("--asvd-loops", type=int, nargs="+", default=[1, 4, 8, 16, 32],
                   help="Recurrent loops to sample for calibration")
    args = p.parse_args()

    args.device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"GPU: {torch.cuda.get_device_name(0)}   torch {torch.__version__}")
    print(f"config: {vars(args)}", flush=True)

    t0 = time.perf_counter()
    model, tok = load_huginn(args.device)
    print(f"loaded {MODEL_ID}: {count_params(model) / 1e9:.2f}B params "
          f"({time.perf_counter() - t0:.0f}s)", flush=True)

    val = build_token_set(tok, args.data_dir, "val", args.val_seqs, args.seq_len, args.out_dir)
    print(f"val set: {tuple(val.shape)}", flush=True)

    print("\n== SVD of core projections ==")
    
    comp = CoreCompressor(model, args.targets, max(args.ranks))
    print(f"  core target params: {comp.core_target_params():,}  "
          f"(core total {count_params(model.transformer.core_block):,})")

    results = {"config": {k: v for k, v in vars(args).items()}}
    run = lambda s: args.stage in ("all", s)

    if run("spectrum"):
        results["spectrum"] = stage_spectrum(comp, args.ranks)
    if run("compress"):
        results["compress"] = stage_compress(model, comp, val, args)
    if run("error"):
        results["error"] = stage_error(model, comp, val[: args.error_seqs], args)
    if run("latency"):
        results["latency"] = stage_latency(model, comp, args)
    if args.stage == "recover":
        train = build_token_set(tok, args.data_dir, "train", args.train_seqs,
                                args.seq_len, args.out_dir)
        results["recover"] = stage_recover(model, comp, train, val, args)
    if args.stage == "eval-loops":
        results["eval_loops"] = stage_eval_loops(model, comp, val, args)
    if args.stage == "asvd":
        results["asvd"] = stage_asvd(model, comp, tok, val, args)

    path = os.path.join(args.out_dir, f"results_{args.stage}.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nwrote {path}  (total {time.perf_counter() - t0:.0f}s)")


if __name__ == "__main__":
    main()
