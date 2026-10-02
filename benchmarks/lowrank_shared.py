"""
Phase 8 — Low-rank shared weights on a FineWeb-trained loop LM.

Question: if the shared Core weight W is replaced by a rank-r product UV
(the weight itself, not a LoRA update), how much language-modelling quality
is lost — and does the loss behave differently because the same UV is
re-applied at every loop?

Stages (run all with --stage all):
  train     train full-rank baselines: shared core (loop) + unshared control
  spectrum  singular-value energy of each trained core projection
  compress  post-training SVD compression of the core at several ranks:
            val loss + per-loop hidden-state error vs. the full model
  recover   compression + short fine-tune of the factors (recovery)
  scratch   train a low-rank shared core from scratch at each rank
  latency   core-block forward time, full vs. low-rank

Post-training compression and compression-with-recovery are reported
separately on purpose: they are different experiments.
"""

import argparse
import copy
import json
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from attention_v2 import ExplicitTransformerBlock
from loop_lm import DEFAULT_DATA_DIR, FineWeb, LoopLM
from lowrank import ALL_TARGETS, TARGETS, _get, compress_block, count_params, lowrank_block


# ============================================================
# Training / evaluation
# ============================================================

def lr_at(step, steps, lr, warmup):
    if step < warmup:
        return lr * (step + 1) / warmup
    t = (step - warmup) / max(1, steps - warmup)
    return lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * t)))


def train(model, data, args, steps, lr, label, params=None):
    model.train()
    params = list(params if params is not None else model.parameters())
    opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.95),
                            weight_decay=0.1, fused=True)
    warmup = min(200, steps // 10)

    t0 = time.perf_counter()
    for step in range(steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, steps, lr, warmup)

        x, y = data.batch(args.batch_size, args.seq_len, args.device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(x, y)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()

        if step % args.log_every == 0 or step == steps - 1:
            print(f"  [{label}] step {step:5d}/{steps}  loss {loss.item():.4f}  "
                  f"({time.perf_counter() - t0:.0f}s)", flush=True)

    model.eval()
    return model


@torch.no_grad()
def evaluate(model, val_batches):
    model.eval()
    total = 0.0
    for x, y in val_batches:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(x, y)
        total += loss.item()
    return total / len(val_batches)


@torch.no_grad()
def per_loop_error(ref, model, val_batches):
    """Relative error ||h_t - h_t^ref|| / ||h_t^ref|| after each core loop."""
    errs = torch.zeros(ref.loops)
    for x, _ in val_batches:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, _, s_ref = ref(x, return_states=True)
            _, _, s = model(x, return_states=True)
        for t, (a, b) in enumerate(zip(s_ref, s)):
            errs[t] += ((b.float() - a.float()).norm() / a.float().norm()).item()
    return (errs / len(val_batches)).tolist()


# ============================================================
# Compression helpers
# ============================================================

def compress_model(model, rank, targets, fresh=False):
    """Copy of `model` with every core block's projections replaced by rank-r
    factors (SVD of the trained weights, or fresh init if `fresh`)."""
    m = copy.deepcopy(model)
    fn = lowrank_block if fresh else compress_block
    if m.shared:
        m.core = fn(m.core, rank, targets)
    else:
        for i in range(m.loops):
            m.cores[i] = fn(m.cores[i], rank, targets)
    return m


def core_params(model):
    return count_params(model.core) if model.shared else count_params(model.cores)


def core_factor_params(model):
    """Parameters of the core's compressible projections only."""
    blocks = [model.core] if model.shared else list(model.cores)
    return sum(count_params(_get(b, TARGETS[t])) for b in blocks for t in ALL_TARGETS)


def make_model(args, shared):
    return LoopLM(d_model=args.d_model, n_heads=args.n_heads, loops=args.loops,
                  max_seq_len=args.seq_len, shared=shared).to(args.device)


def ckpt_path(args, name):
    return os.path.join(args.out_dir, f"{name}.pt")


def load_or_train(args, data, shared):
    name = "baseline_shared" if shared else "baseline_unshared"
    model = make_model(args, shared)
    path = ckpt_path(args, name)
    if os.path.exists(path) and not args.retrain:
        model.load_state_dict(torch.load(path, map_location=args.device))
        print(f"  loaded {path}")
    else:
        train(model, data, args, args.steps, args.lr, name)
        torch.save(model.state_dict(), path)
    model.eval()
    return model


# ============================================================
# Stages
# ============================================================

def stage_spectrum(models, ranks):
    """Fraction of squared-Frobenius energy kept by the top-r singular values."""
    print("\n" + "=" * 78)
    print("SPECTRUM — energy kept at rank r (sum σ_i² for i<r / sum σ_i²)")
    print("=" * 78)
    out = {}
    for kind, model in models.items():
        blocks = {"core": model.core} if model.shared else \
                 {f"core{i}": b for i, b in enumerate(model.cores)}
        for bname, block in blocks.items():
            for t in ALL_TARGETS:
                W = _get(block, TARGETS[t]).weight.detach().float()
                s2 = torch.linalg.svdvals(W) ** 2
                cum = (s2.cumsum(0) / s2.sum()).tolist()
                row = {r: cum[min(r, len(cum)) - 1] for r in ranks}
                out[f"{kind}/{bname}/{t}"] = row
                print(f"  {kind:>8}/{bname:<6}/{t:<5} " +
                      "  ".join(f"r{r}:{row[r]:.3f}" for r in ranks))
    return out


def stage_compress(models, val_batches, ranks, targets):
    print("\n" + "=" * 78)
    print(f"POST-TRAINING SVD COMPRESSION  (targets={','.join(targets)})")
    print("=" * 78)
    out = {}
    for kind, model in models.items():
        base = evaluate(model, val_batches)
        full_p = core_factor_params(model)
        print(f"\n  {kind}: full-rank val loss {base:.4f}  "
              f"(core projection params {full_p:,})")
        print(f"  {'rank':>5} {'params':>10} {'ratio':>6} {'loss':>8} {'Δloss':>8}  "
              f"per-loop rel. error")
        rows = []
        for r in ranks:
            m = compress_model(model, r, targets)
            loss = evaluate(m, val_batches)
            errs = per_loop_error(model, m, val_batches)
            p = core_factor_params(m)
            rows.append(dict(rank=r, params=p, loss=loss, dloss=loss - base,
                             per_loop_err=errs))
            print(f"  {r:>5} {p:>10,} {full_p / p:>5.1f}x {loss:>8.4f} "
                  f"{loss - base:>+8.4f}  " + " ".join(f"{e:.3f}" for e in errs))
            del m
        out[kind] = dict(full_loss=base, full_params=full_p, rows=rows)
    return out


def stage_recover(models, data, val_batches, ranks, targets, args):
    print("\n" + "=" * 78)
    print(f"COMPRESSION + RECOVERY  ({args.recover_steps} fine-tune steps, all params)")
    print("=" * 78)
    out = {}
    for kind, model in models.items():
        rows = []
        for r in ranks:
            m = compress_model(model, r, targets)
            before = evaluate(m, val_batches)
            train(m, data, args, args.recover_steps, args.recover_lr,
                  f"{kind} r={r} recover")
            after = evaluate(m, val_batches)
            rows.append(dict(rank=r, loss_before=before, loss_after=after))
            print(f"  {kind} rank {r}: {before:.4f} → {after:.4f}")
            del m
        out[kind] = rows
    return out


def stage_scratch(args, data, val_batches, ranks, targets):
    print("\n" + "=" * 78)
    print(f"LOW-RANK SHARED CORE TRAINED FROM SCRATCH  ({args.steps} steps)")
    print("=" * 78)
    rows = []
    for r in ranks:
        torch.manual_seed(args.seed)
        # Default nn.Linear init on U and V gives std(UV) ≈ 1/(3·sqrt(d_in))
        # ≈ 0.015 for d=512, close to the 0.02 used for the rest of the model.
        m = compress_model(make_model(args, shared=True), r, targets, fresh=True)
        train(m, data, args, args.steps, args.lr, f"scratch r={r}")
        loss = evaluate(m, val_batches)
        rows.append(dict(rank=r, params=core_factor_params(m), loss=loss))
        print(f"  rank {r}: val loss {loss:.4f}  (core projection params "
              f"{core_factor_params(m):,})")
        del m
    return rows


@torch.no_grad()
def stage_latency(args, ranks, targets, B=32, S=512, iters=200, warmup=20):
    print("\n" + "=" * 78)
    print(f"CORE-BLOCK LATENCY  (B={B}, S={S}, D={args.d_model}, bf16, "
          f"{args.loops} loops, eager)")
    print("=" * 78)

    def time_loop(block, x):
        for _ in range(warmup):
            h = x
            for _ in range(args.loops):
                h = block(h)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            h = x
            for _ in range(args.loops):
                h = block(h)
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / iters

    torch.manual_seed(args.seed)
    full = ExplicitTransformerBlock(args.d_model, args.n_heads, causal=True) \
        .to(args.device).bfloat16().eval()
    x = torch.randn(B, S, args.d_model, device=args.device, dtype=torch.bfloat16)

    base = time_loop(full, x)
    rows = [dict(rank="full", ms=base, params=count_params(full))]
    print(f"  {'full':>5}: {base:.4f} ms  ({count_params(full):,} params)")
    for r in ranks:
        b = compress_block(full.float(), r, targets).bfloat16().eval()
        full.bfloat16()
        ms = time_loop(b, x)
        rows.append(dict(rank=r, ms=ms, params=count_params(b)))
        print(f"  {r:>5}: {ms:.4f} ms  ({count_params(b):,} params)  "
              f"{base / ms:.2f}x vs full")
    return rows


# ============================================================
# Main
# ============================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", default="all",
                   choices=["all", "train", "spectrum", "compress", "recover",
                            "scratch", "latency"])
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--out-dir", default="results/lowrank")
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--loops", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--recover-steps", type=int, default=400)
    p.add_argument("--recover-lr", type=float, default=2e-4)
    p.add_argument("--ranks", type=int, nargs="+", default=[16, 32, 64, 128, 256])
    p.add_argument("--targets", nargs="+", default=list(ALL_TARGETS),
                   choices=list(ALL_TARGETS))
    p.add_argument("--val-batches", type=int, default=20)
    p.add_argument("--log-every", type=int, default=250)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--retrain", action="store_true")
    p.add_argument("--no-unshared", action="store_true",
                   help="skip the unshared control model")
    args = p.parse_args()

    args.device = "cuda"
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"GPU: {torch.cuda.get_device_name(0)}   torch {torch.__version__}")
    print(f"config: {vars(args)}")

    results = {"config": {k: v for k, v in vars(args).items()}}
    run = lambda s: args.stage in ("all", s)

    if args.stage == "latency":
        results["latency"] = stage_latency(args, args.ranks, args.targets)
    else:
        train_data = FineWeb(args.data_dir, "train", seed=args.seed)
        val_data = FineWeb(args.data_dir, "val")
        val_batches = val_data.fixed_batches(args.val_batches, args.batch_size,
                                             args.seq_len, args.device)

        if args.stage == "scratch":
            models = {}
        else:
            print("\n== baselines ==")
            models = {"shared": load_or_train(args, train_data, shared=True)}
            if not args.no_unshared:
                models["unshared"] = load_or_train(args, train_data, shared=False)
            for k, m in models.items():
                print(f"  {k}: total params {count_params(m):,}  "
                      f"core params {core_params(m):,}")

        if run("spectrum"):
            results["spectrum"] = stage_spectrum(models, args.ranks)
        if run("compress"):
            results["compress"] = stage_compress(models, val_batches,
                                                 args.ranks, args.targets)
        if run("recover"):
            results["recover"] = stage_recover(models, train_data, val_batches,
                                               args.ranks, args.targets, args)
        if run("scratch"):
            results["scratch"] = stage_scratch(args, train_data, val_batches,
                                               args.ranks, args.targets)
        if run("latency"):
            results["latency"] = stage_latency(args, args.ranks, args.targets)

    path = os.path.join(args.out_dir, f"results_{args.stage}.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
