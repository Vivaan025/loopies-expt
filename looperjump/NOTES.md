# NOTES.md — Certified Latent Jumping for Looped Transformers
# Updated as work progresses. Each section is timestamped.

## [2026-09-30] Stage 0: Literature Review

### Paper Summaries

#### 1. Huginn (arXiv 2502.05171) — "Scaling up Test-Time Compute with Latent Reasoning"
- **What:** 3.5B-parameter looped transformer (prelude/core/coda). Core is 4 SandwichBlocks
  iterated up to 32+ times. Uses linear adapter injection (NOT addition) to combine state
  with prelude output. State initialized with "like-init" distribution.
- **Compare against:** Standard transformers of equivalent FLOPs.
- **Not tested:** Fixed-point acceleration, Anderson acceleration, learned jump predictors.
  They use fixed depth or Huginn's own adaptive halting. No residual-based certification.

#### 2. Parallel Samplers (arXiv 2510.14961) — "Efficient Parallel Samplers for Recurrent-Depth Models"
- **What:** Treats looped transformer as a diffusion process. New token decoded at every
  forward pass while latent states refine in parallel. Up to 5× speedup on existing 3.5B models.
- **Compare against:** Standard autoregressive generation.
- **Not tested:** Fixed-point certification, Anderson acceleration, per-token convergence analysis.
  Focus is on token-level parallelism, not loop-level skipping.

#### 3. DCP (arXiv 2609.19934) — "Beyond Depth Truncation"
- **What:** Shows that naive depth truncation conflates three effects: fewer block applications,
  less distinct computation, and out-of-distribution readout head input. Proposes Depth Control
  Protocol (DCP) to disentangle them.
- **Key diagnostic:** "linchpin control" — run the full budget of block applications while
  executing only a single distinct iteration. Only realizable in weight-sharing architectures.
- **DCP results on Huginn truncation cost:** PENDING — need to extract specific numbers from
  the full paper (HTML version was not fully readable). TODO: read the actual paper PDF and
  record the depth-vs-quality curves and truncation numbers here.
- **Compare against:** Naive truncation baselines.
- **Not tested:** Anderson acceleration, learned jumps, certification. Pure diagnostic.
- **CRITICAL for us:** If truncation is already nearly free for Huginn (as some prior work
  suggests), our method has limited headroom. This is why Stage 2 comes before method work.

#### 4. LoopSpec (arXiv 2609.17184) — "Pipelined Self-Speculative Decoding for Looped Transformers"
- **What:** Uses intermediate loop states as draft tokens for speculative decoding. Pipelined:
  draft generation overlaps with verification. Selective second proposal from deeper states.
  Up to 6.83× speedup. Training-free, lossless.
- **Compare against:** Standard autoregressive, other self-speculative methods.
- **Not tested:** Fixed-point certification, Anderson acceleration. Focus is token-level
  speculation, not loop-level skipping.

### Literature Search: Prior Work on Extrapolation + Verify + Skip

**Queries run:**
1. `"looped transformer" OR "recurrent depth" AND "Anderson acceleration"` → found "Training-Free Looped Transformers" (2026)
2. `"latent jump" OR "learned solver" OR "skip iterations" looped transformer fixed point verify certificate` → found VeriLoop framework
3. `Anderson acceleration deep equilibrium model inference skip verify residual certificate` → found Consistency DEQs, Neural DEQ Solvers

**Key finding — POTENTIAL OVERLAP:**
- **"Training-Free Looped Transformers" (2026):** Benchmarked Anderson acceleration, heavy-ball,
  Runge-Kutta on frozen looped transformers. CONCLUSION: looped blocks are "often not contractive,"
  so Anderson acceleration was not effective. RK methods outperformed AA.
  
  **CRITICAL DIFFERENCE from our work:** That paper applies acceleration to NON-LOOPED models
  (retrofitting recurrence onto frozen standard transformers). Huginn is a NATIVELY looped model
  trained with weight-sharing. The contractivity properties may differ. We MUST verify this
  empirically in Stage 1 (measure rho_k on Huginn specifically).

- **VeriLoop (2026):** Framework for "Evidence-Governed Recurrence" — formal verification of
  code/math agents, not loop iteration skipping. Different usage of "certificate." NOT a direct
  overlap with our work.

- **Neural DEQ Solvers (ICLR 2025):** Learn a solver for DEQs. Different: DEQs are trained
  to converge; looped LLMs are trained with fixed depth. The convergence behavior differs.

- **Consistency DEQs (2026):** Distill equilibrium mapping for few-step inference. Requires
  training. Our approach is training-free (except optional learned predictor in Stage 5).

**CONCLUSION:** No paper does exactly {Anderson/learned extrapolation + residual certificate +
skip iterations} on a natively looped LLM like Huginn. The "Training-Free Looped Transformers"
paper comes closest but works on different models (retrofitted recurrence, not native).
**Literature check: no redundant prior work found.**

---

## [2026-09-30] Huginn Model Configuration (from config.json)

**Source:** https://huggingface.co/tomg-group-umd/huginn-0125/raw/main/config.json

```json
{
  "n_embd": 5280,
  "n_heads": 55,
  "num_key_value_heads": 55,     // NO GQA — all heads are KV heads
  "head_dim": 96,
  "intermediate_size": 17920,
  "vocab_size": 65536,
  "padded_vocab_size": 65536,
  "n_layers": 8,
  "n_layers_in_prelude": 2,
  "n_layers_in_recurrent_block": 4,
  "n_layers_in_coda": 2,
  "injection_type": "linear",     // Linear adapter, NOT addition
  "state_init": "like-init",      // Initialized like weight init (std ≈ 0.0087)
  "mean_recurrence": 32,
  "mean_backprop_depth": 8,
  "effective_expected_depth": 132,
  "block_class_name": "SandwichBlock",
  "mlp_class_name": "GatedMLP",
  "nonlin_name": "SiLU",
  "norm_class_name": "RMSNorm_llama",
  "norm_eps": 1e-06,
  "qk_bias": true,
  "tie_embeddings": true,
  "rope_base": 50000,
  "block_size": 4096,
  "bias": false,
  "init_strategy": "takase",
  "init_values": {
    "std": 0.008703882797784892,
    "out_proj": 0.0005356869554443541,
    "embedding": 0.008703882797784892,
    "embed_scale": 72.6636084983398
  },
  "sampling_scheme": "poisson-lognormal-filling"
}
```

### Input Injection (injection_type = "linear")

The input injection is a **learned linear adapter** on `cat([state, prelude_out])`,
NOT simple addition:

```python
# R(s, e) — one core iteration:
h = adapter(cat([s, e], dim=-1))   # Linear(2*5280 → 5280)
h = core_layer_0(h)                # SandwichBlock
h = core_layer_1(h)                # SandwichBlock
h = core_layer_2(h)                # SandwichBlock
h = core_layer_3(h)                # SandwichBlock
# h is now s_{k+1}
```

The adapter's exact location in the module tree needs to be discovered by running
`discover_huginn.py` on the 5090. Possible locations:
- `model.model.adapter`
- `model.adapter`
- Named differently — search by shape `Linear(10560, 5280)`

### State Initialization (state_init = "like-init")

Initial state s_0 is drawn from truncated normal with:
- mean = 0, std = 0.008703882797784892 (same as weight init)
- truncated at ±3σ

This is NOT standard torch.randn(). Seeds must be fixed and logged.

### VRAM Estimates (corrected)

**Model weights:**
- ~3.5B params × 2 bytes (bf16) = **7.0 GB**

**KV cache per core iteration:**
- K per layer: B × 55 × S × 96 × 2 bytes (bf16) = B × S × 10560 bytes
- V per layer: same
- Per iteration: 4 layers × 2 (K+V) × B × S × 10560 bytes = 8 × B × S × 10560 bytes
- 32 iterations: 32 × 8 × B × S × 10560 = **2.6 GB** for B=1, S=512

**Activation memory (per forward pass):**
- Hidden states: B × S × 5280 × 2 = B × S × 10560 bytes
- QKV: B × S × 3 × 5280 × 2 = B × S × 31680 bytes
- FFN up: B × S × 2 × 17920 × 2 (GatedMLP) = B × S × 71680 bytes
- Peak per layer: ~B × S × 120 KB
- Total peak: ~**0.5 GB** for B=1, S=512

**Total:** 7.0 + 2.6 + 0.5 + overhead ≈ **12 GB**. Fits comfortably on 5090 (32 GB).

### KV-Cache Lookup Strategies (from source code)

Confirmed from `HuginnDynamicCache` and `HuginnStaticCache` source code:
- `"full"`: all entries present, materialize normally
- `"latest-m4"`: for missing entries, use latest available from same modulo-4 step
- `"available-m4"`: use current step if available, else latest matching mod-4
- `"always-last-m4"`: always use last available matching mod-4
- `"skip"`: only use entries that actually exist
- `"randomized"`: random valid step from same mod-4 group (sanity check)
- `"compress-sN"`: modulo-N compression of step indices
- `"compress-rN"`: floor-division compression
- `"compress-anchor"`: anchor first 8 recurrence steps, reuse next 4 for all future

**Modulo-4 pattern:** Huginn has 4 core layers. Each "recurrence step" increments
through 4 sub-steps (one per core layer). The mod-4 lookup ensures the right layer's
KV is retrieved. Step indices: 0,1 = prelude layers; 2,3,4,5 = first core iteration;
6,7,8,9 = second; etc.

---

## [2026-09-30] Stage 0: Baseline Reproduction

**Status: IN PROGRESS**

Gate 0 requires:
1. ✅ Literature check — no redundant prior work found
2. ⬜ Decomposed forward pass matches library output numerically
   - Need to run `discover_huginn.py` on 5090 to verify module tree
   - Need to locate the injection adapter and verify R(s,e) definition
   - Need to match state initialization seeds
3. ⬜ Verify prelude/core_step/coda decomposition reproduces full forward pass

**Gate 0: NOT YET EVALUATED**

---

## [2026-09-30] DCP Paper Truncation Numbers

**Status: PENDING** — Could not extract numbers from arxiv HTML (content too short).
Need to read the full paper. TODO: Download PDF and extract:
- Perplexity vs. retained depth fraction for Huginn
- The DCP control results
- Specific truncation cost (e.g., "truncating to 50% depth costs X% perplexity")

---

## [2026-09-30] Stage 3 Design Note: Ablation Extrapolators

Per plan correction, Stage 3 will test three extrapolators as ablations:
1. **Anderson acceleration (Type-II)** — primary method
2. **Aitken acceleration** — Δ² extrapolation on scalar/per-component sequences
3. **Heavy-ball** — momentum-based: x_{k+1} = g(x_k) + β(x_k - x_{k-1})

All three use the same certificate residual c(h) = ‖R(h,e) − h‖ / ‖h‖ for stopping.

---

## Deviations from Plan

1. [2026-09-30] Could not download full raven_modeling_minimal.py (~1000+ lines,
   truncated at 462). Created discover_huginn.py to run on 5090 instead of
   relying on remote code reading.
2. [2026-09-30] Config numbers corrected from actual config.json (was wrong:
   had n_embd=2560, 20 heads, GQA with 4 KV heads. Actual: n_embd=5280,
   55 heads, no GQA, 55 KV heads).
3. [2026-09-30] Input injection corrected: is Linear adapter on concat, NOT
   simple addition. The adapter is Linear(10560 → 5280).
4. [2026-09-30] DCP paper numbers not yet extracted (HTML parsing incomplete).
5. [2026-09-30] Added Aitken and heavy-ball to Stage 3 ablations.
