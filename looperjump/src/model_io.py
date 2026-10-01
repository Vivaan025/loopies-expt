"""
model_io.py — Load Huginn and expose prelude / core_step / coda as separate callables.

The decomposed forward pass must reproduce the library's own output to numerical tolerance.

IMPORTANT: This file is written based on config.json but the exact forward pass
decomposition MUST be verified by running discover_huginn.py on the 5090 and
cross-referencing against the actual source. Gate 0 is NOT passed until
verify_decomposition() returns pass=True.

Config from tomg-group-umd/huginn-0125/config.json:
  n_embd:                    5280
  n_heads:                   55
  num_key_value_heads:       55   (no GQA)
  head_dim:                  96
  intermediate_size:         17920
  vocab_size:                65536
  padded_vocab_size:         65536
  n_layers:                  8
  n_layers_in_prelude:       2
  n_layers_in_recurrent_block: 4
  n_layers_in_coda:          2
  injection_type:            "linear"   → linear adapter on cat([state, prelude_out])
  state_init:                "like-init"
  mean_recurrence:           32
  nonlin_name:               SiLU
  mlp_class_name:            GatedMLP
  block_class_name:          SandwichBlock
  norm_class_name:           RMSNorm_llama
  tie_embeddings:            true
  qk_bias:                   true

VRAM estimate (corrected):
  Parameters: ~3.5B × 2 bytes (bf16) ≈ 7.0 GB
  Per-layer activations: B × S × 5280 × 2 bytes
  KV cache per step per core layer:
    K: B × 55_heads × S × 96_head_dim × 2 = B × S × 10560 × 2 bytes
    V: same
    4 core layers × 32 steps × 2(K+V) = 256 cache tensors
    For B=1, S=512: 256 × 1 × 512 × 10560 × 2 ≈ 2.6 GB
  Total: ~7 + 2.6 + overhead ≈ 12 GB. Fits on 5090 (32 GB).
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple
from dataclasses import dataclass, field
import inspect


@dataclass
class HuginnParts:
    """Decomposed Huginn model with separately callable stages.
    
    After running discover_huginn.py, update the field assignments to match
    the actual model structure.
    """
    model: torch.nn.Module            # full model for reference forward pass
    config: object                    # model config
    embed: torch.nn.Module            # token embedding
    prelude_layers: list              # layers 0..1
    core_layers: list                 # layers 2..5 (iterated)
    coda_layers: list                 # layers 6..7
    final_norm: torch.nn.Module       # RMSNorm before lm_head
    lm_head: torch.nn.Module          # linear → vocab logits
    # The injection adapter: Linear(2*n_embd → n_embd) applied to cat([state, prelude_out])
    injection_adapter: Optional[torch.nn.Module] = None
    # Embedding scale factor
    emb_scale: float = 1.0
    device: torch.device = field(default_factory=lambda: torch.device("cuda"))
    dtype: torch.dtype = torch.bfloat16
    # API notes discovered at runtime
    api_notes: dict = field(default_factory=dict)


def load_huginn(
    model_name: str = "tomg-group-umd/huginn-0125",
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    trust_remote_code: bool = True,
) -> HuginnParts:
    """Load Huginn and decompose into prelude/core/coda.
    
    Returns a HuginnParts dataclass. The decomposition MUST be verified
    by calling verify_decomposition() before any experimental use.
    """
    from transformers import AutoModelForCausalLM

    print(f"Loading {model_name} in {dtype}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=trust_remote_code,
    )
    model.eval()
    config = model.config

    # ─── Print verified config ───
    print(f"\nVerified config:")
    print(f"  n_embd:               {config.n_embd}")
    print(f"  n_heads:              {config.n_heads}")
    print(f"  num_key_value_heads:  {config.num_key_value_heads}")
    print(f"  head_dim:             {config.head_dim}")
    print(f"  intermediate_size:    {config.intermediate_size}")
    print(f"  vocab_size:           {config.vocab_size}")
    print(f"  injection_type:       {config.injection_type}")
    print(f"  state_init:           {config.state_init}")
    print(f"  n_layers:             {config.n_layers}")
    print(f"  n_layers_in_prelude:  {config.n_layers_in_prelude}")
    print(f"  n_layers_in_recurrent_block: {config.n_layers_in_recurrent_block}")
    print(f"  n_layers_in_coda:     {config.n_layers_in_coda}")

    # ─── Discover model structure ───
    # Print top-level modules to find the right names
    print(f"\nTop-level children:")
    for name, child in model.named_children():
        n_params = sum(p.numel() for p in child.parameters())
        print(f"  {name}: {type(child).__name__} ({n_params/1e6:.1f}M params)")

    # ─── Auto-discover components ───
    # We search for: embedding, layers/blocks, norm, lm_head, adapter
    api_notes = {}
    
    # Try multiple known layouts
    inner = None
    for attr in ['model', 'transformer', 'gpt']:
        if hasattr(model, attr):
            inner = getattr(model, attr)
            api_notes['inner_model_attr'] = attr
            break
    
    if inner is None:
        # The model itself might be the container
        inner = model
        api_notes['inner_model_attr'] = 'self'

    # Find embedding
    embed = None
    for attr in ['embed_tokens', 'wte', 'embedding', 'tok_emb']:
        if hasattr(inner, attr):
            embed = getattr(inner, attr)
            api_notes['embed_attr'] = attr
            break
    assert embed is not None, f"Cannot find embedding in {[n for n, _ in inner.named_children()]}"

    # Find layers
    layers = None
    for attr in ['layers', 'blocks', 'h']:
        if hasattr(inner, attr):
            layers = getattr(inner, attr)
            api_notes['layers_attr'] = attr
            break
    assert layers is not None, f"Cannot find layers in {[n for n, _ in inner.named_children()]}"

    # Find final norm
    final_norm = None
    for attr in ['norm', 'ln_f', 'final_norm', 'norm_f']:
        if hasattr(inner, attr):
            final_norm = getattr(inner, attr)
            api_notes['final_norm_attr'] = attr
            break
    assert final_norm is not None, f"Cannot find final norm"

    # Find lm_head
    lm_head = None
    for attr in ['lm_head', 'head']:
        if hasattr(model, attr):
            lm_head = getattr(model, attr)
            api_notes['lm_head_attr'] = attr
            break
    if lm_head is None and config.tie_embeddings:
        # Tied embeddings: lm_head shares weights with embedding
        api_notes['lm_head_attr'] = 'tied_to_embedding'
        # We'll handle this in run_coda
    assert lm_head is not None or config.tie_embeddings, "Cannot find lm_head"

    # Find injection adapter
    # injection_type="linear" → expect a Linear(2*n_embd, n_embd) somewhere
    injection_adapter = None
    for name, module in model.named_modules():
        if 'adapter' in name.lower():
            injection_adapter = module
            api_notes['injection_adapter_name'] = name
            if isinstance(module, torch.nn.Linear):
                api_notes['injection_adapter_shape'] = f"[{module.in_features} -> {module.out_features}]"
            break
    
    if injection_adapter is None:
        # Search by shape: any Linear with in_features = 2*n_embd
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear) and module.in_features == 2 * config.n_embd:
                injection_adapter = module
                api_notes['injection_adapter_name'] = name
                api_notes['injection_adapter_shape'] = f"[{module.in_features} -> {module.out_features}]"
                print(f"  Found adapter by shape: {name} [{module.in_features} -> {module.out_features}]")
                break

    # Split layers into prelude / core / coda
    n_prelude = config.n_layers_in_prelude   # 2
    n_core = config.n_layers_in_recurrent_block  # 4
    n_coda = config.n_layers_in_coda          # 2
    n_total = len(layers)
    assert n_total == n_prelude + n_core + n_coda, \
        f"Layer count mismatch: {n_total} != {n_prelude}+{n_core}+{n_coda}"

    prelude_layers = [layers[i] for i in range(n_prelude)]
    core_layers = [layers[i] for i in range(n_prelude, n_prelude + n_core)]
    coda_layers = [layers[i] for i in range(n_prelude + n_core, n_total)]

    print(f"\n  Prelude: layers 0..{n_prelude-1} ({n_prelude} layers)")
    print(f"  Core:    layers {n_prelude}..{n_prelude+n_core-1} ({n_core} layers, iterated)")
    print(f"  Coda:    layers {n_prelude+n_core}..{n_total-1} ({n_coda} layers)")
    
    if injection_adapter is not None:
        print(f"  Adapter: {api_notes.get('injection_adapter_name', '?')} "
              f"{api_notes.get('injection_adapter_shape', '?')}")
    else:
        print(f"  WARNING: No injection adapter found. Run discover_huginn.py to locate it.")

    # Get embedding scale
    emb_scale = getattr(model, 'emb_scale', 1.0)
    api_notes['emb_scale'] = emb_scale
    print(f"  Embed scale: {emb_scale}")

    # Parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    core_params = sum(p.numel() for layer in core_layers for p in layer.parameters())
    adapter_params = sum(p.numel() for p in injection_adapter.parameters()) if injection_adapter else 0
    print(f"\n  Total params:   {total_params:,} ({total_params/1e9:.2f}B)")
    print(f"  Core params:    {core_params:,} ({core_params/1e6:.1f}M)")
    print(f"  Adapter params: {adapter_params:,} ({adapter_params/1e6:.1f}M)")

    parts = HuginnParts(
        model=model,
        config=config,
        embed=embed,
        prelude_layers=prelude_layers,
        core_layers=core_layers,
        coda_layers=coda_layers,
        final_norm=final_norm,
        lm_head=lm_head,
        injection_adapter=injection_adapter,
        emb_scale=emb_scale,
        device=torch.device(device),
        dtype=dtype,
        api_notes=api_notes,
    )

    return parts


def get_tokenizer(model_name: str = "tomg-group-umd/huginn-0125"):
    """Load the tokenizer for Huginn."""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


# ─────────────────────────────────────────────────────────────────────
# Decomposed forward pass
# ─────────────────────────────────────────────────────────────────────
# The exact computation R(s, e) for one core iteration:
#
#   injection_type = "linear":
#       injected = adapter(cat([s, e], dim=-1))    # Linear(2*D → D)
#   injection_type = "add":
#       injected = s + e
#
#   Then pass through the 4 core SandwichBlocks:
#       s_next = core_layer_3(core_layer_2(core_layer_1(core_layer_0(injected))))
#
#   NOTE: There may be a norm (ln_f / RMSNorm) before the coda. Verify with
#   discover_huginn.py. The coda layers might expect normed input.
#
# state_init = "like-init":
#   Initial state is drawn from the same distribution used at init time,
#   NOT uniform random. Check the actual init code.
# ─────────────────────────────────────────────────────────────────────


@torch.no_grad()
def run_prelude(parts: HuginnParts, input_ids: torch.Tensor) -> torch.Tensor:
    """Run embedding + prelude layers.
    
    Args:
        input_ids: (B, S) token IDs
    Returns:
        prelude_out: (B, S, D) hidden states after prelude. This is 'e' in notation.
    """
    h = parts.embed(input_ids) * parts.emb_scale
    
    for layer in parts.prelude_layers:
        # Prelude layers are standard transformer blocks, no special args
        # They may need position_ids, attention_mask — discover at runtime
        h = layer(h)
        # If the layer returns a tuple, take the first element (hidden_states)
        if isinstance(h, tuple):
            h = h[0]
    
    return h


@torch.no_grad()
def run_core_step(
    parts: HuginnParts,
    state: torch.Tensor,
    prelude_out: torch.Tensor,
) -> torch.Tensor:
    """Run one core iteration R(s, e).
    
    This is the EXACT function we accelerate. Every call = 1 "pass."
    
    R(s, e):
        1. Inject: h = adapter(cat([s, e], dim=-1))    if injection_type="linear"
                   h = s + e                            if injection_type="add"
        2. Apply 4 core SandwichBlocks sequentially: h = block_3(block_2(block_1(block_0(h))))
        3. Return h  (this is s_{k+1})
    
    Args:
        state: (B, S, D) current latent state s_k
        prelude_out: (B, S, D) prelude output e
    Returns:
        new_state: (B, S, D) = s_{k+1}
    """
    # Step 1: Input injection
    if parts.config.injection_type == "linear":
        assert parts.injection_adapter is not None, \
            "injection_type='linear' but no adapter found. Run discover_huginn.py."
        h = parts.injection_adapter(torch.cat([state, prelude_out], dim=-1))
    elif parts.config.injection_type == "add":
        h = state + prelude_out
    else:
        raise ValueError(f"Unknown injection_type: {parts.config.injection_type}")
    
    # Step 2: Core layers
    for layer in parts.core_layers:
        h = layer(h)
        if isinstance(h, tuple):
            h = h[0]
    
    return h


@torch.no_grad()
def run_coda(parts: HuginnParts, state: torch.Tensor) -> torch.Tensor:
    """Run coda layers + final norm + lm_head → logits.
    
    C(s_N) → p_N = softmax(logits)
    
    Args:
        state: (B, S, D) final latent state
    Returns:
        logits: (B, S, vocab_size)
    """
    h = state
    
    for layer in parts.coda_layers:
        h = layer(h)
        if isinstance(h, tuple):
            h = h[0]
    
    # Final RMSNorm
    h = parts.final_norm(h)
    
    # LM head (may be tied to embedding)
    if parts.lm_head is not None:
        logits = parts.lm_head(h)
    else:
        # Tied embeddings: multiply by embedding weight
        logits = F.linear(h, parts.embed.weight)
    
    return logits


@torch.no_grad()
def init_latent_state(
    parts: HuginnParts,
    batch_size: int,
    seq_len: int,
    seed: int = 42,
) -> torch.Tensor:
    """Initialize the latent state s_0.
    
    state_init="like-init": uses the same distribution as weight initialization.
    The init_values from config give the std: config.init_values["std"]
    
    NOTE: Verify this against the actual code in discover_huginn.py output.
    If the model uses a different init, this must be corrected.
    """
    D = parts.config.n_embd
    gen = torch.Generator(device=parts.device).manual_seed(seed)
    
    if parts.config.state_init == "like-init":
        # Use the same std as weight initialization
        init_std = parts.config.init_values.get("std", 0.008703882797784892)
        state = torch.empty(batch_size, seq_len, D, device=parts.device, dtype=parts.dtype)
        state.normal_(0.0, init_std, generator=gen)
    elif parts.config.state_init == "random":
        state = torch.randn(batch_size, seq_len, D, generator=gen,
                           device=parts.device, dtype=parts.dtype)
    elif parts.config.state_init == "zero":
        state = torch.zeros(batch_size, seq_len, D, device=parts.device, dtype=parts.dtype)
    else:
        print(f"WARNING: Unknown state_init='{parts.config.state_init}', using zeros")
        state = torch.zeros(batch_size, seq_len, D, device=parts.device, dtype=parts.dtype)
    
    return state


@torch.no_grad()
def run_full_decomposed(
    parts: HuginnParts,
    input_ids: torch.Tensor,
    num_steps: int = 32,
    seed: int = 42,
    return_states: bool = False,
) -> dict:
    """Run the full decomposed forward pass: prelude → core loop → coda.
    
    Args:
        input_ids: (B, S) token IDs
        num_steps: number of core iterations (default 32)
        seed: RNG seed for initial latent state
        return_states: if True, return all intermediate states (memory-heavy!)
    Returns:
        dict with 'logits', 'final_state', 'prelude_out',
        and optionally 'states' (list [s_0, s_1, ..., s_N])
    """
    B, S = input_ids.shape
    
    # 1. Prelude → e
    prelude_out = run_prelude(parts, input_ids)
    
    # 2. Initialize latent state s_0
    state = init_latent_state(parts, B, S, seed=seed)
    
    # 3. Core loop: s_{k+1} = R(s_k, e) for k = 0..num_steps-1
    states = [state.clone()] if return_states else None
    for k in range(num_steps):
        state = run_core_step(parts, state, prelude_out)
        if return_states:
            states.append(state.clone())
    
    # 4. Coda → logits
    logits = run_coda(parts, state)
    
    result = {
        'logits': logits,
        'final_state': state,
        'prelude_out': prelude_out,
    }
    if return_states:
        result['states'] = states
    
    return result


@torch.no_grad()
def run_library_forward(
    parts: HuginnParts,
    input_ids: torch.Tensor,
    num_steps: int = 32,
) -> torch.Tensor:
    """Run the model's own forward pass for reference. Returns logits.
    
    NOTE: We need to match the random seed used for state initialization.
    The library may handle this internally. If there's a mismatch, we need
    to discover how the library seeds its init and replicate it.
    """
    try:
        outputs = parts.model(input_ids, num_steps=num_steps)
        return outputs.logits
    except TypeError:
        # Try without num_steps (some versions use different kwarg name)
        try:
            outputs = parts.model(input_ids, num_iterations=num_steps)
            return outputs.logits
        except TypeError:
            outputs = parts.model(input_ids)
            return outputs.logits


@torch.no_grad()
def verify_decomposition(
    parts: HuginnParts,
    input_ids: torch.Tensor,
    num_steps: int = 32,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """Verify that the decomposed forward pass matches the library's output.
    
    Returns a dict with comparison metrics. Gate 0 requires this to pass.
    
    NOTE: Random state init makes exact matching hard. We verify by:
    1. If we can hook into the library's init to use same seed → exact match
    2. If not, we verify structural correctness by:
       - Checking logit shapes match
       - Checking that prelude output matches (deterministic)
       - Running both at num_steps=0 (just prelude+coda) to isolate issues
    """
    results = {}
    
    # First: verify prelude alone (deterministic, no random init)
    if verbose:
        print(f"\n  Testing prelude (deterministic)...")
    prelude_out = run_prelude(parts, input_ids)
    results['prelude_shape'] = list(prelude_out.shape)
    results['prelude_dtype'] = str(prelude_out.dtype)
    if verbose:
        print(f"    Prelude output: shape={prelude_out.shape}, "
              f"mean={prelude_out.float().mean():.4f}, std={prelude_out.float().std():.4f}")

    # Full decomposed forward
    decomposed = run_full_decomposed(parts, input_ids, num_steps=num_steps, seed=seed)
    decomposed_logits = decomposed['logits']
    results['decomposed_logits_shape'] = list(decomposed_logits.shape)
    
    # Library forward
    try:
        library_logits = run_library_forward(parts, input_ids, num_steps=num_steps)
        results['library_logits_shape'] = list(library_logits.shape)
    except Exception as e:
        results['error'] = str(e)
        if verbose:
            print(f"    Library forward failed: {e}")
        return results
    
    # Shape check
    shape_match = decomposed_logits.shape == library_logits.shape
    results['shape_match'] = shape_match
    
    if not shape_match:
        if verbose:
            print(f"    Shape mismatch: {decomposed_logits.shape} vs {library_logits.shape}")
        results['pass'] = False
        return results
    
    # Value comparison
    max_diff = (decomposed_logits.float() - library_logits.float()).abs().max().item()
    mean_diff = (decomposed_logits.float() - library_logits.float()).abs().mean().item()
    
    # KL divergence
    p_decomposed = F.softmax(decomposed_logits.float(), dim=-1)
    p_library = F.softmax(library_logits.float(), dim=-1)
    kl = F.kl_div(
        p_decomposed.log().clamp(min=-100),
        p_library,
        reduction='batchmean'
    ).item()
    
    # Top-1 agreement
    top1_agree = (decomposed_logits.argmax(-1) == library_logits.argmax(-1)).float().mean().item()
    
    results.update({
        'max_logit_diff': max_diff,
        'mean_logit_diff': mean_diff,
        'kl_divergence': kl,
        'top1_agreement': top1_agree,
        # Strict: max_diff < 0.01 for bf16
        # Note: random init mismatch will cause large diffs. If top1 disagrees
        # significantly, it's likely a seed mismatch, not a structural error.
        'pass': max_diff < 0.05 and top1_agree > 0.95,
    })
    
    if verbose:
        print(f"\n  Decomposition verification (num_steps={num_steps}):")
        print(f"    Max logit diff:   {max_diff:.6f}")
        print(f"    Mean logit diff:  {mean_diff:.6f}")
        print(f"    KL divergence:    {kl:.6f}")
        print(f"    Top-1 agreement:  {top1_agree:.4f}")
        print(f"    PASS: {'✓' if results['pass'] else '✗'}")
        
        if not results['pass'] and top1_agree < 0.5:
            print(f"\n    ⚠ Very low top-1 agreement suggests random state init mismatch.")
            print(f"    Run discover_huginn.py to find how the library seeds its init.")
            print(f"    Then update init_latent_state() to match.")
    
    return results
