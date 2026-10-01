"""
discover_huginn.py — Run on the 5090 to discover the exact Huginn API.

Prints the full model structure, layer types, injection mechanism, adapter shapes,
and verifies the forward pass decomposition.

Usage:
    python scripts/discover_huginn.py

Paste the FULL output into NOTES.md.
"""

import torch
import sys
import json
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

MODEL = "tomg-group-umd/huginn-0125"

print("=" * 80)
print("HUGINN MODEL DISCOVERY")
print("=" * 80)

# 1. Config
print("\n--- CONFIG ---")
config = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
for key in ['n_embd', 'n_heads', 'num_key_value_heads', 'head_dim',
            'intermediate_size', 'vocab_size', 'padded_vocab_size',
            'injection_type', 'state_init', 'n_layers',
            'n_layers_in_prelude', 'n_layers_in_coda', 'n_layers_in_recurrent_block',
            'mean_recurrence', 'mean_backprop_depth', 'effective_expected_depth',
            'block_size', 'norm_class_name', 'nonlin_name', 'mlp_class_name',
            'qk_bias', 'tie_embeddings', 'rope_base', 'norm_eps',
            'block_class_name', 'sampling_scheme']:
    val = getattr(config, key, 'NOT FOUND')
    print(f"  {key}: {val}")

# 2. Load model
print("\n--- LOADING MODEL (bf16) ---")
model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True
)
model.eval()

# 3. Full module tree (2 levels deep)
print("\n--- MODULE TREE (depth=3) ---")
def print_tree(module, prefix="", depth=0, max_depth=3):
    for name, child in module.named_children():
        n_params = sum(p.numel() for p in child.parameters(recurse=False))
        n_params_total = sum(p.numel() for p in child.parameters())
        shape_info = ""
        if isinstance(child, torch.nn.Linear):
            shape_info = f" [{child.in_features} -> {child.out_features}, bias={child.bias is not None}]"
        elif isinstance(child, torch.nn.Embedding):
            shape_info = f" [{child.num_embeddings} x {child.embedding_dim}]"
        print(f"{prefix}{name}: {type(child).__name__}{shape_info} "
              f"(own={n_params/1e6:.2f}M, total={n_params_total/1e6:.2f}M)")
        if depth < max_depth:
            print_tree(child, prefix + "  ", depth + 1, max_depth)

print_tree(model)

# 4. Find the injection/adapter mechanism
print("\n--- INJECTION / ADAPTER SEARCH ---")
for name, module in model.named_modules():
    if 'adapter' in name.lower() or 'inject' in name.lower() or 'gate' in name.lower():
        print(f"  FOUND: {name}: {type(module).__name__}")
        if isinstance(module, torch.nn.Linear):
            print(f"         shape: [{module.in_features} -> {module.out_features}]")
        for pname, p in module.named_parameters(recurse=False):
            print(f"         param: {pname} shape={list(p.shape)}")

# 5. Find all Linear layers and their shapes
print("\n--- ALL LINEAR LAYERS ---")
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        print(f"  {name}: [{module.in_features} -> {module.out_features}, bias={module.bias is not None}]")

# 6. Count parameters
print("\n--- PARAMETER COUNTS ---")
total = sum(p.numel() for p in model.parameters())
print(f"  Total parameters: {total:,} ({total/1e9:.2f}B)")

# Count by component (try to identify prelude/core/coda)
for name, child in model.named_children():
    n = sum(p.numel() for p in child.parameters())
    print(f"  {name}: {n:,} ({n/1e6:.1f}M)")
    for sub_name, sub_child in child.named_children():
        n2 = sum(p.numel() for p in sub_child.parameters())
        print(f"    {sub_name}: {n2:,} ({n2/1e6:.1f}M)")

# 7. Forward pass inspection
print("\n--- FORWARD PASS SOURCE ---")
# Print the forward method source of the top-level model and its inner model
import inspect
for name in ['forward', 'generate']:
    method = getattr(model, name, None)
    if method:
        try:
            src = inspect.getsource(method)
            # Print first 100 lines
            lines = src.split('\n')[:100]
            print(f"\n  {type(model).__name__}.{name}() — first 100 lines:")
            for i, line in enumerate(lines):
                print(f"    {i+1:3d}: {line}")
        except:
            print(f"  Could not get source for {name}")

# Also get the inner model's forward if it exists
inner = getattr(model, 'model', None) or getattr(model, 'transformer', None)
if inner:
    try:
        src = inspect.getsource(inner.forward)
        lines = src.split('\n')[:150]
        print(f"\n  {type(inner).__name__}.forward() — first 150 lines:")
        for i, line in enumerate(lines):
            print(f"    {i+1:3d}: {line}")
    except:
        print(f"  Could not get source for inner model forward")

# 8. Find the recurrence loop
print("\n--- RECURRENCE LOOP SEARCH ---")
for cls_name in [type(model).__name__]:
    try:
        src = inspect.getsource(type(model))
        # Find lines mentioning 'recur' or 'loop' or 'num_steps' or 'iteration'
        for i, line in enumerate(src.split('\n')):
            low = line.lower()
            if any(kw in low for kw in ['recur', 'loop', 'num_step', 'iteration', 'state_init', 'inject']):
                print(f"  L{i+1}: {line.rstrip()}")
    except:
        print(f"  Could not inspect {cls_name}")

# 9. Test a simple forward pass
print("\n--- TEST FORWARD PASS ---")
tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
text = "The capital of France is"
inputs = tokenizer(text, return_tensors="pt").to("cuda")
print(f"  Input: '{text}'")
print(f"  Token IDs: {inputs.input_ids.tolist()}")
print(f"  Shape: {inputs.input_ids.shape}")

with torch.no_grad():
    # Try with different num_steps
    for ns in [1, 4, 16, 32]:
        try:
            out = model(inputs.input_ids, num_steps=ns)
            logits = out.logits
            pred_id = logits[0, -1].argmax().item()
            pred_token = tokenizer.decode(pred_id)
            print(f"  num_steps={ns:2d}: logits shape={logits.shape}, "
                  f"pred='{pred_token}', max_logit={logits[0,-1].max().item():.2f}")
        except Exception as e:
            print(f"  num_steps={ns}: ERROR: {e}")

# 10. Check KV cache lookup strategies available
print("\n--- KV CACHE LOOKUP STRATEGIES ---")
# Search for lookup_strategy in source
try:
    src = inspect.getsource(type(model))
    for i, line in enumerate(src.split('\n')):
        if 'lookup_strategy' in line or 'lookup' in line.lower():
            print(f"  L{i+1}: {line.rstrip()}")
except:
    pass

# Also check the cache class
for name, module in model.named_modules():
    cls = type(module)
    if 'cache' in cls.__name__.lower():
        print(f"  Cache class found: {name}: {cls.__name__}")
        try:
            src = inspect.getsource(cls)
            for i, line in enumerate(src.split('\n')):
                if 'lookup_strategy' in line:
                    print(f"    L{i+1}: {line.rstrip()}")
        except:
            pass

print("\n" + "=" * 80)
print("DISCOVERY COMPLETE — paste this output into NOTES.md")
print("=" * 80)
