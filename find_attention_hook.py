"""Script to find attention probability hooks for models"""
import torch as th
from nnterp import StandardizedTransformer
import sys

model_name = sys.argv[1] if len(sys.argv) > 1 else "yujiepan/qwen1.5-moe-tiny-random"

print(f"\n{'='*70}")
print(f"Finding attention hook for: {model_name}")
print(f"{'='*70}\n")

# Load model without attention probs first
print("Loading model...")
model = StandardizedTransformer(
    model_name,
    enable_attention_probs=False,
    device_map="cpu"
)
print(f"Model type: {type(model._model).__name__}\n")

# Display attention module source
print("="*70)
print("ATTENTION MODULE SOURCE CODE:")
print("="*70)
try:
    with model.scan(th.tensor([[1, 2, 3]])) as scanner:
        attn_source = model.attentions[0].source
        print(f"\nSource:\n{attn_source.source}\n")

        # List all operations
        print("="*70)
        print("AVAILABLE OPERATIONS:")
        print("="*70)
        ops = [attr for attr in dir(attn_source) if not attr.startswith('_')]
        for op in sorted(ops):
            if not op in ['line_numbers', 'operations', 'save', 'source', 'stop']:
                print(f"  - {op}")
except Exception as e:
    print(f"Scan failed, trying with trace: {e}")
    with model.trace(th.tensor([[1, 2, 3]])) as tracer:
        attn_source = model.attentions[0].source
        print(f"\nSource:\n{attn_source.source}\n")

        # List all operations
        print("="*70)
        print("AVAILABLE OPERATIONS:")
        print("="*70)
        ops = [attr for attr in dir(attn_source) if not attr.startswith('_')]
        for op in sorted(ops):
            if not op in ['line_numbers', 'operations', 'save', 'source', 'stop']:
                print(f"  - {op}")

print("\n" + "="*70)
print("INSTRUCTIONS:")
print("="*70)
print("""
1. Look for operations containing 'softmax' or 'dropout' on attention weights
2. Common patterns:
   - nn_functional_dropout_0 (after softmax on attention weights)
   - scaled_dot_product_attention_0 (fused attention - harder to access)
   - attention_interface_0 (custom attention implementations)
3. The hook should be AFTER softmax and BEFORE matmul with values
4. Test the shape with model.scan()/trace() to ensure it's (batch, heads, seq, seq)
""")
