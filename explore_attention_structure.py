"""Script to explore attention module structure for different models"""
import torch as th
from nnterp import StandardizedTransformer
import sys

model_name = sys.argv[1] if len(sys.argv) > 1 else "yujiepan/qwen1.5-moe-tiny-random"

print(f"Loading model: {model_name}...")
# Don't enable attention probs yet - we want to explore the structure first
model = StandardizedTransformer(
    model_name,
    enable_attention_probs=False,
    device_map="cpu"
)
print(f"Model loaded: {type(model._model)}")

print("\n=== Exploring Attention Module Structure ===")
with model.scan(th.tensor([[1, 2, 3]])) as scanner:
    attn_module = model.attentions[0]
    print(f"Attention module type: {type(attn_module.source)}")
    print(f"\nAttention module attributes:")

    # List all attributes
    attrs = [attr for attr in dir(attn_module.source) if not attr.startswith('_')]
    for attr in sorted(attrs):
        try:
            obj = getattr(attn_module.source, attr)
            print(f"  {attr}: {type(obj)}")
        except Exception as e:
            print(f"  {attr}: ERROR - {e}")

    # Try to find where attention probabilities might be
    print(f"\n=== Looking for attention-related attributes ===")
    for attr in attrs:
        if 'attn' in attr.lower() or 'attention' in attr.lower() or 'softmax' in attr.lower():
            try:
                obj = getattr(attn_module.source, attr)
                print(f"  {attr}: {type(obj)}")
                if hasattr(obj, 'source'):
                    sub_attrs = [a for a in dir(obj.source) if not a.startswith('_')]
                    print(f"    └─ sub-attributes: {', '.join(sub_attrs[:10])}")
            except Exception as e:
                print(f"  {attr}: ERROR - {e}")

print("\n=== Attempting to trace with enable_attention_probs=True ===")
try:
    model_with_probs = StandardizedTransformer(
        model_name,
        enable_attention_probs=True,
        device_map="cpu"
    )
    print("SUCCESS: Model loaded with attention probs enabled")
except Exception as e:
    print(f"FAILED: {type(e).__name__}: {e}")
