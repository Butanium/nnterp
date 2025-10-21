"""Find attention hooks for models using eager attention"""
import torch as th
from nnsight import LanguageModel
import sys

model_name = sys.argv[1] if len(sys.argv) > 1 else "yujiepan/qwen1.5-moe-tiny-random"

print(f"\n{'='*70}")
print(f"Finding EAGER attention for: {model_name}")
print(f"{'='*70}\n")

# Load with eager attention implementation
print("Loading model with attn_implementation='eager'...")
model = LanguageModel(
    model_name,
    device_map="cpu",
    attn_implementation="eager"
)
print(f"Model type: {type(model._model).__name__}\n")

# Try to access renamed self_attn module
try:
    with model.trace(th.tensor([[1, 2, 3]])) as tracer:
        # Get first layer's attention
        if hasattr(model._model, 'model'):
            if hasattr(model._model.model, 'layers'):
                attn = model._model.model.layers[0].self_attn
            elif hasattr(model._model.model, 'h'):
                attn = model._model.model.h[0].attn
            else:
                print("Can't find layers!")
                sys.exit(1)
        elif hasattr(model._model, 'transformer'):
            if hasattr(model._model.transformer, 'h'):
                attn = model._model.transformer.h[0].attn
            else:
                print("Can't find h!")
                sys.exit(1)
        else:
            print("Unknown model structure!")
            sys.exit(1)

        source_text = attn.source
        ops = [attr for attr in dir(attn) if not attr.startswith('_')]

    print("="*70)
    print("EAGER ATTENTION MODULE SOURCE:")
    print("="*70)
    print(f"\n{source_text}\n")

    print("="*70)
    print("AVAILABLE OPERATIONS:")
    print("="*70)
    for op in sorted(ops):
        if op not in ['line_numbers', 'operations', 'save', 'source', 'stop']:
            print(f"  - {op}")

    # Look for dropout operations
    print("\n" + "="*70)
    print("DROPOUT/SOFTMAX OPERATIONS:")
    print("="*70)
    for op in sorted(ops):
        if 'dropout' in op.lower() or 'softmax' in op.lower():
            print(f"  *** {op}")

except Exception as e:
    import traceback
    print(f"Error: {e}")
    traceback.print_exc()
