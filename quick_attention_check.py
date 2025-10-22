"""Quick script to find attention hooks by printing source and operations"""
import torch as th
from transformers import AutoModelForCausalLM
from nnsight import LanguageModel

model_names = [
    "yujiepan/qwen1.5-moe-tiny-random",
    "yujiepan/gpt-oss-tiny-random",
    "yujiepan/dbrx-tiny-random",
    "yujiepan/stablelm-2-tiny-random"
]

for model_name in model_names:
    print(f"\n{'█'*70}")
    print(f"Model: {model_name}")
    print(f"{'█'*70}")

    # Load with eager attention
    model = LanguageModel(model_name, device_map="cpu", attn_implementation="eager")

    with model.trace(th.tensor([[1, 2, 3]])):
        # Find attention module location
        if hasattr(model._model, 'model') and hasattr(model._model.model, 'layers'):
            attn = model._model.model.layers[0].self_attn
        elif hasattr(model._model, 'transformer') and hasattr(model._model.transformer, 'h'):
            attn = model._model.transformer.h[0].attn
        else:
            print("  ⚠️  Unknown structure")
            continue

        # Get source
        src = attn.source

        # Find dropout operations
        ops = [a for a in dir(attn) if 'dropout' in a.lower() or 'softmax' in a.lower()]

        print(f"\n  Attention ops with dropout/softmax:")
        if ops:
            for op in ops:
                print(f"    ✓ {op}")
        else:
            print(f"    ✗ None found - likely using scaled_dot_product_attention")

            # Check if there's a super() call that might have eager impl
            all_ops = [a for a in dir(attn) if not a.startswith('_') and a not in ['save', 'stop', 'source', 'line_numbers', 'operations']]
            super_ops = [a for a in all_ops if 'super' in a.lower()]
            if super_ops:
                print(f"    Found super() calls: {super_ops}")
