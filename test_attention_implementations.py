"""Test attention probability access for newly implemented models"""
import torch as th
from nnterp import StandardizedTransformer

models_to_test = [
    "yujiepan/qwen1.5-moe-tiny-random",
    "yujiepan/gpt-oss-tiny-random",
    "yujiepan/dbrx-tiny-random",
    "yujiepan/stablelm-2-tiny-random",
]

for model_name in models_to_test:
    print(f"\n{'='*70}")
    print(f"Testing: {model_name}")
    print(f"{'='*70}")

    try:
        # Load with attention probs enabled
        model = StandardizedTransformer(
            model_name,
            enable_attention_probs=True,
            device_map="cpu"
        )

        # Test accessing attention probs
        with model.trace(th.tensor([[1, 2, 3]])):
            attn_probs = model.attention_probabilities[0].save()

        print(f"✓ SUCCESS - Attention probs shape: {attn_probs.shape}")
        print(f"  Sum of last dim (should be ~1.0): {attn_probs.sum(dim=-1).mean().item():.4f}")

    except Exception as e:
        print(f"✗ FAILED - {type(e).__name__}: {str(e)[:100]}")
        import traceback
        traceback.print_exc()

print(f"\n{'='*70}")
print("Testing complete!")
print(f"{'='*70}")
