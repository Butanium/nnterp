"""Quick test script to check qwen1.5-moe attention probabilities"""
import torch as th
from nnterp import StandardizedTransformer

print("Loading qwen1.5-moe model...")
model = StandardizedTransformer(
    "yujiepan/qwen1.5-moe-tiny-random",
    enable_attention_probs=True,
    device_map="cpu"
)
print(f"Model loaded: {type(model._model)}")

print("\nExploring attention module structure...")
with model.scan(th.tensor([[1, 2, 3]])):
    print(f"Attention module type: {type(model.attentions[0].source)}")
    print(f"Attention module source: {model.attentions[0].source}")

print("\nAttempting to access attention probabilities...")
try:
    with model.trace(th.tensor([[1, 2, 3]])) as tracer:
        attn_probs = model.attention_probabilities[0].save()

    print(f"Success! Attention probs shape: {attn_probs.shape}")
    print(f"Attention probs:\n{attn_probs}")
except Exception as e:
    print(f"FAILED: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
