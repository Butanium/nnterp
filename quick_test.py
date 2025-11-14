from nnsight import LanguageModel
import torch as th

print("Loading model...")
m = LanguageModel("gpt2", rename={"transformer.h": "layers"})
print("Model loaded")

print("Running trace...")
with th.no_grad():
    with m.trace("Hi") as t:
        cache = t.cache(modules=[m.layers[0]]).save()
print("Trace complete")

print("Keys:", list(cache.keys()))

# Test dictionary access
print("\n=== Dictionary access: cache['model.layers.0'].output ===")
try:
    result = cache['model.layers.0'].output
    print("SUCCESS - Shape:", result[0].shape)
except Exception as e:
    print(f"FAILED: {e}")

# Test attribute access
print("\n=== Attribute access: cache.model.layers[0].output ===")
try:
    result = cache.model.layers[0].output
    print("SUCCESS - Shape:", result[0].shape)
except Exception as e:
    print(f"FAILED: {e}")
