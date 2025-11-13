from nnterp import StandardizedTransformer
import torch as th

model = StandardizedTransformer("gpt2")

print(f"model.layers[0]: {model.layers[0]}")

# Cache using model.layers[0]
with th.no_grad():
    with model.trace("Hi") as tracer:
        cache = tracer.cache(modules=[model.layers[0]]).save()

print(f"\nCache keys: {list(cache.keys())}")

# Try access that should work according to the test
print("\n=== Test: cache.model.layers[0].output ===")
try:
    result = cache.model.layers[0].output
    print(f"✅ SUCCESS! Shape: {result[0].shape}")
except AttributeError as e:
    print(f"❌ FAILED: {e}")

# Try dictionary access
print("\n=== Test: cache['model.layers.0'].output ===")
try:
    result = cache["model.layers.0"].output
    print(f"✅ SUCCESS! Shape: {result[0].shape}")
except (KeyError, AttributeError) as e:
    print(f"❌ FAILED: {e}")
