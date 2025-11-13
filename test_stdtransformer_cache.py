from nnterp import StandardizedTransformer
import torch as th

model = StandardizedTransformer("gpt2")

print(f"Can access model.layers? {hasattr(model, 'layers')}")
print(f"Type of model.layers: {type(model.layers)}")
print(f"model.layers[0]: {model.layers[0]}")

# What's the actual module path?
import nnsight
print(f"\nActual module path for layers[0]:")
print(f"  {nnsight.util.WrapperModule.path(model.layers[0])}")

# Now cache it
with th.no_grad():
    with model.trace("Hi") as tracer:
        cache = tracer.cache(modules=[model.layers[0]]).save()

print(f"\nCache keys: {list(cache.keys())}")

# Try the access that should work
print("\nTrying cache.model.layers[0].output:")
try:
    result = cache.model.layers[0].output
    print(f"✅ SUCCESS! Shape: {result[0].shape}")
except AttributeError as e:
    print(f"❌ FAILED: {e}")

# What about dictionary access?
print("\nTrying cache['model.layers.0'].output:")
try:
    result = cache["model.layers.0"].output
    print(f"✅ SUCCESS! Shape: {result[0].shape}")
except (KeyError, AttributeError) as e:
    print(f"❌ FAILED: {e}")
