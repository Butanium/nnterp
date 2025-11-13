from nnterp import StandardizedTransformer

m = StandardizedTransformer("gpt2")
with m.trace("Hello") as tracer:
    cache = tracer.cache(modules=[m.layers[0]]).save()

print("Keys:", list(cache.keys()))
print("\nTrying cache.model.layers[0].output:")
try:
    print(cache.model.layers[0].output)
except AttributeError as e:
    print(f"FAILED: {e}")
