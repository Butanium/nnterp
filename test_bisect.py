from nnsight import LanguageModel

# Test 1: Just transformer -> model
print("=== Test 1: {'transformer': 'model'} ===")
m = LanguageModel("gpt2", rename={"transformer": "model"})
with m.trace("Hi") as t:
    cache = t.cache(modules=[m.model.h[0]]).save()
print("Keys:", list(cache.keys()))
print(cache.model.h[0].output)

# Test 2: Just h -> layers
print("\n=== Test 2: {'h': 'layers'} ===")
m = LanguageModel("gpt2", rename={"h": "layers"})
with m.trace("Hi") as t:
    cache = t.cache(modules=[m.transformer.layers[0]]).save()
print("Keys:", list(cache.keys()))
print(cache.transformer.layers[0].output)

# Test 3: Both simple renames
print("\n=== Test 3: {'transformer': 'model', 'h': 'layers'} ===")
m = LanguageModel("gpt2", rename={"transformer": "model", "h": "layers"})
with m.trace("Hi") as t:
    cache = t.cache(modules=[m.model.layers[0]]).save()
print("Keys:", list(cache.keys()))
print(cache.model.layers[0].output)
