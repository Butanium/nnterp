from nnsight import LanguageModel

m = LanguageModel("gpt2", rename={"transformer.h": "layers"})

with m.trace("Hi") as t:
    cache = t.cache(modules=[m.layers[0]]).save()

print("Keys:", list(cache.keys()))
print("Trying: cache.layers[0].output")
print(cache.layers[0].output)
