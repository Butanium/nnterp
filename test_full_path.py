from nnsight import LanguageModel

m = LanguageModel("gpt2", rename={"transformer": "model"})
with m.trace("Hi") as t:
    cache = t.cache(modules=[m.model.h[0]]).save()

print("Cached keys:", list(cache.keys()))
print("\nTrying cache.model.model.h[0].output:")
print(cache.model.model.h[0].output)
