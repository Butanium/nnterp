from nnsight import LanguageModel

m = LanguageModel(
    "yujiepan/llama-3.2-tiny-random",
    rename={"model": "foo", "model.language_model": "foo"},
)
with m.trace("Hello, world!") as tracer:
    cache = tracer.cache(modules=[m.foo.layers[0]])
print(cache.model.foo.layers[0].output)
