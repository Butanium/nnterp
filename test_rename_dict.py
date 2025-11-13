from nnsight import LanguageModel
from nnterp.rename_utils import get_rename_dict

rename = get_rename_dict(rename_config=None)
print(f"Using rename with {len(rename)} entries\n")

m = LanguageModel("gpt2", rename=rename)
with m.trace("Hello") as tracer:
    cache = tracer.cache(modules=[m.model.layers[0]]).save()

print("Keys:", list(cache.keys()))
print("\nTrying cache.model.layers[0].output:")
print(cache.model.layers[0].output)
