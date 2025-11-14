from nnsight import LanguageModel
from nnterp.rename_utils import get_rename_dict
import torch as th

# Get the exact rename dict that StandardizedTransformer uses
rename = get_rename_dict(rename_config=None)

print(f"Using {len(rename)} rename entries")
print("\nRelevant renames for GPT-2:")
for k, v in sorted(rename.items()):
    if 'transformer' in k or 'h' in k:
        print(f"  {k!r}: {v!r}")

m = LanguageModel("gpt2", rename=rename)

with th.no_grad():
    with m.trace("Hi") as t:
        cache = t.cache(modules=[m.model.layers[0]]).save()

print("\nCache keys:", list(cache.keys()))

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
