#!/usr/bin/env python3
"""Test if basic rename works with cache attribute access"""

from nnsight import LanguageModel
import torch as th

print("Testing basic rename with cache attribute access...\n")

# Test 1: User's exact example
print("=== Test 1: User's example with model->foo rename ===")
try:
    m = LanguageModel(
        "yujiepan/llama-3.2-tiny-random",
        rename={"model": "foo", "model.language_model": "foo"},
    )
    with th.no_grad():
        with m.trace("Hello, world!") as tracer:
            cache = tracer.cache(modules=[m.foo.layers[0]]).save()

    print(f"Cache keys: {list(cache.keys())}")
    result = cache.model.foo.layers[0].output
    print(f"✅ SUCCESS: cache.model.foo.layers[0].output works!")
    print(f"   Shape: {result[0].shape}")
except Exception as e:
    print(f"❌ FAILED: {type(e).__name__}: {e}")

# Test 2: Simpler rename (single level)
print("\n=== Test 2: Simple single rename model->foo ===")
try:
    m = LanguageModel(
        "yujiepan/llama-3.2-tiny-random",
        rename={"model": "foo"},
    )
    with th.no_grad():
        with m.trace("Hello, world!") as tracer:
            cache = tracer.cache(modules=[m.foo.language_model.layers[0]]).save()

    print(f"Cache keys: {list(cache.keys())}")
    result = cache.model.foo.language_model.layers[0].output
    print(f"✅ SUCCESS: cache.model.foo.language_model.layers[0].output works!")
    print(f"   Shape: {result[0].shape}")
except Exception as e:
    print(f"❌ FAILED: {type(e).__name__}: {e}")

# Test 3: StandardizedTransformer case (what we're actually testing)
print("\n=== Test 3: StandardizedTransformer rename pattern ===")
from nnterp import StandardizedTransformer
try:
    model = StandardizedTransformer("gpt2")

    # Check what the rename actually is
    print(f"Model has 'layers' attribute: {hasattr(model, 'layers')}")
    print(f"Actual path to layer 0: {model.layers[0]}")

    with th.no_grad():
        with model.trace("Hello, world!") as tracer:
            cache = tracer.cache(modules=[model.layers[0]]).save()

    print(f"\nCache keys: {list(cache.keys())}")

    # Try attribute access with renamed path
    try:
        result = cache.model.layers[0].output
        print(f"✅ SUCCESS: cache.model.layers[0].output works!")
        print(f"   Shape: {result[0].shape}")
    except AttributeError as e:
        print(f"❌ FAILED (attribute): {e}")

    # Try dictionary access with renamed path
    try:
        result = cache["model.layers.0"].output
        print(f"✅ SUCCESS: cache['model.layers.0'].output works!")
        print(f"   Shape: {result[0].shape}")
    except (KeyError, AttributeError) as e:
        print(f"❌ FAILED (dict): {e}")

except Exception as e:
    print(f"❌ FAILED: {type(e).__name__}: {e}")
