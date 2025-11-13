#!/usr/bin/env python3
"""Debug script to test cache functionality with renamed modules"""

import torch as th
from nnterp import StandardizedTransformer

def main():
    print("Testing cache with renamed modules...")

    model_name = "gpt2"
    model = StandardizedTransformer(model_name)
    prompt = "Hello, world!"

    print(f"\nModel: {model_name}")
    print(f"Number of layers: {model.num_layers}")
    print(f"Module path for layer 0: {model.layers[0]}")

    # Cache layers using renamed module references
    with th.no_grad():
        with model.trace(prompt) as tracer:
            print("\nCaching layers 0, 2, 4, ...")
            layers_to_cache = [model.layers[i] for i in range(0, model.num_layers, 2)]
            print(f"Layers to cache: {layers_to_cache}")
            cache = tracer.cache(modules=layers_to_cache).save()

    print("\n=== Cache keys ===")
    if hasattr(cache, 'keys'):
        print(f"Cache keys: {list(cache.keys())}")
    else:
        print(f"Cache type: {type(cache)}")
        print(f"Cache attributes: {dir(cache)}")

    # Try to access using different paths
    print("\n=== Trying to access cached data ===")

    # Try renamed path
    try:
        print("Trying cache.model.layers[0]...")
        result = cache.model.layers[0]
        print(f"SUCCESS: {type(result)}")
    except AttributeError as e:
        print(f"FAILED: {e}")

    # Try original path
    try:
        print("Trying cache.model.language_model.layers[0]...")
        result = cache.model.language_model.layers[0]
        print(f"SUCCESS: {type(result)}")
    except AttributeError as e:
        print(f"FAILED: {e}")

    # Try dictionary access
    try:
        print("Trying cache['model.layers.0']...")
        result = cache["model.layers.0"]
        print(f"SUCCESS: {type(result)}")
    except (KeyError, AttributeError) as e:
        print(f"FAILED: {e}")

    # Try original dictionary access
    try:
        print("Trying cache['model.language_model.layers.0']...")
        result = cache["model.language_model.layers.0"]
        print(f"SUCCESS: {type(result)}")
    except (KeyError, AttributeError) as e:
        print(f"FAILED: {e}")

if __name__ == "__main__":
    main()
