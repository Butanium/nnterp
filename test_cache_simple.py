#!/usr/bin/env python3
"""Simple test to validate cache functionality with rename-cache branch"""

import torch as th
from nnterp import StandardizedTransformer

def test_cache_with_renamed_modules():
    """Test exactly what the pytest test does"""
    print("Testing cache with renamed modules (from actual test)...")

    model_name = "gpt2"
    print(f"\nModel: {model_name}")

    with th.no_grad():
        model = StandardizedTransformer(model_name)
        prompt = "Hello, world!"

        # Cache layers using renamed module references (from test)
        with model.trace(prompt) as tracer:
            print(f"Caching even layers (0, 2, 4, ..., {model.num_layers-2})")
            cache = tracer.cache(
                modules=[model.layers[i] for i in range(0, model.num_layers, 2)]
            ).save()

        print(f"\nCache keys: {list(cache.keys())}")

        # Test 1: Access cached modules using renamed names (attribute notation)
        print("\n=== Test 1: Attribute Access ===")
        try:
            layer_0_output_attr = cache.model.layers[0].output
            batch_size, seq_len, hidden_size = layer_0_output_attr[0].shape
            assert batch_size == 1
            assert hidden_size == model.hidden_size
            print(f"✅ PASS: cache.model.layers[0].output works!")
            print(f"   Shape: {layer_0_output_attr[0].shape}")
        except AttributeError as e:
            print(f"❌ FAIL: cache.model.layers[0].output failed")
            print(f"   Error: {e}")

        # Test 2: Access using dictionary notation with renamed path
        print("\n=== Test 2: Dictionary Access with Renamed Path ===")
        try:
            layer_0_output_dict = cache["model.layers.0"].output
            print(f"✅ PASS: cache['model.layers.0'].output works!")
            print(f"   Shape: {layer_0_output_dict[0].shape}")

            # Test 3: Verify both methods give same result (if both work)
            print("\n=== Test 3: Comparing Results ===")
            try:
                assert th.allclose(layer_0_output_attr[0], layer_0_output_dict[0])
                print(f"✅ PASS: Both methods return same tensor!")
            except NameError:
                print(f"⚠️  SKIP: Can't compare - attribute access failed")
        except (KeyError, AttributeError) as e:
            print(f"❌ FAIL: cache['model.layers.0'].output failed")
            print(f"   Error: {e}")

if __name__ == "__main__":
    test_cache_with_renamed_modules()
