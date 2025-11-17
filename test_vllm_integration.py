"""
Test script to verify vLLM integration with nnsight.

This script tests basic vLLM functionality without nnterp to understand
the API before integrating into StandardizedTransformer.
"""

print("Importing nnsight...")
import nnsight
print(f"nnsight version: {nnsight.__version__}")

print(f"\nChecking required vLLM version: {nnsight.NNS_VLLM_VERSION}")

try:
    print("\nAttempting to import VLLM from nnsight...")
    from nnsight.modeling.vllm import VLLM
    print("✓ VLLM import successful!")

    print("\nAttempting to load TinyLLama with vLLM backend...")
    print("Note: This requires GPU and vllm==0.9.2")

    # This will likely fail without GPU, but we can see the error
    model = VLLM("Maykeye/TinyLLama-v0", tensor_parallel_size=1, dispatch=True)

    print("✓ Model loaded successfully!")

    # Test basic trace
    print("\nTesting basic trace with sampling parameters...")
    with model.trace("Hello world", temperature=0.0, top_p=1.0, max_tokens=1):
        logits = model.logits.output.save()

    print(f"✓ Trace successful! Logits shape: {logits.shape}")

    # Test module access
    print("\nTesting module access...")
    with model.trace("Test"):
        h5_output = model.transformer.h[5].output.save()

    print(f"✓ Module access successful! Shape: {h5_output.shape}")

except ImportError as e:
    print(f"\n✗ Import failed: {e}")
    print("\nThis is expected if vLLM 0.9.2 is not installed.")
    print("To install: uv pip install vllm==0.9.2")

except Exception as e:
    print(f"\n✗ Error: {type(e).__name__}: {e}")
    print("\nThis is expected without GPU or if vLLM is not properly configured.")

print("\n" + "="*60)
print("Test complete. See errors above for issues.")
print("="*60)
