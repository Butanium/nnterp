#!/usr/bin/env python3

import pytest
import torch as th
from nnterp import StandardizedTransformer
from nnterp.tests.utils import TEST_MOE_MODELS, get_all_test_models

def is_moe_model(model_name):
    """Check if a model is a MoE model."""
    return model_name in TEST_MOE_MODELS


def test_router_probabilities_normalization(model_name):
    """Test that router probabilities are properly normalized."""
    if model_name not in TEST_MOE_MODELS:
        pytest.skip(f"Skipping router probability test for non-MoE model {model_name}")
    
    with th.no_grad():
        model = StandardizedTransformer(model_name)
        
        # All MoE models should have routers available
        assert model.routers_available, f"Router should be available for MoE model {model_name}"
            
        with model.trace("Hello world"):
            router_layers = model.layers_with_routers
            assert len(router_layers) > 0, f"Should find router layers in {model_name}"
            
            layer = router_layers[0]
            
            # Get router components
            router_output = model.routers_output[layer]  # Raw logits
            router_probs = model.router_probabilities[layer]  # Computed probabilities
            top_k = model.router_probabilities.get_top_k()
            
            # Test shapes match
            assert router_probs.shape == router_output.shape, "Probabilities and logits should have same shape"
            
            # Test that probabilities are different from logits
            assert not th.allclose(router_probs, router_output), "Probabilities should differ from raw logits"
            
            # Test normalization
            prob_sums = router_probs.sum(dim=-1)
            assert th.allclose(prob_sums, th.ones_like(prob_sums), atol=1e-5), \
                f"All probabilities should sum to 1, got sums: {prob_sums}"
            
            # Test non-zero count
            non_zero_counts = (router_probs > 1e-8).sum(dim=-1)
            expected_counts = th.full_like(non_zero_counts, top_k)
            assert th.equal(non_zero_counts, expected_counts), \
                f"Expected {top_k} non-zero experts per token, got {non_zero_counts}"
            
            # All probabilities should be non-negative
            assert (router_probs >= 0).all(), "All probabilities should be non-negative"
            
            # Top-k probabilities should be positive
            top_k_values, _ = th.topk(router_probs, k=top_k, dim=-1)
            assert (top_k_values > 0).all(), "Top-k probabilities should be positive"


def test_router_probabilities_top_k_selection(model_name):
    """Test that router probabilities correctly select top-k experts."""
    if model_name not in TEST_MOE_MODELS:
        pytest.skip(f"Skipping router probability test for non-MoE model {model_name}")
    
    with th.no_grad():
        model = StandardizedTransformer(model_name)
        
        # All MoE models should have routers available
        assert model.routers_available, f"Router should be available for MoE model {model_name}"
            
        with model.trace("Hello world"):
            router_layers = model.layers_with_routers
            layer = router_layers[0]
            
            router_output = model.routers_output[layer]
            router_probs = model.router_probabilities[layer]
            top_k = model.router_probabilities.get_top_k()
            
            # Get top-k indices from logits (what we expect to be selected)
            _, expected_top_k_indices = th.topk(router_output, k=top_k, dim=-1)
            expected_sorted = th.sort(expected_top_k_indices, dim=-1)[0]
            
            # Get non-zero indices from probabilities (what was actually selected)
            non_zero_mask = router_probs > 1e-8
            actual_indices = th.nonzero(non_zero_mask)
            actual_sorted = th.sort(actual_indices[:, 1].view(router_output.shape[0], top_k), dim=-1)[0]
            
            # Verify that all tokens have matching top-k selection
            assert th.equal(expected_sorted, actual_sorted), \
                f"Top-k selection mismatch: expected {expected_sorted}, got {actual_sorted}"


def test_router_probabilities_consistency(model_name):
    """Test that router probabilities are consistent across multiple calls."""
    if model_name not in TEST_MOE_MODELS:
        pytest.skip(f"Skipping router probability test for non-MoE model {model_name}")
    
    with th.no_grad():
        model = StandardizedTransformer(model_name)
        
        # All MoE models should have routers available
        assert model.routers_available, f"Router should be available for MoE model {model_name}"
            
        # Test with same input multiple times
        with model.trace("Hello world"):
            router_layers = model.layers_with_routers
            layer = router_layers[0]
            
            probs1 = model.router_probabilities[layer]
            
        with model.trace("Hello world"):
            probs2 = model.router_probabilities[layer]
            
        # Should get identical results for same input
        assert th.allclose(probs1, probs2, atol=1e-6), "Router probabilities should be consistent"


def test_compute_default_router_probabilities_function():
    """Test the compute_default_router_probabilities function directly."""
    from nnterp.rename_utils import compute_default_router_probabilities
    
    # Create test logits
    batch_size, seq_len, num_experts = 2, 3, 8
    top_k = 2
    
    # Create some test logits
    logits = th.randn(batch_size, seq_len, num_experts)
    
    # Compute probabilities
    probs = compute_default_router_probabilities(logits, top_k)
    
    # Test shape
    assert probs.shape == logits.shape, "Output shape should match input shape"
    
    # Test normalization
    prob_sums = probs.sum(dim=-1)
    assert th.allclose(prob_sums, th.ones_like(prob_sums), atol=1e-6), \
        "All probabilities should sum to 1"
    
    # Test non-zero count
    non_zero_counts = (probs > 1e-8).sum(dim=-1)
    expected_counts = th.full_like(non_zero_counts, top_k)
    assert th.equal(non_zero_counts, expected_counts), \
        f"Expected {top_k} non-zero values per token"
    
    # All values should be non-negative
    assert (probs >= 0).all(), "All probabilities should be non-negative"


def test_compute_router_probabilities_variants():
    """Test different router probability computation variants."""
    from nnterp.rename_utils import (
        compute_default_router_probabilities, 
        compute_unnormalized_router_probabilities,
        compute_router_probabilities
    )
    
    # Test logits
    logits = th.tensor([[1.0, 2.0, 0.5, -1.0]])
    top_k = 2
    
    # Test normalized version (default)
    normalized_probs = compute_default_router_probabilities(logits, top_k)
    assert th.allclose(normalized_probs.sum(dim=-1), th.ones(1), atol=1e-6), \
        "Normalized probabilities should sum to 1"
    
    # Test unnormalized version
    unnormalized_probs = compute_unnormalized_router_probabilities(logits, top_k)
    assert not th.allclose(unnormalized_probs.sum(dim=-1), th.ones(1), atol=1e-6), \
        "Unnormalized probabilities should not sum to 1"
    
    # Test base function with explicit parameters
    explicit_normalized = compute_router_probabilities(logits, top_k, norm_topk_prob=True)
    explicit_unnormalized = compute_router_probabilities(logits, top_k, norm_topk_prob=False)
    
    assert th.allclose(normalized_probs, explicit_normalized), \
        "Default and explicit normalized should match"
    assert th.allclose(unnormalized_probs, explicit_unnormalized), \
        "Unnormalized and explicit unnormalized should match"


def test_compute_router_probabilities_edge_cases():
    """Test edge cases for router probability computation functions."""
    from nnterp.rename_utils import compute_default_router_probabilities
    
    # Test with top_k = 1
    logits = th.tensor([[1.0, 2.0, 0.5]])
    probs = compute_default_router_probabilities(logits, top_k=1)
    expected = th.tensor([[0.0, 1.0, 0.0]])
    assert th.allclose(probs, expected), f"Expected {expected}, got {probs}"
    
    # Test with all equal logits
    logits = th.ones(1, 4)
    probs = compute_default_router_probabilities(logits, top_k=2)
    assert th.allclose(probs.sum(dim=-1), th.ones(1), atol=1e-6), "Should still sum to 1"
    assert (probs[probs > 0] > 0.4).all(), "Selected experts should have reasonable probability"
    
    # Test with negative logits
    logits = th.tensor([[-1.0, -2.0, -0.5, -3.0]])
    probs = compute_default_router_probabilities(logits, top_k=2)
    assert th.allclose(probs.sum(dim=-1), th.ones(1), atol=1e-6), \
        "Should sum to 1 even with negative logits"
    non_zero_indices = th.nonzero(probs > 1e-8)[:, 1]  # Get column indices
    expected_indices = th.tensor([2, 0])  # -0.5 and -1.0 are the largest
    assert th.equal(th.sort(non_zero_indices)[0], th.sort(expected_indices)[0])


def test_router_probabilities_setitem_function():
    """Test the __setitem__ functionality for setting custom probability distributions."""
    from nnterp.rename_utils import RouterProbabilitiesAccessor
    import torch as th
    
    # Create mock model structure
    class MockRouter:
        def __init__(self):
            self.output = None
    
    class MockMLP:
        def __init__(self):
            self.router = MockRouter()
    
    class MockLayer:
        def __init__(self):
            self.mlp = MockMLP()
    
    class MockModel:
        def __init__(self):
            self.layers = [MockLayer()]
            self._model = None
    
    # Test the conversion from probabilities to logits
    model = MockModel()
    accessor = RouterProbabilitiesAccessor(model)
    
    # Create a custom probability distribution
    custom_probs = th.tensor([[[0.1, 0.3, 0.6, 0.0]]])  # batch=1, seq=1, experts=4
    
    # Set the custom probabilities
    accessor[0] = custom_probs
    
    # Verify that router.output was set to logits
    router_logits = model.layers[0].mlp.router.output
    assert router_logits is not None, "Router output should be set"
    
    # Convert back to probabilities to verify
    recovered_probs = th.softmax(router_logits, dim=-1)
    
    # Should be close to original probabilities (within numerical precision)
    assert th.allclose(recovered_probs, custom_probs, atol=1e-6), \
        f"Recovered probabilities {recovered_probs} should match original {custom_probs}"
    
    # Test with zero probabilities (should be handled with epsilon)
    zero_probs = th.tensor([[[1.0, 0.0, 0.0, 0.0]]])
    accessor[0] = zero_probs
    
    router_logits = model.layers[0].mlp.router.output
    recovered_probs = th.softmax(router_logits, dim=-1)
    
    # Should still be close to original (epsilon prevents log(0))
    assert th.allclose(recovered_probs, zero_probs, atol=1e-5), \
        "Should handle zero probabilities correctly"


@pytest.mark.parametrize("model_name", get_all_test_models())
def test_router_probabilities_setitem_integration(model_name):
    """Test __setitem__ integration with actual MoE models."""
    if not is_moe_model(model_name):
        pytest.skip(f"Skipping router probability setitem test for non-MoE model {model_name}")
    
    with th.no_grad():
        model = StandardizedTransformer(model_name)
        
        if not model.routers_available:
            pytest.skip(f"Router not available for {model_name}")
        
        with model.trace("Hello world test"):
            router_layers = model.layers_with_routers
            if len(router_layers) == 0:
                pytest.skip(f"No router layers found in {model_name}")
            
            layer = router_layers[0]
            
            # Get original probabilities
            original_probs = model.router_probabilities[layer]
            batch_size, seq_len, num_experts = original_probs.shape
            
            # Create custom probability distribution (uniform over top-2 experts)
            top_k = model.router_probabilities.get_top_k()
            custom_probs = th.zeros_like(original_probs)
            
            # Set uniform probabilities for first top_k experts
            for i in range(min(top_k, num_experts)):
                custom_probs[:, :, i] = 1.0 / top_k
            
            # Set the custom probabilities
            model.router_probabilities[layer] = custom_probs
            
            # Get the probabilities back
            retrieved_probs = model.router_probabilities[layer]
            
            # Should be close to our custom probabilities
            assert th.allclose(retrieved_probs, custom_probs, atol=1e-4), \
                f"Retrieved probabilities should match custom probabilities"
            
            # Verify they still sum to 1
            prob_sums = retrieved_probs.sum(dim=-1)
            assert th.allclose(prob_sums, th.ones_like(prob_sums), atol=1e-5), \
                "Custom probabilities should still sum to 1"
            
            # Verify only top_k experts are active
            non_zero_counts = (retrieved_probs > 1e-6).sum(dim=-1)
            assert (non_zero_counts <= top_k).all(), \
                f"Should have at most {top_k} active experts"


def test_router_probabilities_setitem_edge_cases():
    """Test edge cases for __setitem__ functionality."""
    from nnterp.rename_utils import RouterProbabilitiesAccessor
    import torch as th
    
    # Create mock model structure
    class MockRouter:
        def __init__(self):
            self.output = None
    
    class MockMLP:
        def __init__(self):
            self.router = MockRouter()
    
    class MockLayer:
        def __init__(self):
            self.mlp = MockMLP()
    
    class MockModel:
        def __init__(self):
            self.layers = [MockLayer()]
            self._model = None
    
    model = MockModel()
    accessor = RouterProbabilitiesAccessor(model)
    
    # Test with very small probabilities (near epsilon)
    small_probs = th.tensor([[[1e-10, 1e-10, 1.0 - 2e-10, 0.0]]])
    accessor[0] = small_probs
    
    router_logits = model.layers[0].mlp.router.output
    recovered_probs = th.softmax(router_logits, dim=-1)
    
    # Should handle small probabilities gracefully
    assert th.allclose(recovered_probs, small_probs, atol=1e-6), \
        "Should handle very small probabilities"
    
    # Test with probabilities that don't sum exactly to 1 (numerical precision)
    imperfect_probs = th.tensor([[[0.33, 0.33, 0.34, 0.0]]])  # Sum = 1.0
    accessor[0] = imperfect_probs
    
    router_logits = model.layers[0].mlp.router.output
    recovered_probs = th.softmax(router_logits, dim=-1)
    
    # Should still work reasonably well
    assert th.allclose(recovered_probs, imperfect_probs, atol=1e-5), \
        "Should handle imperfect normalization"
