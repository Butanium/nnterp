#!/usr/bin/env python3

import pytest
import torch as th
from nnterp import StandardizedTransformer
from nnterp.rename_utils import RouterProbabilitiesAccessor
from nnterp.tests.utils import TEST_MOE_MODELS, get_all_test_models, is_moe_model


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


def test_router_probabilities_setitem_basic(model_name):
    """Test basic __setitem__ functionality with valid top-k distributions."""
    if not is_moe_model(model_name):
        pytest.skip(f"Model {model_name} is not a MoE model")
    
    with th.no_grad():
        model = StandardizedTransformer(model_name)
        
        assert model.routers_available, f"Router should be available for {model_name}"
        
        with model.trace("Hello world test"):
            router_layers = model.layers_with_routers
            assert len(router_layers) > 0, f"Should have router layers in {model_name}"
            
            layer = router_layers[0]
            
            # Get original probabilities to understand the shape
            original_probs = model.router_probabilities[layer]
            batch_size, seq_len, num_experts = original_probs.shape
            top_k = model.router_probabilities.get_top_k()
            
            # Test with valid top-k distributions
            valid_test_cases = []
            
            # Case 1: Uniform distribution over top_k experts
            uniform_probs = th.zeros_like(original_probs)
            for i in range(min(top_k, num_experts)):
                uniform_probs[:, :, i] = 1.0 / top_k
            valid_test_cases.append(uniform_probs)
            
            # Case 2: One-hot distribution (first expert)
            onehot_probs = th.zeros_like(original_probs)
            onehot_probs[:, :, 0] = 1.0
            valid_test_cases.append(onehot_probs)
            
            # Case 3: Different valid top-k distribution
            if top_k >= 2 and num_experts >= 2:
                weighted_probs = th.zeros_like(original_probs)
                weighted_probs[:, :, 0] = 0.7
                weighted_probs[:, :, 1] = 0.3
                valid_test_cases.append(weighted_probs)
            
            for i, test_probs in enumerate(valid_test_cases):
                model.router_probabilities[layer] = test_probs
                retrieved = model.router_probabilities[layer]
                assert th.allclose(retrieved, test_probs, atol=1e-5), \
                    f"Valid test case {i+1}: Retrieved != Set probabilities"


def test_router_probabilities_setitem_topk_normalization(model_name):
    """Test that top-k normalization works correctly."""
    if not is_moe_model(model_name):
        pytest.skip(f"Model {model_name} is not a MoE model")
    
    with th.no_grad():
        model = StandardizedTransformer(model_name)
        
        assert model.routers_available, f"Router should be available for {model_name}"
        
        with model.trace("Hello world test"):
            router_layers = model.layers_with_routers
            assert len(router_layers) > 0, f"Should have router layers in {model_name}"
            
            layer = router_layers[0]
            
            # Get original probabilities to understand the shape
            original_probs = model.router_probabilities[layer]
            batch_size, seq_len, num_experts = original_probs.shape
            top_k = model.router_probabilities.get_top_k()
            
            # Only test if we have more experts than top_k
            if num_experts > top_k:
                # Set probabilities with more than top_k non-zero values
                over_topk_probs = th.zeros_like(original_probs)
                # Set uniform probabilities for all experts (more than top_k)
                for i in range(num_experts):
                    over_topk_probs[:, :, i] = 1.0 / num_experts
                
                model.router_probabilities[layer] = over_topk_probs
                retrieved = model.router_probabilities[layer]
                
                # Should only have top_k non-zero values
                non_zero_count = (retrieved > 1e-6).sum(dim=-1)
                assert (non_zero_count <= top_k).all(), \
                    f"Should have at most {top_k} non-zero values per token"
                
                # Should still sum to 1
                prob_sums = retrieved.sum(dim=-1)
                assert th.allclose(prob_sums, th.ones_like(prob_sums), atol=1e-5), \
                    "Probabilities should sum to 1 after top-k normalization"


def test_router_probabilities_setitem_integration(model_name):
    """Test __setitem__ integration with actual MoE models."""
    if not is_moe_model(model_name):
        pytest.skip(f"Model {model_name} is not a MoE model")
    
    with th.no_grad():
        model = StandardizedTransformer(model_name)
        
        assert model.routers_available, f"Router should be available for {model_name}"
        
        with model.trace("Hello world test"):
            router_layers = model.layers_with_routers
            assert len(router_layers) > 0, f"Should have router layers in {model_name}"
            
            layer = router_layers[0]
            
            # Get original probabilities
            original_probs = model.router_probabilities[layer]
            batch_size, seq_len, num_experts = original_probs.shape
            
            # Create custom probability distribution (uniform over top-k experts)
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


def test_router_probabilities_setitem_edge_cases(model_name):
    """Test edge cases for __setitem__ functionality."""
    if not is_moe_model(model_name):
        pytest.skip(f"Model {model_name} is not a MoE model")
    
    with th.no_grad():
        model = StandardizedTransformer(model_name)
        
        assert model.routers_available, f"Router should be available for {model_name}"
        
        with model.trace("Hello world test"):
            router_layers = model.layers_with_routers
            assert len(router_layers) > 0, f"Should have router layers in {model_name}"
            
            layer = router_layers[0]
            
            # Get original probabilities to understand the shape
            original_probs = model.router_probabilities[layer]
            batch_size, seq_len, num_experts = original_probs.shape
            top_k = model.router_probabilities.get_top_k()
            
            # Test with very small probabilities that respect top-k constraint
            if num_experts >= 2:
                small_probs = th.zeros_like(original_probs)
                small_probs[:, :, 0] = 1e-10
                small_probs[:, :, 1] = 1.0 - 1e-10
                
                model.router_probabilities[layer] = small_probs
                retrieved = model.router_probabilities[layer]
                assert th.allclose(retrieved, small_probs, atol=1e-6), \
                    "Should handle very small probabilities"
            
            # Test with probabilities that sum exactly to 1 and respect top-k
            if num_experts >= 2 and top_k >= 2:
                exact_probs = th.zeros_like(original_probs)
                exact_probs[:, :, 0] = 0.33
                exact_probs[:, :, 1] = 0.67
                
                model.router_probabilities[layer] = exact_probs
                retrieved = model.router_probabilities[layer]
                assert th.allclose(retrieved, exact_probs, atol=1e-5), \
                    "Should handle exact normalization"
    
            # Test that zero probabilities are handled correctly
            zero_probs = th.zeros_like(original_probs)
            zero_probs[:, :, 0] = 1.0  # One-hot distribution
            
            model.router_probabilities[layer] = zero_probs
            retrieved = model.router_probabilities[layer]
            assert th.allclose(retrieved, zero_probs, atol=1e-6), \
                "Should handle zero probabilities correctly"


def test_router_probabilities_setitem_validation_errors(model_name):
    """Test that __setitem__ properly validates input probabilities."""
    if not is_moe_model(model_name):
        pytest.skip(f"Model {model_name} is not a MoE model")
    
    with th.no_grad():
        model = StandardizedTransformer(model_name)
        
        assert model.routers_available, f"Router should be available for {model_name}"
        
        with model.trace("Hello world test"):
            router_layers = model.layers_with_routers
            assert len(router_layers) > 0, f"Should have router layers in {model_name}"
            
            layer = router_layers[0]
            
            # Get original probabilities to understand the shape
            original_probs = model.router_probabilities[layer]
            batch_size, seq_len, num_experts = original_probs.shape
            
            # Test negative probabilities
            negative_probs = th.zeros_like(original_probs)
            negative_probs[:, :, 0] = -0.1
            negative_probs[:, :, 1] = 1.1
            
            with pytest.raises(AssertionError, match="All probabilities must be non-negative"):
                model.router_probabilities[layer] = negative_probs
            
            # Test probabilities that don't sum to 1
            bad_sum_probs = th.zeros_like(original_probs)
            bad_sum_probs[:, :, 0] = 0.3
            bad_sum_probs[:, :, 1] = 0.3  # Sum = 0.6, not 1.0
            
            with pytest.raises(AssertionError, match="Probabilities must sum to 1 for each token"):
                model.router_probabilities[layer] = bad_sum_probs


def test_router_probabilities_setitem_real_model_topk_normalization(model_name):
    """Test that setting probabilities with non-top-k experts gets normalized correctly."""
    if not is_moe_model(model_name):
        pytest.skip(f"Model {model_name} is not a MoE model")
    
    with th.no_grad():
        model = StandardizedTransformer(model_name)
        
        assert model.routers_available, f"Router should be available for {model_name}"
        
        with model.trace("Hello world test"):
            router_layers = model.layers_with_routers
            assert len(router_layers) > 0, f"Should have router layers in {model_name}"
            
            layer = router_layers[0]
            top_k = model.router_probabilities.get_top_k()
            
            # Get original probabilities to understand the shape
            original_probs = model.router_probabilities[layer]
            batch_size, seq_len, num_experts = original_probs.shape
            
            # Create a probability distribution that assigns mass to ALL experts
            # This should get normalized to only top-k experts
            uniform_all_experts = th.ones_like(original_probs) / num_experts
            
            # Set the uniform distribution across all experts
            model.router_probabilities[layer] = uniform_all_experts
            
            # Get the probabilities back - should be normalized to top-k only
            retrieved_probs = model.router_probabilities[layer]
            
            # Verify they still sum to 1
            prob_sums = retrieved_probs.sum(dim=-1)
            assert th.allclose(prob_sums, th.ones_like(prob_sums), atol=1e-5), \
                "Probabilities should still sum to 1 after top-k normalization"
            
            # Verify only top_k experts are active (if the model uses top-k normalization)
            non_zero_counts = (retrieved_probs > 1e-6).sum(dim=-1)
            assert (non_zero_counts <= top_k).all(), \
                f"Should have at most {top_k} active experts after normalization"


def test_router_probabilities_setitem_preserves_layer_output(model_name):
    """Test that setting router probabilities to original values preserves layer output."""
    if not is_moe_model(model_name):
        pytest.skip(f"Model {model_name} is not a MoE model")
    
    with th.no_grad():
        model = StandardizedTransformer(model_name)
        
        assert model.routers_available, f"Router should be available for {model_name}"
        
        with model.trace("Hello world test"):
            router_layers = model.layers_with_routers
            assert len(router_layers) > 0, f"Should have router layers in {model_name}"
            
            layer = router_layers[0]
            
            # Get original probabilities and layer output
            original_probs = model.router_probabilities[layer]
            original_layer_output = model.layers[layer].output.clone()
            
            # Set router probabilities to the original values
            model.router_probabilities[layer] = original_probs
            
            # Get the layer output after setting probabilities
            modified_layer_output = model.layers[layer].output
            
            # Layer output should be the same as the original
            assert th.allclose(modified_layer_output, original_layer_output, atol=1e-4), \
                "Layer output should be unchanged when setting router probabilities to original values"
