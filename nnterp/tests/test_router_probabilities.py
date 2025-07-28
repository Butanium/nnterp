#!/usr/bin/env python3

import pytest
import torch as th
from nnterp import StandardizedTransformer
from nnterp.tests.utils import TEST_MOE_MODELS


@pytest.mark.parametrize("model_name", TEST_MOE_MODELS)
def test_router_probabilities_normalization(model_name):
    """Test that router probabilities are properly normalized."""
    with th.no_grad():
        model = StandardizedTransformer(model_name)
        
        if not model.routers_available:
            pytest.skip(f"Router not available for {model_name}")
            
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
            
            # Test normalization for each token
            for token_idx in range(router_probs.shape[0]):
                token_probs = router_probs[token_idx]
                
                # Probabilities should sum to 1
                prob_sum = token_probs.sum()
                assert abs(prob_sum.item() - 1.0) < 1e-5, f"Probabilities should sum to 1, got {prob_sum}"
                
                # Should have exactly top_k non-zero probabilities
                non_zero_count = (token_probs > 1e-8).sum().item()
                assert non_zero_count == top_k, f"Expected {top_k} non-zero experts, got {non_zero_count}"
                
                # All probabilities should be non-negative
                assert (token_probs >= 0).all(), "All probabilities should be non-negative"
                
                # Top-k probabilities should be the largest ones
                top_k_values, top_k_indices = th.topk(token_probs, k=top_k)
                assert (top_k_values > 0).all(), "Top-k probabilities should be positive"


@pytest.mark.parametrize("model_name", TEST_MOE_MODELS)
def test_router_probabilities_top_k_selection(model_name):
    """Test that router probabilities correctly select top-k experts."""
    with th.no_grad():
        model = StandardizedTransformer(model_name)
        
        if not model.routers_available:
            pytest.skip(f"Router not available for {model_name}")
            
        with model.trace("Hello world"):
            router_layers = model.layers_with_routers
            layer = router_layers[0]
            
            router_output = model.routers_output[layer]
            router_probs = model.router_probabilities[layer]
            top_k = model.router_probabilities.get_top_k()
            
            for token_idx in range(router_probs.shape[0]):
                logits = router_output[token_idx]
                probs = router_probs[token_idx]
                
                # Get top-k from logits (what we expect to be selected)
                _, expected_top_k_indices = th.topk(logits, k=top_k)
                
                # Get non-zero indices from probabilities (what was actually selected)
                actual_non_zero_indices = th.nonzero(probs > 1e-8).squeeze(-1)
                
                # Sort both for comparison
                expected_sorted = th.sort(expected_top_k_indices)[0]
                actual_sorted = th.sort(actual_non_zero_indices)[0]
                
                assert th.equal(expected_sorted, actual_sorted), \
                    f"Top-k selection mismatch: expected {expected_sorted}, got {actual_sorted}"


@pytest.mark.parametrize("model_name", TEST_MOE_MODELS)
def test_router_probabilities_consistency(model_name):
    """Test that router probabilities are consistent across multiple calls."""
    with th.no_grad():
        model = StandardizedTransformer(model_name)
        
        if not model.routers_available:
            pytest.skip(f"Router not available for {model_name}")
            
        # Test with same input multiple times
        with model.trace("Hello world"):
            router_layers = model.layers_with_routers
            layer = router_layers[0]
            
            probs1 = model.router_probabilities[layer]
            
        with model.trace("Hello world"):
            probs2 = model.router_probabilities[layer]
            
        # Should get identical results for same input
        assert th.allclose(probs1, probs2, atol=1e-6), "Router probabilities should be consistent"


def test_compute_router_probabilities_function():
    """Test the compute_router_probabilities function directly."""
    from nnterp.rename_utils import compute_router_probabilities
    
    # Create test logits
    batch_size, seq_len, num_experts = 2, 3, 8
    top_k = 2
    
    # Create some test logits
    logits = th.randn(batch_size, seq_len, num_experts)
    
    # Compute probabilities
    probs = compute_router_probabilities(logits, top_k)
    
    # Test shape
    assert probs.shape == logits.shape, "Output shape should match input shape"
    
    # Test normalization
    for b in range(batch_size):
        for s in range(seq_len):
            token_probs = probs[b, s]
            
            # Should sum to 1
            assert abs(token_probs.sum().item() - 1.0) < 1e-6, "Probabilities should sum to 1"
            
            # Should have exactly top_k non-zero values
            non_zero_count = (token_probs > 1e-8).sum().item()
            assert non_zero_count == top_k, f"Expected {top_k} non-zero values, got {non_zero_count}"
            
            # Non-negative values
            assert (token_probs >= 0).all(), "All probabilities should be non-negative"


def test_compute_router_probabilities_edge_cases():
    """Test edge cases for compute_router_probabilities function."""
    from nnterp.rename_utils import compute_router_probabilities
    
    # Test with top_k = 1
    logits = th.tensor([[1.0, 2.0, 0.5]])
    probs = compute_router_probabilities(logits, top_k=1)
    expected = th.tensor([[0.0, 1.0, 0.0]])
    assert th.allclose(probs, expected), f"Expected {expected}, got {probs}"
    
    # Test with all equal logits
    logits = th.ones(1, 4)
    probs = compute_router_probabilities(logits, top_k=2)
    assert abs(probs.sum().item() - 1.0) < 1e-6, "Should still sum to 1"
    assert (probs[probs > 0] > 0.4).all(), "Selected experts should have reasonable probability"
    
    # Test with negative logits
    logits = th.tensor([[-1.0, -2.0, -0.5, -3.0]])
    probs = compute_router_probabilities(logits, top_k=2)
    assert abs(probs.sum().item() - 1.0) < 1e-6, "Should sum to 1 even with negative logits"
    non_zero_indices = th.nonzero(probs > 1e-8).squeeze(-1)
    expected_indices = th.tensor([2, 0])  # -0.5 and -1.0 are the largest
    assert th.equal(th.sort(non_zero_indices)[0], th.sort(expected_indices)[0])
