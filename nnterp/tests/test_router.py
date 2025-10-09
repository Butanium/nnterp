import torch as th
import pytest
from nnterp import StandardizedTransformer
from nnterp.rename_utils import RenamingError, RenameConfig
from nnterp.tests.utils import TEST_MODELS, TEST_MOE_MODELS


def test_router_detection_moe_models(model_name):
    """Test that router components are properly detected in MoE models."""
    if model_name not in TEST_MOE_MODELS:
        pytest.skip(f"Skipping router test for non-MoE model {model_name}")
    
    with th.no_grad():
        model = StandardizedTransformer(model_name)
        
        # Router should be enabled for MoE models
        assert model.routers_available, f"Router should be available for MoE model {model_name}"
        
        # Should be able to access router components
        with model.trace("Hello world"):
            # Use layers_with_routers to find router layers
            router_layers = model.layers_with_routers
            assert len(router_layers) > 0, f"Should find at least one router layer in {model_name}"
            
            # Test access to first router layer
            router_layer = router_layers[0]
            router = model.routers[router_layer]
            assert router is not None, f"Router should be accessible at layer {router_layer}"


def test_router_detection_non_moe_models(model_name):
    """Test that router components are properly disabled for non-MoE models."""
    if model_name in TEST_MOE_MODELS:
        pytest.skip(f"Skipping non-MoE test for MoE model {model_name}")
    
    with th.no_grad():
        model = StandardizedTransformer(model_name)
        
        # Router should be disabled for non-MoE models
        assert not model.routers_available, f"Router should not be available for non-MoE model {model_name}"
        
        # Router probabilities should be disabled
        assert not model.router_probabilities.enabled, "Router probabilities accessor should be disabled for non-MoE models"


def test_router_module_access(model_name):
    """Test accessing router modules and their weights."""
    if model_name not in TEST_MOE_MODELS:
        pytest.skip(f"Skipping router test for non-MoE model {model_name}")
    
    with th.no_grad():
        model = StandardizedTransformer(model_name)
        
        assert model.routers_available, f"Router should be available for MoE model {model_name}"
        
        with model.trace("Hello world"):
            # Use layers_with_routers to find router layers
            router_layers = model.layers_with_routers
            assert len(router_layers) > 0, f"Should find at least one router layer in {model_name}"
            
            # Find a layer with a router that has weights
            router_layer = None
            for layer_idx in router_layers:
                try:
                    router = model.routers[layer_idx]
                    if router is not None and hasattr(router, 'weight'):
                        router_layer = layer_idx
                        break
                except:
                    continue
            
            assert router_layer is not None, f"Should find a router with weights in {model_name}"
            
            # Test router module access
            router = model.routers[router_layer]
            assert router is not None, "Router module should be accessible"
            assert hasattr(router, 'weight'), "Router should have weight attribute"
            
            # Test router weights
            router_weights = router.weight
            assert isinstance(router_weights, th.Tensor), "Router weights should be a tensor"
            assert router_weights.ndim >= 2, "Router weights should be at least 2D"
            
            # Test that weights have reasonable shape (hidden_size x num_experts)
            assert router_weights.shape[0] > 0, "Router weights should have positive first dimension"
            assert router_weights.shape[1] > 0, "Router weights should have positive second dimension"


def test_router_io_access(model_name):
    """Test accessing router inputs and outputs."""
    if model_name not in TEST_MOE_MODELS:
        pytest.skip(f"Skipping router test for non-MoE model {model_name}")
    
    with th.no_grad():
        model = StandardizedTransformer(model_name)
        
        assert model.routers_available, f"Router should be available for MoE model {model_name}"
        
        with model.trace("Hello world"):
            # Use layers_with_routers to find router layers
            router_layers = model.layers_with_routers
            assert len(router_layers) > 0, f"Should find at least one router layer in {model_name}"
            
            # Use first router layer
            router_layer = router_layers[0]
            
            # Test router input/output access
            router_input = model.routers_input[router_layer]
            router_output = model.routers_output[router_layer]
            
            assert isinstance(router_input, th.Tensor), "Router input should be a tensor"
            assert isinstance(router_output, th.Tensor), "Router output should be a tensor"
            
            # Router I/O should have reasonable shapes
            batch_size, seq_len = model.input_size
            assert router_input.shape[0] == batch_size, f"Router input batch size should be {batch_size}"
            assert router_input.shape[1] == seq_len, f"Router input sequence length should be {seq_len}"
            assert router_output.shape[0] == batch_size, f"Router output batch size should be {batch_size}"
            assert router_output.shape[1] == seq_len, f"Router output sequence length should be {seq_len}"
            
            # Get num_experts from router weights for validation
            router = model.routers[router_layer]
            num_experts = router.weight.shape[1]
            assert router_output.shape[2] == num_experts, \
                f"Router outputs should have {num_experts} experts (got {router_output.shape[2]})"


def test_router_probabilities_and_top_k(model_name):
    """Test accessing router probabilities and top_k parameter."""
    if model_name not in TEST_MOE_MODELS:
        pytest.skip(f"Skipping router test for non-MoE model {model_name}")
    
    with th.no_grad():
        model = StandardizedTransformer(model_name)
        
        assert model.routers_available, f"Router should be available for MoE model {model_name}"
        
        with model.trace("Hello world"):
            # Use layers_with_routers to find router layers
            router_layers = model.layers_with_routers
            assert len(router_layers) > 0, f"Should find at least one router layer in {model_name}"
            
            # Use first router layer
            router_layer = router_layers[0]
            
            # Test router probabilities access
            router_probs = model.router_probabilities[router_layer]
            assert isinstance(router_probs, th.Tensor), "Router probabilities should be a tensor"
            
            # Router probabilities should have reasonable shape
            batch_size, seq_len = model.input_size
            assert router_probs.shape[0] == batch_size, f"Router probs batch size should be {batch_size}"
            assert router_probs.shape[1] == seq_len, f"Router probs sequence length should be {seq_len}"
            
            # Get num_experts from router weights for validation
            router = model.routers[router_layer]
            num_experts = router.weight.shape[1]
            assert router_probs.shape[2] == num_experts, \
                f"Router probabilities should have {num_experts} experts (got {router_probs.shape[2]})"
        
        # Test getting top_k parameter
        top_k = model.router_probabilities.get_top_k()
        assert isinstance(top_k, int), "top_k should be an integer"
        assert top_k > 0, "top_k should be positive"
        assert top_k <= 8, "top_k should be reasonable (≤ 8 for typical MoE models)"


def test_router_custom_naming(model_name):
    """Test router with custom naming configuration."""
    # Test with custom router name
    custom_config = RenameConfig(router_name="custom_gate")
    
    # This test is more conceptual since we don't have models with custom naming
    # But we can test that the configuration is accepted
    with th.no_grad():
        model = StandardizedTransformer(model_name, rename_config=custom_config)
        # Should not crash, router should be disabled for non-MoE model
        if model_name not in TEST_MOE_MODELS:
            assert not model.routers_available


def test_router_ignore_configuration(model_name):
    """Test router with ignore configuration."""
    # Test with router ignored
    ignore_config = RenameConfig(ignore_router=True)
    
    with th.no_grad():
        model = StandardizedTransformer(model_name, rename_config=ignore_config)
        # Router should be disabled even for MoE model when ignored
        assert not model.routers_available
        assert not model.router_probabilities.enabled


def test_router_structure_validation(model_name):
    """Test router structure validation for MoE models."""
    if model_name not in TEST_MOE_MODELS:
        pytest.skip(f"Skipping router test for non-MoE model {model_name}")
    
    with th.no_grad():
        model = StandardizedTransformer(model_name)
        
        assert model.routers_available, f"Router should be available for MoE model {model_name}"
        
        with model.trace("Hello world"):
            # Use layers_with_routers to find router layers
            router_layers = model.layers_with_routers
            assert len(router_layers) > 0, f"Should find at least one router layer in {model_name}"
            
            # Test each router layer
            for layer_idx in router_layers:
                router = model.routers[layer_idx]
                assert router is not None, f"Router should be accessible at layer {layer_idx}"
                assert hasattr(router, 'weight'), f"Router at layer {layer_idx} should have weight attribute"
                
                # Test router weights shape
                router_weights = router.weight
                assert router_weights.ndim == 2, f"Router weights should be 2D at layer {layer_idx}"
                assert router_weights.shape[0] > 0, f"Router weights should have positive input dimension at layer {layer_idx}"
                assert router_weights.shape[1] > 0, f"Router weights should have positive expert dimension at layer {layer_idx}"


def test_router_multiple_layers(model_name):
    """Test router access across multiple layers."""
    if model_name not in TEST_MOE_MODELS:
        pytest.skip(f"Skipping router test for non-MoE model {model_name}")
    
    with th.no_grad():
        model = StandardizedTransformer(model_name)
        
        assert model.routers_available, f"Router should be available for MoE model {model_name}"
        
        with model.trace("Hello world"):
            # Use layers_with_routers to find router layers
            router_layers = model.layers_with_routers
            assert len(router_layers) > 0, f"Should find at least one router layer in {model_name}"
            
            # Test that each layer in layers_with_routers has a router
            for layer_idx in router_layers:
                router = model.routers[layer_idx]
                assert router is not None, f"Layer {layer_idx} should have a router (it's in layers_with_routers)"
                assert hasattr(router, 'weight'), f"Router at layer {layer_idx} should have weights"
            
            # Test that layers NOT in layers_with_routers don't have routers
            for layer_idx in range(model.num_layers):
                if layer_idx not in router_layers:
                    try:
                        router = model.routers[layer_idx]
                        # If we get here, the router exists but shouldn't
                        assert False, f"Layer {layer_idx} should not have a router (not in layers_with_routers)"
                    except:
                        # Expected - layer doesn't have router
                        pass


def test_router_error_handling(model_name):
    """Test error handling for router access on non-MoE models."""
    if model_name in TEST_MOE_MODELS:
        pytest.skip(f"Skipping error handling test for MoE model {model_name}")
    
    with th.no_grad():
        model = StandardizedTransformer(model_name)
        
        # Router should be disabled for non-MoE models
        assert not model.routers_available, f"Router should not be available for non-MoE model {model_name}"
        
        # Router probabilities should be disabled
        assert not model.router_probabilities.enabled, "Router probabilities accessor should be disabled for non-MoE models"


def test_router_tensor_shapes_comprehensive(model_name):
    """Test comprehensive tensor shape validation for router components."""
    if model_name not in TEST_MOE_MODELS:
        pytest.skip(f"Skipping router test for non-MoE model {model_name}")
    
    with th.no_grad():
        model = StandardizedTransformer(model_name)
        
        assert model.routers_available, f"Router should be available for MoE model {model_name}"
        
        with model.trace("Hello world"):
            batch_size, seq_len = model.input_size
            
            # Use layers_with_routers to find router layers
            router_layers = model.layers_with_routers
            assert len(router_layers) > 0, f"Should find at least one router layer in {model_name}"
            
            # Use first router layer
            router_layer = router_layers[0]
            
            # Get router components
            router = model.routers[router_layer]
            router_input = model.routers_input[router_layer]
            router_output = model.routers_output[router_layer]
            router_probs = model.router_probabilities[router_layer]
            
            # Get num_experts from router weights
            num_experts = router.weight.shape[1]
            
            # Test all tensor shapes
            assert router_input.shape == (batch_size, seq_len, router.weight.shape[0]), \
                f"Router input shape {router_input.shape} should be ({batch_size}, {seq_len}, {router.weight.shape[0]})"
            
            assert router_output.shape == (batch_size, seq_len, num_experts), \
                f"Router outputs shape {router_output.shape} should be ({batch_size}, {seq_len}, {num_experts})"
            
            assert router_probs.shape == router_output.shape, \
                f"Router probabilities shape should match output shape"
            
            # Validate num_experts consistency
            assert router_output.shape[2] == num_experts, \
                f"Router output experts dimension ({router_output.shape[2]}) should match router weight experts ({num_experts})"
            assert router_probs.shape[2] == num_experts, \
                f"Router probabilities experts dimension ({router_probs.shape[2]}) should match router weight experts ({num_experts})"
            
            # top_k should be reasonable relative to num_experts
            top_k = model.router_probabilities.get_top_k()
            assert top_k <= num_experts, f"top_k ({top_k}) should not exceed num_experts ({num_experts})"
            assert top_k > 0, f"top_k ({top_k}) should be positive"
            
            # Hidden size should match model's hidden size if available
            if model.hidden_size is not None:
                assert router.weight.shape[0] == model.hidden_size, \
                    f"Router input dimension should match hidden size"
