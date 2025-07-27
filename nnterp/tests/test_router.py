import torch as th
import pytest
from nnterp import StandardizedTransformer
from nnterp.rename_utils import RenamingError, RenameConfig
from nnterp.tests.utils import TEST_MODELS


def get_moe_models():
    """Get list of MoE models for testing."""
    moe_models = [
        "yujiepan/mixtral-8xtiny-random",
        "yujiepan/qwen1.5-moe-tiny-random", 
        "yujiepan/qwen3-moe-tiny-random",
    ]
    # Filter to only include models that are in TEST_MODELS
    return [model for model in moe_models if model in TEST_MODELS]


def get_non_moe_models():
    """Get list of non-MoE models for testing."""
    non_moe_models = [
        "gpt2",
        "bigscience/bigscience-small-testing",
        "yujiepan/opt-tiny-2layers-random",
    ]
    return [model for model in non_moe_models if model in TEST_MODELS]


@pytest.mark.parametrize("model_name", get_moe_models())
def test_router_detection_moe_models(model_name):
    """Test that router components are properly detected in MoE models."""
    with th.no_grad():
        model = StandardizedTransformer(model_name)
        
        # Router should be enabled for MoE models
        assert model.routers_available, f"Router should be available for MoE model {model_name}"
        assert model.routers.enabled, f"Router should be enabled for MoE model {model_name}"
        
        # Should be able to access router for first layer
        with model.trace("Hello world"):
            router = model.routers[0]
            assert router is not None, "Router should be accessible"


@pytest.mark.parametrize("model_name", get_non_moe_models())
def test_router_detection_non_moe_models(model_name):
    """Test that router components are properly disabled for non-MoE models."""
    with th.no_grad():
        model = StandardizedTransformer(model_name)
        
        # Router should be disabled for non-MoE models
        assert not model.routers_available, f"Router should not be available for non-MoE model {model_name}"
        assert not model.routers.enabled, f"Router should be disabled for non-MoE model {model_name}"
        
        # Accessing router should raise an error
        with pytest.raises(RenamingError, match="Router access is disabled"):
            with model.trace("Hello world"):
                _ = model.routers[0]


@pytest.mark.parametrize("model_name", get_moe_models())
def test_router_weights_access(model_name):
    """Test accessing router weights."""
    with th.no_grad():
        model = StandardizedTransformer(model_name)
        
        if not model.routers_available:
            pytest.skip(f"Router not available for {model_name}")
        
        with model.trace("Hello world"):
            # Test router weights access
            router_weights = model.routers.router_weights[0]
            assert isinstance(router_weights, th.Tensor), "Router weights should be a tensor"
            assert router_weights.ndim >= 2, "Router weights should be at least 2D"
            
            # Test that weights have reasonable shape (hidden_size x num_experts)
            assert router_weights.shape[0] > 0, "Router weights should have positive first dimension"
            assert router_weights.shape[1] > 0, "Router weights should have positive second dimension"


@pytest.mark.parametrize("model_name", get_moe_models())
def test_router_outputs_access(model_name):
    """Test accessing router outputs."""
    with th.no_grad():
        model = StandardizedTransformer(model_name)
        
        if not model.routers_available:
            pytest.skip(f"Router not available for {model_name}")
        
        with model.trace("Hello world"):
            # Test router outputs access
            router_outputs = model.routers.router_outputs[0]
            assert isinstance(router_outputs, th.Tensor), "Router outputs should be a tensor"
            
            # Router outputs should have shape (batch_size, seq_len, num_experts)
            batch_size, seq_len = model.input_size
            assert router_outputs.shape[0] == batch_size, f"Router output batch size should be {batch_size}"
            assert router_outputs.shape[1] == seq_len, f"Router output sequence length should be {seq_len}"
            assert router_outputs.shape[2] > 0, "Router outputs should have positive number of experts"


@pytest.mark.parametrize("model_name", get_moe_models())
def test_router_top_k_parameter(model_name):
    """Test accessing the top_k parameter."""
    with th.no_grad():
        model = StandardizedTransformer(model_name)
        
        if not model.routers_available:
            pytest.skip(f"Router not available for {model_name}")
        
        # Test getting top_k parameter
        top_k = model.routers.get_top_k()
        assert isinstance(top_k, int), "top_k should be an integer"
        assert top_k > 0, "top_k should be positive"
        assert top_k <= 8, "top_k should be reasonable (≤ 8 for typical MoE models)"
        
        # Test getting top_k for different layers
        if model.num_layers > 1:
            top_k_layer_1 = model.routers.get_top_k(1)
            assert top_k_layer_1 == top_k, "top_k should be consistent across layers"


@pytest.mark.parametrize("model_name", get_moe_models())
def test_router_structure_validation(model_name):
    """Test router structure validation."""
    with th.no_grad():
        model = StandardizedTransformer(model_name)
        
        if not model.routers_available:
            pytest.skip(f"Router not available for {model_name}")
        
        # Structure validation should not raise errors
        try:
            model.routers.check_router_structure()
        except RenamingError as e:
            pytest.fail(f"Router structure validation failed: {e}")


def test_router_custom_naming():
    """Test router with custom naming configuration."""
    # Test with custom router name
    custom_config = RenameConfig(router_name="custom_gate")
    
    # This test is more conceptual since we don't have models with custom naming
    # But we can test that the configuration is accepted
    with th.no_grad():
        model = StandardizedTransformer("gpt2", rename_config=custom_config)
        # Should not crash, router should be disabled for non-MoE model
        assert not model.routers_available


def test_router_ignore_configuration():
    """Test router with ignore configuration."""
    # Test with router ignored
    ignore_config = RenameConfig(ignore_router=True)
    
    moe_models = get_moe_models()
    if not moe_models:
        pytest.skip("No MoE models available for testing")
    
    with th.no_grad():
        model = StandardizedTransformer(moe_models[0], rename_config=ignore_config)
        # Router should be disabled even for MoE model when ignored
        assert not model.routers_available


@pytest.mark.parametrize("model_name", get_moe_models())
def test_router_multiple_layers(model_name):
    """Test router access across multiple layers."""
    with th.no_grad():
        model = StandardizedTransformer(model_name)
        
        if not model.routers_available:
            pytest.skip(f"Router not available for {model_name}")
        
        with model.trace("Hello world"):
            # Test accessing routers in multiple layers
            for layer_idx in range(min(3, model.num_layers)):  # Test first 3 layers
                router = model.routers[layer_idx]
                assert router is not None, f"Router should be accessible at layer {layer_idx}"
                
                router_weights = model.routers.router_weights[layer_idx]
                assert isinstance(router_weights, th.Tensor), f"Router weights should be tensor at layer {layer_idx}"
                
                router_outputs = model.routers.router_outputs[layer_idx]
                assert isinstance(router_outputs, th.Tensor), f"Router outputs should be tensor at layer {layer_idx}"


@pytest.mark.parametrize("model_name", get_moe_models())
def test_router_tensor_shapes_consistency(model_name):
    """Test that router tensor shapes are consistent and reasonable."""
    with th.no_grad():
        model = StandardizedTransformer(model_name)
        
        if not model.routers_available:
            pytest.skip(f"Router not available for {model_name}")
        
        with model.trace("Hello world"):
            batch_size, seq_len = model.input_size
            
            # Get router components
            router_weights = model.routers.router_weights[0]
            router_outputs = model.routers.router_outputs[0]
            top_k = model.routers.get_top_k()
            
            # Validate shapes
            assert router_weights.ndim == 2, "Router weights should be 2D"
            hidden_size, num_experts = router_weights.shape
            
            assert router_outputs.shape == (batch_size, seq_len, num_experts), \
                f"Router outputs shape {router_outputs.shape} should be ({batch_size}, {seq_len}, {num_experts})"
            
            # top_k should be reasonable relative to num_experts
            assert top_k <= num_experts, f"top_k ({top_k}) should not exceed num_experts ({num_experts})"
            
            # Hidden size should match model's hidden size if available
            if model.hidden_size is not None:
                assert hidden_size == model.hidden_size, \
                    f"Router input size ({hidden_size}) should match model hidden size ({model.hidden_size})"


def test_router_error_handling():
    """Test error handling for router access."""
    with th.no_grad():
        # Test with non-MoE model
        model = StandardizedTransformer("gpt2")
        
        # Should raise appropriate errors
        with pytest.raises(RenamingError, match="Router access is disabled"):
            model.routers.get_top_k()
        
        with pytest.raises(RenamingError, match="Router access is disabled"):
            model.routers.check_router_structure()
        
        with pytest.raises(RenamingError, match="Router access is disabled"):
            with model.trace("Hello world"):
                _ = model.routers[0]
